import copy

import torch
import torch.nn.functional as F

from src.plugins.base import BaseClientPlugin, BaseServerPlugin
from src.plugins.fedfed_modules import build_fedfed_generator


class FedFedImageClientPlugin(BaseClientPlugin):
    def __init__(self, options, model, device):
        self.options = options
        self.model = model
        self.device = device
        self.distill_classifier = copy.deepcopy(model).to(device)
        self.input_channels = self._resolve_input_channels()
        self.generator = build_fedfed_generator(
            options.get('fedfed_generator_type', 'beta_vae'),
            self.input_channels,
            latent_channels=int(options.get('fedfed_vae_latent_channels', 64)),
            z_dim=int(options.get('fedfed_vae_z_dim', 2048)),
        ).to(device)
        self.model_optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=options.get('lr', 0.001),
        )
        self.distill_optimizer = self._build_distill_optimizer()
        self.shared_x = None
        self.shared_y = None
        self.current_round = 0
        self.upload_x = []
        self.upload_y = []
        self.upload_counts = {}
        self.in_warmup = False
        self.last_distill_stats = {}

    def _resolve_input_channels(self):
        dataset_name = str(self.options.get('dataset_name', '')).lower()
        if dataset_name.startswith('mnist'):
            return 1
        if dataset_name in {'cifar10', 'cifar-10'}:
            return 3
        return int(self.options.get('fedfed_input_channels', 3))

    def to_device(self, device):
        self.device = device
        self.distill_classifier.to(device)
        self.generator.to(device)
        self._move_optimizer_state(self.model_optimizer, device)
        self._move_optimizer_state(self.distill_optimizer, device)
        if self.shared_x is not None:
            self.shared_x = self.shared_x.to(device)
        if self.shared_y is not None:
            self.shared_y = self.shared_y.to(device)

    def _move_optimizer_state(self, optimizer, device):
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)

    def _set_optimizer_lr(self, learning_rate):
        for group in self.model_optimizer.param_groups:
            group['lr'] = learning_rate
        for group in self.distill_optimizer.param_groups:
            group['lr'] = float(self.options.get('fedfed_distill_lr', learning_rate))

    def _build_distill_optimizer(self):
        params = list(self.distill_classifier.parameters()) + list(self.generator.parameters())
        lr = float(self.options.get('fedfed_distill_lr', self.options.get('lr', 0.001)))
        weight_decay = float(self.options.get(
            'fedfed_distill_weight_decay',
            self.options.get('fedfed_generator_weight_decay', 0.0),
        ))
        optimizer_name = str(self.options.get('fedfed_distill_optimizer', 'adamw')).lower()
        if optimizer_name == 'sgd':
            return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
        if optimizer_name == 'adam':
            return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)

    def on_round_start(self, learning_rate, server_payload):
        self._set_optimizer_lr(learning_rate)
        self.model.train()
        self.distill_classifier.eval()
        self.generator.train()
        self.shared_x = None
        self.shared_y = None
        if server_payload is not None:
            self.current_round = int(server_payload.get('round_index', self.current_round))
            distill_classifier_state = server_payload.get('distill_classifier_state')
            if distill_classifier_state is not None:
                self.distill_classifier.load_state_dict(
                    {key: value.to(self.device) for key, value in distill_classifier_state.items()},
                    strict=True,
                )
            generator_state = server_payload.get('generator_state')
            if generator_state is not None:
                self.generator.load_state_dict(
                    {key: value.to(self.device) for key, value in generator_state.items()},
                    strict=True,
                )
            shared_x = server_payload.get('shared_x')
            shared_y = server_payload.get('shared_y')
            if shared_x is not None and shared_y is not None and len(shared_y) > 0:
                self.shared_x = shared_x.to(self.device)
                self.shared_y = shared_y.to(self.device)
        warmup_rounds = int(self.options.get('fedfed_hard_warmup_rounds', 10))
        self.in_warmup = (
            self.current_round < warmup_rounds
            and not bool(self.options.get('fedfed_two_stage', True))
        )
        self.upload_x = []
        self.upload_y = []
        self.upload_counts = {}

    def _sensitive_feature(self, x):
        robust = self.generator(x)
        return self._clip_sensitive_feature(x, x - robust)

    def _raw_sensitive_feature(self, x):
        robust = self.generator(x)
        return x - robust

    def _clip_sensitive_feature(self, x, xs):
        rho = float(self.options.get('fedfed_rho', 0.0))
        if rho <= 0:
            return xs
        xs_flat = xs.flatten(1)
        x_flat = x.flatten(1)
        xs_norm = xs_flat.norm(p=2, dim=1).clamp_min(1e-12)
        x_budget = rho * x_flat.norm(p=2, dim=1).clamp_min(1e-12)
        scale = torch.minimum(torch.ones_like(xs_norm), x_budget / xs_norm)
        return xs * scale.view(-1, *([1] * (xs.dim() - 1)))

    def _rho_violation(self, x, xs):
        rho = float(self.options.get('fedfed_rho', 0.0))
        if rho <= 0:
            return xs.new_tensor(0.0)
        xs_norm = xs.flatten(1).norm(p=2, dim=1)
        x_budget = rho * x.flatten(1).norm(p=2, dim=1).detach()
        return F.relu(xs_norm - x_budget).pow(2).mean()

    def _distill_objective(self, X, y):
        xs_raw = self._raw_sensitive_feature(X)
        robust = X - xs_raw
        xs = self._clip_sensitive_feature(X, xs_raw)
        pred_sensitive = self.distill_classifier(xs)
        loss_fd = F.cross_entropy(pred_sensitive, y)
        loss_recon = F.mse_loss(robust, X)
        pred_raw = self.distill_classifier(X)
        loss_x_ce = F.cross_entropy(pred_raw, y)
        rho_penalty = self._rho_violation(X, xs_raw)
        kl_loss = self.generator.last_kl
        loss = float(self.options.get('fedfed_lambda_fd', 1.0)) * loss_fd
        loss = loss + float(self.options.get('fedfed_lambda_recon', 0.0)) * loss_recon
        loss = loss + float(self.options.get('fedfed_lambda_x_ce', 0.0)) * loss_x_ce
        loss = loss + float(self.options.get('fedfed_lambda_rho', 10.0)) * rho_penalty
        if kl_loss is not None:
            loss = loss + float(self.options.get('fedfed_beta_kl', 0.001)) * kl_loss
        return pred_sensitive, loss, {
            'fd_loss': loss_fd,
            'recon_loss': loss_recon,
            'x_ce_loss': loss_x_ce,
            'rho_penalty': rho_penalty,
            'xs_norm': xs.flatten(1).norm(p=2, dim=1).mean(),
            'kl_loss': kl_loss if kl_loss is not None else xs.new_tensor(0.0),
        }

    def _sample_shared_batch(self, batch_size):
        if self.shared_x is None or self.shared_y is None or len(self.shared_y) == 0:
            return None, None
        sample_size = min(
            int(self.options.get('fedfed_shared_batch_size', batch_size)),
            len(self.shared_y),
        )
        if sample_size <= 0:
            return None, None
        indices = torch.randint(0, len(self.shared_y), (sample_size,), device=self.device)
        return self.shared_x[indices], self.shared_y[indices]

    def _maybe_collect_upload_samples(self, xs, y):
        per_class_limit = int(self.options.get('fedfed_upload_per_class', 4))
        total_limit = int(self.options.get('fedfed_upload_per_client', 40))
        if per_class_limit <= 0 or total_limit <= 0:
            return
        xs = xs.detach().cpu()
        y = y.detach().cpu()
        order = torch.randperm(y.numel()).tolist()
        for idx in order:
            if len(self.upload_y) >= total_limit:
                break
            class_id = int(y[idx].item())
            if self.upload_counts.get(class_id, 0) >= per_class_limit:
                continue
            self.upload_x.append(xs[idx].clone())
            self.upload_y.append(y[idx].clone())
            self.upload_counts[class_id] = self.upload_counts.get(class_id, 0) + 1

    def train_batch(self, X, y):
        self.model_optimizer.zero_grad()

        pred_local = self.model(X)
        loss = F.cross_entropy(pred_local, y)

        use_online_distill = (
            not bool(self.options.get('fedfed_two_stage', True))
            or bool(self.options.get('fedfed_formal_online_distill', False))
        )
        if use_online_distill:
            self.distill_optimizer.zero_grad()
            _, distill_loss, _ = self._distill_objective(X, y)
            distill_loss.backward()
            self.distill_optimizer.step()

        shared_x, shared_y = (None, None) if self.in_warmup else self._sample_shared_batch(X.size(0))
        if shared_x is not None:
            if str(self.options.get('fedfed_shared_mix_mode', 'concat')).lower() == 'concat':
                mixed_x = torch.cat([X, shared_x], dim=0)
                mixed_y = torch.cat([y, shared_y], dim=0)
                pred_mixed = self.model(mixed_x)
                loss = F.cross_entropy(pred_mixed, mixed_y)
                pred_local = pred_mixed[:X.size(0)]
            else:
                pred_shared = self.model(shared_x)
                loss_shared = F.cross_entropy(pred_shared, shared_y)
                loss = loss + float(self.options.get('fedfed_lambda_shared', 1.0)) * loss_shared

        if not self.in_warmup and not bool(self.options.get('fedfed_two_stage', True)):
            xs = self._sensitive_feature(X)
            self._maybe_collect_upload_samples(xs, y)
        loss.backward()
        self.model_optimizer.step()
        return pred_local, loss.detach()

    def on_distill_start(self, learning_rate, server_payload):
        self.on_round_start(learning_rate, server_payload)
        self.in_warmup = True
        self.model.eval()
        self.distill_classifier.train()

    def distill_batch(self, X, y):
        self.distill_optimizer.zero_grad()
        pred_sensitive, loss, stats = self._distill_objective(X, y)
        loss.backward()
        self.distill_optimizer.step()
        self.last_distill_stats = {
            'fd_loss': float(stats['fd_loss'].detach().item()),
            'recon_loss': float(stats['recon_loss'].detach().item()),
            'x_ce_loss': float(stats['x_ce_loss'].detach().item()),
            'rho_penalty': float(stats['rho_penalty'].detach().item()),
            'xs_norm': float(stats['xs_norm'].detach().item()),
            'kl_loss': float(stats['kl_loss'].detach().item()),
        }
        return pred_sensitive, loss.detach()

    def collect_shared_batch(self, X, y):
        with torch.no_grad():
            xs = self._sensitive_feature(X)
        self._maybe_collect_upload_samples(xs, y)

    def build_upload_payload(self):
        payload = {
            'generator_state': {
                key: value.detach().cpu().clone()
                for key, value in self.generator.state_dict().items()
            },
            'distill_classifier_state': {
                key: value.detach().cpu().clone()
                for key, value in self.distill_classifier.state_dict().items()
            }
        }
        if self.upload_y:
            payload['sensitive_x'] = torch.stack(self.upload_x, dim=0)
            payload['sensitive_y'] = torch.stack(self.upload_y, dim=0).long()
        return payload


class FedFedImageServerPlugin(BaseServerPlugin):
    def __init__(self, options, device):
        self.options = options
        self.device = device
        self.current_round = 0
        self.generator_state = None
        self.distill_classifier_state = None
        self.shared_by_class = {}

    def set_round_index(self, round_index):
        self.current_round = int(round_index)

    def build_broadcast_payload(self):
        payload = {'round_index': self.current_round}
        if self.generator_state is not None:
            payload['generator_state'] = self.generator_state
        warmup_rounds = int(self.options.get('fedfed_hard_warmup_rounds', 10))
        shared_x, shared_y = self._flatten_shared_buffer()
        shared_ready = (
            bool(self.options.get('fedfed_two_stage', True))
            or self.current_round >= warmup_rounds
        )
        if shared_ready and shared_y:
            payload['shared_x'] = torch.stack(shared_x, dim=0)
            payload['shared_y'] = torch.tensor(shared_y, dtype=torch.long)
        return payload

    def build_feature_distill_payload(self):
        payload = {'round_index': self.current_round}
        if self.distill_classifier_state is not None:
            payload['distill_classifier_state'] = self.distill_classifier_state
        if self.generator_state is not None:
            payload['generator_state'] = self.generator_state
        return payload

    def reset_shared_buffer(self):
        self.shared_by_class = {}

    def aggregate_generator_states(self, local_model_paras_set):
        self._aggregate_generator(local_model_paras_set)
        self._aggregate_distill_classifier(local_model_paras_set)

    def collect_shared_payloads(self, local_model_paras_set):
        self._update_shared_buffer(local_model_paras_set, force=True)

    def aggregate_client_payloads(self, local_model_paras_set):
        self._aggregate_generator(local_model_paras_set)
        self._update_shared_buffer(local_model_paras_set)

    def _aggregate_generator(self, local_model_paras_set):
        state_sums = {}
        total_weight = 0
        for update in local_model_paras_set:
            aux = update.get('aux')
            if not aux or 'generator_state' not in aux:
                continue
            weight = int(update.get('num_samples', 1))
            for key, value in aux['generator_state'].items():
                value = value.detach().cpu()
                if not value.is_floating_point():
                    state_sums[key] = value.clone()
                    continue
                if key not in state_sums:
                    state_sums[key] = value.clone() * weight
                else:
                    state_sums[key] += value * weight
            total_weight += weight
        if not state_sums or total_weight <= 0:
            return
        self.generator_state = {
            key: ((value / total_weight) if value.is_floating_point() else value).to(self.device)
            for key, value in state_sums.items()
        }

    def _aggregate_distill_classifier(self, local_model_paras_set):
        state_sums = {}
        total_weight = 0
        for update in local_model_paras_set:
            aux = update.get('aux')
            if not aux or 'distill_classifier_state' not in aux:
                continue
            weight = int(update.get('num_samples', 1))
            for key, value in aux['distill_classifier_state'].items():
                value = value.detach().cpu()
                if not value.is_floating_point():
                    state_sums[key] = value.clone()
                    continue
                if key not in state_sums:
                    state_sums[key] = value.clone() * weight
                else:
                    state_sums[key] += value * weight
            total_weight += weight
        if not state_sums or total_weight <= 0:
            return
        self.distill_classifier_state = {
            key: ((value / total_weight) if value.is_floating_point() else value).to(self.device)
            for key, value in state_sums.items()
        }

    def _update_shared_buffer(self, local_model_paras_set, force=False):
        warmup_rounds = int(self.options.get('fedfed_hard_warmup_rounds', 10))
        if not force and self.current_round < warmup_rounds:
            return
        max_size = int(self.options.get('fedfed_shared_buffer_size', 800))
        if max_size <= 0:
            self.shared_by_class = {}
            return
        per_class_size = int(self.options.get('fedfed_shared_per_class_size', 80))
        num_classes = max(int(self.options.get('fedfed_num_classes', 10)), 1)
        if per_class_size <= 0:
            per_class_size = max(max_size // num_classes, 1)
        for update in local_model_paras_set:
            aux = update.get('aux')
            if not aux or 'sensitive_x' not in aux:
                continue
            xs = aux['sensitive_x'].detach().cpu()
            ys = aux['sensitive_y'].detach().cpu().long()
            for x_item, y_item in zip(xs, ys):
                class_id = int(y_item.item())
                bucket = self.shared_by_class.setdefault(class_id, [])
                bucket.append(x_item.clone())
                if len(bucket) > per_class_size:
                    del bucket[:len(bucket) - per_class_size]
        self._trim_global_overflow(max_size)

    def _flatten_shared_buffer(self):
        shared_x = []
        shared_y = []
        for class_id in sorted(self.shared_by_class):
            for x_item in self.shared_by_class[class_id]:
                shared_x.append(x_item)
                shared_y.append(class_id)
        return shared_x, shared_y

    def _trim_global_overflow(self, max_size):
        total = sum(len(bucket) for bucket in self.shared_by_class.values())
        if total <= max_size:
            return
        overflow = total - max_size
        class_ids = sorted(self.shared_by_class)
        while overflow > 0 and class_ids:
            changed = False
            for class_id in list(class_ids):
                bucket = self.shared_by_class.get(class_id, [])
                if bucket:
                    del bucket[0]
                    overflow -= 1
                    changed = True
                    if overflow <= 0:
                        break
                if not bucket:
                    self.shared_by_class.pop(class_id, None)
                    class_ids = sorted(self.shared_by_class)
            if not changed:
                break
