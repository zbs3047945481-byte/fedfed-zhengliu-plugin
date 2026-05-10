import copy
import random

import torch
import torch.nn.functional as F

from src.plugins.base import BaseClientPlugin, BaseServerPlugin
from src.plugins.fedfed_modules import build_fedfed_generator
from src.optimizers.build import build_main_optimizer


def _batch_random_crop(x, padding=4):
    if padding <= 0:
        return x
    padded = F.pad(x, (padding, padding, padding, padding), mode='reflect')
    output = torch.empty_like(x)
    height, width = x.shape[-2:]
    max_offset = padding * 2
    for idx in range(x.size(0)):
        top = torch.randint(0, max_offset + 1, (1,), device=x.device).item()
        left = torch.randint(0, max_offset + 1, (1,), device=x.device).item()
        output[idx] = padded[idx, :, top:top + height, left:left + width]
    return output


def _batch_horizontal_flip(x, probability=0.5):
    mask = torch.rand(x.size(0), device=x.device) < probability
    if not mask.any():
        return x
    output = x.clone()
    output[mask] = torch.flip(output[mask], dims=(-1,))
    return output


def _train_augment(x, enabled=True):
    if not enabled:
        return x
    return _batch_horizontal_flip(_batch_random_crop(x, padding=4), probability=0.5)


def _mixup_data(x, y, alpha):
    if alpha <= 0.0 or x.size(0) < 2:
        return x, y, y, 1.0
    lam = torch.distributions.Beta(alpha, alpha).sample().to(x.device).item()
    indices = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[indices]
    return mixed_x, y, y[indices], lam


def _cutout(image, length):
    _, height, width = image.shape
    center_y = random.randint(0, height - 1)
    center_x = random.randint(0, width - 1)
    y1 = max(center_y - length // 2, 0)
    y2 = min(center_y + length // 2, height)
    x1 = max(center_x - length // 2, 0)
    x2 = min(center_x + length // 2, width)
    image[:, y1:y2, x1:x2] = 0.0
    return image


def _mosaic_batch(x, output_count):
    if x.size(0) < 4:
        return x
    output_count = x.size(0) if output_count <= 0 else output_count
    channels, height, width = x.shape[1:]
    half_h = height // 2
    half_w = width // 2
    output = torch.empty((output_count, channels, height, width), device=x.device, dtype=x.dtype)
    for out_idx in range(output_count):
        sample_ids = torch.randperm(x.size(0), device=x.device)[:4]
        patch_top = torch.randint(0, height - half_h + 1, (4,), device=x.device)
        patch_left = torch.randint(0, width - half_w + 1, (4,), device=x.device)
        image = torch.zeros((channels, height, width), device=x.device, dtype=x.dtype)
        slots = ((0, half_h, 0, half_w), (0, half_h, half_w, width),
                 (half_h, height, 0, half_w), (half_h, height, half_w, width))
        for slot_idx, sample_id in enumerate(sample_ids):
            y1, y2, x1, x2 = slots[slot_idx]
            top = int(patch_top[slot_idx].item())
            left = int(patch_left[slot_idx].item())
            image[:, y1:y2, x1:x2] = x[sample_id, :, top:top + half_h, left:left + half_w]
        output[out_idx] = _cutout(image, length=min(half_h, half_w))
    return output


class FedFedImageClientPlugin(BaseClientPlugin):
    def __init__(self, options, model, device):
        self.options = options
        self.model = model
        self.device = device
        self.distill_classifier = copy.deepcopy(model).to(device)
        self.input_channels = self._resolve_input_channels()
        self.generator = build_fedfed_generator(
            options.get('fedfed_generator_type', 'paper_beta_vae'),
            self.input_channels,
            latent_channels=int(options.get('fedfed_vae_latent_channels', 32)),
            z_dim=int(options.get('fedfed_vae_z_dim', 2048)),
        ).to(device)
        self.model_optimizer = build_main_optimizer(self.model.parameters(), options)
        self.distill_optimizer = self._build_distill_optimizer()
        self.shared_x1 = None
        self.shared_x2 = None
        self.shared_y = None
        self.current_round = 0
        self.upload_x = []
        self.upload_y = []
        self.upload_counts = {}
        self.in_warmup = False
        self.last_distill_stats = {}
        self.distill_epoch = 1

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
        if self._shared_resident_device() == 'cuda' and device.type == 'cuda':
            if self.shared_x1 is not None:
                self.shared_x1 = self.shared_x1.to(device, non_blocking=True)
            if self.shared_x2 is not None:
                self.shared_x2 = self.shared_x2.to(device, non_blocking=True)
            if self.shared_y is not None:
                self.shared_y = self.shared_y.to(device, non_blocking=True)

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
            0.0,
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
        self.shared_x1 = None
        self.shared_x2 = None
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
            shared_x1 = server_payload.get('shared_x1')
            shared_x2 = server_payload.get('shared_x2')
            shared_y = server_payload.get('shared_y')
            if shared_x1 is None:
                shared_x1 = server_payload.get('shared_x')
            if shared_x2 is None:
                shared_x2 = shared_x1
            if shared_x1 is not None and shared_y is not None and len(shared_y) > 0:
                resident_device = self.device if self._shared_resident_device() == 'cuda' and self.device.type == 'cuda' else torch.device('cpu')
                self.shared_x1 = shared_x1.detach().to(resident_device, non_blocking=True)
                self.shared_x2 = shared_x2.detach().to(resident_device, non_blocking=True)
                self.shared_y = shared_y.detach().to(resident_device, non_blocking=True).long()
        self.in_warmup = False
        self.upload_x = []
        self.upload_y = []
        self.upload_counts = {}

    def _sensitive_feature(self, x):
        robust = self.generator(x)
        return x - robust

    def _raw_sensitive_feature(self, x):
        robust = self.generator(x)
        return x - robust

    def _add_shared_noise(self, x_item, std_key):
        noise_type = str(self.options.get('fedfed_noise_type', 'gaussian')).lower()
        std = float(self.options.get(std_key, 0.0))
        mean = float(self.options.get('fedfed_noise_mean', 0.0))
        if noise_type == 'none' or std <= 0.0:
            return x_item.clone()
        if noise_type == 'gaussian':
            noise = torch.normal(mean=mean, std=std, size=x_item.shape, device=x_item.device)
        elif noise_type == 'laplace':
            dist = torch.distributions.Laplace(
                torch.tensor(mean, dtype=x_item.dtype, device=x_item.device),
                torch.tensor(std, dtype=x_item.dtype, device=x_item.device),
            )
            noise = dist.sample(x_item.shape)
        else:
            raise ValueError('Unsupported fedfed_noise_type: {}'.format(noise_type))
        return x_item.clone() + noise.to(dtype=x_item.dtype)

    def _shared_resident_device(self):
        return str(self.options.get('fedfed_shared_resident_device', 'cpu')).lower()

    def _shared_noise_disabled(self):
        return (
            str(self.options.get('fedfed_noise_type', 'gaussian')).lower() == 'none'
            or (
                float(self.options.get('fedfed_noise_std1', 0.0)) <= 0.0
                and float(self.options.get('fedfed_noise_std2', 0.0)) <= 0.0
            )
        )

    def _collapse_duplicate_shared_views(self):
        return (
            bool(self.options.get('fedfed_collapse_duplicate_shared_no_noise', False))
            and self._shared_noise_disabled()
        )

    def _distill_objective(self, X, y):
        X = _train_augment(X, bool(self.options.get('fedfed_use_augmentation', True)))
        xs_raw = self._raw_sensitive_feature(X)
        robust = X - xs_raw
        xs = xs_raw
        if bool(self.options.get('fedfed_distill_ce_on_noisy_xs', False)):
            xs1 = self._add_shared_noise(xs, 'fedfed_noise_std1')
            xs2 = self._add_shared_noise(xs, 'fedfed_noise_std2')
            pred_sensitive1, feature_sensitive1 = self.distill_classifier(xs1, return_feature=True)
            pred_sensitive2 = self.distill_classifier(xs2)
            pred_sensitive = torch.cat([pred_sensitive1, pred_sensitive2], dim=0)
            loss_fd = F.cross_entropy(pred_sensitive, y.repeat(2))
        else:
            xs1 = xs
            pred_sensitive1, feature_sensitive1 = self.distill_classifier(xs1, return_feature=True)
            loss_fd = F.cross_entropy(pred_sensitive1, y)
        loss_recon = F.mse_loss(robust, X)
        pred_raw, feature_raw = self.distill_classifier(X, return_feature=True)
        loss_x_ce = F.cross_entropy(pred_raw, y)
        loss_align = F.mse_loss(feature_sensitive1, feature_raw.detach())
        temperature = max(float(self.options.get('fedfed_logit_align_temperature', 2.0)), 1e-6)
        teacher_prob = F.softmax(pred_raw.detach() / temperature, dim=1)
        student_log_prob = F.log_softmax(pred_sensitive1 / temperature, dim=1)
        loss_logit_align = F.kl_div(student_log_prob, teacher_prob, reduction='batchmean') * (temperature ** 2)
        kl_loss = self.generator.last_kl
        loss = float(self.options.get('fedfed_lambda_fd', 1.0)) * loss_fd
        loss = loss + self._reconstruction_weight(self.distill_epoch) * loss_recon
        loss = loss + float(self.options.get('fedfed_lambda_x_ce', 0.0)) * loss_x_ce
        loss = loss + float(self.options.get('fedfed_lambda_align', 0.0)) * loss_align
        loss = loss + float(self.options.get('fedfed_lambda_logit_align', 0.0)) * loss_logit_align
        if kl_loss is not None:
            loss = loss + float(self.options.get('fedfed_beta_kl', 0.001)) * kl_loss
        return pred_sensitive1, loss, {
            'fd_loss': loss_fd,
            'recon_loss': loss_recon,
            'x_ce_loss': loss_x_ce,
            'align_loss': loss_align,
            'logit_align_loss': loss_logit_align,
            'xs_norm': xs.flatten(1).norm(p=2, dim=1).mean(),
            'kl_loss': kl_loss if kl_loss is not None else xs.new_tensor(0.0),
        }

    def _reconstruction_weight(self, epoch=None):
        weight = float(self.options.get('fedfed_lambda_recon', 0.0))
        if bool(self.options.get('fedfed_vae_curriculum', True)):
            epoch = self.distill_epoch if epoch is None else int(epoch)
            if epoch < 10:
                return 10.0 * weight
            if epoch < 20:
                return 5.0 * weight
        return weight

    def _sample_shared_batch(self, batch_size):
        if self.shared_x1 is None or self.shared_x2 is None or self.shared_y is None or len(self.shared_y) == 0:
            return None, None, None
        configured_size = int(self.options.get('fedfed_shared_batch_size', 0))
        sample_size = min(configured_size if configured_size > 0 else batch_size, len(self.shared_y))
        if sample_size <= 0:
            return None, None, None
        indices = self._balanced_shared_indices(sample_size)
        shared_x1 = self.shared_x1[indices].to(self.device, non_blocking=True).float()
        shared_x2 = self.shared_x2[indices].to(self.device, non_blocking=True).float()
        shared_y = self.shared_y[indices].to(self.device, non_blocking=True).long()
        return shared_x1, shared_x2, shared_y

    def _balanced_shared_indices(self, sample_size):
        classes = torch.unique(self.shared_y)
        if classes.numel() == 0:
            return torch.randint(0, len(self.shared_y), (sample_size,), device=self.shared_y.device)
        per_class = sample_size // classes.numel()
        remainder = sample_size - per_class * classes.numel()
        selected = []
        for position, class_id in enumerate(classes.tolist()):
            class_indices = torch.nonzero(self.shared_y == int(class_id), as_tuple=False).flatten()
            take = per_class + (1 if position < remainder else 0)
            if take <= 0 or class_indices.numel() == 0:
                continue
            draw = torch.randint(0, class_indices.numel(), (take,), device=class_indices.device)
            selected.append(class_indices[draw])
        if not selected:
            return torch.randint(0, len(self.shared_y), (sample_size,), device=self.shared_y.device)
        indices = torch.cat(selected, dim=0)
        if indices.numel() < sample_size:
            extra = torch.randint(0, len(self.shared_y), (sample_size - indices.numel(),), device=indices.device)
            indices = torch.cat([indices, extra], dim=0)
        return indices[torch.randperm(indices.numel(), device=indices.device)]

    def _maybe_collect_upload_samples(self, xs, y):
        per_class_limit = int(self.options.get('fedfed_upload_per_class', 0))
        total_limit = int(self.options.get('fedfed_upload_per_client', 0))
        xs = xs.detach().cpu()
        y = y.detach().cpu()
        order = torch.randperm(y.numel()).tolist()
        for idx in order:
            if total_limit > 0 and len(self.upload_y) >= total_limit:
                break
            class_id = int(y[idx].item())
            if per_class_limit > 0 and self.upload_counts.get(class_id, 0) >= per_class_limit:
                continue
            self.upload_x.append(xs[idx].clone())
            self.upload_y.append(y[idx].clone())
            self.upload_counts[class_id] = self.upload_counts.get(class_id, 0) + 1

    def train_batch(self, X, y):
        self.model_optimizer.zero_grad()
        X_train = _train_augment(X, bool(self.options.get('fedfed_use_augmentation', True)))
        shared_x1, shared_x2, shared_y = self._sample_shared_batch(X_train.size(0))
        if shared_x1 is not None:
            shared_x1 = _train_augment(shared_x1, bool(self.options.get('fedfed_use_augmentation', True)))
            if self._collapse_duplicate_shared_views():
                mixed_x = torch.cat((X_train, shared_x1), dim=0)
                pred_mixed = self.model(mixed_x)
                pred_local = pred_mixed[:X_train.size(0)]
                pred_shared = pred_mixed[X_train.size(0):]
                loss_local = F.cross_entropy(pred_local, y)
                loss_shared = F.cross_entropy(pred_shared, shared_y)
                loss = (loss_local + 2.0 * loss_shared) / 3.0
            else:
                shared_x2 = _train_augment(shared_x2, bool(self.options.get('fedfed_use_augmentation', True)))
                mixed_x = torch.cat((X_train, shared_x1, shared_x2), dim=0)
                mixed_y = torch.cat((y, shared_y, shared_y), dim=0)
                pred_mixed = self.model(mixed_x)
                loss = F.cross_entropy(pred_mixed, mixed_y)
                pred_local = pred_mixed[:X_train.size(0)]
            loss.backward()
        else:
            pred_local = self.model(X_train)
            loss = F.cross_entropy(pred_local, y)
            loss.backward()
        self.model_optimizer.step()
        return pred_local, loss.detach()

    def on_distill_start(self, learning_rate, server_payload):
        self.on_round_start(learning_rate, server_payload)
        self.in_warmup = True
        self.model.eval()
        self.distill_classifier.train()

    def set_distill_epoch(self, epoch):
        self.distill_epoch = int(epoch)

    def _aug_classifier_train_batch(self, X, y):
        if not bool(self.options.get('fedfed_use_augmentation', True)):
            return
        X = _train_augment(X, True)
        X, y_a, y_b, lam = _mixup_data(X, y, float(self.options.get('fedfed_mixup_alpha', 2.0)))
        self.distill_optimizer.zero_grad()
        for parameter in self.generator.parameters():
            parameter.requires_grad_(False)
        pred = self.distill_classifier(X)
        loss = lam * F.cross_entropy(pred, y_a) + (1.0 - lam) * F.cross_entropy(pred, y_b)
        loss.backward()
        self.distill_optimizer.step()
        for parameter in self.generator.parameters():
            parameter.requires_grad_(True)

    def _aug_vae_train_batch(self, X):
        if not bool(self.options.get('fedfed_use_augmentation', True)) or X.size(0) < 4:
            return
        X = _train_augment(X, True)
        mosaic_count = int(self.options.get('fedfed_mosaic_batch_size', 0))
        aug_x = _mosaic_batch(X, mosaic_count)
        self.distill_optimizer.zero_grad()
        robust = self.generator(aug_x)
        loss_recon = F.mse_loss(robust, aug_x)
        kl_loss = self.generator.last_kl
        loss = self._reconstruction_weight(self.distill_epoch) * loss_recon
        if kl_loss is not None:
            loss = loss + float(self.options.get('fedfed_beta_kl', 0.001)) * kl_loss
        loss.backward()
        self.distill_optimizer.step()

    def distill_batch(self, X, y):
        self._aug_classifier_train_batch(X, y)
        self._aug_vae_train_batch(X)
        self.distill_optimizer.zero_grad()
        pred_sensitive, loss, stats = self._distill_objective(X, y)
        loss.backward()
        self.distill_optimizer.step()
        self.last_distill_stats = {
            'fd_loss': float(stats['fd_loss'].detach().item()),
            'recon_loss': float(stats['recon_loss'].detach().item()),
            'x_ce_loss': float(stats['x_ce_loss'].detach().item()),
            'align_loss': float(stats['align_loss'].detach().item()),
            'logit_align_loss': float(stats['logit_align_loss'].detach().item()),
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

    @property
    def requires_pretraining(self):
        return True

    def set_round_index(self, round_index):
        self.current_round = int(round_index)

    def build_broadcast_payload(self):
        payload = {'round_index': self.current_round}
        if self.generator_state is not None:
            payload['generator_state'] = self.generator_state
        shared_x1, shared_x2, shared_y = self._flatten_shared_buffer()
        if shared_y:
            shared_x1_tensor = torch.stack(shared_x1, dim=0)
            shared_x2_tensor = torch.stack(shared_x2, dim=0)
            if bool(self.options.get('fedfed_shared_cpu_float16', True)):
                shared_x1_tensor = shared_x1_tensor.half()
                shared_x2_tensor = shared_x2_tensor.half()
            payload['shared_x1'] = shared_x1_tensor
            payload['shared_x2'] = shared_x2_tensor
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
        return

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
        max_size = int(self.options.get('fedfed_shared_buffer_size', 0))
        per_class_size = int(self.options.get('fedfed_shared_per_class_size', 0))
        num_classes = max(int(self.options.get('fedfed_num_classes', 10)), 1)
        if per_class_size <= 0 and max_size > 0:
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
                bucket.append((
                    self._add_shared_noise(x_item, 'fedfed_noise_std1'),
                    self._add_shared_noise(x_item, 'fedfed_noise_std2'),
                ))
                if per_class_size > 0 and len(bucket) > per_class_size:
                    del bucket[:len(bucket) - per_class_size]
        if max_size > 0:
            self._trim_global_overflow(max_size)

    def _add_shared_noise(self, x_item, std_key):
        noise_type = str(self.options.get('fedfed_noise_type', 'gaussian')).lower()
        std = float(self.options.get(std_key, 0.0))
        mean = float(self.options.get('fedfed_noise_mean', 0.0))
        if noise_type == 'none' or std <= 0.0:
            return x_item.clone()
        if noise_type == 'gaussian':
            noise = torch.normal(mean=mean, std=std, size=x_item.shape, device=x_item.device)
        elif noise_type == 'laplace':
            dist = torch.distributions.Laplace(
                torch.tensor(mean, dtype=x_item.dtype, device=x_item.device),
                torch.tensor(std, dtype=x_item.dtype, device=x_item.device),
            )
            noise = dist.sample(x_item.shape)
        else:
            raise ValueError('Unsupported fedfed_noise_type: {}'.format(noise_type))
        return x_item.clone() + noise.to(dtype=x_item.dtype)

    def _flatten_shared_buffer(self):
        shared_x1 = []
        shared_x2 = []
        shared_y = []
        for class_id in sorted(self.shared_by_class):
            for x_item1, x_item2 in self.shared_by_class[class_id]:
                shared_x1.append(x_item1)
                shared_x2.append(x_item2)
                shared_y.append(class_id)
        return shared_x1, shared_x2, shared_y

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
