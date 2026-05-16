from torch.utils.data import DataLoader
import torch.nn.functional as F
import time
import torch
import copy
from src.plugins import build_client_plugin
from src.utils.tools import get_runtime_device
criterion = F.cross_entropy


class BaseClient():
    def __init__(self, options, id, local_dataset, model, optimizer, ):
        self.options = options
        self.id = id
        self.local_dataset = local_dataset
        self.model = model
        self.device = get_runtime_device(options)
        self.storage_device = torch.device('cpu')
        self.gpu = self.device.type != 'cpu'
        self.optimizer = optimizer

        self.model.to(self.storage_device)
        self.plugin = build_client_plugin(options, self.model, self.storage_device)
        self.plugin_payload = None
        self.fedprox_mu = 0.0
        self.fedprox_global_params = None
        self.scaffold_server_controls = None
        self.scaffold_client_controls = None
        self.scaffold_global_params = None

    def set_plugin_payload(self, payload):
        self.plugin_payload = payload

    def set_global_sensitive_feature(self, global_sensitive_feature):
        """Backward-compatible alias for legacy server hook names."""
        self.set_plugin_payload(global_sensitive_feature)

    def set_learning_rate(self, learning_rate):
        for group in self.optimizer.param_groups:
            group['lr'] = learning_rate

    def configure_fedprox(self, global_params, mu):
        self.fedprox_mu = float(mu)
        self.fedprox_global_params = {
            key: value.detach().to(self.device)
            for key, value in global_params.items()
            if value.is_floating_point()
        } if self.fedprox_mu > 0.0 else None

    def clear_fedprox(self):
        self.fedprox_mu = 0.0
        self.fedprox_global_params = None

    def configure_scaffold(self, global_params, server_controls):
        if self.scaffold_client_controls is None:
            self.scaffold_client_controls = {
                key: torch.zeros_like(value.detach().cpu())
                for key, value in global_params.items()
                if value.is_floating_point()
            }
        self.scaffold_global_params = {
            key: value.detach().cpu().clone()
            for key, value in global_params.items()
            if value.is_floating_point()
        }
        self.scaffold_server_controls = {
            key: value.detach().to(self.device)
            for key, value in server_controls.items()
        }

    def clear_scaffold(self):
        self.scaffold_server_controls = None
        self.scaffold_global_params = None

    def get_model_parameters(self):
        state_dict = self.model.state_dict()
        return state_dict

    def get_model_parameters_cpu(self):
        return {
            key: value.detach().cpu().clone()
            for key, value in self.model.state_dict().items()
        }

    def set_model_parameters(self, model_parameters_dict):
        state_dict = self.model.state_dict()
        for key, value in state_dict.items():
            state_dict[key] = model_parameters_dict[key].detach().to(value.device)
        self.model.load_state_dict(state_dict)

    def local_train(self, ):
        begin_time = time.time()
        self._move_to_training_device()
        try:
            local_model_paras, return_dict, aux, algorithm_aux = self.local_update(self.local_dataset, self.options, )
        finally:
            self._move_to_storage_device()
        end_time = time.time()
        stats = {'id': self.id, "time": round(end_time - begin_time, 2)}
        stats.update(return_dict)
        # Update structure: weights (FedAvg) + num_samples + optional aux (FedFed)
        update = {
            "weights": local_model_paras,
            "num_samples": len(self.local_dataset),
            "num_steps": return_dict.get("num_steps", 0),
            "aux": aux,
            "algorithm_aux": algorithm_aux,
        }
        return update, stats

    def plugin_feature_distill(self, payload):
        if self.plugin is None or not hasattr(self.plugin, 'distill_batch'):
            raise RuntimeError('Current plugin does not support feature distillation.')
        begin_time = time.time()
        self._move_to_training_device()
        try:
            update, stats = self._plugin_feature_distill_update(payload)
        finally:
            self._move_to_storage_device()
        stats['time'] = round(time.time() - begin_time, 2)
        return update, stats

    def _plugin_feature_distill_update(self, payload):
        pin_memory = self.gpu and self.options.get('dataloader_pin_memory', True)
        loader = DataLoader(
            self.local_dataset,
            batch_size=self.options['batch_size'],
            shuffle=True,
            num_workers=max(int(self.options.get('dataloader_num_workers', 0)), 0),
            pin_memory=pin_memory,
        )
        self.model.train()
        self.plugin.on_distill_start(self.optimizer.param_groups[0]['lr'], payload)
        train_loss = train_acc = train_total = 0
        fd_loss_sum = recon_loss_sum = x_ce_loss_sum = align_loss_sum = logit_align_loss_sum = xs_norm_sum = kl_loss_sum = 0.0
        local_epoch = int(self.options.get('fedfed_distill_local_epoch', 1))
        for epoch in range(1, local_epoch + 1):
            if hasattr(self.plugin, 'set_distill_epoch'):
                self.plugin.set_distill_epoch(epoch)
            for X, y in loader:
                if self.gpu:
                    X = X.to(self.device, non_blocking=pin_memory)
                    y = y.to(self.device, non_blocking=pin_memory)
                pred, loss = self.plugin.distill_batch(X, y)
                distill_stats = getattr(self.plugin, 'last_distill_stats', {})
                _, predicted = torch.max(pred, 1)
                train_acc += predicted.eq(y).sum().item()
                train_loss += loss.item() * y.size(0)
                fd_loss_sum += float(distill_stats.get('fd_loss', 0.0)) * y.size(0)
                recon_loss_sum += float(distill_stats.get('recon_loss', 0.0)) * y.size(0)
                x_ce_loss_sum += float(distill_stats.get('x_ce_loss', 0.0)) * y.size(0)
                align_loss_sum += float(distill_stats.get('align_loss', 0.0)) * y.size(0)
                logit_align_loss_sum += float(distill_stats.get('logit_align_loss', 0.0)) * y.size(0)
                xs_norm_sum += float(distill_stats.get('xs_norm', 0.0)) * y.size(0)
                kl_loss_sum += float(distill_stats.get('kl_loss', 0.0)) * y.size(0)
                train_total += y.size(0)
        update = {
            "weights": self.get_model_parameters_cpu(),
            "num_samples": len(self.local_dataset),
            "aux": self.plugin.build_upload_payload(),
        }
        stats = {
            "id": self.id,
            "loss": train_loss / max(train_total, 1),
            "acc": train_acc / max(train_total, 1),
            "fd_loss": fd_loss_sum / max(train_total, 1),
            "recon_loss": recon_loss_sum / max(train_total, 1),
            "x_ce_loss": x_ce_loss_sum / max(train_total, 1),
            "align_loss": align_loss_sum / max(train_total, 1),
            "logit_align_loss": logit_align_loss_sum / max(train_total, 1),
            "xs_norm": xs_norm_sum / max(train_total, 1),
            "kl_loss": kl_loss_sum / max(train_total, 1),
        }
        return update, stats

    def plugin_collect_shared_features(self, payload):
        if self.plugin is None or not hasattr(self.plugin, 'collect_shared_batch'):
            raise RuntimeError('Current plugin does not support shared feature collection.')
        self._move_to_training_device()
        try:
            update = self._plugin_collect_shared_features(payload)
        finally:
            self._move_to_storage_device()
        return update

    def _plugin_collect_shared_features(self, payload):
        pin_memory = self.gpu and self.options.get('dataloader_pin_memory', True)
        loader = DataLoader(
            self.local_dataset,
            batch_size=self.options['batch_size'],
            shuffle=True,
            num_workers=max(int(self.options.get('dataloader_num_workers', 0)), 0),
            pin_memory=pin_memory,
        )
        self.model.eval()
        self.plugin.on_distill_start(self.optimizer.param_groups[0]['lr'], payload)
        for X, y in loader:
            if self.gpu:
                X = X.to(self.device, non_blocking=pin_memory)
                y = y.to(self.device, non_blocking=pin_memory)
            self.plugin.collect_shared_batch(X, y)
        return {
            "weights": self.get_model_parameters_cpu(),
            "num_samples": len(self.local_dataset),
            "aux": self.plugin.build_upload_payload(),
        }

    def local_update(self, local_dataset, options, ):
        use_plugin = self.plugin is not None
        pin_memory = self.gpu and options.get('dataloader_pin_memory', True)
        localTrainDataLoader = DataLoader(
            local_dataset,
            batch_size=options['batch_size'],
            shuffle=True,
            num_workers=max(int(options.get('dataloader_num_workers', 0)), 0),
            pin_memory=pin_memory,
        )
        self.model.train() #把模型设置为训练模式
        if use_plugin:
            self.plugin.on_round_start(self.optimizer.param_groups[0]['lr'], self.plugin_payload)#正式训练前，先把插件状态准备好。
        train_loss = train_acc = train_total = num_steps = 0
        for epoch in range(options['local_epoch']):  #表示这个客户端会把自己的本地数据完整训练 local_epoch 遍。
            for X, y in localTrainDataLoader:
                if self.gpu:
                    X = X.to(self.device, non_blocking=pin_memory)
                    y = y.to(self.device, non_blocking=pin_memory)
                if use_plugin:
                    if hasattr(self.plugin, 'train_batch_with_hooks'):
                        pred, loss = self.plugin.train_batch_with_hooks(
                            X,
                            y,
                            extra_loss_fn=self._fedprox_loss,
                            before_step=self._apply_scaffold_gradient_correction,
                        )
                    else:
                        pred, loss = self.plugin.train_batch(X, y)
                else:
                    self.optimizer.zero_grad()
                    pred = self.model(X)
                    loss = criterion(pred, y)
                    loss = loss + self._fedprox_loss()
                    loss.backward()
                    self._apply_scaffold_gradient_correction()
                    self.optimizer.step()
                num_steps += 1
                _, predicted = torch.max(pred, 1) #从预测 logits 中取每个样本得分最高的类别，作为预测标签。
                correct = predicted.eq(y).sum().item() #统计当前 batch 中预测正确的样本数。
                target_size = y.size(0) #得到当前 batch 的样本数量。
                train_loss += loss.item() * y.size(0)
                train_acc += correct
                train_total += target_size
        local_model_paras = self.get_model_parameters_cpu() #训练结束后，把上传权重固定到CPU，避免客户端模型常驻GPU。
        algorithm_aux = self._build_scaffold_upload(local_model_paras, num_steps)
        return_dict = {"id": self.id,
                       "loss": train_loss / train_total,
                       "acc": train_acc / train_total,
                       "num_steps": num_steps}
        aux = self.plugin.build_upload_payload() if use_plugin else None
        return local_model_paras, return_dict, aux, algorithm_aux

    def _fedprox_loss(self):
        if self.fedprox_mu <= 0.0 or self.fedprox_global_params is None:
            return torch.zeros((), device=self.device)
        prox = torch.zeros((), device=self.device)
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad or name not in self.fedprox_global_params:
                continue
            prox = prox + torch.sum((parameter - self.fedprox_global_params[name]) ** 2)
        return 0.5 * self.fedprox_mu * prox

    def _apply_scaffold_gradient_correction(self):
        if self.scaffold_server_controls is None or self.scaffold_client_controls is None:
            return
        for name, parameter in self.model.named_parameters():
            if parameter.grad is None or name not in self.scaffold_server_controls:
                continue
            server_control = self.scaffold_server_controls[name].to(parameter.grad.device)
            client_control = self.scaffold_client_controls[name].to(parameter.grad.device)
            parameter.grad.add_(server_control - client_control)
        grad_clip = float(self.options.get('scaffold_grad_clip', 0.0) or 0.0)
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

    def _build_scaffold_upload(self, local_model_paras, num_steps):
        if (
            self.scaffold_server_controls is None
            or self.scaffold_client_controls is None
            or self.scaffold_global_params is None
            or num_steps <= 0
        ):
            return None
        lr = float(self.optimizer.param_groups[0]['lr'])
        denom = max(num_steps * lr, 1e-12)
        delta_controls = {}
        for name, global_value in self.scaffold_global_params.items():
            if name not in local_model_paras:
                continue
            old_client = self.scaffold_client_controls[name]
            server_control = self.scaffold_server_controls[name].detach().cpu()
            local_value = local_model_paras[name].detach().cpu()
            new_client = old_client - server_control + (global_value - local_value) / denom
            control_clip = float(self.options.get('scaffold_control_clip', 0.0) or 0.0)
            if control_clip > 0.0:
                new_client = torch.clamp(new_client, -control_clip, control_clip)
            delta_controls[name] = new_client - old_client
            self.scaffold_client_controls[name] = new_client
        return {"control_delta": delta_controls, "num_steps": num_steps}

    def _move_to_training_device(self):
        self.model.to(self.device)
        self._move_optimizer_state(self.optimizer, self.device)
        if self.plugin is not None and hasattr(self.plugin, 'to_device'):
            self.plugin.to_device(self.device)

    def _move_to_storage_device(self):
        self.model.to(self.storage_device)
        self._move_optimizer_state(self.optimizer, self.storage_device)
        if self.plugin is not None and hasattr(self.plugin, 'to_device'):
            self.plugin.to_device(self.storage_device)
        if self.gpu and self.device.type == 'cuda' and bool(self.options.get('client_empty_cache', True)):
            torch.cuda.empty_cache()

    @staticmethod
    def _move_optimizer_state(optimizer, device):
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
