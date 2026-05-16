import torch

from src.fed_server.fedavg import FedAvgTrainer


class FedNovaTrainer(FedAvgTrainer):
    def __init__(self, options, dataset, clients_label):
        super().__init__(options, dataset, clients_label)
        self.server_momentum = float(options.get('fednova_server_momentum', 0.0))
        self.server_lr = float(options.get('fednova_server_lr', 1.0))
        self.server_velocity = self._zero_float_state(self.latest_global_model)
        self.trainable_parameter_names = {
            name for name, parameter in self.model.named_parameters() if parameter.requires_grad
        }

    @staticmethod
    def _zero_float_state(model_state):
        return {
            key: torch.zeros_like(value.detach().cpu())
            for key, value in model_state.items()
            if value.is_floating_point()
        }

    @staticmethod
    def _normalizing_vector(num_steps, momentum):
        num_steps = max(int(num_steps), 1)
        momentum = float(momentum)
        if abs(momentum) < 1e-12:
            return float(num_steps)
        counter = 0.0
        normalizer = 0.0
        for _ in range(num_steps):
            counter = counter * momentum + 1.0
            normalizer += counter
        return max(normalizer, 1e-12)

    def aggregate_parameters(self, local_model_paras_set):
        if not local_model_paras_set:
            return self.latest_global_model
        total_samples = sum(update["num_samples"] for update in local_model_paras_set)
        if total_samples <= 0:
            raise ValueError('FedNova requires at least one local sample.')

        momentum = float(self.options.get('momentum', 0.0) or 0.0)
        normalizers = [
            self._normalizing_vector(update.get("num_steps", 1), momentum)
            for update in local_model_paras_set
        ]
        tau_eff = sum(
            (update["num_samples"] / total_samples) * normalizer
            for update, normalizer in zip(local_model_paras_set, normalizers)
        )
        fedavg_state = self._aggregate_weights_only(local_model_paras_set)
        next_state = {}
        for key, current_value in self.latest_global_model.items():
            if current_value.is_floating_point() and key in self.trainable_parameter_names:
                current_cpu = current_value.detach().cpu()
                delta = torch.zeros_like(current_cpu)
                for update, normalizer in zip(local_model_paras_set, normalizers):
                    weight = update["num_samples"] / total_samples
                    local_value = update["weights"][key].detach().cpu()
                    delta.add_(local_value - current_cpu, alpha=weight * tau_eff / normalizer)
                if self.server_momentum > 0.0:
                    velocity = self.server_momentum * self.server_velocity[key] + delta
                    self.server_velocity[key] = velocity
                    delta = velocity
                next_state[key] = (current_cpu + self.server_lr * delta).to(current_value.device)
            else:
                next_state[key] = fedavg_state[key]
        if self.server_plugin is not None:
            self.server_plugin.aggregate_client_payloads(local_model_paras_set)
        return next_state
