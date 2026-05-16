import torch

from src.fed_server.fedavg import FedAvgTrainer


class FedAvgMTrainer(FedAvgTrainer):
    def __init__(self, options, dataset, clients_label):
        super().__init__(options, dataset, clients_label)
        self.server_momentum = float(options.get('fedavgm_momentum', 0.9))
        self.server_lr = float(options.get('fedavgm_server_lr', 1.0))
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

    def aggregate_parameters(self, local_model_paras_set):
        fedavg_state = super().aggregate_parameters(local_model_paras_set)
        next_state = {}
        for key, current_value in self.latest_global_model.items():
            if current_value.is_floating_point() and key in self.trainable_parameter_names:
                delta = fedavg_state[key].detach().cpu() - current_value.detach().cpu()
                velocity = self.server_momentum * self.server_velocity[key] + delta
                self.server_velocity[key] = velocity
                next_state[key] = (current_value.detach().cpu() + self.server_lr * velocity).to(current_value.device)
            else:
                next_state[key] = fedavg_state[key]
        return next_state
