import torch

from src.fed_server.fedavg import FedAvgTrainer


class ScaffoldTrainer(FedAvgTrainer):
    def __init__(self, options, dataset, clients_label):
        super().__init__(options, dataset, clients_label)
        self.server_controls = self._zero_control_like(self.latest_global_model)

    @staticmethod
    def _zero_control_like(model_state):
        return {
            key: torch.zeros_like(value.detach().cpu())
            for key, value in model_state.items()
            if value.is_floating_point()
        }

    def local_train(self, round_i, select_clients):
        for client in select_clients:
            client.configure_scaffold(self.latest_global_model, self.server_controls)
        try:
            local_model_paras_set, stats = super().local_train(round_i, select_clients)
            self._update_server_controls(local_model_paras_set)
            return local_model_paras_set, stats
        finally:
            for client in select_clients:
                client.clear_scaffold()

    def _update_server_controls(self, local_model_paras_set):
        if not local_model_paras_set:
            return
        scale = 1.0 / max(self.clients_num, 1)
        for update in local_model_paras_set:
            algorithm_aux = update.get("algorithm_aux") or {}
            control_delta = algorithm_aux.get("control_delta")
            if not control_delta:
                continue
            for key, delta in control_delta.items():
                if key in self.server_controls:
                    updated = self.server_controls[key] + scale * delta.detach().cpu()
                    control_clip = float(self.options.get('scaffold_control_clip', 0.0) or 0.0)
                    if control_clip > 0.0:
                        updated = torch.clamp(updated, -control_clip, control_clip)
                    self.server_controls[key] = updated
