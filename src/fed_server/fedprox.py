from src.fed_server.fedavg import FedAvgTrainer


class FedProxTrainer(FedAvgTrainer):
    def local_train(self, round_i, select_clients):
        mu = float(self.options.get('fedprox_mu', 0.01))
        for client in select_clients:
            client.configure_fedprox(self.latest_global_model, mu)
        try:
            return super().local_train(round_i, select_clients)
        finally:
            for client in select_clients:
                client.clear_fedprox()
