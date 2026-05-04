# FedAvg + FedFed Plugin

This repository keeps one baseline and one plugin:

- `plugin_name none`: vanilla FedAvg.
- `plugin_name fedfed_image`: paper-style FedFed image-space feature distillation plugin.

The FedFed plugin follows the NeurIPS 2023 FedFed code path: train a beta-VAE generator `q(x)`,
share performance-sensitive features `x_s = x - q(x)`, build a global shared dataset, then train
FedAvg clients on local data plus two shared sensitive-feature views.

## Quick Start

FedAvg:

```bash
python main.py \
  --round_num 150 \
  --num_of_clients 20 \
  --c_fraction 0.2 \
  --local_epoch 5 \
  --batch_size 64 \
  --dataset_name cifar10 \
  --partition_strategy dirichlet \
  --dirichlet_alpha 0.3 \
  --enable_quantity_skew true \
  --enable_feature_skew false \
  --plugin_name none
```

FedAvg + FedFed:

```bash
python main.py \
  --round_num 150 \
  --num_of_clients 20 \
  --c_fraction 0.2 \
  --local_epoch 5 \
  --batch_size 64 \
  --dataset_name cifar10 \
  --partition_strategy dirichlet \
  --dirichlet_alpha 0.3 \
  --enable_quantity_skew true \
  --enable_feature_skew false \
  --plugin_name fedfed_image
```

## Key Files

- `main.py`: experiment entrypoint.
- `src/fed_server/fedavg.py`: FedAvg orchestration and FedFed two-stage hook.
- `src/fed_client/client.py`: local client training and FedFed distillation calls.
- `src/plugins/fedfed_image_plugin.py`: client/server FedFed plugin.
- `src/plugins/fedfed_modules.py`: paper-style beta-VAE generator.
- `src/options.py`: FedAvg and FedFed configuration.

## Outputs

Each experiment writes results under `result/<dataset>/<experiment>/`, including:

- `metrics.json`
- `live_metrics.json`
- `metrics_table.csv`
- accuracy/loss plots
