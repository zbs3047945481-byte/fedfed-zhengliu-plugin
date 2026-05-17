# Official FedFed Reference Comparison

## Code Layout

- `reference_implementations/FedFed_official/`: untouched upstream clone of `https://github.com/tmlr-group/FedFed`.
- `reference_implementations/official_fedfed_compare/`: local config, runner, and this experiment plan.

The official code must not import from this project's `src/` tree. The only connection is the CIFAR-10 data path.

## Upstream Snapshot

Current upstream commit:

```text
c4a84371403a3cdab7307cea48d03732b888c2e9
```

## Target Plugin Result For Comparison

Use the existing plugin result:

```text
remote_results/fedfed_thesis_15run_4090_fast_20260506T140411Z/fedfed_thesis_15run_4090_fast_20260506T140411Z_summary.csv
```

Main comparable row:

- Dataset: CIFAR-10
- Model: ResNet18 family
- Clients: 10 total, 5 sampled per communication round
- Non-IID: Dirichlet alpha = 0.1
- Local epoch: 1
- Batch size: 64
- Seed: 3001
- Communication rounds: 300 maximum
- Plugin result: best_acc = 0.8895, final_acc = 0.8853

## Official Reference Configuration

Run config:

```text
reference_implementations/official_fedfed_compare/official_fedfed_cifar10_a0p1_e1.yaml
```

Matched task-level settings:

- CIFAR-10
- 10 clients, 5 clients per FedAvg round
- Dirichlet alpha = 0.1
- local epoch = 1
- batch size = 64
- learning rate = 0.01
- seed = 3001
- 300 communication rounds

Kept official FedFed implementation settings:

- `VAE_re = 5.0`
- `VAE_ce = 2.0`
- `VAE_kl = 0.005`
- `VAE_x_ce = 0.4`
- `VAE_comm_round = 15`
- `VAE_client_num_per_round = 10`
- `noise_type = Gaussian`
- `VAE_std1 = 0.2`
- `VAE_std2 = 0.25`

This is intentional: changing these would no longer be a strict reference-implementation comparison.

## Run Command

From the repository root:

```bash
PYTHON_BIN=/root/miniconda3/bin/python \
OUT_ROOT=/root/runs/official_fedfed_reference \
RUN_ID=official_fedfed_cifar10_a0p1_e1_$(date -u +%Y%m%dT%H%M%SZ) \
bash reference_implementations/official_fedfed_compare/run_official_fedfed_cifar10_a0p1_e1.sh
```

Local run output defaults to:

```text
remote_results/official_fedfed_reference/<run_id>/
```

The runner saves:

- `config.yaml`
- `upstream_commit.txt`
- `stdout.log`
- `stderr.log`
- `summary.json`
- `summary.csv`

## Reporting

Report the official reference as:

> Under the same CIFAR-10 federated task setting, the original FedFed implementation is run from the unmodified upstream repository. Task-level hyperparameters are matched to the plugin experiment, while FedFed-internal VAE and noise settings are kept as the original implementation defaults.

Then compare official best/final accuracy against the existing plugin result.
