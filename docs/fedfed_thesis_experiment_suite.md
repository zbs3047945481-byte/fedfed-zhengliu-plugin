# FedFed Plugin Thesis Experiment Suite

This suite follows the FedFed paper's experiment logic, but narrows the scope to this project's goal:
verifying a plug-and-play FedFed plugin for a FedAvg baseline.

## Core Settings

| Item | Value |
|---|---|
| Dataset | CIFAR-10 |
| Model | ResNet18 |
| Partition | LDA / Dirichlet |
| Quantity skew | enabled |
| Feature skew | disabled |
| Clients | 10 |
| Clients per round | 5 |
| Local epoch | 1 by default |
| Rounds | 300 with early stopping |
| Main optimizer | SGD, lr=0.01, momentum=0.9, weight decay=1e-4 |
| FedFed generator | paper-style beta-VAE |
| VAE optimizer | AdamW, lr=1e-3, betas=(0.9, 0.999), weight decay=1e-6 |
| Shared data | paper full shared sensitive dataset by default |
| DP noise | Gaussian std1=0.2, std2=0.25 by default |

## Experiments

### E1 Main Result

Purpose: prove that `FedAvg + FedFed plugin` improves vanilla FedAvg.

Runs:

| Method | alpha | E | K | plugin |
|---|---:|---:|---:|---|
| FedAvg | 0.1 | 1 | 10 | none |
| FedFedPlugin | 0.1 | 1 | 10 | fedfed_image |

Metrics: best accuracy, final accuracy, best round, target round, speedup.

Figures/tables: main result table, accuracy-round curve, best/final accuracy bar.

### E2 Heterogeneity Strength

Purpose: show that FedFed gain depends on Non-IID strength.

Grid: `alpha=0.1, 0.3, 0.5`.

Runs: FedAvg and FedFedPlugin for every alpha.

Metrics: best accuracy, final accuracy, FedFed gain.

Figures/tables: alpha-gain line chart, alpha result table.

### E3 Local Epoch Sensitivity

Purpose: compare the paper setting `E=1` with stronger local drift `E=5`.

Grid: `local_epoch=1, 5` under `alpha=0.1`.

Runs: FedAvg and FedFedPlugin.

Metrics: best accuracy, final accuracy, gain.

Figures/tables: epoch result table, convergence curves.

### E4 Shared Pool Size

Purpose: verify whether larger/full sensitive feature sharing improves the plugin.

Runs:

| Method | upload_per_class | upload_per_client | shared_per_class_size | shared_buffer_size |
|---|---:|---:|---:|---:|
| FedFedSmallPool | 4 | 100 | 80 | 800 |
| FedFedMediumPool | 20 | 1000 | 400 | 4000 |
| FedFedFullPool | 0 | 0 | 0 | 0 |

`0` means paper full shared dataset.

Metrics: best accuracy, final accuracy, shared dataset size.

Figures/tables: shared pool size vs accuracy.

### E5 DP Noise

Purpose: analyze the privacy-performance tradeoff.

Runs:

| Method | noise type | std1 | std2 |
|---|---|---:|---:|
| FedFedNoNoise | none | 0.0 | 0.0 |
| FedFedGaussianLow | gaussian | 0.1 | 0.15 |
| FedFedGaussianPaper | gaussian | 0.2 | 0.25 |
| FedFedGaussianHigh | gaussian | 0.3 | 0.35 |
| FedFedLaplacePaper | laplace | 0.2 | 0.25 |

Metrics: best accuracy, final accuracy, noise level.

Figures/tables: noise level vs accuracy, sample visualization of `x`, `xs`, noisy `xp`.

### E6 Distillation Sufficiency

Purpose: check whether generator training is sufficient.

Runs:

| Method | distill rounds | distill local epoch |
|---|---:|---:|
| Distill5 | 5 | 1 |
| Distill15 | 15 | 1 |
| Distill30 | 30 | 1 |
| Distill30E2 | 30 | 2 |

Metrics: best accuracy, final accuracy, L_fd, distill accuracy, xs_norm.

Figures/tables: distillation metrics over rounds, distill rounds vs final accuracy.

### E7 x / xs / xr Diagnostic

Purpose: verify that performance-sensitive features are classifiable while robust features preserve more visual/private information.

Runs:

| Classifier | Train input | Test input |
|---|---|---|
| x-only | raw x | raw x |
| xs-only | sensitive feature `xs=x-q(x)` | sensitive feature |
| xr-only | robust feature `xr=q(x)` | robust feature |

Metrics: classifier accuracy for x, xs, xr.

Figures/tables: diagnostic accuracy bar, optional feature visualization.

### E8 Overhead

Purpose: make the plugin implementation credible as an engineering artifact.

Metrics: FedAvg time, FedFed time, distillation time, GPU memory, CPU memory, shared dataset size.

Figures/tables: overhead table and compact overhead bar.

## Minimal Thesis Figure Set

1. Plugin architecture diagram.
2. Client feature distillation diagram.
3. Server shared dataset construction diagram.
4. Main FedAvg vs FedFed accuracy curve.
5. Main result table.
6. Alpha sensitivity plot.
7. x/xs/xr diagnostic bar.
8. Distillation metric curves.
9. Shared pool ablation table.
10. DP noise accuracy plot.
11. Overhead table.

## DP Noise + Simple Visualization

This means taking a small fixed batch of CIFAR-10 images and visualizing:

1. raw image `x`;
2. performance-sensitive feature `xs = x - q(x)`;
3. protected shared feature `xp = xs + noise`;
4. robust feature `xr = q(x)`.

The point is not to claim a full privacy proof. The point is to give an intuitive picture that adding
DP noise makes the shared sensitive feature harder to visually map back to the original image, while
the accuracy table shows the performance cost of different noise strengths.
