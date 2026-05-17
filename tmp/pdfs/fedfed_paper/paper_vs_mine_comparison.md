# FedFed Paper vs This Project Results

## Comparability Notes

- Paper values come from FedFed NeurIPS 2023 tables. In paper tables, `ACC` is reported as `with FedFed(without FedFed)`.
- This project values come from local `remote_results/`.
- Same task-level settings mainly mean CIFAR-10, Dirichlet alpha, local epoch E, clients K=10, 5 clients sampled per round.
- Not fully identical: paper uses its original FedFed code, 1000 classifier rounds, DP noise in main settings; this project uses pluginized `fedfed_image`, 300 max rounds with early stop, and the final G1 no-noise configuration.
- Therefore, compare trend and relative gain, not exact numeric equality.

## Directly Comparable CIFAR-10 Main Settings

| Purpose | Setting | Paper baseline | Paper FedFed | Paper gain | This project baseline | This project plugin | This project gain | Comment |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Main result | alpha=0.1,E=1,K=10,FedAvg | 79.35 | 92.34 | +12.99 | 63.89 best / 53.83 final | 88.95 best / 88.53 final | +25.06 best | Same core setting; project reproduces strong improvement trend, but exact values are not strictly comparable. |
| Stronger heterogeneity | alpha=0.05,E=1,K=10,FedAvg | 69.36 | 90.02 | +20.66 | 38.15 best / 34.93 final | 84.80 best / 78.64 final | +46.65 best | Same heterogeneity purpose; both show larger gain under stronger Non-IID. |
| More local training | alpha=0.1,E=5,K=10,FedAvg | 83.79 | 93.24 | +9.45 | 59.60 best / 58.57 final | 91.90 best / 91.01 final | +32.30 best | Same E=5 purpose; plugin strongly reduces local-drift damage in this project. |

## Different FL Backbone Comparison, CIFAR-10 alpha=0.1,E=1,K=10

| Backbone | Paper baseline | Paper + FedFed | Paper gain | This project baseline | This project + plugin | This project gain | Comment |
|---|---:|---:|---:|---:|---:|---:|---|
| FedAvg | 79.35 | 92.34 | +12.99 | 63.89 best / 53.83 final | 88.95 best / 88.53 final | +25.06 best | Core comparison. |
| FedProx | 83.06 | 92.12 | +9.06 | 59.91 best / 58.65 final | 90.49 best / 90.25 final | +30.58 best | Same purpose; project plugin gain is larger because baseline is weaker. |
| SCAFFOLD | 83.67 | 89.66 | +5.99 | 51.68 best / 40.33 final | 56.27 best / 53.54 final | +4.59 best | Trend is positive but much weaker; do not overstate. |
| FedNova | 80.95 | 92.23 | +11.28 | 47.04 best / 42.73 final | 70.77 best / 62.87 final | +23.73 best | Positive trend, but absolute performance remains below paper. |
| FedAvgM | no paper counterpart | no paper counterpart | - | 49.52 best / 43.87 final | 80.20 best / 70.86 final | +30.68 best | Useful as extra project evidence, not a paper comparison. |

## Paper Results With No Direct Final-Plugin Counterpart Yet

| Paper experiment | Paper result summary | This project status | How to use in thesis |
|---|---|---|---|
| CIFAR-10 alpha=0.1,E=1,K=100 | FedAvg 49.72 -> FedFed 84.06 | No corresponding final plugin K=100 run | Cite as paper evidence only; do not compare to your implementation. |
| FMNIST main tables | FedFed improves all listed algorithms, gains about +2.5 to +12.35 depending setting | No final `fedfed_image` FMNIST run | Use only if discussing original paper generality. |
| CIFAR-100 main tables | FedFed improves all listed algorithms, gains about +0.2 to +19.59 | No final `fedfed_image` CIFAR-100 run | Use as original paper generality, not your result. |
| SVHN main tables | FedFed improves most settings, but gains are limited in some K=100 cases | No final `fedfed_image` SVHN run | Use as original paper generality. |
| Different Non-IID partitions on CIFAR-10 | #C=2 and Subset show large improvements with FedFed | No direct project counterpart | Optional background evidence. |
| Gaussian vs Laplacian noise | Similar FedFed accuracy across noise type on CIFAR-10 | Project treats noise only as privacy validation, not main setting | Do not mix with your main no-noise performance table. |
| Sampling rate K=100 | Higher sampling rate improves accuracy and target-round speed | No direct project counterpart | Not needed unless discussing scalability. |

## Project Component Ablation vs Paper

| Project ablation | Project result | Paper counterpart | Interpretation |
|---|---:|---|---|
| C2_NoDistill | 75.44 best / 74.17 final | No exact paper ablation table | Shows distillation stage is necessary in plugin implementation. |
| C3_NoSharedMainTrain | 71.36 best / 63.79 final | No exact paper ablation table | Shows generated/shared features must enter main training. |
| C4_NoLogitAlign | 90.37 best / 87.89 final | Not in original paper | Auxiliary term contributes limited benefit in current config. |
| C5_NoRawCE | 90.55 best / 89.11 final | Not in original paper | Auxiliary raw CE is not main performance source. |
| C6_NoAugMixup | 76.38 best / 76.16 final | Paper uses augmentation/mixup in implementation details | Strong evidence that augmentation/mixup is critical. |

## Suggested Thesis Wording

The original FedFed paper reports that under CIFAR-10 Dirichlet alpha=0.1, E=1, K=10, FedAvg improves from 79.35% to 92.34% after applying FedFed. Under the same task-level setting, this project observes the same qualitative trend: the FedAvg baseline reaches 63.89% best accuracy, while the pluginized FedFed implementation reaches 88.95%. Although the absolute values differ because the original paper and this project use different codebases, training budgets, noise configurations, and early-stopping rules, both results support the same conclusion: sharing performance-sensitive features through FedFed-style distillation can substantially mitigate data heterogeneity.

For stricter wording, describe this as "trend-level consistency with the original FedFed results" rather than "exact reproduction of the original paper".
