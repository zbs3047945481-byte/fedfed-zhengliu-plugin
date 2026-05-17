# FedFed Paper Extracted Experimental Results

Source PDF: `/Users/zhubingshuo/论文合集/NeurIPS-2023-fedfed-feature-distillation-against-data-heterogeneity-in-federated-learning-Paper-Conference.pdf`

Notes:
- In Tables 1-4, values are reported as `FedFed(without FedFed)`.
- `Round` is communication rounds to target accuracy.
- `None` means target accuracy was not reached.

## Experimental Setup

- Datasets: CIFAR-10, CIFAR-100, Fashion-MNIST (FMNIST), SVHN.
- Partition: latent Dirichlet allocation, alpha = 0.1 or 0.05; additional #C=2 and Subset partitions.
- Models: ResNet-18 for feature distillation and FL classifier.
- Algorithms: FedAvg, FedProx, SCAFFOLD, FedNova.
- Clients: K = 10 or 100.
- Sampling: K=10 selects 5 clients per round; K=100 selects 10 clients per round unless otherwise specified.
- Training: feature distillation rounds Td = 15; classifier rounds Tr = 1000; local distillation epoch Ed = 1; classifier local epoch E = 1 or 5.
- Batch size: 64 for K=10, 32 for K=100.
- Noise level in main experiments: N(0, 0.15).

## Table 1 - CIFAR-10 Top-1 Accuracy

Centralized training ACC = 95.48%.

| Setting | Algorithm | ACC | Gain | Round | Speedup |
|---|---|---:|---:|---|---|
| alpha=0.1,E=1,K=10,target=79% | FedAvg | 92.34(79.35) | 12.99 | 39(284) | x7.3(x1.0) |
| alpha=0.1,E=1,K=10,target=79% | FedProx | 92.12(83.06) | 9.06 | 62(192) | x4.6(x1.5) |
| alpha=0.1,E=1,K=10,target=79% | SCAFFOLD | 89.66(83.67) | 5.99 | 34(288) | x8.4(x1.0) |
| alpha=0.1,E=1,K=10,target=79% | FedNova | 92.23(80.95) | 11.28 | 33(349) | x8.6(x0.8) |
| alpha=0.05,E=1,K=10,target=69% | FedAvg | 90.02(69.36) | 20.66 | 44(405) | x9.2(x1.0) |
| alpha=0.05,E=1,K=10,target=69% | FedProx | 90.73(78.98) | 11.75 | 48(203) | x8.4(x2.0) |
| alpha=0.05,E=1,K=10,target=69% | SCAFFOLD | 81.04(37.87) | 43.17 | 37(None) | x10.9(None) |
| alpha=0.05,E=1,K=10,target=69% | FedNova | 91.21(65.08) | 26.13 | 32(None) | x12.7(None) |
| alpha=0.1,E=5,K=10,target=85% | FedAvg | 93.24(83.79) | 9.45 | 17(261) | x15.4(x1.0) |
| alpha=0.1,E=5,K=10,target=85% | FedProx | 91.39(82.32) | 8.97 | 76(None) | x3.4(None) |
| alpha=0.1,E=5,K=10,target=85% | SCAFFOLD | 92.34(85.31) | 7.03 | 15(66) | x17.0(x4.0) |
| alpha=0.1,E=5,K=10,target=85% | FedNova | 92.85(86.21) | 6.64 | 31(120) | x8.4(x2.2) |
| alpha=0.1,E=1,K=100,target=49% | FedAvg | 84.06(49.72) | 34.34 | 163(967) | x5.9(x1.0) |
| alpha=0.1,E=1,K=100,target=49% | FedProx | 87.01(50.01) | 37.00 | 127(831) | x7.6(x1.2) |
| alpha=0.1,E=1,K=100,target=49% | SCAFFOLD | 79.60(52.76) | 26.84 | 171(627) | x5.7(x1.5) |
| alpha=0.1,E=1,K=100,target=49% | FedNova | 86.64(45.97) | 40.67 | 199(None) | x4.9(None) |

## Table 2 - FMNIST Top-1 Accuracy

Centralized training ACC = 95.64%.

| Setting | Algorithm | ACC | Gain | Round | Speedup |
|---|---|---:|---:|---|---|
| alpha=0.1,E=1,K=10,target=86% | FedAvg | 92.34(86.73) | 5.61 | 14(121) | x8.6(x1.0) |
| alpha=0.1,E=1,K=10,target=86% | FedProx | 92.09(87.73) | 4.36 | 32(129) | x2.1(x0.9) |
| alpha=0.1,E=1,K=10,target=86% | SCAFFOLD | 91.62(86.31) | 3.89 | 29(147) | x4.2(x0.8) |
| alpha=0.1,E=1,K=10,target=86% | FedNova | 92.39(87.03) | 5.36 | 18(88) | x6.7(x1.4) |
| alpha=0.05,E=1,K=10,target=78% | FedAvg | 90.69(78.34) | 12.35 | 16(420) | x26.3(x1.0) |
| alpha=0.05,E=1,K=10,target=78% | FedProx | 89.68(82.03) | 7.65 | 16(44) | x26.3(9.5) |
| alpha=0.05,E=1,K=10,target=78% | SCAFFOLD | 80.48(76.63) | 3.85 | 139(None) | x6.2(None) |
| alpha=0.05,E=1,K=10,target=78% | FedNova | 89.72(79.98) | 9.74 | 16(531) | x26.3(x0.8) |
| alpha=0.1,E=5,K=10,target=87% | FedAvg | 92.26(87.43) | 4.83 | 19(276) | x14.5(x1.0) |
| alpha=0.1,E=5,K=10,target=87% | FedProx | 91.79(86.63) | 5.16 | 34(None) | x8.1(None) |
| alpha=0.1,E=5,K=10,target=87% | SCAFFOLD | 92.92(87.21) | 5.71 | 8(112) | x34.5(x2.5) |
| alpha=0.1,E=5,K=10,target=87% | FedNova | 92.30(87.67) | 4.63 | 8(187) | x34.5(x1.5) |
| alpha=0.1,E=1,K=100,target=90% | FedAvg | 92.71(90.21) | 2.50 | 243(687) | x2.8(x1.0) |
| alpha=0.1,E=1,K=100,target=90% | FedProx | 92.82(90.17) | 2.65 | 284(501) | x2.4(x1.4) |
| alpha=0.1,E=1,K=100,target=90% | SCAFFOLD | 90.28(84.87) | 5.41 | 952(None) | x0.7(None) |
| alpha=0.1,E=1,K=100,target=90% | FedNova | 91.04(85.32) | 5.72 | 589(None) | x1.2(None) |

## Table 3 - CIFAR-100 Top-1 Accuracy

Centralized training ACC = 75.56%.

| Setting | Algorithm | ACC | Gain | Round | Speedup |
|---|---|---:|---:|---|---|
| alpha=0.1,E=1,K=10,target=67% | FedAvg | 69.64(67.84) | 1.80 | 283(495) | x1.7(x1.0) |
| alpha=0.1,E=1,K=10,target=67% | FedProx | 70.02(65.34) | 4.68 | 233(None) | x2.1(None) |
| alpha=0.1,E=1,K=10,target=67% | SCAFFOLD | 70.14(67.23) | 2.91 | 198(769) | x2.5(x0.6) |
| alpha=0.1,E=1,K=10,target=67% | FedNova | 70.48(67.98) | 2.50 | 147(432) | x3.4(x1.1) |
| alpha=0.05,E=1,K=10,target=61% | FedAvg | 68.49(62.01) | 6.48 | 137(503) | x3.7(x1.0) |
| alpha=0.05,E=1,K=10,target=61% | FedProx | 69.03(61.29) | 7.74 | 141(485) | x3.6(1.0) |
| alpha=0.05,E=1,K=10,target=61% | SCAFFOLD | 69.32(58.78) | 10.54 | 81(None) | x6.2(None) |
| alpha=0.05,E=1,K=10,target=61% | FedNova | 68.92(60.53) | 8.39 | 87(None) | x5.8(None) |
| alpha=0.1,E=5,K=10,target=69% | FedAvg | 70.96(69.34) | 1.62 | 79(276) | x3.5(x1.0) |
| alpha=0.1,E=5,K=10,target=69% | FedProx | 69.66(62.32) | 7.34 | 285(None) | x1.0(None) |
| alpha=0.1,E=5,K=10,target=69% | SCAFFOLD | 70.76(70.23) | 0.53 | 108(174) | x2.6(x1.6) |
| alpha=0.1,E=5,K=10,target=69% | FedNova | 69.98(69.78) | 0.20 | 89(290) | x3.1(x1.0) |
| alpha=0.1,E=1,K=100,target=48% | FedAvg | 60.58(48.21) | 12.37 | 448(967) | x2.2(x1.0) |
| alpha=0.1,E=1,K=100,target=48% | FedProx | 67.69(48.78) | 18.91 | 200(932) | x4.8(x1.0) |
| alpha=0.1,E=1,K=100,target=48% | SCAFFOLD | 66.67(51.03) | 15.64 | 181(832) | x5.3(x1.2) |
| alpha=0.1,E=1,K=100,target=48% | FedNova | 67.62(48.03) | 19.59 | 198(976) | x4.9(x1.0) |

## Table 4 - SVHN Top-1 Accuracy

Centralized training ACC = 96.56%.

| Setting | Algorithm | ACC | Gain | Round | Speedup |
|---|---|---:|---:|---|---|
| alpha=0.1,E=1,K=10,target=88% | FedAvg | 93.21(88.34) | 4.87 | 105(264) | x2.5(x1.0) |
| alpha=0.1,E=1,K=10,target=88% | FedProx | 91.80(86.23) | 5.574 | 233(None) | x1.1(None) |
| alpha=0.1,E=1,K=10,target=88% | SCAFFOLD | 88.41(80.12) | 8.29 | 357(None) | x0.(None) |
| alpha=0.1,E=1,K=10,target=88% | FedNova | 92.98(89.23) | 3.75 | 113(276) | x2.3(x1.0) |
| alpha=0.05,E=1,K=10,target=82% | FedAvg | 93.49(82.76) | 10.73 | 194(365) | x1.9(x1.0) |
| alpha=0.05,E=1,K=10,target=82% | FedProx | 93.21(79.43) | 13.78 | 37(None) | x9.9(None) |
| alpha=0.05,E=1,K=10,target=82% | SCAFFOLD | 90.27(75.87) | 14.40 | 64(None) | x5.7(None) |
| alpha=0.05,E=1,K=10,target=82% | FedNova | 93.05(82.32) | 10.73 | 37(731) | x9.9(x0.5) |
| alpha=0.1,E=5,K=10,target=87% | FedAvg | 93.77(87.24) | 6.53 | 105(128) | x1.2(x1.0) |
| alpha=0.1,E=5,K=10,target=87% | FedProx | 91.15(77.21) | 13.94 | 142(None) | x0.9(None) |
| alpha=0.1,E=5,K=10,target=87% | SCAFFOLD | 93.78(80.98) | 12.80 | 20(None) | x6.4(None) |
| alpha=0.1,E=5,K=10,target=87% | FedNova | 93.66(89.03) | 4.63 | 52(177) | x2.5(x0.7) |
| alpha=0.1,E=1,K=100,target=89% | FedAvg | 91.04(89.32) | 1.72 | 763(623) | x0.8(x1.0) |
| alpha=0.1,E=1,K=100,target=89% | FedProx | 91.41(88.76) | 2.65 | 733(645) | x0.8(x1.0) |
| alpha=0.1,E=1,K=100,target=89% | SCAFFOLD | 92.73(88.32) | 4.41 | 507(687) | x1.2(x0.9) |
| alpha=0.1,E=1,K=100,target=89% | FedNova | 84.05(81.87) | 2.18 | None(None) | None(None) |

## Table 5 - Different Non-IID Partition Methods on CIFAR-10, K=10

Test accuracy w/(w/o) FedFed.

| Partition | FedAvg | FedProx | SCAFFOLD | FedNova |
|---|---:|---:|---:|---:|
| alpha=0.1 | 92.34(79.35) | 92.12(83.06) | 89.66(83.67) | 92.23(80.95) |
| #C=2 | 89.23/42.54 | 88.17/58.45 | 84.43/46.82 | 89.54/45.42 |
| Subset | 90.29/39.53 | 89.11/32.87 | 89.92/35.26 | 90.00/38.52 |

## Table 6 - Noise Type on CIFAR-10

Test accuracy with FedFed.

| Noise Type | FedAvg | FedProx | SCAFFOLD | FedNova |
|---|---:|---:|---:|---:|
| Gaussian | 92.34 | 92.12 | 89.66 | 92.23 |
| Laplacian | 92.30 | 91.36 | 91.24 | 91.73 |

## Table 9 - Sampling Rates, CIFAR-10, alpha=0.1,E=1,K=100

| Sampling rate | 5% | 10% | 20% | 40% | 60% |
|---|---:|---:|---:|---:|---:|
| Accuracy | 83.67 | 84.06 | 87.98 | 89.17 | 89.62 |
| Round to target | 182 | 163 | 90 | 70 | 61 |

## Table 10 - alpha=0.5,E=1,K=10, 50% Sampling

| Method | CIFAR-10 | FMNIST | SVHN | CIFAR-100 |
|---|---:|---:|---:|---:|
| FedAvg | 87.68 | 90.32 | 91.11 | 68.98 |
| FedFed | 93.21 | 94.01 | 93.71 | 69.52 |

## Table 11 - alpha=1.0,E=1,K=10, 50% Sampling

| Method | CIFAR-10 | FMNIST | SVHN | CIFAR-100 |
|---|---:|---:|---:|---:|
| FedAvg | 89.45 | 92.38 | 92.03 | 69.02 |
| FedFed | 94.01 | 94.31 | 93.89 | 70.31 |

## Table 12 - Visual Similarity PSNR

| Pair | PSNR |
|---|---:|
| x and xr | 31.47 +/- 1.57 |
| x and xs | 18.87 +/- 0.98 |

## Table 13 - Training Time

| Item | Time |
|---|---:|
| Generator | 5,251 s |
| Classifier | 62,901 s |

## Table 14 - Parameters

| Item | Size |
|---|---:|
| Generator | 48M |
| Classifier | 42MB |

## Table 15 - FLOPs

| Item | FLOPs |
|---|---:|
| Training generator | 16.03M |
| Training classifier | 556.66M |

## Table 16 - Local Classifier vs Globally Shared Data Size

| Item | Size |
|---|---:|
| Local classifier | 42MB |
| Globally shared data | 586MB |

## Communication Overhead Examples

| Setting | Extra communication cost |
|---|---:|
| K=10, Td=15, Tr=1000, beta=50% | 4.54% |
| K=100, Td=15, Tr=1000, beta=10% | 22.07% |

## Figures With Qualitative/Curve Results

- Figure 2(a): beta-VAE converges/performs better than ResNet generator; paper uses beta-VAE in later results.
- Figure 2(b): compares FedFed plugged into FedAvg/FedProx/SCAFFOLD/FedNova under alpha=0.1,E=1,K=100.
- Figure 2(c), Figure 16: higher DP noise improves privacy but reduces test accuracy.
- Figure 3, Figure 6, Figure 7: model inversion attacks fail to reconstruct protected shared features clearly.
- Figure 8: membership inference recall for shared xp is lower than sharing raw x under similar/noise settings.
- Figure 11: sharing protected partial features with DP performs better than sharing protected raw data with DP.
- Figures 12-15: convergence curves on CIFAR-10, CIFAR-100, SVHN, and FMNIST.
- Figure 17: classifiers trained on x and xs have comparable generalization ability; xr is weaker for task performance.
