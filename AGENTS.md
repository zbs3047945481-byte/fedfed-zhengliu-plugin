# Project Context

每次在本仓库 `/Users/zhubingshuo/plug-and-play-fedfed-fedavg-plugin` 中开始新对话或新任务时，先参考本文件同步背景信息。

## 交流原则

- 不要一味迎合用户判断；用户想法正确时指出正确点，想法不正确时明确指出问题在哪里。
- 所有操作在保证效果质量的前提下，尽量用最少 token 和最少无关动作完成。
- 不要改动与当前任务无关的代码、结果文件或实验产物。

## 项目任务

本项目用于毕业设计。

- 毕设题目：一种即插即用的联邦学习特征蒸馏模块设计与实现
- 核心目标：复现参考文献中的 FedFed / feature distillation 思路，并改造成可插拔插件形式，接入 FedAvg 主流程
- 当前代码定位：保留 `plugin_name none` 作为 FedAvg baseline，使用 `plugin_name fedfed_image` 作为论文式 FedFed 图像空间特征蒸馏插件
- 主要入口：`main.py`
- 主要插件文件：`src/plugins/fedfed_image_plugin.py`、`src/plugins/fedfed_modules.py`
- 主要联邦流程：`src/fed_server/fedavg.py`、`src/fed_client/client.py`
- 主要文档：`README.md`、`docs/FEDFED_PLUGIN.md`、`docs/THESIS_EXPERIMENT_PLAN.md`、`docs/THESIS_EXECUTION_CHECKLIST.md`

## 论文写作规范

本仓库中所有与毕业论文写作、改写、扩写、排版、章节生成、图表说明、摘要、实验分析相关的对话，都必须先参考本节。

- 格式模板必须严格遵循：`/Users/zhubingshuo/毕设资料/1.doc`
- 论文方法与行文风格可参考：`/Users/zhubingshuo/论文合集/NeurIPS-2023-fedfed-feature-distillation-against-data-heterogeneity-in-federated-learning-Paper-Conference.pdf`
- 写作语言要求：简洁、明了、凝练，删除冗余解释；能用一句话说清楚的，不拆成两句话。
- 行文质量目标：尽量接近顶刊、顶会论文的表达方式，强调问题定义、方法动机、技术设计和实验结论，避免口语化、宣传化、空泛化表述。
- 中文论文正文中不要堆砌形容词；每句话都应服务于论证、定义、实验解释或结论。
- 生成或修改 `.doc` / `.docx` 论文文件时，必须优先保留模板的标题层级、字体、段落、页眉页脚、目录和参考文献格式。

## 当前已确认的远程执行脚本

本地仓库中已确认的远程执行脚本主要面向“Mac 编辑，远程 Windows GPU 运行”工作流，公共配置来自 `scripts/remote_common.sh`，默认读取 `scripts/remote_windows.env`。

- `scripts/deploy_to_windows.sh`：增量同步本地代码到远程 Windows 项目目录
- `scripts/run_remote_train.sh`：提交后台训练任务
- `scripts/check_remote_run.sh`：查看远程运行状态、日志和 GPU 状态
- `scripts/check_remote_run_compact.sh`：精简查看远程运行状态
- `scripts/run_windows_experiment.sh`：部署并前台运行一次实验，适合日常 smoke run
- `scripts/start_remote_thesis_suite.sh`：提交论文实验队列
- `scripts/start_remote_fedfed_final_queue.sh`：提交最终实验队列
- `scripts/start_remote_fedavg_fedfed_paper_compare.sh`：提交 FedAvg 和 FedFed 对比实验队列
- `scripts/start_remote_six_classifier_diag_offline.sh`：部署代码并通过 Windows 计划任务离线提交六分类器诊断队列
- 其他 `scripts/start_remote_*_queue.sh`：不同诊断、消融或参数网格实验队列

默认远程配置：

- SSH alias：`win-gpu`
- 远程项目目录：`D:\ml\plug-and-play-feature-distillation-fl`
- 远程运行目录：`D:\runs`
- 远程临时脚本目录：`C:/Users/zbs30/codex`
- 远程 Python：`C:\Users\zbs30\miniconda3\envs\t\python.exe`

## GPU 主机背景

用户期望本地可连接 4 台远程 GPU 主机：

- 1 台 RTX 5090
- 2 台 RTX 4090
- 1 台 RTX 3060

当前仓库脚本和本机 SSH 配置中，可直接确认到的训练主机 alias 是 `win-gpu`。没有在仓库脚本中直接确认到 4 个分别对应 5090、4090、4090、3060 的独立主机 alias。后续如果要真正把四台机器纳入统一调度，需要补充每台机器的 SSH alias、远程项目目录、运行目录、Python 路径，并最好拆成多个 `.env` 配置，例如：

- `scripts/remote_5090.env`
- `scripts/remote_4090_a.env`
- `scripts/remote_4090_b.env`
- `scripts/remote_3060.env`

现有 `scripts/remote_common.sh` 已支持通过环境变量 `REMOTE_ENV_FILE` 切换配置文件，因此可以沿用同一套远程脚本连接不同主机。

示例：

```bash
REMOTE_ENV_FILE=scripts/remote_4090_a.env ./scripts/run_windows_experiment.sh --preset plugin-smoke
```

### 已新增：SeetaCloud RTX 5090 Linux 主机

已为 5090 Linux 主机新增独立配置和脚本：

- 配置：`scripts/remote_5090.env`
- 增量部署：`scripts/deploy_to_5090.sh`
- 离线提交 privacy probe：`scripts/start_remote_5090_privacy_probe.sh`
- 离线提交 FedProx/SCAFFOLD 插件主对比：`scripts/start_remote_5090_baseline_plugin_compare.sh`
- 查看状态：`scripts/check_remote_5090_run.sh`

SSH 登录命令：

```bash
ssh -p 53848 root@connect.bjb2.seetacloud.com
```

使用时只在当前 shell 临时设置密码，不要写入仓库文件或 `AGENTS.md`：

```bash
export REMOTE_5090_PASSWORD='<password>'
./scripts/deploy_to_5090.sh
./scripts/start_remote_5090_privacy_probe.sh
./scripts/start_remote_5090_baseline_plugin_compare.sh
./scripts/check_remote_5090_run.sh --run-id <run_id>
```

默认 5090 配置：

- SSH：`root@connect.bjb2.seetacloud.com -p 53848`
- 远程项目目录：`/root/FedFed`
- 远程运行目录：`/root/runs`
- 远程临时脚本目录：`/root/codex`
- 远程 Python：`/root/miniconda3/bin/python`

### 已新增：SeetaCloud bjb3 Linux 主机

已为 bjb3 Linux 主机新增独立配置和脚本：

- 配置：`scripts/remote_bjb3.env`
- 增量部署：`scripts/deploy_to_bjb3.sh`
- 只上传 CIFAR-10 数据压缩包：`scripts/deploy_cifar10_to_bjb3.sh`
- 离线提交 CIFAR-10 C2-C6 组件消融：`scripts/start_remote_bjb3_cifar10_ablation_c2_c6.sh`
- 查看状态：`scripts/check_remote_bjb3_run.sh`

SSH 登录命令：

```bash
ssh -p 31942 root@connect.bjb3.seetacloud.com
```

使用时只在当前 shell 临时设置密码，不要写入仓库文件或 `AGENTS.md`：

```bash
export REMOTE_BJB3_PASSWORD='<password>'
./scripts/deploy_to_bjb3.sh
./scripts/deploy_cifar10_to_bjb3.sh
./scripts/start_remote_bjb3_cifar10_ablation_c2_c6.sh
./scripts/check_remote_bjb3_run.sh --run-id <run_id>
```

默认 bjb3 配置：

- SSH：`root@connect.bjb3.seetacloud.com -p 31942`
- 远程项目目录：`/root/FedFed`
- 远程运行目录：`/root/runs`
- 远程临时脚本目录：`/root/codex`
- 远程 Python：`/root/miniconda3/bin/python`

bjb3 CIFAR-10 C2-C6 组件消融默认配置：

- 实验 suite：`experiments/fedfed_cifar10_ablation_c2_c6_bjb3.json`
- 五组实验：`C2_NoDistill`、`C3_NoSharedMainTrain`、`C4_NoLogitAlign`、`C5_NoRawCE`、`C6_NoAugMixup`
- 默认主配置沿用 2026-05-06 无噪声 G1 配置：CIFAR-10、ResNet18、`dirichlet_alpha=0.1`、`local_epoch=1`、10 客户端、每轮 5 客户端、`batch_size=64`、`seed=3001`
- 不在主实验和消融实验中开启噪声；噪声只用于隐私验证部分
- 如果远程没有 CIFAR-10 数据且在线下载很慢，可以先运行 `scripts/deploy_cifar10_to_bjb3.sh`，只上传 `data/cifar-10-python.tar.gz`，不上传实验结果或其他数据集

5090 FedProx/SCAFFOLD 插件主对比默认配置：

- 实验 suite：`experiments/baseline_plugin_5090_compare.json`
- 四组实验：`FedProx`、`FedProx + fedfed_image`、`SCAFFOLD`、`SCAFFOLD + fedfed_image`
- 默认主配置：CIFAR-10、ResNet18、`dirichlet_alpha=0.1`、`local_epoch=1`、10 客户端、每轮 5 客户端、`batch_size=64`
- 速度配置：`dataloader_num_workers=8`、`client_empty_cache=false`、FedFed 共享数据常驻 CUDA；不减少主训练轮数、本地 epoch 或 FedFed 蒸馏轮数

5090 privacy probe 脚本会先调用 `scripts/deploy_to_5090.sh` 增量同步最新代码，再通过 `nohup bash <worker>` 离线启动十组裁剪/噪声实验。默认十组为：

- `A0_noclip_none`
- `A1_noclip_g020_025`
- `C14_g020_025`
- `C12_g020_025`
- `C10_g020_025`
- `C12_g010_015`
- `C12_g015_020`
- `C12_g025_030`
- `C14_g015_020`
- `C10_g015_020`

其中 `C12_*` 使用 `clip_norm=12`，`C14_*` 使用 `clip_norm=14`，`C10_*` 使用 `clip_norm=10`，`noise_shape=paper`，`distill_ce_on_noisy_xs=true`。脚本为了不改变性能代理指标，保留 `diagnostic_train_limit=12000`、`diagnostic_test_limit=4000`、`diagnostic_epochs=6`、`batch_size=64`；主要通过 5090 GPU、更高 DataLoader worker、较小 shadow/inversion 诊断规模缩短时间。

## 实验和结果习惯

- 远程训练前优先运行 `./scripts/deploy_to_windows.sh` 或使用会自动部署的 `./scripts/run_windows_experiment.sh`
- smoke run 优先使用：

```bash
./scripts/run_windows_experiment.sh --preset plugin-smoke
./scripts/run_windows_experiment.sh --preset fedavg-smoke
```

- 长实验优先使用队列脚本或 `scripts/run_remote_train.sh`
- 六分类器敏感特征诊断使用：

```bash
./scripts/start_remote_six_classifier_diag_offline.sh
```

该脚本会先调用 `scripts/deploy_to_windows.sh` 同步代码，再上传远程 `.bat` 到 `REMOTE_Codex_DIR`，最后用 Windows `schtasks /Create` + `schtasks /Run` 离线启动。提交后本地 SSH 会话不需要保持打开。默认实验为 5 月 6 日主实验同配置的 FedFed 蒸馏阶段诊断：`x->x`、`xs->xs`、`xr->xr`、`xs->x`、`x+xs->x`、`x+xp->x`，其中 `xp=xs+Gaussian(0,0.2)`，只用一路噪声。
- 检查远程状态优先使用：

```bash
./scripts/check_remote_run_compact.sh --run-id <run_id>
./scripts/check_remote_run.sh --run-id <run_id>
```

- 论文实验结果通常整理到 `remote_results/`、`result/`、`outputs/` 或相关 `docs/` 文档中；不要随意删除已有结果。
