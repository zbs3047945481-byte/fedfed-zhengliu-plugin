from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

OUT = Path('tmp/docs/figures_54')
OUT.mkdir(parents=True, exist_ok=True)

font_path = '/System/Library/Fonts/STHeiti Light.ttc'
font_manager.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'STHeiti'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 140
plt.rcParams['savefig.dpi'] = 300

blue = '#4C78A8'
orange = '#F58518'
green = '#54A24B'
red = '#E45756'
grid_kw = dict(axis='y', linestyle='--', alpha=0.28)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f'{name}.png', bbox_inches='tight')
    plt.close(fig)


def grouped_bar(labels, best, final, title, name, ylim=(0, 100)):
    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar(x - width / 2, best, width, label='BestAcc', color=blue)
    ax.bar(x + width / 2, final, width, label='FinalAcc', color=orange)
    ax.set_title(title, pad=8)
    ax.set_ylabel('准确率 (%)')
    ax.set_ylim(*ylim)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.grid(**grid_kw)
    ax.legend(frameon=False, ncols=2, loc='upper left')
    for xi, v in zip(x - width / 2, best):
        ax.text(xi, v + 1.0, f'{v:.1f}', ha='center', va='bottom', fontsize=8)
    for xi, v in zip(x + width / 2, final):
        ax.text(xi, v + 1.0, f'{v:.1f}', ha='center', va='bottom', fontsize=8)
    save(fig, name)


grouped_bar(
    ['无插件基线', 'FedFed 插件'],
    [63.89, 88.95],
    [53.83, 88.53],
    '强异构场景主性能对比',
    'fig_5_5_main_bar',
)

# Main curve: derived from original metrics when available; otherwise use compact trend.
import json
ROOT = Path('/Users/zhubingshuo/plug-and-play-fedfed-fedavg-plugin')
summary = ROOT / 'remote_results/fedfed_thesis_15run_4090_fast_20260506T140411Z/fedfed_thesis_15run_4090_fast_20260506T140411Z_summary.csv'
metrics_paths = {}
import csv
with summary.open(encoding='utf-8') as f:
    for row in csv.DictReader(f):
        if row['experiment_id'] == 'main_a0p1_e1':
            metrics_paths[row['method']] = Path(row['metrics_path'].replace('/root/FedFed-thesis-suite', str(ROOT)))

fig, ax = plt.subplots(figsize=(6.4, 3.6))
for method, label, color in [('FedAvg', '无插件基线', blue), ('FedFedPlugin', 'FedFed 插件', orange)]:
    p = metrics_paths.get(method)
    if p and p.exists():
        m = json.loads(p.read_text(encoding='utf-8'))
        rounds = m.get('rounds', [])
        acc = [v * 100 for v in m.get('acc_on_g_test_data', [])]
        ax.plot(rounds, acc, label=label, color=color, linewidth=1.8)
ax.set_title('强异构场景收敛曲线', pad=8)
ax.set_xlabel('通信轮次')
ax.set_ylabel('测试准确率 (%)')
ax.set_ylim(0, 100)
ax.grid(True, linestyle='--', alpha=0.28)
ax.legend(frameon=False)
save(fig, 'fig_5_6_main_curve')

labels = ['α=0.3', 'α=0.1', 'α=0.05']
fedavg = [71.95, 63.89, 38.15]
fedfed = [91.98, 88.95, 84.80]
fig, ax = plt.subplots(figsize=(6.4, 3.6))
x = np.arange(len(labels))
w = 0.34
ax.bar(x - w / 2, fedavg, w, label='无插件基线', color=blue)
ax.bar(x + w / 2, fedfed, w, label='FedFed 插件', color=orange)
ax.set_title('不同数据异构强度下的最高准确率', pad=8)
ax.set_ylabel('BestAcc (%)')
ax.set_ylim(0, 100)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.grid(**grid_kw)
ax.legend(frameon=False, ncols=2, loc='upper left')
for xi, v in zip(x - w / 2, fedavg):
    ax.text(xi, v + 1, f'{v:.1f}', ha='center', fontsize=8)
for xi, v in zip(x + w / 2, fedfed):
    ax.text(xi, v + 1, f'{v:.1f}', ha='center', fontsize=8)
save(fig, 'fig_5_7_heter_bar')

gain = [20.03, 25.06, 46.65]
fig, ax = plt.subplots(figsize=(6.0, 3.4))
ax.plot(labels, gain, marker='o', color=red, linewidth=2)
ax.set_title('不同数据异构强度下的插件增益', pad=8)
ax.set_ylabel('BestAcc 提升 (百分点)')
ax.set_ylim(0, 55)
ax.grid(True, linestyle='--', alpha=0.28)
for i, v in enumerate(gain):
    ax.text(i, v + 1.5, f'+{v:.2f}', ha='center', fontsize=8)
save(fig, 'fig_5_8_heter_gain')

grouped_bar(
    ['无插件基线', 'FedFed 插件'],
    [59.60, 91.90],
    [58.57, 91.01],
    '本地训练增强场景性能对比',
    'fig_5_9_local_bar',
)

grouped_bar(
    ['低容量共享池', '中等容量共享池', '不设额外限制'],
    [80.31, 80.12, 90.33],
    [75.30, 75.48, 89.50],
    '不同共享特征池规模下的性能对比',
    'fig_5_10_pool_bar',
)

grouped_bar(
    ['短程蒸馏', '标准蒸馏', '长程蒸馏', '长程增强'],
    [89.73, 88.41, 89.26, 90.30],
    [89.43, 87.50, 88.73, 88.19],
    '不同特征蒸馏训练量下的性能对比',
    'fig_5_11_distill_bar',
)

print(OUT.resolve())
