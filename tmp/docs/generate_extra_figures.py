from pathlib import Path
import csv
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ROOT = Path('/Users/zhubingshuo/plug-and-play-fedfed-fedavg-plugin')
OUT = ROOT / 'tmp/docs/extra_figures'
OUT.mkdir(parents=True, exist_ok=True)

font_path = '/System/Library/Fonts/STHeiti Light.ttc'
font_manager.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'STHeiti'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 300

BLUE = '#4C78A8'
ORANGE = '#F58518'
GREEN = '#54A24B'
RED = '#E45756'
GRID = dict(axis='y', linestyle='--', alpha=0.28)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f'{name}.png', bbox_inches='tight')
    plt.close(fig)


# 5-12: feature distillation diagnostic.
diag_path = ROOT / 'remote_results/fedfed_six_classifier_diag_20260513T013033Z/diagnostic_summary.csv'
diag_rows = list(csv.DictReader(diag_path.open(encoding='utf-8')))
label_map = {
    'x_to_x': 'x→x',
    'xs_to_xs': 'xs→xs',
    'xr_to_xr': 'xr→xr',
    'xs_to_x': 'xs→x',
    'x_plus_xs_to_x': 'x+xs→x',
    'x_plus_xp_to_x': 'x+xp→x',
}
labels = [label_map[r['mode']] for r in diag_rows]
best = [float(r['best_acc']) * 100 for r in diag_rows]
final = [float(r['final_acc']) * 100 for r in diag_rows]
x = np.arange(len(labels))
w = 0.34
fig, ax = plt.subplots(figsize=(6.6, 3.8))
ax.bar(x - w / 2, best, w, label='BestAcc', color=BLUE)
ax.bar(x + w / 2, final, w, label='FinalAcc', color=ORANGE)
ax.set_title('特征蒸馏机制诊断结果', pad=8)
ax.set_ylabel('准确率 (%)')
ax.set_ylim(0, 85)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.grid(**GRID)
ax.legend(frameon=False, ncols=2, loc='upper left')
for xi, v in zip(x - w / 2, best):
    ax.text(xi, v + 1.0, f'{v:.1f}', ha='center', fontsize=8)
save(fig, 'fig_5_12_diagnostic')


# 5-13: cross-optimizer plugin comparison.
cross_path = ROOT / 'remote_results/baseline_plugin_8way_20260515/combined_summary.csv'
rows = list(csv.DictReader(cross_path.open(encoding='utf-8')))
pairs = [('FedProx', 'FedProxFedFedPlugin'), ('SCAFFOLD', 'SCAFFOLDFedFedPlugin'),
         ('FedAvgM', 'FedAvgMFedFedPlugin'), ('FedNova', 'FedNovaFedFedPlugin')]
lookup = {r['method']: r for r in rows}
labels = [p[0] for p in pairs]
base = [float(lookup[p[0]]['best_acc']) * 100 for p in pairs]
plug = [float(lookup[p[1]]['best_acc']) * 100 for p in pairs]
x = np.arange(len(labels))
w = 0.34
fig, ax = plt.subplots(figsize=(6.6, 3.8))
ax.bar(x - w / 2, base, w, label='关闭插件', color=BLUE)
ax.bar(x + w / 2, plug, w, label='开启插件', color=ORANGE)
ax.set_title('不同联邦优化器下的插件性能', pad=8)
ax.set_ylabel('BestAcc (%)')
ax.set_ylim(0, 100)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.grid(**GRID)
ax.legend(frameon=False, ncols=2, loc='upper left')
for xi, b, p in zip(x, base, plug):
    ax.text(xi + w / 2, p + 1.0, f'+{p-b:.1f}', ha='center', fontsize=8)
save(fig, 'fig_5_13_cross_optimizer')


# 5-14: overhead.
thesis_path = ROOT / 'remote_results/fedfed_thesis_15run_4090_fast_20260506T140411Z/fedfed_thesis_15run_4090_fast_20260506T140411Z_summary.csv'
rows = list(csv.DictReader(thesis_path.open(encoding='utf-8')))
settings = [
    ('α=0.1, E=1', 'main_a0p1_e1'),
    ('α=0.1, E=5', 'local_epoch'),
]
base_times = []
plug_times = []
for _, exp in settings:
    subset = [r for r in rows if r['experiment_id'] == exp and r['method'] in {'FedAvg', 'FedFedPlugin'}]
    base_times.append(float([r for r in subset if r['method'] == 'FedAvg'][0]['duration_min']))
    plug_times.append(float([r for r in subset if r['method'] == 'FedFedPlugin'][0]['duration_min']))
labels = [s[0] for s in settings]
x = np.arange(len(labels))
w = 0.34
fig, ax = plt.subplots(figsize=(6.2, 3.6))
ax.bar(x - w / 2, base_times, w, label='无插件基线', color=BLUE)
ax.bar(x + w / 2, plug_times, w, label='FedFed 插件', color=ORANGE)
ax.set_title('训练时间开销对比', pad=8)
ax.set_ylabel('训练时间 (min)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.grid(**GRID)
ax.legend(frameon=False, ncols=2, loc='upper left')
for xi, b, p in zip(x, base_times, plug_times):
    ax.text(xi + w / 2, p + 12, f'{p/b:.2f}×', ha='center', fontsize=8)
save(fig, 'fig_5_14_overhead')

print(OUT)
