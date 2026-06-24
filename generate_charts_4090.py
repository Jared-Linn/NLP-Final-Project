"""
从 RTX 4090D 云端训练日志生成 matplotlib 图表
输出到 outputs_cloud/charts/
"""
import re, json, os, sys
sys.stdout.reconfigure(encoding='utf-8')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ==================== 配置 ====================
plt.rcParams.update({
    'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'axes.unicode_minus': False,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

OUTPUT_DIR = 'E:/work/Claude code default/自然语言处理期末/outputs_cloud/charts'
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOG_FILE = 'E:/work/Claude code default/自然语言处理期末/outputs_cloud/train_output.log'
GEN_REPORT = 'E:/work/Claude code default/自然语言处理期末/outputs_cloud/gen_report_v2.json'

# ==================== 1. 解析训练日志 ====================
def parse_training_log(log_path):
    """解析训练日志，提取 loss/grad_norm/lr 数据"""
    steps, losses, grad_norms, lrs = [], [], [], []

    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            match = re.search(r"\{'loss':\s*'([\d.]+)',\s*'grad_norm':\s*'([\d.na]+)',\s*'learning_rate':\s*'([\d.e\-]+)',\s*'epoch':\s*'([\d.]+)'\}", line)
            if match:
                loss = float(match.group(1))
                gn = match.group(2)
                lr = float(match.group(3))

                steps.append(len(steps) + 1)
                losses.append(loss)
                grad_norms.append(float(gn) if gn != 'nan' else None)
                lrs.append(lr)

    return steps, losses, grad_norms, lrs

# ==================== 2. 生成图表 ====================

def chart_loss_curve(steps, losses, output_path):
    """Loss 收敛曲线（4090D）"""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(steps, losses, color='#2c7bb6', linewidth=0.6, alpha=0.7, label='Training Loss')

    # 平滑曲线
    if len(losses) > 20:
        window = max(10, len(losses) // 40)
        smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
        smooth_steps = steps[window//2:window//2+len(smoothed)]
        ax.plot(smooth_steps, smoothed, color='#d7191c', linewidth=2, alpha=0.9, label=f'Smoothed (window={window})')

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss')
    ax.set_title('RTX 4090D Training Loss Convergence (Qwen3.5-0.8B + LoRA · 8,000 samples)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, len(steps))

    # 标注起点和终点
    ax.annotate(f'Start: {losses[0]:.2f}', xy=(steps[0], losses[0]),
                xytext=(steps[0]+len(steps)*0.05, losses[0]+0.5),
                arrowprops=dict(arrowstyle='->', color='gray'), fontsize=9, fontweight='bold')
    ax.annotate(f'Final: {losses[-1]:.2f}', xy=(steps[-1], losses[-1]),
                xytext=(steps[-1]-len(steps)*0.18, losses[-1]+0.6),
                arrowprops=dict(arrowstyle='->', color='gray'), fontsize=9, fontweight='bold')
    # 标注最低点
    min_idx = np.argmin(losses)
    ax.annotate(f'Min: {losses[min_idx]:.2f} @ step {steps[min_idx]}',
                xy=(steps[min_idx], losses[min_idx]),
                xytext=(steps[min_idx]-len(steps)*0.15, losses[min_idx]-1.0),
                arrowprops=dict(arrowstyle='->', color='green'), fontsize=8, color='green')

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f'  [OK] {os.path.basename(output_path)}')

def chart_loss_and_lr(steps, losses, lrs, output_path):
    """Loss + 学习率双轴图（4090D）"""
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color1 = '#2c7bb6'
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Loss', color=color1)
    ax1.plot(steps, losses, color=color1, linewidth=0.5, alpha=0.7)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color2 = '#d7191c'
    ax2.set_ylabel('Learning Rate', color=color2)
    ax2.plot(steps, lrs, color=color2, linewidth=1.2, alpha=0.8)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0, max(lrs)*1.1)

    plt.title('Loss Convergence & Cosine LR Schedule (RTX 4090D · 8,000 samples · 3 epochs)')
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f'  [OK] {os.path.basename(output_path)}')

def chart_grad_norm(steps, grad_norms, output_path):
    """Gradient Norm 变化（4090D）"""
    valid = [(s, g) for s, g in zip(steps, grad_norms) if g is not None]
    if not valid:
        return
    s, g = zip(*valid)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(s, g, color='#fdae61', linewidth=0.5, alpha=0.7)
    mean_gn = np.mean(g)
    ax.axhline(y=mean_gn, color='#d7191c', linestyle='--', linewidth=1.5, label=f'Mean: {mean_gn:.2f}')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Gradient Norm')
    ax.set_title('Gradient Norm Stability (RTX 4090D · 12,000 steps)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f'  [OK] {os.path.basename(output_path)}')

def chart_gen_test_report(report_path, output_path):
    """泛化测试结果柱状图"""
    if not os.path.exists(report_path):
        print(f'  [SKIP] gen_report not found')
        return

    with open(report_path, 'r', encoding='utf-8') as f:
        report = json.load(f)

    results = report.get('results', [])
    if not results:
        return

    names = [r['name'].replace('-', '-\n', 1) if '-' in r['name'] else r['name'] for r in results]
    rates = [r['pass_rate'] for r in results]
    totals = [r['total'] for r in results]

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(names))
    colors = ['#1f78b4' if r >= 60 else '#e31a1c' if r < 50 else '#fdbf6f' for r in rates]
    bars = ax.bar(x, rates, color=colors, width=0.6, edgecolor='white', linewidth=0.5)

    # 标注数值
    for bar, rate, total in zip(bars, rates, totals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f'{rate:.0f}%\n({total} cases)', ha='center', fontsize=9, fontweight='bold')

    ax.set_ylabel('Pass Rate (%)')
    ax.set_title('Generalization Test Results (v3 · RTX 4090D · LoRA · 8,000 samples)')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 115)
    ax.axhline(y=60, color='green', linestyle='--', linewidth=1, alpha=0.7, label='Pass threshold (60%)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f'  [OK] {os.path.basename(output_path)}')

def chart_speed_comparison(output_path):
    """GTX 1060 vs RTX 3090 vs RTX 4090D 训练速度对比"""
    labels = ['Speed\n(s/step)', 'Train\nSamples', 'Epochs', 'Final\nLoss']
    gtx = [50, 150, 2, 4.3]       # GTX 1060 本地
    rtx3090 = [2.81, 1600, 3, 3.08]  # RTX 3090 云端
    rtx4090d = [1.27, 8000, 3, 2.79]  # RTX 4090D 云端

    fig, axes = plt.subplots(1, 4, figsize=(14, 4.5))

    comparisons = [
        ('Training Speed (s/step)\nLower = Better', [50, 2.81, 1.27]),
        ('Training Samples\nMore = Better', [150, 1600, 8000]),
        ('Training Epochs', [2, 3, 3]),
        ('Final Loss\nLower = Better', [4.3, 3.08, 2.79]),
    ]

    for i, (title, values) in enumerate(comparisons):
        ax = axes[i]
        bars = ax.bar(['GTX 1060', 'RTX 3090', 'RTX 4090D'], values,
                       color=['#fc8d59', '#2c7bb6', '#33a02c'])
        ax.set_title(title, fontsize=10, fontweight='bold')
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.03,
                    str(val), ha='center', fontsize=10, fontweight='bold')

    fig.suptitle('Training Comparison: GTX 1060 vs RTX 3090 vs RTX 4090D', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f'  [OK] {os.path.basename(output_path)}')

def chart_training_time_comparison(output_path):
    """训练耗时对比"""
    fig, ax = plt.subplots(figsize=(8, 5))

    gpus = ['GTX 1060\n(Local)', 'RTX 3090\n(创投云)', 'RTX 4090D\n(AutoDL)']
    times_min = [50, 56, 270]  # minutes
    samples = [150, 1600, 8000]
    colors = ['#fc8d59', '#2c7bb6', '#33a02c']

    bars = ax.bar(gpus, times_min, color=colors, width=0.5, edgecolor='white', linewidth=0.5)
    for bar, t, s in zip(bars, times_min, samples):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{t} min\n({s} samples)', ha='center', fontsize=10, fontweight='bold')

    ax.set_ylabel('Training Time (minutes)')
    ax.set_title('Training Time Comparison Across GPUs')
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f'  [OK] {os.path.basename(output_path)}')


# ==================== 主流程 ====================
if __name__ == '__main__':
    print(f'Generating charts from RTX 4090D cloud training log...\n')

    # 解析日志
    steps, losses, grad_norms, lrs = parse_training_log(LOG_FILE)
    print(f'Parsed {len(steps)} data points from training log')
    print(f'  Loss: {losses[1]:.2f} → {losses[-1]:.2f} (min: {min(losses):.2f})')
    print(f'  Recorded steps: {len(steps)} of ~12,000 total steps\n')

    # 生成 4090D 专属图表
    chart_loss_curve(steps, losses, f'{OUTPUT_DIR}/loss_curve_4090.png')
    chart_loss_and_lr(steps, losses, lrs, f'{OUTPUT_DIR}/loss_and_lr_4090.png')
    chart_grad_norm(steps, grad_norms, f'{OUTPUT_DIR}/grad_norm_4090.png')
    chart_gen_test_report(GEN_REPORT, f'{OUTPUT_DIR}/gen_test_report_4090.png')
    chart_speed_comparison(f'{OUTPUT_DIR}/speed_comparison_4090.png')
    chart_training_time_comparison(f'{OUTPUT_DIR}/training_time_4090.png')

    print(f'\nDone! Charts saved to {OUTPUT_DIR}/')
    print(f'Total: 6 charts generated')
