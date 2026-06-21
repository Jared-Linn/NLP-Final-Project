"""
从 RTX 3090 训练日志生成 matplotlib 图表
输出到 outputs_3090/charts/
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

OUTPUT_DIR = 'E:/work/Claude code default/自然语言处理期末/outputs_3090/charts'
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOG_FILE = 'E:/work/Claude code default/自然语言处理期末/RTX3090_training.log'
EVAL_FILE = 'E:/work/Claude code default/自然语言处理期末/outputs_3090/evaluation_report.json'

# ==================== 1. 解析训练日志 ====================
def parse_training_log(log_path):
    """解析训练日志，提取 loss/grad_norm/lr 数据"""
    steps, losses, grad_norms, lrs = [], [], [], []

    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            # 匹配 loss 记录行: {'loss': '3.19', 'grad_norm': '1.156', 'learning_rate': '0.0001995', 'epoch': '0.125'}
            match = re.search(r"\{'loss':\s*'([\d.]+)',\s*'grad_norm':\s*'([\d.na]+)',\s*'learning_rate':\s*'([\d.e\-]+)',\s*'epoch':\s*'([\d.]+)'\}", line)
            if match:
                loss = float(match.group(1))
                gn = match.group(2)
                lr = float(match.group(3))

                steps.append(len(steps) + 1)  # 累加步数索引
                losses.append(loss)
                grad_norms.append(float(gn) if gn != 'nan' else None)
                lrs.append(lr)

    return steps, losses, grad_norms, lrs

# ==================== 2. 生成图表 ====================

def chart_loss_curve(steps, losses, output_path):
    """Loss 收敛曲线"""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(steps, losses, color='#2c7bb6', linewidth=1.2, alpha=0.9, label='Training Loss')

    # 平滑曲线
    if len(losses) > 20:
        window = max(5, len(losses) // 30)
        smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
        smooth_steps = steps[window//2:window//2+len(smoothed)]
        ax.plot(smooth_steps, smoothed, color='#d7191c', linewidth=2, alpha=0.8, label=f'Smoothed (window={window})')

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss')
    ax.set_title('RTX 3090 Training Loss Convergence (Qwen3.5-0.8B + LoRA)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, len(steps))

    # 标注起点和终点
    ax.annotate(f'Start: {losses[0]:.2f}', xy=(steps[0], losses[0]),
                xytext=(steps[0]+len(steps)*0.05, losses[0]+0.3),
                arrowprops=dict(arrowstyle='->', color='gray'), fontsize=9)
    ax.annotate(f'Final: {losses[-1]:.2f}', xy=(steps[-1], losses[-1]),
                xytext=(steps[-1]-len(steps)*0.2, losses[-1]+0.3),
                arrowprops=dict(arrowstyle='->', color='gray'), fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f'  [OK] {os.path.basename(output_path)}')

def chart_loss_and_lr(steps, losses, lrs, output_path):
    """Loss + 学习率双轴图"""
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color1 = '#2c7bb6'
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Loss', color=color1)
    ax1.plot(steps, losses, color=color1, linewidth=1.0, alpha=0.8)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color2 = '#d7191c'
    ax2.set_ylabel('Learning Rate', color=color2)
    ax2.plot(steps, lrs, color=color2, linewidth=1.2, alpha=0.7)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0, max(lrs)*1.1)

    plt.title('Loss Convergence & Cosine LR Schedule (RTX 3090)')
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f'  [OK] {os.path.basename(output_path)}')

def chart_grad_norm(steps, grad_norms, output_path):
    """Gradient Norm 变化"""
    valid = [(s, g) for s, g in zip(steps, grad_norms) if g is not None]
    if not valid:
        return
    s, g = zip(*valid)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(s, g, color='#fdae61', linewidth=0.8, alpha=0.8)
    ax.axhline(y=np.mean(g), color='#d7191c', linestyle='--', linewidth=1, label=f'Mean: {np.mean(g):.2f}')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Gradient Norm')
    ax.set_title('Gradient Norm Stability (RTX 3090)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f'  [OK] {os.path.basename(output_path)}')

def chart_eval_comparison(eval_path, output_path):
    """微调前后评价对比柱状图"""
    if not os.path.exists(eval_path):
        print(f'  [SKIP] eval file not found')
        return

    with open(eval_path, 'r', encoding='utf-8') as f:
        report = json.load(f)

    cases = report.get('scores', {}).get('rule_scores', [])
    if not cases:
        return

    labels = [f"{c['id']}\n{c['dimension']}" for c in cases]
    base_scores = [c['rule_score']['base'] for c in cases]
    ft_scores = [c['rule_score']['ft'] for c in cases]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, base_scores, width, label='Base Model (Before FT)', color='#2c7bb6')
    bars2 = ax.bar(x + width/2, ft_scores, width, label='Fine-tuned Model (After FT)', color='#d7191c')

    ax.set_ylabel('Score (1-5)')
    ax.set_title('Model Evaluation: Before vs After Fine-tuning (Rule-based)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 5.5)
    ax.grid(True, alpha=0.3, axis='y')

    # 数值标注
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{bar.get_height():.1f}', ha='center', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{bar.get_height():.1f}', ha='center', fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f'  [OK] {os.path.basename(output_path)}')

def chart_bleu_rouge(eval_path, output_path):
    """BLEU/ROUGE-L 对比图"""
    if not os.path.exists(eval_path):
        return

    with open(eval_path, 'r', encoding='utf-8') as f:
        report = json.load(f)

    br = report.get('scores', {}).get('bleu_rouge', [])
    if not br:
        return

    labels = [c['id'] for c in br]
    bleu_base = [c['bleu1']['base'] for c in br]
    bleu_ft = [c['bleu1']['ft'] for c in br]
    rouge_base = [c['rouge_l']['base'] for c in br]
    rouge_ft = [c['rouge_l']['ft'] for c in br]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(labels))
    width = 0.35

    ax1.bar(x - width/2, bleu_base, width, label='Base', color='#2c7bb6')
    ax1.bar(x + width/2, bleu_ft, width, label='Fine-tuned', color='#d7191c')
    ax1.set_title('BLEU-1 Score')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    ax2.bar(x - width/2, rouge_base, width, label='Base', color='#2c7bb6')
    ax2.bar(x + width/2, rouge_ft, width, label='Fine-tuned', color='#d7191c')
    ax2.set_title('ROUGE-L Score')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Automatic Metrics: BLEU-1 & ROUGE-L Comparison', fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f'  [OK] {os.path.basename(output_path)}')

def chart_emotion_distribution(output_path):
    """情感分布饼图"""
    emotions = {
        '日常': 589, '焦虑': 526, '人际': 371,
        '抑郁': 192, '家庭': 160, '职场': 84, '学业': 78
    }

    fig, ax = plt.subplots(figsize=(8, 8))
    colors = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f']
    explode = (0, 0.05, 0, 0, 0, 0, 0)

    wedges, texts, autotexts = ax.pie(
        emotions.values(), labels=emotions.keys(), autopct='%1.1f%%',
        colors=colors, explode=explode, startangle=90,
        pctdistance=0.75
    )
    for at in autotexts:
        at.set_fontsize(9)

    ax.set_title('Emotion Distribution in Fused Dialogues (2000 samples)')
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f'  [OK] {os.path.basename(output_path)}')

def chart_speed_comparison(output_path):
    """GTX 1060 vs RTX 3090 对比图"""
    metrics = ['Speed\n(s/step)', 'Samples', 'Epochs', 'Loss\nReduction']
    gtx = [50, 150, 2, 17]
    rtx = [2.81, 1600, 3, 29]

    fig, axes = plt.subplots(1, 4, figsize=(12, 4))

    comparisons = [
        ('Training Speed (s/step)', gtx[0], rtx[0], 'Lower is better'),
        ('Training Samples', gtx[1], rtx[1], 'More is better'),
        ('Training Epochs', gtx[2], rtx[2], 'More is better'),
        ('Loss Reduction (%)', gtx[3], rtx[3], 'More is better'),
    ]

    for i, (title, g, r, note) in enumerate(comparisons):
        ax = axes[i]
        bars = ax.bar(['GTX 1060', 'RTX 3090'], [g, r], color=['#fc8d59', '#2c7bb6'])
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(note, fontsize=8)
        for bar, val in zip(bars, [g, r]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(g,r)*0.02,
                    str(val), ha='center', fontsize=11, fontweight='bold')

    fig.suptitle('GTX 1060 (Local) vs RTX 3090 (Cloud) Training Comparison', fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f'  [OK] {os.path.basename(output_path)}')

# ==================== 主流程 ====================
if __name__ == '__main__':
    print('Generating charts from RTX 3090 training log...\n')

    # 解析日志
    steps, losses, grad_norms, lrs = parse_training_log(LOG_FILE)
    print(f'Parsed {len(steps)} data points from training log\n')

    # 生成图表
    chart_loss_curve(steps, losses, f'{OUTPUT_DIR}/loss_curve.png')
    chart_loss_and_lr(steps, losses, lrs, f'{OUTPUT_DIR}/loss_and_lr.png')
    chart_grad_norm(steps, grad_norms, f'{OUTPUT_DIR}/grad_norm.png')
    chart_eval_comparison(EVAL_FILE, f'{OUTPUT_DIR}/eval_comparison.png')
    chart_bleu_rouge(EVAL_FILE, f'{OUTPUT_DIR}/bleu_rouge.png')
    chart_emotion_distribution(f'{OUTPUT_DIR}/emotion_distribution.png')
    chart_speed_comparison(f'{OUTPUT_DIR}/speed_comparison.png')

    print(f'\nDone! Charts saved to {OUTPUT_DIR}/')
