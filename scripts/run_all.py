"""
run_all.py — 一键全流程执行

按顺序执行 Step 1 → Step 8，每步完成后打印状态。
每个步骤完成后可截图控制台输出作为报告素材。
"""
import os
import sys
import subprocess
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")

STEPS = [
    ("Step 1", "step1_load_data.py", "📂 多源语料读取"),
    ("Step 2", "step2_clean_data.py", "🧹 脏数据清洗"),
    ("Step 3", "step3_classify.py", "🏷️  故事/笑话分类+关键词"),
    ("Step 4", "step4_fusion.py", "🔄 跨文件融合多轮对话"),
    ("Step 5", "step5_tokenize_split.py", "✂️  分词+格式统一+划分"),
    ("Step 6", "step6_setup_lora.py", "🏗️  模型搭建+LoRA配置"),
    ("Step 7", "step7_train_test.py", "🚀 训练+推理测试"),
    ("Step 8", "step8_evaluate.py", "📊 模型评价"),
]


def print_banner(text):
    print(f"\n{'=' * 70}")
    print(f"  {text}")
    print(f"{'=' * 70}")


def run_step(step_num, script_name, desc):
    """运行单个步骤"""
    script_path = os.path.join(SCRIPTS_DIR, script_name)

    if not os.path.exists(script_path):
        print(f"\n  ❌ 脚本不存在：{script_path}")
        return False

    print_banner(f"{step_num}: {desc}")
    print(f"  脚本：{script_name}")
    print(f"  开始时间：{time.strftime('%H:%M:%S')}\n")

    start = time.time()
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=False,
        cwd=BASE_DIR,
    )
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"\n  ✅ {step_num} 完成！耗时 {elapsed:.1f} 秒")
        return True
    else:
        print(f"\n  ❌ {step_num} 失败（返回码 {result.returncode}）")
        return False


def main():
    print("=" * 70)
    print("  自然语言处理期末项目 — 一键全流程")
    print(f"  开始时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    total = len(STEPS)
    success_count = 0

    for i, (step_num, script_name, desc) in enumerate(STEPS, 1):
        print(f"\n{'─' * 70}")
        print(f"  [{i}/{total}] {step_num}: {desc}")
        print(f"{'─' * 70}")

        ok = run_step(step_num, script_name, desc)
        if ok:
            success_count += 1
        else:
            print(f"\n  ⚠️  {step_num} 执行出错，是否继续？")
            print(f"     （按 Enter 继续下一个步骤，Ctrl+C 终止）")
            try:
                input()
            except KeyboardInterrupt:
                print(f"\n  用户中断")
                break

    # ========== 汇总 ==========
    print_banner("全流程执行汇总")
    print(f"  总步骤：{total}")
    print(f"  成功：{success_count}")
    print(f"  失败：{total - success_count}")
    print(f"  完成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")

    if success_count == total:
        print(f"\n  🎉 全流程执行成功！")
    else:
        print(f"\n  ⚠️  部分步骤失败，请检查日志")

    print(f"\n  📁 输出文件：")
    print(f"    data/cleaned/    — 清洗后数据")
    print(f"    data/classified/ — 分类后数据")
    print(f"    data/fused/      — 融合多轮对话")
    print(f"    data/split/      — 训练集/测试集")
    print(f"    outputs/lora_adapter/ — LoRA 权重")
    print(f"    outputs/evaluation_report.json — 评价报告")


if __name__ == "__main__":
    main()
