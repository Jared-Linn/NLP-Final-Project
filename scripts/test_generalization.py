"""
泛化能力测试脚本 — 覆盖多种对话场景

测试场景：
  A. 纯共情：用户倾诉，检测模型是否不强制推荐故事/笑话
  B. 拒绝处理：模型推荐→用户拒绝→检测模型是否尊重意愿
  C. 直接索要：用户跳过情感倾诉，直接要故事/笑话
  D. 连续多轮：5轮+对话不涉及故事/笑话
  E. 话题切换：对话中途改变话题

输出：JSON 测试报告，包含通过率和详细结果
"""
import json
import os
import sys
import time
sys.stdout.reconfigure(encoding='utf-8')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "Qwen3.5-0.8B")
LORA_PATH = os.path.join(BASE_DIR, "outputs", "lora_adapter", "lora_adapter")

SYSTEM_PROMPT = "你是一位温暖专业的心理咨询师，善于倾听和共情。当来访者需要放松时，你会讲有趣的笑话或温暖的故事来安慰他们。"

# 推荐关键词：检测模型是否在推荐故事/笑话
RECOMMEND_KEYWORDS = ["故事", "笑话", "讲个", "听个", "给你讲", "分享一个"]


def load_model():
    """加载模型和分词器"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"设备：{device.upper()} | 精度：{dtype}")

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=dtype,
        device_map="cpu", trust_remote_code=True,
        attn_implementation="eager",
    )
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    if device == "cuda":
        try:
            model = model.to("cuda").half()
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            model = model.to("cpu").float()
            device = "cpu"
    else:
        model = model.float()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, device


def generate(model, tokenizer, device, prompt, max_new=200):
    """生成回复"""
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) if device == "cuda" else v for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new, temperature=0.7, top_p=0.9,
            repetition_penalty=1.1, eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    if "<|im_start|>assistant\n" in response:
        answer = response.split("<|im_start|>assistant\n")[-1]
        answer = answer.split("<|im_end|>")[0].strip()
    else:
        answer = response.strip()
    return answer


# ============================================================
# 测试 A：纯共情测试 — 模型不应每次都推荐故事/笑话
# ============================================================
def test_pure_empathy(model, tokenizer, device):
    """用户倾诉情绪，模型是否能做纯共情回复（不总是推荐故事/笑话）"""
    print("\n" + "=" * 65)
    print("  测试 A：纯共情场景")
    print("  检测：用户倾诉情绪后，模型是否不强制推荐故事/笑话")
    print("=" * 65)

    test_inputs = [
        "最近工作压力好大，每天都睡不好，总是半夜惊醒",
        "和男朋友分手了，感觉整个世界都塌了",
        "我爸妈总是吵架，我夹在中间好难受",
        "考研复习了半年，模考成绩还是不理想，很沮丧",
        "刚入职什么都不懂，同事都很忙没人带我，很焦虑",
    ]

    results = []
    recommend_count = 0

    for i, user_input in enumerate(test_inputs, 1):
        prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
        reply = generate(model, tokenizer, device, prompt)

        has_recommend = any(kw in reply for kw in RECOMMEND_KEYWORDS)
        if has_recommend:
            recommend_count += 1

        results.append({
            "input": user_input,
            "reply": reply[:200],
            "has_recommend": has_recommend,
        })

        status = "⚠️ 推荐了" if has_recommend else "✅ 纯共情"
        print(f"\n  [{i}] {status}")
        print(f"  👤 {user_input[:50]}...")
        print(f"  🤖 {reply[:120]}...")

    pass_rate = (len(test_inputs) - recommend_count) / len(test_inputs) * 100
    print(f"\n  📊 结果：{len(test_inputs) - recommend_count}/{len(test_inputs)} 纯共情（{pass_rate:.0f}%）")
    print(f"     理想：不推荐比例 ≥ 60%（即 5 条中至少 3 条不做推荐）")

    return {
        "name": "A-纯共情场景",
        "total": len(test_inputs),
        "passed": len(test_inputs) - recommend_count,
        "pass_rate": pass_rate,
        "ideal_threshold": 60,
        "details": results,
    }


# ============================================================
# 测试 B：拒绝处理 — 用户拒绝后模型应尊重
# ============================================================
def test_decline_handling(model, tokenizer, device):
    """模型推荐→用户拒绝→模型是否尊重意愿继续共情"""
    print("\n" + "=" * 65)
    print("  测试 B：拒绝处理场景")
    print("  检测：用户拒绝推荐后，模型是否尊重意愿")
    print("=" * 65)

    test_cases = [
        {
            "name": "焦虑+拒绝",
            "turns": [
                "最近总是失眠，精神状态很差",
                "算了，我不想听笑话，你能陪我聊聊吗？",
            ]
        },
        {
            "name": "抑郁+拒绝",
            "turns": [
                "我对什么都提不起兴趣了，觉得自己好没用",
                "不用讲那些了，我就想有个人听我说说话",
            ]
        },
        {
            "name": "学业+拒绝",
            "turns": [
                "考试压力太大了，我觉得自己要撑不住了",
                "你理解错我的意思了，我不是想听故事，你能不能给我点建议？",
            ]
        },
    ]

    results = []
    for i, case in enumerate(test_cases, 1):
        # 第一轮：用户倾诉
        history = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        prompt1 = history + f"<|im_start|>user\n{case['turns'][0]}<|im_end|>\n<|im_start|>assistant\n"
        reply1 = generate(model, tokenizer, device, prompt1)

        # 第二轮：用户拒绝
        history += f"<|im_start|>user\n{case['turns'][0]}<|im_end|>\n<|im_start|>assistant\n{reply1}<|im_end|>\n"
        prompt2 = history + f"<|im_start|>user\n{case['turns'][1]}<|im_end|>\n<|im_start|>assistant\n"
        reply2 = generate(model, tokenizer, device, prompt2)

        # 检测第二轮是否还在强行推荐
        still_recommending = any(kw in reply2 for kw in RECOMMEND_KEYWORDS)
        has_empathy = any(kw in reply2 for kw in ["理解", "没关系", "好的", "陪", "聊", "说说", "听你"])

        passed = not still_recommending and has_empathy

        results.append({
            "name": case["name"],
            "turn1_user": case["turns"][0],
            "turn1_assistant": reply1[:150],
            "turn2_user": case["turns"][1],
            "turn2_assistant": reply2[:150],
            "still_recommending": still_recommending,
            "has_empathy": has_empathy,
            "passed": passed,
        })

        status = "✅ 尊重意愿" if passed else "❌ 仍在推荐"
        print(f"\n  [{i}] {case['name']}: {status}")
        print(f"  👤 T1: {case['turns'][0][:50]}...")
        print(f"  🤖 T1: {reply1[:80]}...")
        print(f"  👤 T2: {case['turns'][1][:50]}...")
        print(f"  🤖 T2: {reply2[:80]}...")

    passed_count = sum(1 for r in results if r["passed"])
    pass_rate = passed_count / len(test_cases) * 100
    print(f"\n  📊 结果：{passed_count}/{len(test_cases)} 通过（{pass_rate:.0f}%）")

    return {
        "name": "B-拒绝处理",
        "total": len(test_cases),
        "passed": passed_count,
        "pass_rate": pass_rate,
        "details": results,
    }


# ============================================================
# 测试 C：直接索要 — 跳过情感倾诉直接要故事/笑话
# ============================================================
def test_direct_ask(model, tokenizer, device):
    """用户直接索要故事/笑话，无情感倾诉前置"""
    print("\n" + "=" * 65)
    print("  测试 C：直接索要场景")
    print("  检测：用户直接要故事/笑话时，模型能否应对")
    print("=" * 65)

    test_inputs = [
        "你讲个笑话吧",
        "给我讲个故事",
        "有什么好笑的笑话吗？",
        "我今天有点无聊，讲个故事听听",
    ]

    results = []
    for i, user_input in enumerate(test_inputs, 1):
        prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
        reply = generate(model, tokenizer, device, prompt)

        # 检测是否给出了故事/笑话内容（回复够长说明讲了）
        has_content = len(reply) >= 20
        is_repeating = reply.count(reply[:10]) > 3 if len(reply) >= 10 else False

        passed = has_content and not is_repeating

        results.append({
            "input": user_input,
            "reply": reply[:200],
            "has_content": has_content,
            "is_repeating": is_repeating,
            "passed": passed,
        })

        status = "✅" if passed else "❌ 重复或过短"
        print(f"\n  [{i}] {status}")
        print(f"  👤 {user_input}")
        print(f"  🤖 {reply[:120]}...")

    passed_count = sum(1 for r in results if r["passed"])
    pass_rate = passed_count / len(test_inputs) * 100
    print(f"\n  📊 结果：{passed_count}/{len(test_inputs)} 正常回复（{pass_rate:.0f}%）")

    return {
        "name": "C-直接索要",
        "total": len(test_inputs),
        "passed": passed_count,
        "pass_rate": pass_rate,
        "details": results,
    }


# ============================================================
# 测试 D：连续多轮（不涉及故事/笑话）
# ============================================================
def test_multi_turn_no_story(model, tokenizer, device):
    """5轮+对话，全程不涉及故事/笑话推荐"""
    print("\n" + "=" * 65)
    print("  测试 D：连续多轮纯共情场景")
    print("  检测：多轮对话中模型能否保持共情不跑偏")
    print("=" * 65)

    turns = [
        "我最近心情不太好",
        "工作上遇到了一些麻烦，老板总是不理解我",
        "我跟他解释过，但他根本听不进去",
        "我现在每天都害怕去上班",
        "你说的对，只是有时候我觉得自己撑不下去了",
    ]

    history = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
    replies = []
    forced_recommend_count = 0

    for i, user_msg in enumerate(turns, 1):
        prompt = history + f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"
        reply = generate(model, tokenizer, device, prompt)
        history += f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n{reply}<|im_end|>\n"
        replies.append(reply)

        has_forced = any(kw in reply for kw in RECOMMEND_KEYWORDS)
        if has_forced:
            forced_recommend_count += 1

        status = "⚠️ 强行推荐" if has_forced else "✅ 共情"
        print(f"  [{i}] {status}")
        print(f"  👤 {user_msg[:50]}...")
        print(f"  🤖 {reply[:80]}...")
        print()

    passed = forced_recommend_count <= 1  # 允许5轮中最多1次推荐
    pass_rate = (len(turns) - forced_recommend_count) / len(turns) * 100
    print(f"  📊 结果：{len(turns) - forced_recommend_count}/{len(turns)} 轮纯共情（{pass_rate:.0f}%）")

    return {
        "name": "D-连续多轮",
        "total": len(turns),
        "forced_recommend_count": forced_recommend_count,
        "passed": forced_recommend_count,
        "pass_rate": pass_rate,
        "details": [{"turn": i + 1, "user": turns[i], "assistant": replies[i][:150]} for i in range(len(turns))],
    }


# ============================================================
# 测试 E：话题切换
# ============================================================
def test_topic_switch(model, tokenizer, device):
    """对话中途切换话题"""
    print("\n" + "=" * 65)
    print("  测试 E：话题切换场景")
    print("  检测：对话中途切换话题，模型是否能跟随")
    print("=" * 65)

    turns = [
        "我最近失眠很严重",                                     # T1: 睡眠问题
        "其实不只是失眠，我最近跟室友也处不好",                    # T2: 切换到人际关系
        "对啊，他晚上总是打游戏到很晚，严重影响我休息",            # T3: 继续人际关系
    ]

    history = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
    replies = []
    topic_coherence = []

    for i, user_msg in enumerate(turns, 1):
        prompt = history + f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"
        reply = generate(model, tokenizer, device, prompt)
        history += f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n{reply}<|im_end|>\n"
        replies.append(reply)

        # 简单检测：回复不为空且不太短
        has_content = len(reply) >= 15
        not_repeating = not (len(reply) > 30 and reply.count(reply[:10]) > 4)
        passed = has_content and not_repeating
        topic_coherence.append(passed)

        status = "✅" if passed else "❌"
        print(f"  [{i}] {status}")
        print(f"  👤 {user_msg[:50]}...")
        print(f"  🤖 {reply[:80]}...")
        print()

    passed_count = sum(topic_coherence)
    pass_rate = passed_count / len(turns) * 100
    print(f"  📊 结果：{passed_count}/{len(turns)} 轮正常（{pass_rate:.0f}%）")

    return {
        "name": "E-话题切换",
        "total": len(turns),
        "passed": passed_count,
        "pass_rate": pass_rate,
        "details": [{"turn": i + 1, "user": turns[i], "assistant": replies[i][:150]} for i in range(len(turns))],
    }


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 65)
    print("  泛化能力测试 — 多场景综合评估")
    print("=" * 65)

    # 检查 LoRA 权重
    if not os.path.exists(LORA_PATH):
        print(f"❌ 未找到 LoRA 权重：{LORA_PATH}")
        print("   请先训练模型（step7）再运行此测试")
        return

    # 加载模型
    print("\n⏳ 加载模型...")
    model, tokenizer, device = load_model()
    print("✅ 模型加载完成\n")

    # 运行全部测试
    all_results = []
    start_time = time.time()

    all_results.append(test_pure_empathy(model, tokenizer, device))
    all_results.append(test_decline_handling(model, tokenizer, device))
    all_results.append(test_direct_ask(model, tokenizer, device))
    all_results.append(test_multi_turn_no_story(model, tokenizer, device))
    all_results.append(test_topic_switch(model, tokenizer, device))

    elapsed = time.time() - start_time

    # 汇总
    print("\n" + "=" * 65)
    print("  测试汇总")
    print("=" * 65)

    total_tests = len(all_results)
    # 各测试的通过标准不同，使用各测试自己定义的 pass_rate 作为参考
    print(f"\n  {'测试场景':<25} {'样本数':<8} {'通过率':<10} {'评价':<10}")
    print(f"  {'-' * 53}")
    for r in all_results:
        passed_count = r.get("passed", 0)
        # 对于计数型的测试，passed 是实际通过的样本数
        if isinstance(passed_count, bool):
            passed_count = 1 if passed_count else 0
        pct = r.get("pass_rate", 0)
        if r["name"].startswith("D"):
            # 连续多轮：passed 是强行推荐次数，反过来的
            pct = 100 - r.get("pass_rate", 0)
            evaluation = "✅ 好" if r.get("forced_recommend_count", 999) <= 1 else "⚠️ 需改进"
        elif r["name"].startswith("A"):
            evaluation = "✅ 好" if r["pass_rate"] >= 60 else "⚠️ 偏多推荐"
        elif r["name"].startswith("B"):
            evaluation = "✅ 好" if r["pass_rate"] >= 66 else "⚠️ 需改进"
        elif r["name"].startswith("C"):
            evaluation = "✅ 好" if r["pass_rate"] >= 75 else "⚠️ 需改进"
        else:
            evaluation = "✅ 好" if r["pass_rate"] >= 60 else "⚠️ 需改进"

        print(f"  {r['name']:<25} {r['total']:<8} {pct:.0f}%{'':<6} {evaluation:<10}")

    # 保存报告
    report = {
        "test_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(elapsed, 1),
        "model": "Qwen3.5-0.8B + LoRA",
        "lora_path": LORA_PATH,
        "results": all_results,
    }

    output_dir = os.path.join(BASE_DIR, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "generalization_test_report.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n💾 测试报告已保存：{output_path}")
    print(f"⏱  总耗时：{elapsed:.1f} 秒")
    print("✅ 泛化测试完成。")


if __name__ == "__main__":
    main()
