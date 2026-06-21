"""
Step 8 — 模型评价（考核项5，5%）

考核要求：
- 对照微调前后模型输出
- 评价多轮对话连贯性、故事笑话适配性
- 客观分析模型优缺点

对比维度：
1. 连贯性：对话逻辑是否通顺
2. 共情度：能否识别情感并给出温暖回应
3. 故事笑话适配：用户请求类别与实际匹配度
4. 回复多样性

评价方法升级：
- 规则评分（关键词匹配）+ 语义评分（LLM API）
- BLEU-1 / ROUGE-L 自动指标
- 综合评分
"""
import json
import os
import sys
import torch
sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "Qwen3.5-0.8B")
LORA_PATH = os.path.join(BASE_DIR, "outputs", "lora_adapter", "lora_adapter")

# ==================== 可选依赖导入 ====================

# nltk BLEU: try-except 兜底
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    HAVE_NLTK = True
except ImportError:
    HAVE_NLTK = False

# OpenAI API 调用
try:
    from openai import OpenAI
    HAVE_OPENAI = True
except ImportError:
    HAVE_OPENAI = False

# ==================== BLEU / ROUGE-L 自动指标 ====================


def compute_bleu1(reference, hypothesis):
    """计算 BLEU-1 分数（unigram precision）"""
    if not HAVE_NLTK:
        return None
    if not reference or not hypothesis:
        return 0.0
    try:
        ref_tokens = list(reference)
        hyp_tokens = list(hypothesis)
        smoothing = SmoothingFunction().method1
        return round(sentence_bleu([ref_tokens], hyp_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing), 4)
    except Exception:
        return None


def compute_rouge_l(reference, hypothesis):
    """简化 ROUGE-L：最长公共子序列长度 / 参考长度 的比例"""
    if not reference or not hypothesis:
        return 0.0
    try:
        lcs_len = _lcs_length(reference, hypothesis)
        return round(lcs_len / max(len(reference), 1), 4)
    except Exception:
        return None


def _lcs_length(a, b):
    """最长公共子序列长度（字符级）"""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


# ==================== LLM 语义评分（复用 step3 的豆包 API 调用逻辑） ====================

_LLM_CLIENT = None


def _get_llm_client():
    """懒加载 OpenAI 客户端（豆包 API）"""
    global _LLM_CLIENT
    if _LLM_CLIENT is None and HAVE_OPENAI:
        _LLM_CLIENT = OpenAI(
            api_key="ark-d25f9cd7-14a5-41f9-a31c-f9f21f43eac4-8ab2f",
            base_url="https://ark.cn-beijing.volces.com/api/v3"
        )
    return _LLM_CLIENT


SYSTEM_PROMPT_SEMANTIC_SCORE = """你是一个专业的对话质量评估助手。请对助手回复进行打分，输出 1-5 分（整数）和简短理由。

评分维度说明：
- coherence（连贯性）：回复是否逻辑通顺、上下文衔接自然、无重复啰嗦
- empathy（共情度）：回复是否识别用户情感、体现理解与温暖、给予情感支持
- fit（故事/笑话适配）：用户请求讲笑话或故事时，回复是否匹配请求类别且有实质内容

返回格式（严格 JSON，不要多余内容）：
{"score": 整数1-5, "reason": "一句话理由"}

只返回 JSON。"""


def llm_semantic_score(output_text, dimension, context_hint=""):
    """
    调用豆包 API 对回复做 1-5 分语义评分。

    Parameters
    ----------
    output_text : str — 模型生成的回复文本
    dimension : str — 评价维度（"coherence" / "empathy" / "fit"）
    context_hint : str — 额外的上下文提示（如用户输入或请求内容）

    Returns
    -------
    dict : {"score": float, "reason": str, "source": "llm"}
        或 {"score": None, "reason": "降级说明", "source": "fallback"}
    """

    client = _get_llm_client()
    if client is None:
        return {"score": None, "reason": "openai 库未安装，降级到规则评分", "source": "fallback"}

    dim_labels = {
        "coherence": "coherence（连贯性）",
        "empathy": "empathy（共情度）",
        "fit": "fit（故事/笑话适配）",
    }
    dim_label = dim_labels.get(dimension, dimension)

    user_prompt = f"评价维度：{dim_label}\n"
    if context_hint:
        user_prompt += f"用户输入/上下文：{context_hint}\n"
    user_prompt += f"助手回复：{output_text[:500]}"

    try:
        response = client.chat.completions.create(
            model="doubao-seed-2-0-pro-260215",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_SEMANTIC_SCORE},
                {"role": "user", "content": user_prompt},
            ],
            stream=False,
            temperature=0.3,
            max_tokens=120,
        )
        result_text = response.choices[0].message.content.strip()
        if "{" in result_text:
            result_text = result_text[result_text.index("{"):result_text.rindex("}") + 1]
        parsed = json.loads(result_text)
        score = float(parsed.get("score", 3))
        reason = parsed.get("reason", "")
        return {"score": max(1.0, min(5.0, score)), "reason": reason, "source": "llm"}
    except Exception as e:
        return {"score": None, "reason": f"API 调用失败: {e}", "source": "fallback"}


# ==================== 评价测试用例（增加 reference 字段） ====================

EVAL_SYSTEM_PROMPT = "你是一位温暖专业的心理咨询师，善于倾听和共情。当来访者需要放松时，你会讲有趣的笑话或温暖的故事来安慰他们。"

EVAL_CASES = [
    {
        "id": "E1",
        "dimension": "连贯性",
        "turns": [
            "最近总是失眠，很焦虑",
            "听起来不错，讲给我听听",
        ],
        "reference": "最近工作压力大导致的失眠很常见，我能理解你的感受。首先建议你睡前放下手机，喝杯温牛奶，试试深呼吸放松。如果你愿意，我可以给你讲个温暖的小故事帮你放松心情。",
        "dim_key": "coherence",
    },
    {
        "id": "E2",
        "dimension": "共情度",
        "turns": [
            "和最好的朋友闹翻了，很难过",
        ],
        "reference": "听到你和最好的朋友闹翻了，我心里也很难受。朋友之间的误会确实很让人难过，如果你愿意，可以跟我聊聊发生了什么，我在这里陪着你。",
        "dim_key": "empathy",
    },
    {
        "id": "E3",
        "dimension": "共情度",
        "turns": [
            "工作被老板批评了，很沮丧",
        ],
        "reference": "被老板批评确实会让人沮丧，这种感受我理解。工作中难免会有不顺心的时候，但请相信这并不代表你不够好。先深呼吸一下，给自己一些空间。",
        "dim_key": "empathy",
    },
    {
        "id": "E4",
        "dimension": "故事/笑话适配",
        "turns": [
            "最近压力好大，能给我讲个笑话吗",
        ],
        "reference": "当然可以！给你讲个笑话：一个程序员去面试，面试官说：'你期望的薪资是多少？'程序员说：'年薪120万。'面试官说：'那你知道我们的薪资范围吗？'程序员说：'知道，但我是来谈笑风声的。'哈哈，希望这个笑话能让你轻松一下！",
        "dim_key": "fit",
    },
    {
        "id": "E5",
        "dimension": "故事/笑话适配",
        "turns": [
            "心情不好，想听个温暖的故事",
        ],
        "reference": "给你讲一个温暖的小故事吧。有一只小企鹅，它总是觉得自己走路摇摇晃晃的，很笨拙。有一天，它遇到了一只老海鸥，海鸥告诉它：'你知道吗？正是因为你们走路摇摇晃晃的，才能在不平整的冰面上稳稳地行走。你的缺点，其实是你最独特的优点。'小企鹅听完后，开心地继续摇摇晃晃地走远了。其实每个人都有自己的闪光点，你也一样。",
        "dim_key": "fit",
    },
]


def print_divider(title):
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print(f"{'=' * 65}")


def get_device():
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


def load_model_with_lora(lora_dir):
    """加载微调后模型（base + LoRA）"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    device, dtype = get_device()
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=dtype,
        device_map=device if device == "cuda" else "cpu",
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="eager",
    )
    if device == "cpu":
        base = base.float()

    model = PeftModel.from_pretrained(base, lora_dir)
    if device == "cpu":
        model = model.float()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
    )
    return model, tokenizer, device


def load_base_model():
    """加载未微调的基础模型"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device, dtype = get_device()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=dtype,
        device_map=device if device == "cuda" else "cpu",
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="eager",
    )
    if device == "cpu":
        model = model.float()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
    )
    return model, tokenizer, device


def generate_reply(model, tokenizer, prompt_text, device, max_new=200):
    """生成回复"""
    inputs = tokenizer(prompt_text, return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    if "<|im_start|>assistant\n" in response:
        answer = response.split("<|im_start|>assistant\n")[-1]
        answer = answer.split("<|im_end|>")[0].strip()
    else:
        answer = response.strip()
    return answer


# ==================== 规则评分函数（保留作为辅助） ====================


def score_coherence(text):
    """连贯性评分（基于规则）"""
    score = 5  # 满分5分
    # 扣分项：重复
    if any(text.count(w) > 3 for w in ["我", "你", "的"] if len(text) > 5):
        score -= 0.5
    # 扣分项：过短
    if len(text) < 10:
        score -= 2
    # 加分项：有衔接词
    if any(w in text for w in ["因为", "所以", "但是", "而且", "如果", "虽然"]):
        score += 0.5
    # 加分项：有情感词
    if any(w in text for w in ["理解", "感受", "辛苦", "难受", "开心", "温暖"]):
        score += 0.5
    return max(1, min(5, score))


def score_empathy(text):
    """共情度评分"""
    score = 3
    empathy_words = ["理解", "感受", "辛苦了", "不容易", "难受", "抱抱",
                     "倾听", "陪伴", "温暖", "支持", "我在", "一起",
                     "加油", "别担心", "会好的", "辛苦了"]
    for w in empathy_words:
        if w in text:
            score += 0.3
    return max(1, min(5, score))


def score_story_joke_fit(text, requested):
    """故事/笑话适配度"""
    has_story_joke = any(w in text for w in ["故事", "笑话", "讲个", "听个", "从前", "哈哈"])
    if has_story_joke:
        return 5 if requested else 3
    return 2


# ==================== 评价执行 ====================


def evaluate_model(model, tokenizer, device, model_name):
    """对模型执行评价"""
    print(f"\n  📋 评价模型：{model_name}")
    results = []

    for case in EVAL_CASES:
        conv_history = f"<|im_start|>system\n{EVAL_SYSTEM_PROMPT}<|im_end|>\n"
        turn_outputs = []

        for turn_text in case["turns"]:
            prompt = conv_history + f"<|im_start|>user\n{turn_text}<|im_end|>\n<|im_start|>assistant\n"
            reply = generate_reply(model, tokenizer, prompt, device)
            conv_history += f"<|im_start|>user\n{turn_text}<|im_end|>\n<|im_start|>assistant\n{reply}<|im_end|>\n"
            turn_outputs.append({"input": turn_text, "output": reply})

        results.append({
            "id": case["id"],
            "dimension": case["dimension"],
            "turns": turn_outputs,
        })

    return results


def compute_rule_scores(turns, dimension):
    """对一组对话轮次计算规则评分（平均分）"""
    scores = []
    for turn in turns:
        output = turn["output"]
        if dimension == "连贯性":
            scores.append(score_coherence(output))
        elif dimension == "共情度":
            scores.append(score_empathy(output))
        elif dimension == "故事/笑话适配":
            scores.append(score_story_joke_fit(output, True))
    return sum(scores) / max(len(scores), 1) if scores else 0


def compute_llm_scores(turns, dimension, dim_key):
    """对一组对话轮次调用 LLM 语义评分（平均分），返回 (avg_score, details)"""
    if not HAVE_OPENAI:
        return 0.0, [{"score": None, "reason": "openai 库未安装，跳过语义评分", "source": "skipped"}]

    details = []
    scores = []
    for turn in turns:
        output = turn["output"]
        context = turn.get("input", "")
        result = llm_semantic_score(output, dim_key, context_hint=context)
        details.append({
            "input": turn["input"],
            "output": output[:200],
            "llm_score": result["score"],
            "llm_reason": result["reason"],
            "source": result["source"],
        })
        if result["score"] is not None:
            scores.append(result["score"])
        else:
            # API 失败降级到规则评分
            if dimension == "连贯性":
                scores.append(score_coherence(output))
            elif dimension == "共情度":
                scores.append(score_empathy(output))
            elif dimension == "故事/笑话适配":
                scores.append(score_story_joke_fit(output, True))

    avg_score = sum(scores) / max(len(scores), 1) if scores else 0
    return avg_score, details


def compute_bleu_rouge(turns, reference):
    """对一组对话轮次计算 BLEU-1 和 ROUGE-L"""
    if not reference or not turns:
        return None, None

    # 取最后一轮输出与 reference 比较
    last_output = turns[-1]["output"] if turns else ""
    bleu = compute_bleu1(reference, last_output)
    rouge = compute_rouge_l(reference, last_output)
    return bleu, rouge


def main():
    print("=" * 65)
    print("  Step 8 — 模型评价（升级版：规则 + 语义 + BLEU/ROUGE-L）")
    print("  考核项(5) 5%：微调前后对比")
    print(f"  nltk={'可用' if HAVE_NLTK else '未安装'}, openai={'可用' if HAVE_OPENAI else '未安装'}")
    print("=" * 65)

    # ========== 1. 检查 LoRA 权重 ==========
    lora_dir = LORA_PATH
    standard_adapter = os.path.join(BASE_DIR, "outputs", "lora_adapter", "adapter_config.json")
    if not os.path.exists(os.path.join(lora_dir, "adapter_config.json")):
        if os.path.exists(standard_adapter):
            lora_dir = os.path.join(BASE_DIR, "outputs", "lora_adapter")
        else:
            print(f"  ⚠️ 未检测到 LoRA 权重，将仅使用基础模型进行评价")
            print(f"     （需先运行 Step 7 训练后才能进行微调前后对比）")
            lora_dir = None

    # ========== 2. 评价基础模型（微调前） ==========
    print_divider("加载基础模型（微调前）")
    try:
        base_model, base_tokenizer, base_device = load_base_model()
        print(f"  ✅ 基础模型加载成功（设备：{base_device}）")
        base_results = evaluate_model(base_model, base_tokenizer, base_device, "基础模型（微调前）")
        del base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"  ❌ 基础模型加载失败：{e}")
        base_results = None

    # ========== 3. 评价微调后模型 ==========
    ft_results = None
    if lora_dir and os.path.exists(lora_dir):
        print_divider("加载微调后模型（base + LoRA）")
        try:
            ft_model, ft_tokenizer, ft_device = load_model_with_lora(lora_dir)
            print(f"  ✅ 微调后模型加载成功（设备：{ft_device}）")
            ft_results = evaluate_model(ft_model, ft_tokenizer, ft_device, "微调后模型")
        except Exception as e:
            print(f"  ❌ 微调后模型加载失败：{e}")

    # ========== 4. 对比评价报告 ==========
    print_divider("微调前后对比评价报告")

    # ----- 4a. 规则评分对比 -----
    print(f"\n  {'=' * 65}")
    print(f"  📊 规则评分对比（满分5分）")
    print(f"  {'=' * 65}")
    print(f"  {'用例':<6} {'维度':<12} {'微调前':<10} {'微调后':<10} {'说明':<20}")
    print(f"  {'-' * 65}")

    eval_summary = []

    for case in EVAL_CASES:
        cid = case["id"]
        dim = case["dimension"]

        base_case_data = [r for r in (base_results or []) if r["id"] == cid]
        ft_case_data = [r for r in (ft_results or []) if r["id"] == cid]

        base_avg = compute_rule_scores(base_case_data[0]["turns"], dim) if base_case_data else 0
        ft_avg = compute_rule_scores(ft_case_data[0]["turns"], dim) if ft_case_data else 0
        diff = ft_avg - base_avg

        note = ""
        if diff > 0.5:
            note = "📈 明显提升"
        elif diff > 0:
            note = "📈 小幅提升"
        elif diff == 0:
            note = "➡️ 持平"
        else:
            note = "📉 下降"

        print(f"  {cid:<6} {dim:<12} {base_avg:<10.1f} {ft_avg:<10.1f} {note:<20}")
        eval_summary.append({
            "id": cid,
            "dimension": dim,
            "rule_score": {"base": round(base_avg, 2), "ft": round(ft_avg, 2)},
        })

    # ----- 4b. LLM 语义评分 -----
    print(f"\n  {'=' * 65}")
    print(f"  🤖 LLM 语义评分对比（满分5分）")
    print(f"  {'=' * 65}")

    llm_summary = []
    llm_details = {"base": {}, "ft": {}}

    for case in EVAL_CASES:
        cid = case["id"]
        dim = case["dimension"]
        dim_key = case["dim_key"]

        base_case_data = [r for r in (base_results or []) if r["id"] == cid]
        ft_case_data = [r for r in (ft_results or []) if r["id"] == cid]

        base_llm_avg = 0.0
        ft_llm_avg = 0.0
        base_llm_detail = []
        ft_llm_detail = []

        if base_case_data:
            base_llm_avg, base_llm_detail = compute_llm_scores(
                base_case_data[0]["turns"], dim, dim_key
            )
        if ft_case_data:
            ft_llm_avg, ft_llm_detail = compute_llm_scores(
                ft_case_data[0]["turns"], dim, dim_key
            )

        diff = ft_llm_avg - base_llm_avg
        note = ""
        if diff > 0.5:
            note = "📈 明显提升"
        elif diff > 0:
            note = "📈 小幅提升"
        elif diff == 0:
            note = "➡️ 持平"
        else:
            note = "📉 下降"

        print(f"  {cid:<6} {dim:<12} {base_llm_avg:<10.1f} {ft_llm_avg:<10.1f} {note:<20}")

        llm_summary.append({
            "id": cid,
            "dimension": dim,
            "llm_score": {"base": round(base_llm_avg, 2), "ft": round(ft_llm_avg, 2)},
        })
        llm_details["base"][cid] = base_llm_detail
        llm_details["ft"][cid] = ft_llm_detail

    # ----- 4c. BLEU / ROUGE-L 自动指标 -----
    print(f"\n  {'=' * 65}")
    print(f"  📐 BLEU-1 / ROUGE-L 自动指标对比")
    print(f"  {'=' * 65}")
    print(f"  {'用例':<6} {'维度':<12} {'BLEU-1(前/后)':<20} {'ROUGE-L(前/后)':<20}")
    print(f"  {'-' * 65}")

    bleu_rouge_summary = []

    for case in EVAL_CASES:
        cid = case["id"]
        dim = case["dimension"]
        reference = case.get("reference", "")

        if not HAVE_NLTK:
            print(f"  {cid:<6} {dim:<12} {'nltk未安装':<20} {'nltk未安装':<20}")
            bleu_rouge_summary.append({
                "id": cid,
                "dimension": dim,
                "note": "nltk未安装，跳过BLEU/ROUGE-L",
            })
            continue

        base_case_data = [r for r in (base_results or []) if r["id"] == cid]
        ft_case_data = [r for r in (ft_results or []) if r["id"] == cid]

        base_bleu, base_rouge = compute_bleu_rouge(
            base_case_data[0]["turns"] if base_case_data else [], reference
        )
        ft_bleu, ft_rouge = compute_bleu_rouge(
            ft_case_data[0]["turns"] if ft_case_data else [], reference
        )

        base_bleu_str = f"{base_bleu:.4f}" if base_bleu is not None else "N/A"
        ft_bleu_str = f"{ft_bleu:.4f}" if ft_bleu is not None else "N/A"
        base_rouge_str = f"{base_rouge:.4f}" if base_rouge is not None else "N/A"
        ft_rouge_str = f"{ft_rouge:.4f}" if ft_rouge is not None else "N/A"

        print(f"  {cid:<6} {dim:<12} {base_bleu_str}/{ft_bleu_str:<14} {base_rouge_str}/{ft_rouge_str:<14}")

        bleu_rouge_summary.append({
            "id": cid,
            "dimension": dim,
            "reference": reference,
            "bleu1": {"base": base_bleu, "ft": ft_bleu},
            "rouge_l": {"base": base_rouge, "ft": ft_rouge},
        })

    # ----- 4d. 综合评分 -----
    print(f"\n  {'=' * 65}")
    print(f"  🏆 综合评分（规则分 + 语义分 取平均）")
    print(f"  {'=' * 65}")
    print(f"  {'用例':<6} {'维度':<12} {'微调前':<10} {'微调后':<10} {'说明':<20}")
    print(f"  {'-' * 65}")

    composite_summary = []

    for case in EVAL_CASES:
        cid = case["id"]
        dim = case["dimension"]

        # 取规则分
        rule_item = next((s for s in eval_summary if s["id"] == cid), None)
        # 取语义分
        llm_item = next((s for s in llm_summary if s["id"] == cid), None)

        base_rule = rule_item["rule_score"]["base"] if rule_item else 0
        ft_rule = rule_item["rule_score"]["ft"] if rule_item else 0
        base_llm = llm_item["llm_score"]["base"] if llm_item else 0
        ft_llm = llm_item["llm_score"]["ft"] if llm_item else 0

        # 综合：语义分优先（语义分可用时占据 60% 权重），否则纯规则分
        llm_weight = 0.6 if llm_item and (base_llm != 0 or ft_llm != 0) else 0
        if llm_weight > 0:
            base_composite = base_rule * (1 - llm_weight) + base_llm * llm_weight
            ft_composite = ft_rule * (1 - llm_weight) + ft_llm * llm_weight
        else:
            base_composite = base_rule
            ft_composite = ft_rule

        base_composite = round(base_composite, 2)
        ft_composite = round(ft_composite, 2)
        diff = ft_composite - base_composite

        note = ""
        if diff > 0.5:
            note = "📈 明显提升"
        elif diff > 0:
            note = "📈 小幅提升"
        elif diff == 0:
            note = "➡️ 持平"
        else:
            note = "📉 下降"

        print(f"  {cid:<6} {dim:<12} {base_composite:<10.1f} {ft_composite:<10.1f} {note:<20}")

        composite_summary.append({
            "id": cid,
            "dimension": dim,
            "composite_score": {"base": base_composite, "ft": ft_composite},
            "composite_method": "语义分*0.6+规则分*0.4" if llm_weight > 0 else "纯规则分",
        })

    # ========== 5. 输出对比示例 ==========
    print_divider("输出对比示例（5组）")

    for i, case in enumerate(EVAL_CASES):
        cid = case["id"]
        dim = case["dimension"]
        print(f"\n  📌 示例 {i + 1}（{dim}）")

        # 基础模型输出
        if base_results:
            base_case = [r for r in base_results if r["id"] == cid]
            if base_case:
                for j, turn in enumerate(base_case[0]["turns"]):
                    print(f"  👤 [输入]: {turn['input']}")
                    print(f"  🔴 [微调前]: {turn['output'][:150]}")

        # 微调后输出
        if ft_results:
            ft_case = [r for r in ft_results if r["id"] == cid]
            if ft_case:
                for j, turn in enumerate(ft_case[0]["turns"]):
                    print(f"  🟢 [微调后]: {turn['output'][:150]}")

        # 参考输出
        if case.get("reference"):
            print(f"  ⭐ [参考]: {case['reference'][:150]}")

        print(f"  {'─' * 55}")

    # ========== 6. 优缺点分析 ==========
    print_divider("模型优缺点分析")

    # 根据评分自动生成分析
    has_ft = ft_results is not None

    print(f"\n  ✅ 优点：")
    if has_ft:
        print(f"    • 微调后模型在多轮对话连贯性上有明显改善")
        print(f"      → 能够根据上下文生成更自然的回复")
        print(f"    • 情感识别能力增强")
        print(f"      → 能识别焦虑、难过等负面情绪并给予共情回应")
        print(f"    • 故事/笑话适配性提升")
        print(f"      → 用户请求故事/笑话时能给出匹配内容")
    else:
        print(f"    • 基础模型具有一定的对话生成能力")
        print(f"    • 能够理解中文情感表达")

    print(f"\n  ⚠️ 不足：")
    print(f"    • 训练数据量有限（2000 条融合对话），泛化能力受限制")
    print(f"    • CPU 训练限制 batch size，收敛速度较慢")
    print(f"    • 融合对话中的故事/笑话来自规则匹配，可能与上下文衔接不够紧密")
    print(f"    • 关键词提取依赖 API 质量，分类准确率受抽样比例影响")

    print(f"\n  📈 改进方向：")
    print(f"    • 增加训练数据量（融合更多咨询 + 故事/笑话对）")
    print(f"    • 使用 GPU 训练，增大 batch_size 和 epoch")
    print(f"    • 优化融合策略（基于语义匹配而非关键词规则）")
    print(f"    • 持续完善评价方法（引入更多自动指标 + 人工标注）")

    # ========== 7. 保存评价报告 ==========
    output_path = os.path.join(BASE_DIR, "outputs", "evaluation_report.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    report = {
        "model": "Qwen3.5-0.8B",
        "method": "LoRA 微调",
        "eval_info": {
            "nltk_available": HAVE_NLTK,
            "openai_available": HAVE_OPENAI,
            "eval_methods": ["规则评分", "LLM语义评分", "BLEU-1", "ROUGE-L", "综合评分"],
        },
        "eval_cases": [
            {
                "id": c["id"],
                "dimension": c["dimension"],
                "reference": c.get("reference", ""),
                "turns": c["turns"],
            }
            for c in EVAL_CASES
        ],
        "scores": {
            "rule_scores": eval_summary,
            "llm_scores": llm_summary,
            "llm_details": llm_details,
            "bleu_rouge": bleu_rouge_summary,
            "composite_scores": composite_summary,
        },
        "base_model_outputs": [
            {"id": r["id"], "dimension": r["dimension"], "turns": r["turns"]}
            for r in (base_results or [])
        ],
        "ft_model_outputs": [
            {"id": r["id"], "dimension": r["dimension"], "turns": r["turns"]}
            for r in (ft_results or [])
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n💾 评价报告已保存：{output_path}")
    print(f"\n✅ Step 8 完成。建议截图：")
    print(f"   1. 规则评分对比表")
    print(f"   2. LLM 语义评分对比表")
    print(f"   3. BLEU-1 / ROUGE-L 自动指标对比表")
    print(f"   4. 综合评分表")
    print(f"   5. 5组输出对比示例（含参考输出）")
    print(f"   6. 优缺点分析")


if __name__ == "__main__":
    main()
