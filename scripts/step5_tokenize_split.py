"""
Step 5 — Qwen3 分词器编码 + 统一 Prompt 格式 + 训练集/测试集划分（考核项2-⑤，5%）

考核要求：
- 正确调用 Qwen3 官方分词器编码数据
- 统一 Prompt 格式（ChatML）
- 划分训练集测试集

输出：
- data/split/train.json
- data/split/test.json
(已分词并附带 label 掩码)
"""
import json
import os
import sys
import random
sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "Qwen3.5-0.8B")

# ==================== 配置 ====================
MAX_SEQ_LENGTH = 1024
TEST_SPLIT_RATIO = 0.2
RANDOM_SEED = 42


def print_divider(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def format_chatml(dialogue):
    """
    将融合对话数据转换为 Qwen ChatML 格式纯文本

    ChatML 格式：
    <|im_start|>system
    系统提示词<|im_end|>
    <|im_start|>user
    用户消息<|im_end|>
    <|im_start|>assistant
    助手回复<|im_end|>
    """
    # 系统提示词：设定心理咨询师角色 + 故事/笑话助手
    system_prompt = "你是一位温暖专业的心理咨询师，善于倾听和共情。当来访者需要放松时，你会讲有趣的笑话或温暖的故事来安慰他们。"

    text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"

    for i, turn in enumerate(dialogue["conversations"]):
        role = turn["role"]
        content = turn["content"]
        text += f"<|im_start|>{role}\n{content}<|im_end|>\n"

    return text


def main():
    random.seed(RANDOM_SEED)
    print("=" * 60)
    print("  Step 5 — 分词 + 格式统一 + train/test 划分")
    print("  考核项(2)-⑤ 5%")
    print("=" * 60)

    # ========== 1. 加载融合数据 ==========
    fused_path = os.path.join(BASE_DIR, "data", "fused", "fused_dialogue.json")
    if not os.path.exists(fused_path):
        print(f"❌ 未找到融合数据，请先运行 Step 4")
        return

    with open(fused_path, "r", encoding="utf-8") as f:
        dialogues = json.load(f)
    print(f"\n✅ 已加载融合对话：{len(dialogues)} 条")

    # ========== 2. 统一 ChatML 格式 ==========
    print_divider("统一 ChatML 格式")
    chatml_data = []
    for i, d in enumerate(dialogues):
        chatml_text = format_chatml(d)
        chatml_data.append({
            "text": chatml_text,
            "emotion": d.get("emotion", "日常"),
            "category": d.get("category", "其他"),
        })
    print(f"✅ 已转换 {len(chatml_data)} 条 ChatML 格式对话")

    # 打印一条示例
    print(f"\n📝 ChatML 格式示例（前 300 字符）：")
    print(chatml_data[0]["text"][:300])
    print("...")

    # ========== 3. 加载 Qwen3 分词器 ==========
    print_divider("加载 Qwen3 分词器")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            local_files_only=True,
        )
        print(f"✅ Qwen3 分词器加载成功")
        print(f"   词表大小：{len(tokenizer)}")
        print(f"   pad_token: {tokenizer.pad_token}")
        print(f"   eos_token: {tokenizer.eos_token}")

        # 确保有 pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"❌ Qwen3 分词器加载失败：{e}")
        print(f"   请确认模型路径存在：{MODEL_PATH}")
        return

    # ========== 4. 编码数据 ==========
    print_divider("分词编码 + label 掩码")

    IM_START = "<|im_start|>"
    IM_END = "<|im_end|>"

    im_start_id = tokenizer.convert_tokens_to_ids(IM_START)
    im_end_id = tokenizer.convert_tokens_to_ids(IM_END)
    assistant_id = tokenizer.convert_tokens_to_ids("assistant")
    user_id = tokenizer.convert_tokens_to_ids("user")
    system_id = tokenizer.convert_tokens_to_ids("system")
    role_ids = {assistant_id: "assistant", user_id: "user", system_id: "system"}

    tokenized_data = []
    token_lengths = []

    for i, item in enumerate(chatml_data):
        text = item["text"]

        # 分词
        tokens = tokenizer(
            text,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False,
        )
        input_ids = tokens["input_ids"]
        seq_len = len(input_ids)
        token_lengths.append(seq_len)

        # 构造 label 掩码（仅 assistant 回复参与 loss 计算）
        label_ids = [-100] * seq_len
        current_role = None

        for j in range(seq_len):
            tid = input_ids[j]

            if tid == im_start_id and j + 1 < seq_len:
                current_role = role_ids.get(input_ids[j + 1], "user")
            elif tid == im_end_id:
                current_role = None

            if current_role == "assistant":
                label_ids[j] = tid

        tokenized_data.append({
            "input_ids": input_ids,
            "labels": label_ids,
            "emotion": item["emotion"],
            "category": item["category"],
        })

        if (i + 1) % 500 == 0:
            print(f"  已编码 {i + 1}/{len(chatml_data)} 条...")

    # 统计 token 长度
    avg_len = sum(token_lengths) / len(token_lengths)
    max_len = max(token_lengths)
    min_len = min(token_lengths)
    print(f"\n📊 Token 长度统计：")
    print(f"   平均：{avg_len:.0f} | 最长：{max_len} | 最短：{min_len}")
    print(f"   超过 {MAX_SEQ_LENGTH} 被截断：{sum(1 for l in token_lengths if l >= MAX_SEQ_LENGTH)} 条")

    # ========== 5. 划分 train/test ==========
    print_divider("划分训练集/测试集")

    # 打乱数据
    random.shuffle(tokenized_data)

    split_idx = int(len(tokenized_data) * (1 - TEST_SPLIT_RATIO))
    train_data = tokenized_data[:split_idx]
    test_data = tokenized_data[split_idx:]

    print(f"  {'' :<20} {'数量':<10} {'占比':<10}")
    print(f"  {'-' * 40}")
    print(f"  {'训练集 (train)':<20} {len(train_data):<10} {(1 - TEST_SPLIT_RATIO) * 100:.0f}%")
    print(f"  {'测试集 (test)':<20} {len(test_data):<10} {TEST_SPLIT_RATIO * 100:.0f}%")

    # ========== 6. 保存 ==========
    split_dir = os.path.join(BASE_DIR, "data", "split")
    os.makedirs(split_dir, exist_ok=True)

    train_path = os.path.join(split_dir, "train.json")
    test_path = os.path.join(split_dir, "test.json")

    # 保存为轻量格式（只存必要的字段）
    def save_tokenized(path, data):
        """保存为 JSON（标记数据已分词）"""
        # 转换 input_ids/labels 为列表（torch 张量不可 JSON 序列化）
        serializable = []
        for item in data:
            serializable.append({
                "input_ids": item["input_ids"],
                "labels": item["labels"],
                "emotion": item["emotion"],
                "category": item["category"],
            })
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False)

    save_tokenized(train_path, train_data)
    save_tokenized(test_path, test_data)

    print(f"\n💾 已保存训练数据：")
    print(f"   训练集：{train_path}（{len(train_data)} 条）")
    print(f"   测试集：{test_path}（{len(test_data)} 条）")

    # ========== 7. 同时保存 ChatML 纯文本格式（便于直接用 Trainer 加载） ==========
    print_divider("保存 ChatML 纯文本格式（备用）")

    chatml_train = [{"text": item["text"], "emotion": item["emotion"], "category": item["category"]}
                    for item in chatml_data[:split_idx]]
    chatml_test = [{"text": item["text"], "emotion": item["emotion"], "category": item["category"]}
                   for item in chatml_data[split_idx:]]

    chatml_train_path = os.path.join(split_dir, "train_chatml.json")
    chatml_test_path = os.path.join(split_dir, "test_chatml.json")

    with open(chatml_train_path, "w", encoding="utf-8") as f:
        for item in chatml_train:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(chatml_test_path, "w", encoding="utf-8") as f:
        for item in chatml_test:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ 已保存 ChatML 文本格式（JSONL）")
    print(f"   训练集：{chatml_train_path}")
    print(f"   测试集：{chatml_test_path}")

    print(f"\n✅ Step 5 完成。建议截图：")
    print(f"   1. ChatML 格式示例")
    print(f"   2. 分词器加载信息")
    print(f"   3. Token 长度统计")
    print(f"   4. 训练集/测试集划分统计")


if __name__ == "__main__":
    main()
