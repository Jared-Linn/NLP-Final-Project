"""
Step 7 — 模型训练与推理测试（考核项4，10%）

考核要求：
4-① 顺利执行微调训练，损失值平稳收敛，无中断报错，成功保存权重
4-② 功能测试达标：可回复多轮情感问题，可识别负面情绪并联动输出故事/笑话

此脚本先检查是否有已保存的 LoRA 权重。有则跳过训练直接测试。
"""
import json
import os
import sys
import time
sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "Qwen3.5-0.8B")
LORA_OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "lora_adapter")

# ==================== 超参数 ====================
# GPU 环境适配（GTX 1060 3GB）
LEARNING_RATE = 2e-4
NUM_EPOCHS = 2  # GTX 1060 3GB：减少轮数加速
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
MAX_SEQ_LENGTH = 512  # GTX 1060 3GB适配：降低序列长度加速训练

# 训练数据子集（0=使用全部数据）
TRAIN_SUBSET = 150  # GTX 1060 3GB适配：150条子集随机采样，GPU约30-60分钟


def print_divider(title):
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print(f"{'=' * 65}")


def check_existing_lora():
    """检查是否已有训练好的 LoRA 权重"""
    adapter_path = os.path.join(LORA_OUTPUT_DIR, "adapter_config.json")
    if os.path.exists(adapter_path):
        print(f"✅ 检测到已保存的 LoRA 权重：{LORA_OUTPUT_DIR}")
        return True
    return False


def get_device():
    """检测设备"""
    import torch
    if torch.cuda.is_available():
        mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        dtype = torch.float16 if mem_gb < 8 else torch.bfloat16
        return "cuda", dtype
    return "cpu", torch.float32


def train_model():
    """执行 LoRA 微调训练"""
    print_divider("模型训练")
    print(f"  训练参数：lr={LEARNING_RATE}, epochs={NUM_EPOCHS}")

    import torch
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer,
        TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    )
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, TaskType

    device, dtype = get_device()
    print(f"  设备：{device} | 精度：{dtype}")

    # 1. 加载模型
    print(f"\n  ⏳ 加载基础模型...")
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
    print(f"  ✅ 模型加载完成")

    # 2. 加载分词器
    print(f"  ⏳ 加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print(f"  ✅ 分词器加载完成")

    # 3. LoRA 配置
    print(f"  ⏳ 配置 LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    trainable = model.print_trainable_parameters()

    # 4. 加载训练数据
    print(f"  ⏳ 加载训练数据...")
    data_path = os.path.join(BASE_DIR, "data", "split", "train_chatml.json")
    if not os.path.exists(data_path):
        print(f"  ❌ 未找到训练数据：{data_path}")
        return False

    dataset = load_dataset("json", data_files={"train": data_path}, split="train")
    total = len(dataset)
    print(f"  ✅ 数据集加载完成：{total} 条")

    # 随机采样子集（用于调试加速，0=全量训练）
    if TRAIN_SUBSET > 0 and TRAIN_SUBSET < total:
        import random
        random.seed(42)
        indices = sorted(random.sample(range(total), TRAIN_SUBSET))
        dataset = dataset.select(indices)
        print(f"  🔧 使用子集：从 {total} 条中随机采样 {TRAIN_SUBSET} 条")
    else:
        print(f"  ✅ 使用全部训练数据：{total} 条")

    # 5. 分词 + label 掩码
    print(f"  ⏳ 数据预处理...")
    IM_START = "<|im_start|>"
    IM_END = "<|im_end|>"
    im_start_id = tokenizer.convert_tokens_to_ids(IM_START)
    im_end_id = tokenizer.convert_tokens_to_ids(IM_END)
    assistant_id = tokenizer.convert_tokens_to_ids("assistant")
    user_id = tokenizer.convert_tokens_to_ids("user")
    system_id = tokenizer.convert_tokens_to_ids("system")
    role_ids = {assistant_id: "assistant", user_id: "user", system_id: "system"}

    def tokenize_fn(examples):
        tokens = tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False,
        )
        labels = []
        for input_ids in tokens["input_ids"]:
            seq_len = len(input_ids)
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
            labels.append(label_ids)
        tokens["labels"] = labels
        return tokens

    tokenized_dataset = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text", "emotion", "category"],
    )
    print(f"  ✅ 数据预处理完成")

    # ---- 数据量统计 ----
    num_samples = len(tokenized_dataset)
    total_tokens = sum(len(ids) for ids in tokenized_dataset["input_ids"])
    avg_len = total_tokens // num_samples if num_samples > 0 else 0
    effective_batch_size = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    steps_per_epoch = (num_samples + effective_batch_size - 1) // effective_batch_size
    total_steps = steps_per_epoch * NUM_EPOCHS

    print(f"\n  📊 数据统计：")
    print(f"     训练样本数：{num_samples}")
    print(f"     总 token 数：{total_tokens}")
    print(f"     平均序列长度：{avg_len}")
    print(f"     等效批次大小：{effective_batch_size}")
    print(f"     每轮步数：{steps_per_epoch}")
    print(f"     总训练步数：~{total_steps}")

    device_name, _ = get_device()
    if device_name == "cpu":
        print(f"  🕐 CPU 预估耗时：约 {max(1, total_steps // 3)}~{max(2, total_steps // 2)} 秒")
        print(f"     （实际取决于 CPU 性能，建议首次调试用 TRAIN_SUBSET=50~100 快速验证）")
    else:
        print(f"  🕐 GPU 预估耗时：约 {max(1, total_steps // 20)}~{max(2, total_steps // 10)} 秒")

    # 6. 训练参数
    print(f"  ⏳ 配置训练参数...")
    training_args = TrainingArguments(
        output_dir=LORA_OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        logging_steps=5,
        logging_first_step=True,
        save_steps=100,
        save_total_limit=2,
        prediction_loss_only=True,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        use_cpu=(device == "cpu"),
        fp16=(device == "cuda"),
        bf16=False,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # 7. 开始训练
    print(f"\n{'=' * 65}")
    print(f"  🚀 开始训练...")
    print(f"  {'=' * 65}")
    print(f"  训练轮数：{NUM_EPOCHS}")
    print(f"  等效批次：{BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"  样本数：{len(tokenized_dataset)}")
    total_steps = (len(tokenized_dataset) + BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS - 1) // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS) * NUM_EPOCHS
    print(f"  总步数预估：{total_steps}")
    device_name, _ = get_device()
    if device_name == "cpu":
        print(f"  🕐 CPU 训练中，请耐心等待（约 {max(1, total_steps // 3)}~{max(2, total_steps // 2)} 秒）")
    else:
        print(f"  🕐 GPU 训练中...（约 {max(1, total_steps // 20)}~{max(2, total_steps // 10)} 秒）")

    try:
        train_result = trainer.train()
        metrics = train_result.metrics

        print(f"\n  ✅ 训练完成！")
        print(f"\n  📊 训练指标：")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
            else:
                print(f"    {k}: {v}")

        # 打印 loss 变化
        if hasattr(trainer.state, 'log_history'):
            print(f"\n  📈 Loss 变化记录：")
            for log in trainer.state.log_history:
                if "loss" in log:
                    print(f"    Step {log['step']:>4d} | Loss: {log['loss']:.4f}")

    except Exception as e:
        print(f"\n  ❌ 训练失败：{e}")
        # 尝试保存当前状态
        try:
            lora_path = f"{LORA_OUTPUT_DIR}/lora_adapter"
            model.save_pretrained(lora_path)
            tokenizer.save_pretrained(lora_path)
            print(f"  💾 已保存当前 LoRA 权重到 {lora_path}")
        except:
            pass
        return False

    # 8. 保存 LoRA 权重
    print(f"\n  💾 保存 LoRA 权重...")
    lora_save_path = f"{LORA_OUTPUT_DIR}/lora_adapter"
    model.save_pretrained(lora_save_path)
    tokenizer.save_pretrained(lora_save_path)

    # 同时保存到标准路径
    if os.path.exists(os.path.join(LORA_OUTPUT_DIR, "adapter_config.json")):
        print(f"  ✅ LoRA 权重已保存到 {LORA_OUTPUT_DIR}")
    print(f"  ✅ LoRA 权重已保存到 {lora_save_path}")

    return True


def inference_test():
    """推理测试：加载 LoRA 进行多轮对话测试"""
    print_divider("推理测试")
    print(f"  考核项4-②：功能测试 — 情感识别 + 故事/笑话联动")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    device, dtype = get_device()

    # 检查 LoRA 权重
    lora_path = os.path.join(LORA_OUTPUT_DIR, "lora_adapter")
    adapter_path = os.path.join(lora_path, "adapter_config.json")
    standard_adapter_path = os.path.join(LORA_OUTPUT_DIR, "adapter_config.json")

    if os.path.exists(adapter_path):
        lora_dir = lora_path
    elif os.path.exists(standard_adapter_path):
        lora_dir = LORA_OUTPUT_DIR
    else:
        print(f"  ❌ 未找到 LoRA 权重，请先训练")
        return

    print(f"  加载 LoRA 权重：{lora_dir}")

    # 加载 base model
    print(f"\n  ⏳ 加载基础模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=dtype,
        device_map=device if device == "cuda" else "cpu",
        trust_remote_code=True,
        local_files_only=True,
    )
    if device == "cpu":
        base_model = base_model.float()

    # 加载 LoRA
    print(f"  ⏳ 加载 LoRA 适配器...")
    model = PeftModel.from_pretrained(base_model, lora_dir)
    if device == "cpu":
        model = model.float()
    print(f"  ✅ 模型加载完成（设备：{device}）")

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
    )

    # ========== 多轮对话测试用例 ==========
    test_cases = [
        {
            "name": "情感1 - 焦虑学业",
            "turns": [
                "最近要考试了，我每天都睡不好，很焦虑",
                "嗯，那你讲一个吧",
            ]
        },
        {
            "name": "情感2 - 职场压力",
            "turns": [
                "最近工作压力很大，感觉快撑不住了",
                "好啊，讲来听听",
            ]
        },
        {
            "name": "情感3 - 家庭关系",
            "turns": [
                "和父母吵架了，心里很难受",
                "嗯，想听",
            ]
        },
        {
            "name": "情感4 - 孤独感",
            "turns": [
                "一个人在这个城市，感觉很孤独",
                "好啊，说吧",
            ]
        },
    ]

    def generate_reply(prompt_text, max_new=200):
        """生成回复"""
        inputs = tokenizer(prompt_text, return_tensors="pt")
        inputs = {k: v.to(device) if device == "cuda" else v for k, v in inputs.items()}

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
        # 提取 assistant 回复
        if "<|im_start|>assistant\n" in response:
            answer = response.split("<|im_start|>assistant\n")[-1]
            answer = answer.split("<|im_end|>")[0].strip()
        else:
            answer = response.strip()
        return answer

    # ========== 执行测试 ==========
    print(f"\n  {'=' * 60}")
    print(f"  多轮对话功能测试（共 {len(test_cases)} 组）")
    print(f"  {'=' * 60}")

    system_prompt = "你是一位温暖专业的心理咨询师，善于倾听和共情。当来访者需要放松时，你会讲有趣的笑话或温暖的故事来安慰他们。"

    all_results = []
    for i, case in enumerate(test_cases, 1):
        print(f"\n  📌 测试 {i}：{case['name']}")
        print(f"  {'=' * 50}")

        conversation_history = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        turn_results = []

        for j, turn_text in enumerate(case["turns"]):
            current_prompt = conversation_history + f"<|im_start|>user\n{turn_text}<|im_end|>\n<|im_start|>assistant\n"
            reply = generate_reply(current_prompt)

            print(f"\n  👤 [user]: {turn_text}")
            print(f"  🤖 [assistant]: {reply[:200]}")

            # 更新对话历史
            conversation_history += f"<|im_start|>user\n{turn_text}<|im_end|>\n<|im_start|>assistant\n{reply}<|im_end|>\n"
            turn_results.append({"user": turn_text, "assistant": reply})

        all_results.append({
            "name": case["name"],
            "turns": turn_results,
        })
        print(f"  {'=' * 50}")

    # ========== 保存测试结果 ==========
    output_path = os.path.join(BASE_DIR, "outputs", "inference_test_results.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n💾 测试结果已保存：{output_path}")

    # ========== 识别负面情绪+故事笑话联动验证 ==========
    print_divider("负面情绪→故事/笑话联动验证")
    print(f"  检查模型是否能在识别负面情绪后推荐故事/笑话...")

    emotion_triggers = [
        ("焦虑", "焦虑", "放松"),
        ("难过", "难过", "开心"),
        ("压力", "压力", "解压"),
    ]

    for emotion, keyword, expect in emotion_triggers:
        test_text = f"我最近很{emotion}，心里好难受"
        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{test_text}<|im_end|>\n<|im_start|>assistant\n"
        reply = generate_reply(prompt, max_new=150)
        has_story = any(kw in reply for kw in ["故事", "笑话", "讲个", "听个", "分享", "给你讲"])
        print(f"\n  😰 输入：{test_text}")
        print(f"  🤖 回复：{reply[:150]}")
        print(f"  {'✅ 检测到故事/笑话推荐' if has_story else '❌ 未检测到故事/笑话推荐'}")

    print(f"\n✅ Step 7 完成。建议截图：")
    print(f"   1. 训练 loss 收敛曲线")
    print(f"   2. 多轮对话测试 1-2 组")
    print(f"   3. 负面情绪联动测试结果")


def main():
    print("=" * 65)
    print("  Step 7 — 模型训练与推理测试")
    print("  考核项(4) 10%")
    print("=" * 65)

    # Step A: 检查是否有已有 LoRA 权重
    has_lora = check_existing_lora()

    if not has_lora:
        print(f"\n  ⏳ 未检测到 LoRA 权重，开始训练...")
        success = train_model()
        if not success:
            print(f"\n  ❌ 训练失败，跳过推理测试")
            return
    else:
        print(f"\n  ⏭️  跳过训练，直接进行推理测试")

    # Step B: 推理测试
    inference_test()


if __name__ == "__main__":
    main()
