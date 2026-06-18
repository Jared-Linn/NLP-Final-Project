"""
多轮对话微调训练脚本
=====================
使用 LoRA（Low-Rank Adaptation）对 Qwen3.5-0.8B 进行多轮对话微调。

核心流程：
  1. 加载预训练模型（强制 FP32 + eager attention，适配 Windows CPU）
  2. 加载分词器（设置 pad_token + right padding）
  3. 配置 LoRA 适配器（仅微调注意力层，冻结原始权重）
  4. 加载并预处理训练数据（分词 + label 掩码）
  5. 配置训练参数并启动训练
  6. 保存 LoRA 适配器权重

适用环境：Windows / Linux CPU，不适用于 GPU（GPU 请使用 finetune_qwen_gpu.py）
"""
import os

# ==============================================================================
# 关键环境变量设置（必须在 import torch 之前执行）
# ==============================================================================
# 禁用 oneDNN（Intel 的 DNNL 加速库）：
#   PyTorch 默认启用 oneDNN 加速 CPU 运算，但 oneDNN 不支持 bf16/f16 的反向传播，
#   而 Qwen3 模型内部部分层可能以 bf16 加载，导致 RuntimeError。
#   设为 "0" 强制使用原生 CPU 算子，避免该问题。
#os.environ["ONEDNN_ENABLED"] = "0"

import torch
from transformers import (
    AutoModelForCausalLM,         # 自动加载因果语言模型（Causal LM）
    AutoTokenizer,                 # 自动加载与模型匹配的分词器
    TrainingArguments,             # 训练超参数配置（学习率、批次、调度器等）
    Trainer,                       # HuggingFace 训练器，封装训练循环
    DataCollatorForLanguageModeling,  # 数据整理器，自动处理动态 padding 和 label 移位
)
from datasets import load_dataset   # HuggingFace 数据集加载器
from peft import LoraConfig, get_peft_model, TaskType  # PEFT 库：LoRA 配置与应用

# ==================== 配置参数 ====================
# 所有超参数集中管理，方便修改和复用

MODEL_PATH = r"D:\dataset\model\Qwen3.5-0.8B"  # 基础模型路径（Windows 路径需加 r 前缀避免转义）
OUTPUT_DIR = "./outputs_multi_turn"              # 训练输出目录（checkpoint 和最终适配器保存于此）
DATA_PATH = "./data/multi_turn_qa.json"          # 训练数据路径（JSONL 格式，每行一条对话）
MAX_SEQ_LENGTH = 1024  # 最大序列长度：多轮对话拼接后较长，需要足够大的上下文窗口
                       # 超过此长度的 token 会被截断（truncation=True）

# --- LoRA 超参数 ---
LORA_R = 16       # LoRA 秩（rank）：低秩矩阵的维度，越大表达能力越强但参数越多
                  # 推荐范围：8~64，0.8B 小模型用 16 即可
LORA_ALPHA = 32   # LoRA 缩放系数：控制 LoRA 更新的权重，通常设为 2 * r
                  # 实际缩放 = alpha / r = 32 / 16 = 2.0

# --- 训练超参数 ---
LEARNING_RATE = 2e-4   # 学习率：LoRA 微调推荐 1e-4 ~ 5e-4，比全量微调大 10~100 倍
NUM_EPOCHS = 3         # 训练轮数：多轮对话任务建议 3~5 轮，太少学不充分，太多易过拟合
BATCH_SIZE = 1         # 每设备批次大小：CPU 训练内存有限，必须设为 1
GRADIENT_ACCUMULATION_STEPS = 4  # 梯度累积步数：每累积 4 步更新一次权重
                                 # 等效批次 = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS = 4

# 打印训练配置摘要，便于运行前快速确认参数
print("=" * 60)
print("多轮对话微调训练")
print(f"模型路径: {MODEL_PATH}")
print(f"输出目录: {OUTPUT_DIR}")
print(f"数据路径: {DATA_PATH}")
print(f"最大序列长度: {MAX_SEQ_LENGTH}")
print("=" * 60)

# ==================== 1. 加载模型 ====================
print("\n1. 加载基础模型...")

# from_pretrained 关键参数说明：
#   torch_dtype=torch.float32  — 指定加载精度为 FP32，CPU 不支持 bf16/f16 高效运算
#   device_map="cpu"           — 将模型所有层加载到 CPU（避免部分层意外分配到 GPU）
#   trust_remote_code=True     — Qwen3 含自定义建模代码，需信任远程代码才能加载
#   local_files_only=True      — 仅从本地路径加载，禁止自动下载（防止网络超时）
#   attn_implementation="eager" — 【关键】强制使用标准 attention 实现
#       Qwen3 默认的 fast path（Flash Attention / SDPA）在 Windows + transformers 下
#       存在回退逻辑的 dtype 不匹配 bug，会抛出 RuntimeError。设为 "eager" 使用
#       纯 PyTorch 实现的 scaled_dot_product_attention，稳定可靠。
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float32,
    device_map="cpu",
    trust_remote_code=True,
    local_files_only=True,
    attn_implementation="eager",
)

# 二次保险：逐参数强制转换为 FP32
# 原因：from_pretrained 的 torch_dtype 可能不覆盖所有参数（如 embedding 层或某些 buffer），
# 残留的 bf16/f16 参数在 backward 时仍会触发 oneDNN 或 dtype 不匹配错误。
# 这里遍历所有参数逐一转换，确保万无一失。
model = model.float()  # .float() 等价于 .to(torch.float32)，作用于模型整体
for param in model.parameters():
    param.data = param.data.to(torch.float32)  # 对每个参数张量单独转换

print("模型加载完成（已转换为 FP32）")

# ==================== 2. 加载分词器 ====================
print("\n2. 加载分词器...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    local_files_only=True,
)

# 设置 padding token（Qwen 没有默认的 pad_token）
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("已设置 padding token = eos token")

# 训练时使用右填充（Qwen 默认左填充用于生成，训练需要右填充）
tokenizer.padding_side = "right"

# ==================== 3. 配置 LoRA ====================
print("\n3. 配置 LoRA...")
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ==================== 4. 加载数据集 ====================
print("\n4. 加载训练数据...")
dataset = load_dataset("json", data_files={"train": DATA_PATH}, split="train")
print(f"数据集加载完成，共 {len(dataset)} 条对话")

# 打印数据示例
print("\n数据示例:")
print(dataset[0]["text"][:300] + "...")

# ==================== 5. 分词处理（含 label 掩码）====================
print("\n5. 数据预处理（分词 + label 掩码）...")

# Qwen3 使用 ChatML 格式，通过特殊 token 标记对话角色：
#   <|im_start|>  — 一条消息的开始标记
#   <|im_end|>    — 一条消息的结束标记
# 完整格式示例：
#   <|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n你好！<|im_end|>
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


def tokenize_function(examples):
    """分词函数 - 多轮对话带 label 掩码

    核心思想：
      语言模型训练时，loss 只应在 assistant 回复的 token 上计算。
      用户输入和系统提示的 token 不应参与 loss 计算（设为 -100 被 PyTorch 忽略）。
      这样模型只学习「如何回复」，而不是「如何提问」。

    label 掩码逻辑：
      -100  — 掩码值，PyTorch 的 CrossEntropyLoss 会跳过 label=-100 的位置
      正常 token ID — 该位置参与 loss 计算
    """
    # 第一步：对文本进行分词（文本 → token ID 序列）
    #   truncation=True   — 超过 MAX_SEQ_LENGTH 的序列自动截断
    #   padding=False     — 不做预填充，后续由 DataCollator 动态填充到 batch 内最长长度
    #                       动态填充比固定长度填充更高效，避免浪费算力在 padding token 上
    tokens = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding=False,
    )

    # 第二步：预先获取特殊 token 和角色关键词的 ID
    # convert_tokens_to_ids 将 token 字符串转换为对应的整数 ID
    # 提前获取避免在循环中重复调用，提升处理效率
    im_start_id = tokenizer.convert_tokens_to_ids(IM_START)  # <|im_start|> 的 ID
    im_end_id = tokenizer.convert_tokens_to_ids(IM_END)      # <|im_end|> 的 ID

    # 获取三个角色关键词的 token ID
    system_id = tokenizer.convert_tokens_to_ids("system")       # 系统角色
    assistant_id = tokenizer.convert_tokens_to_ids("assistant")  # 助手角色（需要学习的部分）
    user_id = tokenizer.convert_tokens_to_ids("user")            # 用户角色（需要掩码的部分）

    # 构建 token ID → 角色名 的映射表，用于 O(1) 快速查找
    role_ids = {assistant_id: "assistant", user_id: "user", system_id: "system"}

    # 第三步：为每个 token 生成 label，实现角色级别的 loss 掩码
    labels = []
    for input_ids in tokens["input_ids"]:
        seq_len = len(input_ids)
        label_ids = [-100] * seq_len  # 初始化：所有位置都掩码（不参与 loss 计算）
        current_role = None  # 当前所处的角色，None 表示在对话标记之外

        # 遍历每个 token，根据 ChatML 标记切换角色状态
        for j in range(seq_len):
            tid = input_ids[j]

            # 检测到 <|im_start|>：下一个 token 是角色名，据此切换角色
            # 例：<|im_start|>assistant → current_role 切换为 "assistant"
            if tid == im_start_id and j + 1 < seq_len:
                current_role = role_ids.get(input_ids[j + 1], "user")  # 未识别的角色默认为 user
            # 检测到 <|im_end|>：当前消息结束，重置角色状态
            elif tid == im_end_id:
                current_role = None

            # 只有 assistant 角色的 token 参与 loss 计算
            # 包括 <|im_start|>assistant、回复内容、<|im_end|> 全部保留
            if current_role == "assistant":
                label_ids[j] = tid  # 保留原始 token ID，训练时会与模型输出比较计算 loss

        labels.append(label_ids)

    # 将生成的 label 添加到分词结果中
    tokens["labels"] = labels
    return tokens


# dataset.map() 对所有样本批量应用分词函数：
#   batched=True         — 批量处理，比逐条处理快数倍（利用 tokenizer 的批量优化）
#   remove_columns=["text"] — 处理后移除原始文本列，节省内存（模型只需要 token ID）
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
)

print("分词处理完成（已对用户输入和系统提示做 label 掩码）")

# ==================== 6. 配置训练参数 ====================
# TrainingArguments 是 HuggingFace Trainer 的核心配置对象，
# 控制训练过程的所有超参数和行为选项。
print("\n6. 配置训练参数...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,                  # 输出目录：checkpoint、日志、适配器均保存在此路径下
    num_train_epochs=NUM_EPOCHS,            # 训练轮数：整个数据集被遍历的次数（3 轮适合小数据集，避免过拟合）
    per_device_train_batch_size=BATCH_SIZE, # 每设备批次：每次前向传播的样本数（CPU 内存有限，设为 1）
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,  # 梯度累积：累积 N 个 batch 的梯度后再更新权重
                                           # 效果等同于增大 batch_size，但不增加内存占用
    learning_rate=LEARNING_RATE,            # 学习率峰值：LoRA 微调推荐 1e-4~5e-4，过大会发散，过小收敛慢
    weight_decay=0.01,                      # 权重衰减（L2 正则化）：对参数施加惩罚，防止过拟合
                                           # 0.01 是经验值，对小数据集 LoRA 微调效果较好
    lr_scheduler_type="cosine",             # 学习率调度策略：cosine 按余弦曲线从峰值衰减到 0
                                           # 比默认 linear 更平滑，训练后期更稳定
    warmup_steps=10,                        # 预热步数：训练前 10 步学习率从 0 线性增加到峰值
                                           # 避免训练初期梯度过大导致参数剧烈震荡
    logging_steps=5,                        # 日志间隔：每 5 步打印一次 loss 等训练指标
    logging_first_step=True,                # 记录第一步：打印第一个 step 的 loss，用于判断训练是否正常启动
    save_steps=50,                          # 保存间隔：每 50 步保存一次 checkpoint
    save_total_limit=2,                     # 最多保留 2 个 checkpoint，超出后删除最旧的，节省磁盘空间
    prediction_loss_only=True,              # 评估时只计算 loss，不计算其他指标（加速评估）
    report_to="none",                       # 不上报到 wandb/tensorboard 等第三方平台
    dataloader_num_workers=0,               # 数据加载线程数：Windows 下多进程 DataLoader 易报错，设为 0（主线程加载）
    remove_unused_columns=False,            # 不移除模型 forward() 未使用的列
                                           # 必须设为 False，因为我们手动添加了 "labels" 列
    use_cpu=True,                           # 强制使用 CPU 训练
    fp16=False,                             # 禁用 FP16 混合精度：CPU 不支持高效的 FP16 运算
    bf16=False,                             # 禁用 BF16：CPU 的 oneDNN 不支持 bf16 backward
    tf32=False,                             # 禁用 TF32：仅 NVIDIA Ampere+ GPU 支持，CPU 无效
)

# ==================== 7. 初始化 Trainer ====================
# DataCollatorForLanguageModeling 是 HuggingFace 提供的数据整理器，
# 负责在每个 batch 送入模型前自动执行以下操作：
#   1. 动态 padding：将 batch 内的不等长序列填充到该 batch 内最长长度
#      （比固定填充到 MAX_SEQ_LENGTH 更高效，减少无用的 padding token）
#   2. 生成 attention_mask：标记哪些位置是真实 token，哪些是 padding
#   3. label 移位（mlm=False 时）：自动将 input_ids 右移一位作为 labels
#      但这里我们已经手动生成了带掩码的 labels，所以 DataCollator 会直接使用我们提供的 labels
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # False = 因果语言模型（CLM）：预测下一个 token
                # True  = 掩码语言模型（MLM）：随机遮盖 token 并预测，如 BERT
)

# Trainer 封装了完整的训练循环，包括：
#   - 前向传播、反向传播、梯度累积、参数更新
#   - 学习率调度、日志记录、checkpoint 保存
#   - 自动处理分布式训练、混合精度等复杂场景
trainer = Trainer(
    model=model,                       # 待训练的模型（已应用 LoRA）
    args=training_args,                # 训练超参数配置
    train_dataset=tokenized_dataset,   # 训练数据集（已分词 + label 掩码）
    data_collator=data_collator,       # 数据整理器（动态 padding）
)

# ==================== 8. 开始训练 ====================
# 打印训练关键参数摘要，方便运行时快速确认
print("\n" + "=" * 60)
print("8. 开始训练...")
print("=" * 60)
print(f"训练轮数: {NUM_EPOCHS}")
print(f"每设备批次: {BATCH_SIZE}")
print(f"梯度累积步数: {GRADIENT_ACCUMULATION_STEPS}")
print(f"等效批次: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
print(f"学习率: {LEARNING_RATE}")
print("=" * 60)

try:
    # trainer.train() 启动完整训练循环：
    #   1. 初始化优化器（AdamW）和学习率调度器
    #   2. 按 epoch 遍历数据集，每步执行前向传播 + 反向传播
    #   3. 每 gradient_accumulation_steps 步更新一次权重
    #   4. 按 logging_steps 打印 loss，按 save_steps 保存 checkpoint
    #   返回 TrainOutput 对象，包含训练指标
    train_result = trainer.train()
    print("\n训练完成！")

    # 打印训练汇总指标：包括总 loss、训练时长、每秒处理样本数等
    metrics = train_result.metrics
    print(f"\n训练指标:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # 打印每个 step 的 loss 变化，用于判断训练是否正常：
    #   - loss 应整体呈下降趋势（可能有波动）
    #   - 若 loss 持续上升，说明学习率过大或数据有问题
    #   - 若 loss 不变化，说明学习率过小或模型未在学习
    log_history = trainer.state.log_history
    if log_history:
        print("\nLoss 变化:")
        for log in log_history:
            if "loss" in log:
                print(f"  Step {log['step']:>4d} | Loss: {log['loss']:.4f}")

except Exception as e:
    print(f"\n训练失败: {e}")
    raise e

# ==================== 9. 保存模型 ====================
print("\n9. 保存 LoRA 适配器...")

# save_pretrained 只保存 LoRA 适配器的权重（而非整个模型），
# 文件很小（通常几 MB），推理时需要与基础模型合并加载
lora_path = f"{OUTPUT_DIR}/lora_adapter"
model.save_pretrained(lora_path)       # 保存 LoRA 权重（adapter_model.safetensors + adapter_config.json）
tokenizer.save_pretrained(lora_path)   # 保存分词器配置，确保推理时能使用相同的分词规则
print(f"LoRA 适配器已保存至: {lora_path}")

print("\n" + "=" * 60)
print("多轮对话微调完成！")
print("=" * 60)
