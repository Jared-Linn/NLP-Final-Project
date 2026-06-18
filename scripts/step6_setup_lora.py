"""
Step 6 — 模型搭建（考核项3，15%）

考核要求：
① 结合多轮对话任务，合理设置学习率、训练轮数、LoRA秩等微调超参数（5%）
② 正确完成GPU调用、模型量化配置，或者CPU运行环境无报错（10%）
③ 无误搭建Qwen3基础模型及LoRA轻量化微调结构（5%）
④ 匹配对话生成任务，正确设置优化器与损失函数（5%）

此脚本负责：模型加载、LoRA 配置、超参定义、设备检测
训练由 Step 7 执行
"""
import os
import sys
import torch
sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "Qwen3.5-0.8B")


def print_divider(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def detect_device():
    """检测 GPU/CPU 环境"""
    print("  🔍 检测运行环境...")
    print(f"  PyTorch 版本：{torch.__version__}")

    if torch.cuda.is_available():
        device = "cuda"
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  ✅ GPU 可用")
        print(f"     GPU 数量：{gpu_count}")
        print(f"     型号：{gpu_name}")
        print(f"     显存：{mem_gb:.2f} GB")
        dtype = torch.float16 if mem_gb < 8 else torch.bfloat16
        print(f"     推荐精度：{dtype}")
    else:
        device = "cpu"
        dtype = torch.float32
        print(f"  ⚠️ GPU 不可用，使用 CPU")
        print(f"     精度：FP32（CPU 兼容）")
        print(f"     建议：Training 可能需要较长时间")

    print(f"  ✅ 设备确定：{device.upper()}，精度：{dtype}")
    return device, dtype


def setup_lora_config():
    """配置 LoRA 超参数"""
    print("  📋 LoRA 微调配置...")

    lora_config = {
        "r": 16,              # LoRA 秩（rank）
        "alpha": 32,          # 缩放系数 = 2 * r
        "dropout": 0.1,       # Dropout 防过拟合
        "target_modules": [   # 目标模块：注意力层的4个投影矩阵
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
        "bias": "none",       # 不微调 bias
        "task_type": "CAUSAL_LM",  # 因果语言模型
    }

    print(f"     LoRA r（秩）: {lora_config['r']}")
    print(f"     LoRA alpha（缩放）: {lora_config['alpha']}")
    print(f"     等效缩放 = alpha / r = {lora_config['alpha'] / lora_config['r']}")
    print(f"     LoRA dropout: {lora_config['dropout']}")
    print(f"     目标模块: {', '.join(lora_config['target_modules'])}")

    return lora_config


def setup_training_params():
    """定义训练超参数"""
    params = {
        # 学习率（LoRA 微调推荐 1e-4 ~ 5e-4）
        "learning_rate": 2e-4,
        # 训练轮数
        "num_epochs": 5,
        # 批次大小（CPU 设为1，GPU 可根据显存调整）
        "batch_size": 1,
        # 梯度累积步数（等效 batch = batch_size * grad_accum）
        "gradient_accumulation_steps": 4,
        # 最大序列长度
        "max_seq_length": 1024,
        # 优化器（LLM 微调标准选择）
        "optimizer": "AdamW",
        # 损失函数（语言模型标准）
        "loss_function": "CrossEntropyLoss",
        # 学习率调度器
        "lr_scheduler": "cosine",
        # 预热步数
        "warmup_steps": 10,
        # 权重衰减（L2正则化）
        "weight_decay": 0.01,
    }

    print(f"  📋 训练超参数配置...")
    print(f"     学习率 (learning_rate): {params['learning_rate']}")
    print(f"     训练轮数 (num_epochs): {params['num_epochs']}")
    print(f"     批次大小 (batch_size): {params['batch_size']}")
    print(f"     梯度累积步数: {params['gradient_accumulation_steps']}")
    print(f"     等效批次大小: {params['batch_size'] * params['gradient_accumulation_steps']}")
    print(f"     优化器: {params['optimizer']}")
    print(f"     损失函数: {params['loss_function']}")
    print(f"     学习率调度器: {params['lr_scheduler']}")
    print(f"     预热步数: {params['warmup_steps']}")
    print(f"     权重衰减: {params['weight_decay']}")
    print(f"     最大序列长度: {params['max_seq_length']}")

    return params


def main():
    print("=" * 60)
    print("  Step 6 — 模型搭建与 LoRA 配置")
    print("  考核项(3) 15%")
    print("=" * 60)

    # ========== 1. 检测设备 ==========
    print_divider("① 环境检测")
    device, dtype = detect_device()

    # ========== 2. 检查模型文件 ==========
    print_divider("② 检查模型文件")
    if not os.path.exists(MODEL_PATH):
        print(f"  ❌ 模型路径不存在：{MODEL_PATH}")
        print(f"     请确认 Qwen3.5-0.8B 已在正确位置")
        return

    required_files = ["config.json", "tokenizer_config.json", "model.safetensors.index.json"]
    for fname in required_files:
        fpath = os.path.join(MODEL_PATH, fname)
        exists = os.path.exists(fpath)
        print(f"  {'✅' if exists else '❌'} {fname}")

    # 查找实际的 safetensors 权重文件
    safetensors_files = [f for f in os.listdir(MODEL_PATH) if f.endswith(".safetensors")]
    if safetensors_files:
        total_size = sum(os.path.getsize(os.path.join(MODEL_PATH, f)) for f in safetensors_files)
        print(f"  📦 权重文件：{len(safetensors_files)} 个，总计 {total_size / 1024 / 1024:.0f} MB")
    else:
        print(f"  ❌ 未找到 .safetensors 权重文件")

    # ========== 3. LoRA 配置 ==========
    print_divider("③ LoRA 微调配置")
    lora_config = setup_lora_config()

    # ========== 4. 训练超参配置 ==========
    print_divider("④ 训练超参数配置")
    training_params = setup_training_params()

    # 自动调整：CPU 下建议减少轮数
    if device == "cpu":
        print(f"\n  💡 CPU 环境建议：可减少 epoch 到 3 或降低 max_seq_length")
        print(f"     如果训练太慢，在 step7 中调整 num_epochs=3")

    # ========== 5. 实际加载模型验证 ==========
    print_divider("⑤ 加载模型验证（试加载，不保留）")
    print("  ⏳ 正在加载模型...（验证模型文件完整性）")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"  加载路径：{MODEL_PATH}")
        print(f"  设备：{device} | 精度：{dtype}")

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            local_files_only=True,
        )

        # 试加载模型配置（不加载完整权重，仅验证 config）
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
        print(f"  ✅ 模型配置加载成功")
        # 兼容不同 Qwen 版本的属性名
        hidden_size = getattr(config, 'hidden_size', getattr(config, 'd_model', 'N/A'))
        num_heads = getattr(config, 'num_attention_heads', getattr(config, 'num_key_value_heads', 'N/A'))
        num_layers = getattr(config, 'num_hidden_layers', getattr(config, 'num_layers', 'N/A'))
        vocab_size = getattr(config, 'vocab_size', 'N/A')
        arch = config.architectures[0] if hasattr(config, 'architectures') and config.architectures else '未知'
        print(f"     模型类型：{arch}")
        print(f"     隐藏层维度：{hidden_size}")
        print(f"     注意力头数：{num_heads}")
        print(f"     层数：{num_layers}")
        print(f"     词表大小：{vocab_size}")

        # tokenizer 信息
        print(f"\n  ✅ 分词器加载成功")
        print(f"     词表大小：{len(tokenizer)}")
        print(f"     pad_token: {tokenizer.pad_token}")
        print(f"     eos_token: {tokenizer.eos_token}")

        # 打印训练参数量估算
        hidden = getattr(config, 'hidden_size', getattr(config, 'd_model', None))
        num_layers_val = getattr(config, 'num_hidden_layers', getattr(config, 'num_layers', None))
        if hidden and num_layers_val:
            num_heads_model = getattr(config, 'num_attention_heads', getattr(config, 'num_key_value_heads', 128))
            head_dim = hidden // num_heads_model
            # 粗略估算 LoRA 参数量
            lora_r = lora_config["r"]
            lora_params = 0
            for module in lora_config["target_modules"]:
                # 每层每个目标模块：q_proj, k_proj, v_proj, o_proj
                # 每模块：2 * hidden * r 参数量（A和B两个低秩矩阵）
                lora_params += 2 * hidden * lora_r
            lora_params *= layers  # 所有层
            total_params = sum(p.numel() for p in AutoModelForCausalLM.from_config(config).parameters())
            print(f"\n  📊 参数量估算：")
            print(f"     基础模型参数量：~{total_params / 1e6:.0f}M")
            print(f"     LoRA 可训练参数量：~{lora_params / 1e3:.0f}K")
            print(f"     微调比例：~{lora_params / total_params * 100:.3f}%")

        print(f"\n  ✅ 模型搭建验证成功！")
        print(f"     下一步：运行 Step 7 开始训练")

    except Exception as e:
        print(f"  ❌ 模型加载失败：{e}")
        print(f"     请检查模型文件是否完整")
        return

    # ========== 汇总 ==========
    print_divider("Step 6 配置汇总")
    print(f"  {'配置项':<20} {'值'}")
    print(f"  {'-' * 50}")
    print(f"  {'设备':<20} {device.upper()}")
    print(f"  {'精度':<20} {dtype}")
    print(f"  {'LoRA 秩 (r)':<20} {lora_config['r']}")
    print(f"  {'LoRA alpha':<20} {lora_config['alpha']}")
    print(f"  {'LoRA dropout':<20} {lora_config['dropout']}")
    print(f"  {'目标模块':<20} {', '.join(lora_config['target_modules'])}")
    print(f"  {'学习率':<20} {training_params['learning_rate']}")
    print(f"  {'训练轮数':<20} {training_params['num_epochs']}")
    print(f"  {'优化器':<20} {training_params['optimizer']}")
    print(f"  {'损失函数':<20} {training_params['loss_function']}")

    print(f"\n✅ Step 6 完成。建议截图：")
    print(f"   1. 环境检测结果（GPU/CPU + 显存）")
    print(f"   2. LoRA 配置参数")
    print(f"   3. 模型结构信息")
    print(f"   4. 参数量估算")


if __name__ == "__main__":
    main()
