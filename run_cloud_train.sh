#!/bin/bash
# 云GPU一键训练脚本
# 在远程服务器上执行：bash run_cloud_train.sh

set -e
echo "============================================"
echo "  灵犀 — 云端全量训练"
echo "============================================"

# 0. 强制离线模式（新版 transformers 需要）
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
echo "[0/6] 已启用离线模式 (HF_HUB_OFFLINE=1)"

# 1. 环境检查
echo "[1/6] 检查环境..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"

# 2. 安装依赖
echo "[2/6] 安装依赖..."
pip install -r requirements.txt -q

# 3. 下载模型（如果不存在）
MODEL_DIR="Qwen3.5-0.8B"
if [ ! -d "$MODEL_DIR" ]; then
    echo "[3/6] 下载 Qwen3.5-0.8B 模型..."
    bash download_model.sh
else
    echo "[3/6] 模型已存在，跳过下载"
fi

# 4. 生成训练数据（如果 data/split 不存在）
if [ ! -f "data/split/train_chatml.json" ]; then
    echo "[4/6] 生成训练数据..."
    PYTHONIOENCODING=utf-8 python multi_turn_prepare_data.py
    PYTHONIOENCODING=utf-8 python scripts/step1_load_data.py
    PYTHONIOENCODING=utf-8 python scripts/step2_clean_data.py
    PYTHONIOENCODING=utf-8 python scripts/step3_classify.py
    PYTHONIOENCODING=utf-8 python scripts/step4_fusion.py
    PYTHONIOENCODING=utf-8 python scripts/step5_tokenize_split.py
else
    echo "[4/6] 训练数据已存在，跳过生成"
fi

# 5. 全量训练（TRAIN_SUBSET=0 使用全部数据）
echo "[5/6] 开始全量训练..."
# 临时修改 TRAIN_SUBSET 为 0（全量）
sed -i 's/TRAIN_SUBSET = .*/TRAIN_SUBSET = 0  # 云端全量训练/' scripts/step7_train_test.py
sed -i 's/MAX_SEQ_LENGTH = 512/MAX_SEQ_LENGTH = 1024/' scripts/step7_train_test.py
sed -i 's/NUM_EPOCHS = 2/NUM_EPOCHS = 3/' scripts/step7_train_test.py

PYTHONIOENCODING=utf-8 python scripts/step7_train_test.py

# 6. 运行泛化测试
echo "[6/6] 运行泛化测试..."
PYTHONIOENCODING=utf-8 python scripts/test_generalization.py

echo "============================================"
echo "  训练完成！"
echo "  LoRA权重: outputs/lora_adapter/"
echo "============================================"
