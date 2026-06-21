#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate qa_gen
cd /root
rm -rf outputs/lora_adapter
export PYTHONIOENCODING=utf-8
export HF_ENDPOINT=https://hf-mirror.com
nohup python -u scripts/step7_train_test.py > /root/training.log 2>&1 &
echo "PID=$!"
echo "Training started. Check /root/training.log for progress."
