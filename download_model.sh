#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate qa_gen

echo "=== Disk before ==="
df -h / | tail -1

# Download Qwen3.5-0.8B via huggingface mirror (faster in China)
export HF_ENDPOINT=https://hf-mirror.com

python << 'PYEOF'
from transformers import AutoModelForCausalLM, AutoTokenizer
model_id = "Qwen/Qwen3.5-0.8B"
print(f"Downloading {model_id} to /root/Qwen3.5-0.8B/...")
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model.save_pretrained("/root/Qwen3.5-0.8B/")
tokenizer.save_pretrained("/root/Qwen3.5-0.8B/")
print("Model download complete!")
PYEOF

echo "=== Disk after ==="
df -h / | tail -1
echo "=== Files ==="
ls -lh /root/Qwen3.5-0.8B/ | head -15
