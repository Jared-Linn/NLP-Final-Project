import paramiko, time

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.cqa1.seetacloud.com', port=21026, username='root', password='8uiFh3ZS8JWP')

def run(cmd, timeout=120):
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode()
    err = stderr.read().decode()
    if err.strip():
        print(f"  [stderr]: {err[:300]}")
    return out

# 1. 先下载模型
print("=== 步骤1：下载 Qwen3.5-0.8B 模型 ===")
# 用 conda 环境里的 python
download_cmd = """cd /root/nlp_project && \
source /root/miniconda3/etc/profile.d/conda.sh && \
conda activate qa_gen && \
export HF_ENDPOINT=https://hf-mirror.com && \
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model_id = 'Qwen/Qwen3.5-0.8B'
print(f'Downloading {model_id}...')
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model.save_pretrained('/root/nlp_project/Qwen3.5-0.8B')
tokenizer.save_pretrained('/root/nlp_project/Qwen3.5-0.8B')
print('Model downloaded to /root/nlp_project/Qwen3.5-0.8B/')
"
"""
print("下载中（约5-10分钟）...")
out = run(download_cmd, timeout=600)
print(out[-500:] if len(out) > 500 else out)

# 2. 验证模型
print("\n=== 步骤2：验证模型文件 ===")
out = run('ls /root/nlp_project/Qwen3.5-0.8B/*.json 2>/dev/null | head -5')
print(out or "模型下载失败，请检查")

# 3. 清理旧权重
print("=== 步骤3：清理旧LoRA权重 ===")
out = run('rm -rf /root/nlp_project/outputs/lora_adapter && echo "OK"')
print(out)

# 4. 启动训练
print("=== 步骤4：启动全量训练 ===")
train_cmd = """cd /root/nlp_project && \
source /root/miniconda3/etc/profile.d/conda.sh && \
conda activate qa_gen && \
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONIOENCODING=utf-8 && \
sed -i 's/TRAIN_SUBSET = .*/TRAIN_SUBSET = 0  # 云端全量/' scripts/step7_train_test.py && \
sed -i 's/MAX_SEQ_LENGTH = 512/MAX_SEQ_LENGTH = 1024/' scripts/step7_train_test.py && \
sed -i 's/NUM_EPOCHS = 2/NUM_EPOCHS = 3/' scripts/step7_train_test.py && \
nohup python scripts/step7_train_test.py > /root/nlp_project/train_output.log 2>&1 &
echo "训练已启动, PID=$!"
"""
out = run(train_cmd)
print(out)

# 5. 确认训练启动
time.sleep(8)
print("\n=== 步骤5：确认训练启动状态 ===")
out = run('tail -30 /root/nlp_project/train_output.log 2>/dev/null')
print(out[-1500:] if len(out) > 1500 else out)

# 6. 检查GPU
print("\n=== GPU状态 ===")
out = run('source /root/miniconda3/etc/profile.d/conda.sh && conda activate qa_gen && python -c "import torch; print(f\'CUDA: {torch.cuda.is_available()}\'); print(f\'GPU: {torch.cuda.get_device_name(0)}\'); print(f\'VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB\')"')
print(out)

ssh.close()
print("\n=== 完成，训练在云端后台运行中 ===")
