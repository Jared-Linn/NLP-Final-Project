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

# 1. 找 Python 和 conda env
print("=== Conda 环境列表 ===")
out = run('source /root/miniconda3/etc/profile.d/conda.sh && conda info --envs')
print(out)

print("=== 哪个 python 有 torch? ===")
out = run('/root/miniconda3/bin/python -c "import torch; print(\"base python ok, torch\", torch.__version__)" 2>&1')
print(out)

# 2. 用 base python 下载模型
print("=== 下载模型（用 base python）===")
dl_cmd = """cd /root/nlp_project && \
export HF_ENDPOINT=https://hf-mirror.com && \
/root/miniconda3/bin/python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model_id = 'Qwen/Qwen3.5-0.8B'
print('Downloading', model_id, '...')
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model.save_pretrained('/root/nlp_project/Qwen3.5-0.8B')
tokenizer.save_pretrained('/root/nlp_project/Qwen3.5-0.8B')
print('DONE: model saved')
"
"""
print("下载中（5-10分钟）...")
out = run(dl_cmd, timeout=600)
print(out[-800:] if len(out) > 800 else out)

# 3. 验证
print("\n=== 验证模型 ===")
out = run('ls /root/nlp_project/Qwen3.5-0.8B/*.safetensors 2>/dev/null | head -3 && echo "OK" || echo "FAIL"')
print(out)

# 4. 启动训练
print("=== 启动训练 ===")
train_cmd = """cd /root/nlp_project && \
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONIOENCODING=utf-8 && \
sed -i 's/TRAIN_SUBSET = .*/TRAIN_SUBSET = 0  # 云端全量/' scripts/step7_train_test.py && \
sed -i 's/MAX_SEQ_LENGTH = 512/MAX_SEQ_LENGTH = 1024/' scripts/step7_train_test.py && \
sed -i 's/NUM_EPOCHS = 2/NUM_EPOCHS = 3/' scripts/step7_train_test.py && \
nohup /root/miniconda3/bin/python scripts/step7_train_test.py > /root/nlp_project/train_output.log 2>&1 &
echo "PID=$!"
"""
out = run(train_cmd)
print(out)

# 5. 等待并检查
time.sleep(10)
print("\n=== 训练日志 ===")
out = run('tail -30 /root/nlp_project/train_output.log 2>/dev/null')
print(out[-2000:] if len(out) > 2000 else out)

ssh.close()
