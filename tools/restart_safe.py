import paramiko, time
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.cqa1.seetacloud.com', port=21026, username='root', password='8uiFh3ZS8JWP')

def run(cmd, timeout=30):
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    return stdout.read().decode(), stderr.read().decode()

# 1. 恢复安全配置：batch=1, grad_accum=2, seq=2048
print("=== 调整参数 ===")
cmds = [
    'cd /root/nlp_project',
    "sed -i 's/BATCH_SIZE = .*/BATCH_SIZE = 1/' scripts/step7_train_test.py",
    "sed -i 's/GRADIENT_ACCUMULATION_STEPS = .*/GRADIENT_ACCUMULATION_STEPS = 2/' scripts/step7_train_test.py",
    "sed -i 's/MAX_SEQ_LENGTH = .*/MAX_SEQ_LENGTH = 2048/' scripts/step7_train_test.py",
    'sed -i "s/padding=.*/padding=False,/" scripts/step7_train_test.py',
    'grep -E "BATCH_SIZE|GRADIENT|MAX_SEQ|padding" scripts/step7_train_test.py | head -5',
]
out, _ = run(' && '.join(cmds))
print(out)

# 2. 清理
run('rm -rf /root/nlp_project/outputs/lora_adapter /root/nlp_project/train_output.log')

# 3. 启动
print("=== 启动 ===")
cmd = 'cd /root/nlp_project && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONIOENCODING=utf-8 nohup /root/miniconda3/bin/python scripts/step7_train_test.py > /root/nlp_project/train_output.log 2>&1 & echo "PID=$!"'
out, _ = run(cmd)
print(out)

time.sleep(25)
out, _ = run('tail -20 /root/nlp_project/train_output.log')
print("\n=== 日志 ===")
# 只取非tqdm行
lines = [l for l in out.split('\n') if not l.startswith('\r') and l.strip() and 'it/s' not in l and '%' not in l]
print('\n'.join(lines[-15:]))

out, _ = run("grep -E 'Error|失败|loss' /root/nlp_project/train_output.log 2>/dev/null")
if out.strip():
    print("结果:", out[:500])

out, _ = run('nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader')
print("\nGPU:", out.strip())

ssh.close()
