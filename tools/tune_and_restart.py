import paramiko, time
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.cqa1.seetacloud.com', port=21026, username='root', password='8uiFh3ZS8JWP')

def run(cmd, timeout=30):
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    return stdout.read().decode(), stderr.read().decode()

# 1. 杀掉当前训练
print("=== 停止当前训练 ===")
out, _ = run('kill 3364 2>/dev/null; sleep 1; ps aux | grep step7 | grep -v grep || echo "已停止"')
print(out)

# 2. 修改参数：batch_size 1→4, grad_accum 4→1, seq 1024→1536
print("=== 调整参数 ===")
cmds = [
    "cd /root/nlp_project",
    "sed -i 's/BATCH_SIZE = 1/BATCH_SIZE = 2/' scripts/step7_train_test.py",
    "sed -i 's/GRADIENT_ACCUMULATION_STEPS = 4/GRADIENT_ACCUMULATION_STEPS = 2/' scripts/step7_train_test.py",
    "sed -i 's/MAX_SEQ_LENGTH = 1024/MAX_SEQ_LENGTH = 1536/' scripts/step7_train_test.py",
    "grep -E 'BATCH_SIZE|GRADIENT|MAX_SEQ_LENGTH' scripts/step7_train_test.py | head -5",
]
run(' && '.join(cmds))

# 3. 清理旧权重和日志
run('rm -rf /root/nlp_project/outputs/lora_adapter /root/nlp_project/train_output.log')

# 4. 启动新训练
print("\n=== 启动优化后的训练 ===")
cmd = 'cd /root/nlp_project && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONIOENCODING=utf-8 nohup /root/miniconda3/bin/python scripts/step7_train_test.py > /root/nlp_project/train_output.log 2>&1 & echo "PID=$!"'
out, _ = run(cmd)
print(out)

time.sleep(15)
out, _ = run('tail -20 /root/nlp_project/train_output.log')
print("\n=== 启动日志 ===")
print(out[-1500:])

out, _ = run('nvidia-smi | grep -E "4090|MiB"')
print("\n=== GPU ===")
print(out)

ssh.close()
print("\n=== 训练已重启，batch=4 等效批次=4，预计3-4小时 ===")
