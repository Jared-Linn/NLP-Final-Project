import paramiko, time, sys

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.cqa1.seetacloud.com', port=21026, username='root', password='8uiFh3ZS8JWP')

def run(cmd, timeout=30):
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode()
    err = stderr.read().decode()
    return out, err

# 1. 检查状态
out, err = run('echo "=== 模型目录 ===" && ls /root/nlp_project/Qwen3.5-0.8B/*.json 2>/dev/null | head -3 && echo "=== 进程 ===" && ps aux | grep python | grep -v grep || echo "无python进程"')
print(out)

# 2. 修复：设置环境变量并重新训练
print("=== 开始修复训练 ===")
run('export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1')

# 3. 先 clean 旧 LoRA（避免跳过训练）
out, err = run('rm -rf /root/nlp_project/outputs/lora_adapter && echo "旧权重已清理"')
print(out)

# 4. 启动训练（nohup 后台跑）
cmd = 'cd /root/nlp_project && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONIOENCODING=utf-8 nohup python scripts/step7_train_test.py > /root/train.log 2>&1 &'
out, err = run(cmd)
print('训练已启动:', out, err)

# 5. 等几秒确认启动
time.sleep(5)
out, err = run('tail -20 /root/train.log 2>/dev/null')
print('日志尾部:')
print(out)

ssh.close()
