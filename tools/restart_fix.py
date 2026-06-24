import paramiko, time
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.cqa1.seetacloud.com', port=21026, username='root', password='8uiFh3ZS8JWP')

def run(cmd, timeout=30):
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    return stdout.read().decode(), stderr.read().decode()

# 1. 修复 tokenize 函数：把 padding=False 改为 padding='max_length'
print("=== 修复 tokenize padding ===")
run('cd /root/nlp_project && sed -i "s/padding=False/padding=\\"max_length\\"/" scripts/step7_train_test.py')
out, _ = run('grep "padding" /root/nlp_project/scripts/step7_train_test.py')
print(out[:300])

# 2. 清理
run('rm -rf /root/nlp_project/outputs/lora_adapter /root/nlp_project/train_output.log')

# 3. 启动
print("\n=== 启动 ===")
cmd = 'cd /root/nlp_project && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONIOENCODING=utf-8 nohup /root/miniconda3/bin/python scripts/step7_train_test.py > /root/nlp_project/train_output.log 2>&1 & echo "PID=$!"'
out, _ = run(cmd)
print(out)

time.sleep(20)
out, _ = run('tail -15 /root/nlp_project/train_output.log')
print("\n=== 日志 ===")
print(out[-1500:])

out, _ = run("grep 'loss\|Error\|失败' /root/nlp_project/train_output.log 2>/dev/null")
if out:
    print("关键行:", out[:500])

ssh.close()
