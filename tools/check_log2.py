import paramiko
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.cqa1.seetacloud.com', port=21026, username='root', password='8uiFh3ZS8JWP')

def run(cmd):
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=30)
    return stdout.read().decode(), stderr.read().decode()

out, _ = run('grep -A 30 -E "Error|Traceback|OOM|CUDA out|memory" /root/nlp_project/train_output.log 2>/dev/null | tail -80')
print("=== 错误 ===")
print(out)

out, _ = run('tail -30 /root/nlp_project/train_output.log 2>/dev/null')
print("\n=== 日志尾 ===")
print(out)

out, _ = run('nvidia-smi 2>/dev/null | head -15')
print("\n=== GPU ===")
print(out)

ssh.close()
