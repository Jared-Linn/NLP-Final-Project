import paramiko
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.cqa1.seetacloud.com', port=21026, username='root', password='8uiFh3ZS8JWP')

def run(cmd):
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=30)
    return stdout.read().decode(), stderr.read().decode()

# 搜索错误
out, _ = run('grep -n -E "Error|Traceback|OOM|CUDA out|失败|killed" /root/nlp_project/train_output.log 2>/dev/null')
print("=== 错误行 ===")
print(out if out else "无错误")

# 当前进程
out, _ = run('ps aux | grep step7 | grep -v grep')
print("\n=== step7 进程 ===")
print(out if out else "无运行中的 step7 进程")

# loss
out, _ = run("grep 'loss' /root/nlp_project/train_output.log 2>/dev/null | tail -15")
print("\n=== Loss 记录 ===")
print(out if out else "无loss记录（训练还未到记录点）")

# 当前训练进度
out, _ = run('tail -3 /root/nlp_project/train_output.log 2>/dev/null')
print("\n=== 最新3行 ===")
print(out)

ssh.close()
