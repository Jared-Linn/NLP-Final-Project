import paramiko, time

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.cqa1.seetacloud.com', port=21026, username='root', password='8uiFh3ZS8JWP')

def run(cmd, timeout=60):
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode()
    err = stderr.read().decode()
    return out, err

# 1. 诊断：找模型位置和python
print("=== 模型文件探查 ===")
out, _ = run('find /root/nlp_project -name "config.json" -path "*/Qwen*" 2>/dev/null')
print(out or "未找到 Qwen 模型目录")

print("=== Python 路径 ===")
out, _ = run('which python3 && python3 --version')
print(out)

print("=== 项目根目录 ===")
out, _ = run('ls /root/nlp_project/')
print(out)

# 2. 检查是否需要下载模型
out, _ = run('ls /root/nlp_project/Qwen3.5-0.8B/ 2>/dev/null | head -10')
if not out.strip():
    print("模型不存在，需要下载...")
else:
    print("模型存在:", out[:200])

# 3. 检查 download_model.sh
print("=== download_model.sh ===")
out, _ = run('cat /root/nlp_project/download_model.sh 2>/dev/null')
print(out[:500] if out else "脚本不存在")

ssh.close()
