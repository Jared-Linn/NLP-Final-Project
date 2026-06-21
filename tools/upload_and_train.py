import paramiko, sys, time
sys.stdout.reconfigure(encoding='utf-8')

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.nmb2.seetacloud.com', port=42792, username='root', password='cIL5T2R2VKmG', timeout=30)

def run(cmd):
    stdin, stdout, stderr = ssh.exec_command(cmd)
    out = stdout.read().decode()
    err = stderr.read().decode()
    return out, err

# Upload shell script
sftp = ssh.open_sftp()
sftp.put('E:/work/Claude code default/自然语言处理期末/run_training.sh', '/root/run_training.sh')
sftp.close()
print("Script uploaded.")

# Run it
out, err = run('bash /root/run_training.sh 2>&1')
print(out)
if err:
    print("STDERR:", err)

# Wait a bit and check log
time.sleep(8)
out, err = run('tail -10 /root/training.log 2>/dev/null || echo no-log-yet')
print("Training log:")
print(out)

ssh.close()
print("Done. Training is running on server.")
print("Monitor: ssh -p 42792 root@connect.nmb2.seetacloud.com 'tail -f /root/training.log'")
