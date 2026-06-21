import paramiko, sys
sys.stdout.reconfigure(encoding='utf-8')

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.nmb2.seetacloud.com', port=42792, username='root', password='cIL5T2R2VKmG', timeout=15)

# Get last 5 lines for current progress
stdin, stdout, stderr = ssh.exec_command('tail -5 /root/training.log 2>/dev/null')
out = stdout.read().decode('utf-8', errors='replace')
print("=== Last 5 lines ===")
print(out)

# Get training status
stdin2, stdout2, stderr2 = ssh.exec_command('grep -c "loss" /root/training.log 2>/dev/null; grep "训练完成\|训练失败\|Step 7 完成\|100%" /root/training.log 2>/dev/null | tail -3')
out2 = stdout2.read().decode('utf-8', errors='replace')
print("=== Status ===")
print(out2)

ssh.close()
