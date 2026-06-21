import paramiko, sys
sys.stdout.reconfigure(encoding='utf-8')

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.nmb2.seetacloud.com', port=42792, username='root', password='cIL5T2R2VKmG', timeout=15)

stdin, stdout, stderr = ssh.exec_command('tail -30 /root/training.log 2>/dev/null')
out = stdout.read().decode('utf-8', errors='replace')
print(out)

# Check if training completed
stdin2, stdout2, stderr2 = ssh.exec_command('grep -c "Step 7" /root/training.log 2>/dev/null; grep "训练完成\|训练失败\|Step 7 完成" /root/training.log 2>/dev/null || echo "still-training"')
print(stdout2.read().decode('utf-8', errors='replace'))

ssh.close()
