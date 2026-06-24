"""Cloud training monitor - checks every 5min, auto-downloads when done"""
import paramiko, time, os, subprocess, sys
from datetime import datetime

CLOUD_HOST = 'connect.cqa1.seetacloud.com'
CLOUD_PORT = 21026
CLOUD_USER = 'root'
CLOUD_PASS = '8uiFh3ZS8JWP'
PROJECT_DIR = '/root/nlp_project'
LOCAL_DIR = r'E:\work\Claude code default\自然语言处理期末'
LOCAL_OUTPUT = os.path.join(LOCAL_DIR, 'outputs_cloud')

def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line)
    sys.stdout.flush()

def ssh_connect():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(CLOUD_HOST, port=CLOUD_PORT, username=CLOUD_USER, password=CLOUD_PASS, timeout=20)
    return ssh

def check_status(ssh):
    stdin, stdout, stderr = ssh.exec_command(
        'ps aux | grep "step7_train_test" | grep -v grep | wc -l', timeout=10
    )
    count = int(stdout.read().decode().strip())
    stdin, stdout, stderr = ssh.exec_command(
        f'tail -2 {PROJECT_DIR}/train_output.log 2>/dev/null | tr "\\r" "\\n" | tail -1', timeout=10
    )
    progress = stdout.read().decode().strip()[:120]
    return count > 0, progress

def download_weights():
    log('>>> Downloading LoRA weights...')
    os.makedirs(LOCAL_OUTPUT, exist_ok=True)
    # use paramiko SFTP since scp may not work on Windows
    ssh = ssh_connect()
    sftp = ssh.open_sftp()
    remote_dir = f'{PROJECT_DIR}/outputs/lora_adapter'
    try:
        for f in sftp.listdir(remote_dir):
            remote_path = f'{remote_dir}/{f}'
            local_path = os.path.join(LOCAL_OUTPUT, f)
            sftp.get(remote_path, local_path)
            log(f'  Downloaded: {f}')
        # also get subdirectories
        for item in sftp.listdir(remote_dir):
            item_path = f'{remote_dir}/{item}'
            try:
                for f2 in sftp.listdir(item_path):
                    local_dir = os.path.join(LOCAL_OUTPUT, item)
                    os.makedirs(local_dir, exist_ok=True)
                    sftp.get(f'{item_path}/{f2}', os.path.join(local_dir, f2))
                    log(f'  Downloaded: {item}/{f2}')
            except:
                pass
        log('[OK] Weights downloaded')
    except Exception as e:
        log(f'[ERR] Download failed: {e}')
    sftp.close()
    ssh.close()

def monitor():
    log('===== Cloud Training Monitor Started =====')
    log(f'Target: {CLOUD_HOST}:{CLOUD_PORT}')
    log(f'Local output: {LOCAL_OUTPUT}')

    fail_count = 0
    check_num = 0

    while True:
        try:
            check_num += 1
            ssh = ssh_connect()
            running, progress = check_status(ssh)

            if running:
                if 'loss' in progress.lower():
                    log(f'R[{check_num}] Loss: {progress}')
                elif '%' in progress or 'step' in progress.lower() or 'it/' in progress:
                    # extract progress percentage
                    log(f'R[{check_num}] {progress}')
                else:
                    log(f'R[{check_num}] Training...')

                fail_count = 0
                ssh.close()
                time.sleep(300)
                continue

            # Training stopped
            log(f'R[{check_num}] Training process exited, checking result...')
            stdin, stdout, stderr = ssh.exec_command(
                f'grep -E "训练完成|Error|失败|OOM|train_loss" {PROJECT_DIR}/train_output.log 2>/dev/null | tail -10',
                timeout=10
            )
            result = stdout.read().decode()

            if '训练完成' in result or 'train_loss' in result:
                log('[OK] Training completed successfully!')
                for line in result.strip().split('\n'):
                    log(f'  {line[:200]}')

                # Run generalization test on cloud
                log('>>> Running generalization test on cloud...')
                stdin, stdout, stderr = ssh.exec_command(
                    f'cd {PROJECT_DIR} && HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8 '
                    f'/root/miniconda3/bin/python scripts/test_generalization.py > '
                    f'{PROJECT_DIR}/gen_test.log 2>&1',
                    timeout=300
                )
                time.sleep(5)
                stdin2, stdout2, stderr2 = ssh.exec_command(
                    f'tail -30 {PROJECT_DIR}/gen_test.log 2>/dev/null', timeout=10
                )
                log(stdout2.read().decode()[:500])

                ssh.close()

                # Download weights
                download_weights()

                # Download test log
                try:
                    ssh2 = ssh_connect()
                    sftp = ssh2.open_sftp()
                    sftp.get(f'{PROJECT_DIR}/gen_test.log',
                             os.path.join(LOCAL_OUTPUT, 'gen_test.log'))
                    sftp.get(f'{PROJECT_DIR}/train_output.log',
                             os.path.join(LOCAL_OUTPUT, 'train_output.log'))
                    sftp.get(f'{PROJECT_DIR}/outputs/generalization_test_report.json',
                             os.path.join(LOCAL_OUTPUT, 'generalization_test_report.json'))
                    sftp.close()
                    ssh2.close()
                    log('[OK] Test reports downloaded')
                except Exception as e:
                    log(f'[WARN] Report download: {e}')

                log('[DONE] All tasks completed!')
                return

            elif 'Error' in result or '失败' in result:
                log(f'[FAIL] Training failed: {result[:500]}')
                ssh.close()
                return

            else:
                log(f'[WARN] Process exited but no clear status. Log tail:')
                stdin, stdout, stderr = ssh.exec_command(
                    f'tail -10 {PROJECT_DIR}/train_output.log 2>/dev/null', timeout=10
                )
                log(stdout.read().decode()[:500])
                ssh.close()
                return

        except Exception as e:
            fail_count += 1
            log(f'[ERR] Connection failed ({fail_count}/10): {e}')
            if fail_count >= 10:
                log('[FAIL] 10 consecutive failures, giving up')
                return
            time.sleep(60)


if __name__ == '__main__':
    monitor()
