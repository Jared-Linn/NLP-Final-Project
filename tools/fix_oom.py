with open("/root/scripts/step7_train_test.py", "r") as f:
    code = f.read()

# Fix 1: padding back to False (DataCollator handles dynamic padding for batch_size=1)
code = code.replace("padding='max_length'", "padding=False")

# Fix 2: batch_size=1, grad_accum=4 (safe for 3090, effective batch=4)
import re
code = re.sub(r"BATCH_SIZE = \d+.*", "BATCH_SIZE = 1  # safe for OOM", code)
code = re.sub(r"GRADIENT_ACCUMULATION_STEPS = \d+.*", "GRADIENT_ACCUMULATION_STEPS = 4", code)

# Fix 3: max_seq_length=512 to save memory
code = re.sub(r"MAX_SEQ_LENGTH = \d+.*", "MAX_SEQ_LENGTH = 512  # memory safe", code)

with open("/root/scripts/step7_train_test.py", "w") as f:
    f.write(code)

print("Fixed for OOM safety:")
for line in code.split('\n'):
    if any(k in line for k in ['BATCH_SIZE', 'GRADIENT', 'MAX_SEQ', 'TRAIN_SUBSET', 'padding']):
        print(f"  {line.strip()}")
