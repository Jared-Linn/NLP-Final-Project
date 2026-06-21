import re

with open("/root/scripts/step7_train_test.py", "r") as f:
    code = f.read()

# Update for RTX 3090
code = re.sub(r"NUM_EPOCHS = \d+", "NUM_EPOCHS = 3", code)
code = re.sub(r"BATCH_SIZE = \d+", "BATCH_SIZE = 4  # RTX3090 24GB", code)
code = re.sub(r"GRADIENT_ACCUMULATION_STEPS = \d+", "GRADIENT_ACCUMULATION_STEPS = 2", code)
code = re.sub(r"MAX_SEQ_LENGTH = \d+.*", "MAX_SEQ_LENGTH = 1024  # RTX3090 full length", code)
code = re.sub(r"TRAIN_SUBSET = \d+.*", "TRAIN_SUBSET = 0  # RTX3090: full 1600 samples", code)
# Remove eager attention (3090 supports flash)
code = code.replace(', attn_implementation="eager"', '')
code = code.replace(", attn_implementation='eager'", '')

with open("/root/scripts/step7_train_test.py", "w") as f:
    f.write(code)

print("step7 params updated for RTX 3090:")
for line in code.split('\n'):
    if any(k in line for k in ['NUM_EPOCHS', 'BATCH_SIZE', 'GRADIENT', 'MAX_SEQ', 'TRAIN_SUBSET']):
        print(f"  {line.strip()}")
