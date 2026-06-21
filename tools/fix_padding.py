import re

with open("/root/scripts/step7_train_test.py", "r") as f:
    code = f.read()

# Fix padding for batch_size>1
# Change padding=False to padding='max_length' in tokenize_fn
old = "padding=False,"
new = "padding='max_length',"
code = code.replace(old, new)

# Also need to set padding token on the output side for the collator
# Ensure the tokenizer has pad_token set (already done but double check)
with open("/root/scripts/step7_train_test.py", "w") as f:
    f.write(code)

print("Fixed: padding=False -> padding='max_length'")
print("Verifying tokenize_fn section:")
for i, line in enumerate(code.split('\n')):
    if 'padding' in line.lower() and 'tokenizer' in code.split('\n')[max(0,i-2):i+1][0].lower() if i>0 else '':
        print(f"  L{i+1}: {line.strip()}")
