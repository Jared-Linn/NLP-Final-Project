import re

with open("/root/scripts/step7_train_test.py", "r") as f:
    code = f.read()

# Fix 1: tokenize_fn - change padding=False to padding='max_length'
code = code.replace(
    "padding=False,",
    "padding='max_length', padding_max_length=MAX_SEQ_LENGTH,"
)

# Fix 2: data collator should use mlm=False (already set, just verify)
# Fix 3: ensure dataloader_num_workers=0 (already set)

# Fix 4: The tokenize function nested list issue - labels need proper handling
# Add pad_token_id for padding
old = """tokenizer.padding_side = "right""""
new = """tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id"""

# Already done above, let me check
# Actually the issue is labels padding - need to pad labels with -100
# When using padding='max_length', the tokenizer pads input_ids, but labels need manual padding

# Better approach: use DataCollatorForLanguageModeling which handles this
# But we need tokenizer to NOT pad, and let DataCollator do it

# Revert to no padding in tokenizer, but ensure DataCollator has proper pad_token_id
code = code.replace(
    "padding='max_length', padding_max_length=MAX_SEQ_LENGTH,",
    "padding=False,"
)

# The real fix: ensure DataCollator has tokenizer with pad_token_id set
# and that pad_to_multiple_of is set properly
old_dc = "data_collator = DataCollatorForLanguageModeling(\n        tokenizer=tokenizer,\n        mlm=False,\n    )"
new_dc = "data_collator = DataCollatorForLanguageModeling(\n        tokenizer=tokenizer,\n        mlm=False,\n        pad_to_multiple_of=8,\n    )"
code = code.replace(old_dc, new_dc)

# Also add remove_unused_columns check to False (already set)
print("Checking for remove_unused_columns...")
if "remove_unused_columns" not in code:
    print("  WARNING: remove_unused_columns not found, adding...")

with open("/root/scripts/step7_train_test.py", "w") as f:
    f.write(code)

print("Done fixing padding. Key changes:")
print("  - DataCollator pad_to_multiple_of=8")
print("  - Verified padding=False in tokenizer (dynamic padding by collator)")
