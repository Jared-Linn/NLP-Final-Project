import easyocr
import os
import sys
from PIL import Image
import numpy as np

# Fix Windows encoding for Chinese paths
sys.stdout.reconfigure(encoding='utf-8')

reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)

base = r"E:\work\Claude code default\自然语言处理期末\test\测试方案"
output_file = r"E:\work\Claude code default\自然语言处理期末\test\ocr_results.txt"

# Use os.scandir with bytes fallback
results = []

for group in sorted(os.listdir(base)):
    group_path = os.path.join(base, group)
    if not os.path.isdir(group_path):
        continue
    for img_name in sorted(os.listdir(group_path)):
        img_path = os.path.join(group_path, img_name)
        print(f"OCR: {group}/{img_name}")

        # Use PIL to open image (handles Unicode paths better than OpenCV)
        try:
            pil_img = Image.open(img_path).convert('RGB')
            img_array = np.array(pil_img)
            text = reader.readtext(img_array, detail=0)
            results.append(f"\n{'='*60}")
            results.append(f"组: {group}")
            results.append(f"图片: {img_name}")
            results.append(f"{'='*60}")
            results.append('\n'.join(text))
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append(f"\n{'='*60}")
            results.append(f"组: {group} / 图片: {img_name}")
            results.append(f"ERROR: {e}")
            results.append(f"{'='*60}")

with open(output_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(results))

print(f"\nDone! Output: {output_file}")
print(f"Total groups: {sum(1 for g in os.listdir(base) if os.path.isdir(os.path.join(base, g)))}")
