import os
import numpy as np
from PIL import Image
from collections import Counter

value_map = {0:0, 100:1, 200:2, 300:3, 500:4, 550:5, 600:6, 700:7, 800:8, 7100:9, 10000:10}
CLASS_NAMES = ['Background','Trees','Lush Bushes','Dry Grass','Dry Bushes',
               'Ground Clutter','Flowers','Logs','Rocks','Landscape','Sky']

# Update this path to where the training segmentation masks are located
mask_dir = './data/train/Segmentation'

pixel_counts = Counter()
files = os.listdir(mask_dir)
print(f"Processing {len(files)} masks...")

for fname in files:
    try:
        arr = np.array(Image.open(os.path.join(mask_dir, fname))).astype(np.int32)
        for raw, mapped in value_map.items():
            pixel_counts[mapped] += int(np.sum(arr == raw))
    except Exception as e:
        print(f"Skipping {fname} due to error: {e}")

total = sum(pixel_counts.values())
if total == 0:
    print("No valid pixels found. Check the mask_dir path!")
else:
    print("\nClass distribution:")
    for i, name in enumerate(CLASS_NAMES):
        pct = 100 * pixel_counts[i] / total
        bar = '█' * int(pct / 2)
        print(f"  {name:<18}: {pct:5.2f}% {bar}")
