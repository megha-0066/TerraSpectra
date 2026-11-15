```python
import os
from collections import Counter

folder = "data/raw/SWIR"

print("\n========== SWIR DATASET INSPECTION ==========\n")

if not os.path.exists(folder):
    print(f"Dataset folder not found: {folder}")
    exit()

total_files = 0
total_size = 0
formats = Counter()

for root, dirs, files in os.walk(folder):
    level = root.replace(folder, "").count(os.sep)
    indent = "    " * level

    print(f"{indent}{os.path.basename(root)}/")

    for file in files:
        path = os.path.join(root, file)
        size = os.path.getsize(path) / (1024 * 1024)
        extension = os.path.splitext(file)[1].lower() or "No extension"

        total_files += 1
        total_size += size
        formats[extension] += 1

        print(
            f"{indent}    File: {file} | "
            f"Format: {extension} | "
            f"Size: {size:.2f} MB"
        )

print("\n========== DATASET SUMMARY ==========")
print("Total files:", total_files)
print("Total size:", round(total_size, 2), "MB")

print("\nFile formats:")
for extension, count in formats.items():
    print(f"{extension}: {count} files")

print("\n========== INSPECTION COMPLETED ==========")
```
        )
