import os

folder = "data/raw/SWIR"

print("SWIR DATASET INSPECTION")
print("========================")

for root, dirs, files in os.walk(folder):

    for file in files:

        path = os.path.join(root, file)

        size = os.path.getsize(path) / (1024 * 1024)

        extension = os.path.splitext(file)[1]

        print(
            "File:", file,
            "| Format:", extension,
            "| Size:", round(size, 2), "MB"
        )