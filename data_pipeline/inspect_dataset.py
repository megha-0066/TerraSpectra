
import os

folders = {
    "VNIR": "data/raw/VNIR",
    "SWIR": "data/raw/SWIR"
}

for name, path in folders.items():

    print("\n==========", name, "DATASET ==========")

    if not os.path.exists(path):
        print("Folder not found:", path)
        continue

    for root, dirs, files in os.walk(path):

        for file in files:
            full_path = os.path.join(root, file)

            size_mb = os.path.getsize(full_path) / (1024 * 1024)

            print(
                file,
                "|",
                round(size_mb, 2),
                "MB"
            )
