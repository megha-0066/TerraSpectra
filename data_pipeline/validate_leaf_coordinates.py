import pandas as pd
import numpy as np
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
ANNOTATION_DIR = BASE_DIR / "data" / "raw" / "VNIR"

# Hyperspectral image dimensions
HEIGHT = 1370
WIDTH = 1024

annotation_files = [
    "13_G_leaf1.csv",
    "13_G_leaf2.csv"
]

print("=" * 55)
print("TERRASPECTRA: LEAF COORDINATE VALIDATION")
print("=" * 55)

for filename in annotation_files:

    file_path = ANNOTATION_DIR / filename

    coordinates = pd.read_csv(
        file_path,
        header=None,
        names=["row", "column"]
    )

    rows = coordinates["row"].to_numpy()
    columns = coordinates["column"].to_numpy()

    valid = (
        (rows >= 0) &
        (rows < HEIGHT) &
        (columns >= 0) &
        (columns < WIDTH)
    )

    print(f"\nFile: {filename}")
    print(f"Total coordinates: {len(coordinates)}")
    print(f"Valid coordinates: {np.sum(valid)}")
    print(f"Invalid coordinates: {np.sum(~valid)}")
    print(f"Row range: {rows.min()} to {rows.max()}")
    print(f"Column range: {columns.min()} to {columns.max()}")

print("\nValidation completed.")