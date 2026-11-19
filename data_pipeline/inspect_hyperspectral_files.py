import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Common hyperspectral and scientific image formats
SUPPORTED_EXTENSIONS = {
    ".hdr", ".raw", ".dat", ".bil", ".bip", ".bsq",
    ".img", ".envi", ".tif", ".tiff",
    ".npy", ".npz", ".mat", ".h5", ".hdf5"
}

def inspect_files():
    print("\n========================================")
    print(" TERRASPECTRA HYPERSPECTRAL FILE INSPECTION")
    print("========================================")

    if not DATA_DIR.exists():
        print("Data directory not found:", DATA_DIR)
        return

    all_files = [
        file for file in DATA_DIR.rglob("*")
        if file.is_file()
    ]

    print("\nTotal files found:", len(all_files))

    hyperspectral_files = [
        file for file in all_files
        if file.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    print("\nPotential hyperspectral or scientific image files:")

    if hyperspectral_files:
        for file in hyperspectral_files:
            size_mb = file.stat().st_size / (1024 * 1024)

            print("\nFile:", file.name)
            print("Path:", file)
            print(f"Size: {size_mb:.2f} MB")
            print("Format:", file.suffix.lower())
    else:
        print("No files with the supported image/data extensions found.")

    print("\nCSV files found:")

    csv_files = [
        file for file in all_files
        if file.suffix.lower() == ".csv"
    ]

    for file in csv_files:
        size_kb = file.stat().st_size / 1024
        print(f"{file} | {size_kb:.2f} KB")

    print("\n========================================")
    print(" FILE INSPECTION COMPLETED")
    print("========================================")

    if not hyperspectral_files:
        print(
            "\nNext action: Check the original dataset download "
            "for hyperspectral image cubes."
        )

if __name__ == "__main__":
    inspect_files()