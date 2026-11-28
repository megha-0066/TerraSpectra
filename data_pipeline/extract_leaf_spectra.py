import numpy as np
import pandas as pd
import spectral
from pathlib import Path

# ==========================================
# TERRASPECTRA: LEAF SPECTRAL EXTRACTION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw" / "VNIR"

OUTPUT_DIR = BASE_DIR / "data" / "processed" / "leaf_spectra"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HEADER_FILE = RAW_DIR / "13_G.hdr"

ANNOTATION_FILES = {
    "Leaf_1": RAW_DIR / "13_G_leaf1.csv",
    "Leaf_2": RAW_DIR / "13_G_leaf2.csv"
}

print("=" * 60)
print("TERRASPECTRA: LEAF SPECTRAL FEATURE EXTRACTION")
print("=" * 60)

# 1. Load hyperspectral image metadata
print("\nLoading hyperspectral image...")

image = spectral.open_image(str(HEADER_FILE))

cube = image.open_memmap(writable=False)

height, width, bands = cube.shape

print("Image shape:", cube.shape)
print("Number of spectral bands:", bands)

# 2. Read wavelength information
wavelengths = image.metadata.get("wavelength")

if wavelengths is not None:
    wavelengths = np.asarray(wavelengths, dtype=float)
else:
    wavelengths = np.arange(bands)

if len(wavelengths) != bands:
    raise ValueError("Wavelength count does not match spectral bands.")

# 3. Extract spectral information
all_spectra = []
all_leaf_ids = []
all_rows = []
all_columns = []

summary_records = []

for leaf_name, annotation_file in ANNOTATION_FILES.items():

    print(f"\nProcessing {leaf_name}...")

    if not annotation_file.exists():
        raise FileNotFoundError(annotation_file)

    coordinates = pd.read_csv(
        annotation_file,
        header=None,
        names=["row", "column"]
    )

    rows = coordinates["row"].to_numpy(dtype=int)
    columns = coordinates["column"].to_numpy(dtype=int)

    # Check coordinate bounds
    valid = (
        (rows >= 0) &
        (rows < height) &
        (columns >= 0) &
        (columns < width)
    )

    if not np.all(valid):
        raise ValueError(
            f"Invalid coordinates found in {annotation_file.name}"
        )

    # Extract all 216 spectral bands for each pixel
    spectra = np.asarray(cube[rows, columns, :], dtype=np.float32)

    if spectra.shape != (len(rows), bands):
        raise ValueError("Unexpected extracted spectral dimensions.")

    # Check non-finite values
    finite_pixels = np.all(np.isfinite(spectra), axis=1)

    spectra = spectra[finite_pixels]
    valid_rows = rows[finite_pixels]
    valid_columns = columns[finite_pixels]

    if len(spectra) == 0:
        raise ValueError(f"No finite spectra found for {leaf_name}")

    print("Annotated pixels:", len(rows))
    print("Extracted pixels:", len(spectra))
    print("Spectral features per pixel:", spectra.shape[1])

    # Store pixel-level spectra
    all_spectra.append(spectra)

    all_leaf_ids.extend([leaf_name] * len(spectra))
    all_rows.extend(valid_rows)
    all_columns.extend(valid_columns)

    # Calculate summary statistics
    mean_spectrum = np.mean(spectra, axis=0)
    median_spectrum = np.median(spectra, axis=0)
    std_spectrum = np.std(spectra, axis=0)

    for band_index in range(bands):

        summary_records.append({
            "leaf_id": leaf_name,
            "band_index": band_index,
            "wavelength_nm": wavelengths[band_index],
            "mean": mean_spectrum[band_index],
            "median": median_spectrum[band_index],
            "std": std_spectrum[band_index]
        })


# 4. Combine extracted data
pixel_spectra = np.vstack(all_spectra)

print("\nTotal extracted pixels:", len(pixel_spectra))
print("Total spectral features:", pixel_spectra.shape[1])

# 5. Save pixel-level spectral dataset
np.savez_compressed(
    OUTPUT_DIR / "13_G_pixel_spectra.npz",
    spectra=pixel_spectra,
    leaf_ids=np.asarray(all_leaf_ids),
    rows=np.asarray(all_rows),
    columns=np.asarray(all_columns),
    wavelengths=wavelengths
)

# 6. Save summary statistics
summary_df = pd.DataFrame(summary_records)

summary_df.to_csv(
    OUTPUT_DIR / "13_G_leaf_spectral_summary.csv",
    index=False
)

print("\n============================================")
print("SPECTRAL EXTRACTION COMPLETED")
print("============================================")

print("Pixel-level dataset:")
print(OUTPUT_DIR / "13_G_pixel_spectra.npz")

print("\nSpectral summary:")
print(OUTPUT_DIR / "13_G_leaf_spectral_summary.csv")

print("\nFinal pixel-level shape:", pixel_spectra.shape)

print("\nIMPORTANT:")
print("These are extracted sensor measurements.")
print("They are not yet calibrated reflectance values.")
print("Disease labels have not yet been assigned.")