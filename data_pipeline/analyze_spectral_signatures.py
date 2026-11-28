from pathlib import Path
import csv

import numpy as np
import spectral
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

HDR_PATH = ROOT / "data" / "raw" / "VNIR" / "13_G.hdr"

OUTPUT_FOLDER = (
    ROOT / "data" / "processed" / "spectral_signatures"
)

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


def get_wavelengths(image, number_of_bands):

    wavelengths = image.bands.centers

    if wavelengths is not None and len(wavelengths) == number_of_bands:
        return np.asarray(wavelengths, dtype=float)

    print("Wavelength metadata unavailable. Using band numbers.")

    return np.arange(1, number_of_bands + 1)


def find_nearest_valid_pixel(valid_mask, target_row, target_col):

    candidates = np.argwhere(valid_mask)

    if len(candidates) == 0:
        raise ValueError("No pixels have all spectral bands valid.")

    distances = (
        (candidates[:, 0] - target_row) ** 2
        + (candidates[:, 1] - target_col) ** 2
    )

    nearest = candidates[np.argmin(distances)]

    return int(nearest[0]), int(nearest[1])


def analyze_signatures():

    print("\nTERRASPECTRA: SPECTRAL SIGNATURE ANALYSIS")
    print("=" * 60)

    # Load hyperspectral cube
    image = spectral.open_image(str(HDR_PATH))

    print("Loading VNIR hyperspectral cube...", flush=True)

    cube = np.asarray(image.load())

    height, width, bands = cube.shape

    ignore_value = float(
        image.metadata.get("data ignore value", 4095)
    )

    wavelengths = get_wavelengths(image, bands)

    print("\nIMAGE INFORMATION")
    print("Height:", height)
    print("Width:", width)
    print("Spectral bands:", bands)
    print("Wavelength range:",
          wavelengths.min(), "to", wavelengths.max(), "nm")

    # Identify pixels with all bands valid
    valid_measurements = (
        np.isfinite(cube)
        & (cube != ignore_value)
    )

    valid_pixel_mask = np.all(
        valid_measurements,
        axis=2
    )

    print("\nVALID PIXEL INFORMATION")
    print("Valid pixels:", np.sum(valid_pixel_mask))
    print("Invalid pixels:", np.sum(~valid_pixel_mask))

    # Calculate mean and median for each band
    band_means = []
    band_medians = []
    valid_counts = []

    print("\nCalculating band statistics...", flush=True)

    for band in range(bands):

        band_data = cube[:, :, band]

        valid = (
            np.isfinite(band_data)
            & (band_data != ignore_value)
        )

        values = band_data[valid]

        if values.size == 0:
            band_means.append(np.nan)
            band_medians.append(np.nan)
            valid_counts.append(0)

        else:
            band_means.append(float(np.mean(values)))
            band_medians.append(float(np.median(values)))
            valid_counts.append(int(values.size))

    # Save spectral statistics
    csv_path = OUTPUT_FOLDER / "spectral_band_statistics.csv"

    with open(csv_path, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            "Band",
            "Wavelength_nm",
            "Mean_Intensity",
            "Median_Intensity",
            "Valid_Pixel_Count"
        ])

        for band in range(bands):

            writer.writerow([
                band + 1,
                wavelengths[band],
                band_means[band],
                band_medians[band],
                valid_counts[band]
            ])

    # Select representative valid pixel locations
    target_locations = [
        (height // 4, width // 4),
        (height // 4, 3 * width // 4),
        (height // 2, width // 2),
        (3 * height // 4, width // 4),
        (3 * height // 4, 3 * width // 4)
    ]

    selected_locations = []

    for row, col in target_locations:

        actual_row, actual_col = find_nearest_valid_pixel(
            valid_pixel_mask,
            row,
            col
        )

        selected_locations.append((actual_row, actual_col))

    # Plot mean and median spectral signatures
    plt.figure(figsize=(12, 6))

    plt.plot(
        wavelengths,
        band_means,
        label="Mean spectral intensity"
    )

    plt.plot(
        wavelengths,
        band_medians,
        label="Median spectral intensity"
    )

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Raw sensor intensity")
    plt.title("VNIR Mean and Median Spectral Signatures")

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    mean_plot = OUTPUT_FOLDER / "mean_median_spectral_signature.png"

    plt.savefig(mean_plot, dpi=200)
    plt.close()

    # Plot signatures from selected pixels
    plt.figure(figsize=(12, 6))

    for index, (row, col) in enumerate(selected_locations):

        signature = cube[row, col, :]

        plt.plot(
            wavelengths,
            signature,
            label=f"Pixel {index + 1}: ({row}, {col})"
        )

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Raw sensor intensity")
    plt.title("VNIR Spectral Signatures at Selected Pixels")

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    pixel_plot = OUTPUT_FOLDER / "selected_pixel_signatures.png"

    plt.savefig(pixel_plot, dpi=200)
    plt.close()

    print("\nANALYSIS COMPLETED")

    print("\nGenerated files:")
    print(csv_path)
    print(mean_plot)
    print(pixel_plot)

    print("\nSelected pixel coordinates:")

    for location in selected_locations:
        print(location)


if __name__ == "__main__":
    analyze_signatures()