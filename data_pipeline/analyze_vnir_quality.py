from pathlib import Path
import numpy as np
import spectral
import csv

ROOT = Path(__file__).resolve().parents[1]

HDR_PATH = ROOT / "data" / "raw" / "VNIR" / "13_G.hdr"

OUTPUT_FOLDER = ROOT / "data" / "processed"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

REPORT_PATH = OUTPUT_FOLDER / "vnir_band_quality.csv"


def analyze_vnir():

    print("\nTERRASPECTRA: VNIR DATA QUALITY ANALYSIS")
    print("=" * 55)

    # Load hyperspectral image
    image = spectral.open_image(str(HDR_PATH))

    print("Loading hyperspectral cube...", flush=True)

    cube = np.asarray(image.load())

    height, width, bands = cube.shape

    print("\nIMAGE INFORMATION")
    print("Height:", height)
    print("Width:", width)
    print("Bands:", bands)

    # Read ignore value
    ignore_value = float(
        image.metadata.get("data ignore value", 4095)
    )

    print("\nPIXEL QUALITY ANALYSIS")

    finite_mask = np.isfinite(cube)

    ignore_mask = cube == ignore_value

    zero_mask = cube == 0

    invalid_pixels = np.any(
        (~finite_mask) | ignore_mask,
        axis=2
    )

    background_pixels = np.all(
        zero_mask,
        axis=2
    )

    total_pixels = height * width

    print("Total pixels:", total_pixels)
    print("Ignore value:", ignore_value)

    print("Ignore-value entries:", np.sum(ignore_mask))
    print("Non-finite entries:", np.sum(~finite_mask))
    print("Zero-value entries:", np.sum(zero_mask))

    print("Pixels containing invalid values:",
          np.sum(invalid_pixels))

    print("Completely zero pixels:",
          np.sum(background_pixels))

    print("\nBAND-WISE ANALYSIS")

    with open(REPORT_PATH, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            "Band",
            "Minimum",
            "Maximum",
            "Mean",
            "Ignore_Value_Count",
            "Zero_Value_Count"
        ])

        for band in range(bands):

            band_data = cube[:, :, band]

            valid = (
                np.isfinite(band_data)
                & (band_data != ignore_value)
            )

            valid_values = band_data[valid]

            if valid_values.size == 0:
                minimum = maximum = mean = "No valid data"
            else:
                minimum = float(np.min(valid_values))
                maximum = float(np.max(valid_values))
                mean = float(np.mean(valid_values))

            writer.writerow([
                band + 1,
                minimum,
                maximum,
                mean,
                int(np.sum(band_data == ignore_value)),
                int(np.sum(band_data == 0))
            ])

    print("\nQUALITY REPORT SAVED")
    print(REPORT_PATH)

    print("\nVNIR DATA QUALITY ANALYSIS COMPLETED!")


if __name__ == "__main__":
    analyze_vnir()