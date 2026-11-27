
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import spectral


PROJECT_ROOT = Path(__file__).resolve().parents[1]

VNIR_FOLDER = PROJECT_ROOT / "data" / "raw" / "VNIR"

OUTPUT_FOLDER = (
    PROJECT_ROOT / "data" / "processed" / "hyperspectral_visualization"
)

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


def find_band(wavelengths, target):
    """Find the band closest to a target wavelength."""

    return int(np.argmin(np.abs(wavelengths - target)))


def normalize_band(band):
    """Normalize a band for visualization."""

    band = np.asarray(band, dtype=np.float32)

    low, high = np.nanpercentile(band, [2, 98])

    if high <= low:
        return np.zeros_like(band)

    return np.clip((band - low) / (high - low), 0, 1)


def visualize_hyperspectral():

    print("\nTERRASPECTRA: HYPERSPECTRAL VISUALIZATION")
    print("=" * 55)

    hdr_files = list(VNIR_FOLDER.rglob("*.hdr"))

    if not hdr_files:
        print("No VNIR .hdr files found.")
        print("Please download and extract an actual VNIR image archive.")
        return

    hdr_file = hdr_files[0]

    print("Selected header:", hdr_file.name)

    try:
        image = spectral.open_image(str(hdr_file))
        cube = np.asarray(image.load(), dtype=np.float32)

        if cube.ndim != 3:
            raise ValueError("Expected a three-dimensional image cube.")

        height, width, bands = cube.shape

        print("\nImage dimensions")
        print("Height:", height)
        print("Width:", width)
        print("Spectral bands:", bands)

        wavelengths = image.metadata.get("wavelength")

        if wavelengths:
            wavelengths = np.asarray(
                [float(w) for w in wavelengths],
                dtype=np.float32
            )

            if len(wavelengths) != bands:
                raise ValueError(
                    "Wavelength count does not match cube bands."
                )

            red_band = find_band(wavelengths, 650)
            green_band = find_band(wavelengths, 550)
            blue_band = find_band(wavelengths, 450)

            print("\nSelected wavelength bands")
            print("Red:", wavelengths[red_band], "nm")
            print("Green:", wavelengths[green_band], "nm")
            print("Blue:", wavelengths[blue_band], "nm")

        else:
            print("\nWavelength metadata unavailable.")
            print("Using approximate band positions for visualization.")

            red_band = int(bands * 0.70)
            green_band = int(bands * 0.45)
            blue_band = int(bands * 0.20)

        # Individual band visualization
        selected_bands = sorted(set([
            0,
            bands // 4,
            bands // 2,
            (3 * bands) // 4,
            bands - 1
        ]))

        fig, axes = plt.subplots(
            1, len(selected_bands),
            figsize=(18, 5)
        )

        for ax, band_index in zip(axes, selected_bands):

            band_image = normalize_band(cube[:, :, band_index])

            ax.imshow(band_image, cmap="gray")
            ax.set_title(f"Band {band_index}")
            ax.axis("off")

        fig.suptitle("VNIR Individual Spectral Bands")
        plt.tight_layout()

        band_output = OUTPUT_FOLDER / "individual_bands.png"
        plt.savefig(band_output, dpi=200)
        plt.close()

        # False-colour RGB composite
        red = normalize_band(cube[:, :, red_band])
        green = normalize_band(cube[:, :, green_band])
        blue = normalize_band(cube[:, :, blue_band])

        rgb_image = np.dstack((red, green, blue))

        plt.figure(figsize=(9, 7))
        plt.imshow(rgb_image)
        plt.title("VNIR False-Colour RGB Composite")
        plt.axis("off")

        rgb_output = OUTPUT_FOLDER / "vnir_false_colour.png"
        plt.savefig(rgb_output, dpi=200, bbox_inches="tight")
        plt.close()

        print("\nVisualization completed.")
        print("Individual bands:", band_output)
        print("False-colour image:", rgb_output)

    except Exception as error:
        print("\nVisualization failed.")
        print("Error:", error)


if __name__ == "__main__":
    visualize_hyperspectral()
