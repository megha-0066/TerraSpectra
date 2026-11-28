from pathlib import Path

import numpy as np
import spectral
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

HDR_PATH = ROOT / "data" / "raw" / "VNIR" / "13_G.hdr"

OUTPUT_FOLDER = (
    ROOT / "data" / "processed" / "vnir_quality_maps"
)

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


def create_quality_maps():

    print("\nTERRASPECTRA: VNIR SPATIAL QUALITY MAPS")
    print("=" * 55)

    # Load hyperspectral image
    image = spectral.open_image(str(HDR_PATH))

    print("Loading hyperspectral cube...", flush=True)

    cube = np.asarray(image.load())

    height, width, bands = cube.shape

    ignore_value = float(
        image.metadata.get("data ignore value", 4095)
    )

    print("\nImage dimensions:", cube.shape)
    print("Ignore value:", ignore_value)

    # Identify invalid measurements
    finite_mask = np.isfinite(cube)

    ignore_mask = cube == ignore_value

    zero_mask = cube == 0

    # Count measurements for each spatial pixel
    ignore_count = np.sum(ignore_mask, axis=2)

    zero_count = np.sum(zero_mask, axis=2)

    valid_count = np.sum(
        finite_mask & ~ignore_mask,
        axis=2
    )

    # Pixels containing at least one ignore value
    invalid_pixel_map = ignore_count > 0

    # Pixels containing at least one zero measurement
    zero_pixel_map = zero_count > 0

    # Fraction of bands with valid measurements
    valid_fraction = valid_count / bands

    print("\nSPATIAL QUALITY SUMMARY")

    print(
        "Pixels containing ignore values:",
        np.sum(invalid_pixel_map)
    )

    print(
        "Pixels containing zero values:",
        np.sum(zero_pixel_map)
    )

    print(
        "Pixels with all bands valid:",
        np.sum(valid_count == bands)
    )

    # Plot spatial quality maps
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    images = [
        (ignore_count, "Ignore-value count per pixel"),
        (zero_count, "Zero-value count per pixel"),
        (valid_fraction, "Valid spectral measurement fraction"),
        (invalid_pixel_map, "Pixels containing ignore values")
    ]

    for ax, (data, title) in zip(axes.flat, images):

        plot = ax.imshow(data, cmap="viridis")

        ax.set_title(title)
        ax.set_xlabel("Image width (pixels)")
        ax.set_ylabel("Image height (pixels)")

        fig.colorbar(plot, ax=ax, fraction=0.046)

    plt.tight_layout()

    output_path = OUTPUT_FOLDER / "vnir_spatial_quality_maps.png"

    plt.savefig(output_path, dpi=200)

    plt.close()

    print("\nQUALITY MAP SAVED:")
    print(output_path)

    print("\nSpatial quality analysis completed!")


if __name__ == "__main__":
    create_quality_maps()