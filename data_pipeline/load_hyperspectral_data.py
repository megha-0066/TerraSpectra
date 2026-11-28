from pathlib import Path
import numpy as np
import spectral


PROJECT_ROOT = Path(__file__).resolve().parents[1]

VNIR_FOLDER = PROJECT_ROOT / "data" / "raw" / "VNIR"

OUTPUT_FOLDER = (
    PROJECT_ROOT / "data" / "processed" / "hyperspectral"
)

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


def load_vnir_cube():

    print("\nTERRASPECTRA: HYPERSPECTRAL DATA LOADER")
    print("=" * 50)

    hdr_files = list(VNIR_FOLDER.rglob("*.hdr"))

    if not hdr_files:
        print("No hyperspectral header files found.")
        print("Please add VNIR .hdr and corresponding image files.")
        return

    hdr_file = hdr_files[0]

    print("Selected file:", hdr_file.name)

    try:

        image = spectral.open_image(str(hdr_file))

        print("\nLoading hyperspectral cube...")

        cube = image.load()

        cube = np.asarray(cube, dtype=np.float32)

        print("\nLOADING SUCCESSFUL")

        print("Cube shape:", cube.shape)
        print("Height:", cube.shape[0])
        print("Width:", cube.shape[1])
        print("Bands:", cube.shape[2])

        print("\nDATA INFORMATION")

        print("Minimum:", np.nanmin(cube))
        print("Maximum:", np.nanmax(cube))
        print("Mean:", np.nanmean(cube))

        print("NaN values:", np.isnan(cube).sum())
        print("Infinite values:", np.isinf(cube).sum())

        output_file = OUTPUT_FOLDER / "vnir_sample_cube.npy"

        np.save(output_file, cube)

        print("\nSaved cube:", output_file)

    except Exception as error:

        print("\nERROR LOADING HYPERSPECTRAL DATA")
        print(error)


if __name__ == "__main__":
    load_vnir_cube()