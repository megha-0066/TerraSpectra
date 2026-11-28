
from pathlib import Path
import numpy as np
import spectral

print("TerraSpectra VNIR Validation Started!", flush=True)

ROOT = Path(__file__).resolve().parents[1]
VNIR_FOLDER = ROOT / "data" / "raw" / "VNIR"


def validate_vnir():

    hdr_path = VNIR_FOLDER / "13_G.hdr"

    if not hdr_path.exists():
        print("ERROR: Header file not found.")
        return

    print("Header found:", hdr_path.name, flush=True)

    try:
        image = spectral.open_image(str(hdr_path))

        print("\nIMAGE METADATA")
        print("Shape:", image.shape)
        print("Interleave:", image.metadata.get("interleave"))
        print("Data type:", image.metadata.get("data type"))

        print("\nLoading hyperspectral cube. Please wait...", flush=True)

        cube = np.asarray(image.load())

        print("\nVALIDATION RESULTS")
        print("Shape:", cube.shape)
        print("Height:", cube.shape[0])
        print("Width:", cube.shape[1])
        print("Bands:", cube.shape[2])
        print("Data type:", cube.dtype)

        print("\nPIXEL VALUE ANALYSIS")
        print("Minimum:", np.min(cube))
        print("Maximum:", np.max(cube))
        print("Mean:", np.mean(cube))
        print("NaN values:", np.isnan(cube).sum())

        print("\nVNIR validation completed!")

    except Exception as error:
        print("\nVALIDATION FAILED")
        print(type(error).__name__, ":", error)


if __name__ == "__main__":
    validate_vnir()