from pathlib import Path
import sys

try:
    import spectral
except ImportError:
    print("Install the library using: pip install spectral")
    sys.exit(1)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VNIR_FOLDER = PROJECT_ROOT / "data" / "raw" / "VNIR"


def inspect_hyperspectral_data():

    print("\nTERRASPECTRA HYPERSPECTRAL DATA INSPECTION")
    print("=" * 55)

    hdr_files = list(VNIR_FOLDER.rglob("*.hdr"))

    if not hdr_files:
        print("No .hdr files found.")
        print("Download and extract a VNIR hyperspectral archive first.")
        return

    print(f"Header files found: {len(hdr_files)}")

    for hdr_file in hdr_files:
        print("\nInspecting:", hdr_file.name)

        try:
            image = spectral.open_image(str(hdr_file))

            rows, columns, bands = image.shape

            print("Image successfully opened!")
            print("Rows:", rows)
            print("Columns:", columns)
            print("Spectral bands:", bands)

            wavelengths = image.metadata.get("wavelength")

            if wavelengths:
                print("Wavelength count:", len(wavelengths))
                print("First wavelengths:", wavelengths[:5])
                print("Last wavelengths:", wavelengths[-5:])
            else:
                print("Wavelength information not found in header.")

            print("Data type:", image.metadata.get("data type"))
            print("Interleave:", image.metadata.get("interleave"))

            print("\nFirst hyperspectral image inspected successfully.")
            break

        except Exception as error:
            print("Could not open this image:", error)


if __name__ == "__main__":
    inspect_hyperspectral_data()