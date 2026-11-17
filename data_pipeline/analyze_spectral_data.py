
import pandas as pd
from pathlib import Path

# ==========================================
# 1. PROJECT PATHS
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

VNIR_PATH = RAW_DIR / "VNIR_data.csv"
SWIR_PATH = RAW_DIR / "SWIR_data.csv"
DESCRIPTION_PATH = RAW_DIR / "Data_description.csv"


# ==========================================
# 2. LOAD AND CLEAN VNIR / SWIR
# ==========================================


def load_spectral_dataset(file_path, dataset_name):

    print(f"\nLoading {dataset_name} dataset...")

    if not file_path.exists():
        raise FileNotFoundError(file_path)

    # Read the original CSV
    raw = pd.read_csv(
        file_path,
        sep=";",
        header=None,
        skiprows=1,
        encoding="cp1252"
    )

    # First row contains date information
    date_row = raw.iloc[0]

    # Second row contains G/D labels
    side_row = raw.iloc[1]

    columns = ["Pot", "Genotype", "Treatment"]

    # Create measurement column names
    for i in range(3, len(raw.columns)):

        date = date_row.iloc[i]
        side = str(side_row.iloc[i]).strip()

        if pd.isna(date):
            date = date_row.iloc[i - 1]

        if pd.isna(date):
            date = "unknown"

        if side not in ["G", "D"]:
            side = f"side_{i}"

        columns.append(f"dpi_{date}_{side}")

    # Extract records after both header rows
    data = raw.iloc[2:].copy()

    data.columns = columns

    # Remove explanatory rows and completely empty rows
    data = data[
        data["Pot"].astype(str).str.strip().str.fullmatch(r"\d+")
    ].copy()

    # Clean text columns
    for col in ["Pot", "Genotype", "Treatment"]:
        data[col] = data[col].astype("string").str.strip()

    # Convert measurement columns
    measurement_columns = columns[3:]

    for col in measurement_columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    return data.reset_index(drop=True)



# ==========================================
# 3. LOAD DESCRIPTION DATASET
# ==========================================

def load_description_dataset(file_path):

    print("\nLoading description dataset...")

    if not file_path.exists():
        raise FileNotFoundError(file_path)

    data = pd.read_csv(
        file_path,
        sep=";",
        skiprows=2,
        encoding="cp1252"
    )

    # Keep only the six actual data columns.
    data = data.iloc[:, :6].copy()

    data.columns = [
        "Pot",
        "Side",
        "Leaf",
        "Genotype",
        "Treatment",
        "Confidence_Level"
    ]

    # Remove blank and explanatory rows.
    data = data.dropna(
        subset=[
            "Pot",
            "Side",
            "Leaf",
            "Genotype",
            "Treatment"
        ]
    ).copy()

    # Clean text values.
    for col in ["Side", "Genotype", "Treatment"]:
        data[col] = data[col].astype("string").str.strip()

    # Convert numeric columns.
    data["Pot"] = pd.to_numeric(data["Pot"], errors="coerce")
    data["Leaf"] = pd.to_numeric(data["Leaf"], errors="coerce")
    data["Confidence_Level"] = pd.to_numeric(
        data["Confidence_Level"], errors="coerce"
    )

    data = data.dropna(
        subset=["Pot", "Leaf", "Confidence_Level"]
    )

    data["Pot"] = data["Pot"].astype(int)
    data["Leaf"] = data["Leaf"].astype(int)
    data["Confidence_Level"] = data["Confidence_Level"].astype(int)

    return data.reset_index(drop=True)


# ==========================================
# 4. DATASET ANALYSIS
# ==========================================

def analyze_dataset(data, name):

    print("\n" + "=" * 55)
    print(f"{name} DATASET ANALYSIS")
    print("=" * 55)

    print("\nFirst five rows:")
    print(data.head())

    print("\nDataset shape:")
    print(data.shape)

    print("\nColumn names:")
    print(data.columns.tolist())

    print("\nData types:")
    print(data.dtypes)

    print("\nMissing values:")
    print(data.isnull().sum())

    print("\nTotal missing values:")
    print(data.isnull().sum().sum())

    print("\nDuplicate rows:")
    print(data.duplicated().sum())


# ==========================================
# 5. LABEL DISTRIBUTION
# ==========================================

def analyze_labels(data, name):

    print(f"\n{name} Treatment Distribution:")

    print(data["Treatment"].value_counts())

    print(f"\n{name} Genotype Distribution:")

    print(data["Genotype"].value_counts())


# ==========================================
# 6. SAVE PROCESSED DATASETS
# ==========================================

def save_dataset(data, filename):

    output_path = PROCESSED_DIR / filename

    data.to_csv(output_path, index=False)

    print(f"\nSaved: {output_path}")


# ==========================================
# 7. MAIN FUNCTION
# ==========================================

def main():

    # Load datasets.
    vnir = load_spectral_dataset(VNIR_PATH, "VNIR")
    swir = load_spectral_dataset(SWIR_PATH, "SWIR")

    description = load_description_dataset(
        DESCRIPTION_PATH
    )

    # Analyse datasets.
    analyze_dataset(vnir, "VNIR")
    analyze_dataset(swir, "SWIR")
    analyze_dataset(description, "DESCRIPTION")

    # Analyse labels.
    analyze_labels(vnir, "VNIR")
    analyze_labels(swir, "SWIR")
    analyze_labels(description, "DESCRIPTION")

    # Save cleaned datasets.
    save_dataset(vnir, "VNIR_cleaned.csv")
    save_dataset(swir, "SWIR_cleaned.csv")
    save_dataset(description, "Data_description_cleaned.csv")

    print("\n" + "=" * 55)
    print("TERRASPECTRA DATA PROCESSING COMPLETED")
    print("=" * 55)


if __name__ == "__main__":
    main()



