
import pandas as pd
import numpy as np
from pathlib import Path


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. LOAD AND CLEAN SPECTRAL DATA
# ==========================================

def load_spectral_dataset(file_path, dataset_name):

    print(f"\nLoading {dataset_name} dataset...")

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    raw = pd.read_csv(
        file_path,
        sep=";",
        header=None,
        skiprows=2,
        encoding="cp1252"
    )

    # First header row contains dates
    date_row = raw.iloc[0].ffill()

    # Second header row contains G and D
    side_row = raw.iloc[1]

    columns = [
        "Pot",
        "Genotype",
        "Treatment"
    ]

    # Generate measurement column names
    for i in range(3, len(raw.columns)):

        date_value = pd.to_numeric(
            date_row.iloc[i],
            errors="coerce"
        )

        if pd.notna(date_value):
            date = str(int(date_value))
        else:
            date = "unknown"

        side_value = str(
            side_row.iloc[i]
        ).strip()

        if side_value.startswith("G"):
            side = "G"

        elif side_value.startswith("D"):
            side = "D"

        else:
            side = "Unknown"

        columns.append(
            f"dpi_{date}_{side}"
        )

    # Extract actual records
    data = raw.iloc[2:].copy()

    data.columns = columns

    # Keep valid pot records only
    data = data[
        data["Pot"]
        .astype(str)
        .str.strip()
        .str.fullmatch(r"\d+")
    ].copy()

    # Clean metadata
    data["Pot"] = pd.to_numeric(
        data["Pot"],
        errors="coerce"
    )

    for col in ["Genotype", "Treatment"]:

        data[col] = (
            data[col]
            .astype("string")
            .str.strip()
        )

    # Convert measurement columns to numeric
    for col in columns[3:]:

        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    data = data.reset_index(drop=True)

    print(f"{dataset_name} shape: {data.shape}")

    print("Columns:")
    print(data.columns.tolist())

    return data


# ==========================================
# 3. LOAD AND CLEAN DESCRIPTION DATA
# ==========================================

def load_description_dataset(file_path):

    print("\nLoading description dataset...")

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    data = pd.read_csv(
        file_path,
        sep=";",
        header=2,
        encoding="cp1252"
    )

    # Remove unnecessary columns
    data = data.iloc[:, :6].copy()

    # Standardize column names
    data.columns = [
        "Pot",
        "Side",
        "Leaf",
        "Genotype",
        "Treatment",
        "Confidence"
    ]

    # Remove invalid records
    data = data.dropna(
        subset=[
            "Pot",
            "Side",
            "Leaf",
            "Genotype",
            "Treatment"
        ]
    )

    # Clean text columns
    for col in [
        "Side",
        "Genotype",
        "Treatment"
    ]:

        data[col] = (
            data[col]
            .astype("string")
            .str.strip()
        )

    # Convert numeric columns
    for col in [
        "Pot",
        "Leaf",
        "Confidence"
    ]:

        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    data = data.dropna(
        subset=[
            "Pot",
            "Leaf",
            "Confidence"
        ]
    )

    data = data.reset_index(drop=True)

    print("Description shape:", data.shape)

    print("Description columns:")
    print(data.columns.tolist())

    return data


# ==========================================
# 4. BASIC DATASET ANALYSIS
# ==========================================

def analyze_dataset(data, dataset_name):

    print(f"\n{'=' * 45}")
    print(f"{dataset_name} DATASET ANALYSIS")
    print(f"{'=' * 45}")

    print("\nDataset shape:")
    print(data.shape)

    print("\nDataset information:")
    print(data.info())

    print("\nFirst five records:")
    print(data.head())

    print("\nMissing values:")
    print(data.isnull().sum())

    print("\nDuplicate records:")
    print(data.duplicated().sum())

    print("\nStatistical summary:")
    print(data.describe(include="all"))


# ==========================================
# 5. LABEL ANALYSIS
# ==========================================

def analyze_labels(data):

    print("\n========== LABEL ANALYSIS ==========")

    print("\nTreatment distribution:")
    print(data["Treatment"].value_counts())

    print("\nGenotype distribution:")
    print(data["Genotype"].value_counts())

    print("\nConfidence distribution:")
    print(data["Confidence"].value_counts())

    print("\nTreatment by genotype:")
    print(
        pd.crosstab(
            data["Genotype"],
            data["Treatment"]
        )
    )


# ==========================================
# 6. SAVE PROCESSED DATASETS
# ==========================================

def save_dataset(data, file_name):

    output_path = PROCESSED_DIR / file_name

    data.to_csv(
        output_path,
        index=False
    )

    print(f"\nSaved successfully: {output_path}")


# ==========================================
# 7. MAIN EXECUTION
# ==========================================

def main():

    print("\n====================================")
    print(" TERRASPECTRA DATA ANALYSIS")
    print("====================================")

    # Load VNIR dataset
    vnir_data = load_spectral_dataset(
        RAW_DIR / "VNIR_data.csv",
        "VNIR"
    )

    # Load SWIR dataset
    swir_data = load_spectral_dataset(
        RAW_DIR / "SWIR_data.csv",
        "SWIR"
    )

    # Load description dataset
    description_data = load_description_dataset(
        RAW_DIR / "Data_description.csv"
    )

    # Analyze datasets
    analyze_dataset(
        vnir_data,
        "VNIR"
    )

    analyze_dataset(
        swir_data,
        "SWIR"
    )

    analyze_dataset(
        description_data,
        "DESCRIPTION"
    )

    # Analyze infection labels
    analyze_labels(description_data)

    # Save cleaned datasets
    save_dataset(
        vnir_data,
        "VNIR_cleaned.csv"
    )

    save_dataset(
        swir_data,
        "SWIR_cleaned.csv"
    )

    save_dataset(
        description_data,
        "Data_description_cleaned.csv"
    )

    print("\n====================================")
    print(" DATA CLEANING COMPLETED")
    print("====================================")


if __name__ == "__main__":
    main()
