
import pandas as pd
from pathlib import Path


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = BASE_DIR / "data" / "processed"

OUTPUT_DIR = PROCESSED_DIR / "prepared"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. LOAD CLEANED DATASETS
# ==========================================

def load_datasets():

    print("\n[1/6] Loading cleaned datasets...")

    vnir_path = PROCESSED_DIR / "VNIR_cleaned.csv"
    swir_path = PROCESSED_DIR / "SWIR_cleaned.csv"
    description_path = PROCESSED_DIR / "Data_description_cleaned.csv"

    for path in [vnir_path, swir_path, description_path]:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

    vnir = pd.read_csv(vnir_path)
    swir = pd.read_csv(swir_path)
    description = pd.read_csv(description_path)

    print("VNIR shape:", vnir.shape)
    print("SWIR shape:", swir.shape)
    print("Description shape:", description.shape)

    return vnir, swir, description


# ==========================================
# 3. VALIDATE DATASETS
# ==========================================

def validate_datasets(vnir, swir, description):

    print("\n[2/6] Validating datasets...")

    required_columns = [
        "Pot", "Genotype", "Treatment"
    ]

    for name, dataset in [
        ("VNIR", vnir),
        ("SWIR", swir)
    ]:

        missing = set(required_columns) - set(dataset.columns)

        if missing:
            raise ValueError(
                f"{name} missing columns: {missing}"
            )

    description_columns = [
        "Pot", "Side", "Leaf",
        "Genotype", "Treatment", "Confidence"
    ]

    missing = set(description_columns) - set(description.columns)

    if missing:
        raise ValueError(
            f"Description missing columns: {missing}"
        )

    if vnir["Pot"].duplicated().any():
        raise ValueError("Duplicate pot numbers in VNIR")

    if swir["Pot"].duplicated().any():
        raise ValueError("Duplicate pot numbers in SWIR")

    print("Dataset validation successful.")


# ==========================================
# 4. COMBINE VNIR AND SWIR DATA
# ==========================================

def combine_spectral_data(vnir, swir):

    print("\n[3/6] Combining VNIR and SWIR data...")

    vnir = vnir.rename(
        columns={
            col: f"VNIR_{col}"
            for col in vnir.columns
            if col not in ["Pot", "Genotype", "Treatment"]
        }
    )

    swir = swir.rename(
        columns={
            col: f"SWIR_{col}"
            for col in swir.columns
            if col not in ["Pot", "Genotype", "Treatment"]
        }
    )

    combined = pd.merge(
        vnir,
        swir,
        on=["Pot", "Genotype", "Treatment"],
        how="inner",
        validate="one_to_one"
    )

    print("Combined spectral shape:", combined.shape)

    return combined


# ==========================================
# 5. MATCH LEAF METADATA
# ==========================================

def match_leaf_metadata(spectral_data, description):

    print("\n[4/6] Matching leaf-level metadata...")

    merged = pd.merge(
        description,
        spectral_data,
        on=["Pot", "Genotype", "Treatment"],
        how="left",
        validate="many_to_one",
        indicator=True
    )

    unmatched = merged["_merge"].ne("both").sum()

    print("Total leaf records:", len(merged))
    print("Unmatched leaf records:", unmatched)

    merged = merged.drop(columns=["_merge"])

    return merged


# ==========================================
# 6. SAVE PREPARED DATASET
# ==========================================

def save_prepared_dataset(data):

    print("\n[5/6] Saving prepared dataset...")

    output_path = OUTPUT_DIR / "terraspectra_prepared.csv"

    data.to_csv(
        output_path,
        index=False
    )

    print("Saved:", output_path)


# ==========================================
# 7. DATASET SUMMARY
# ==========================================

def dataset_summary(data):

    print("\n[6/6] Prepared dataset summary")

    print("\nShape:", data.shape)

    print("\nTreatment distribution:")
    print(data["Treatment"].value_counts())

    print("\nGenotype distribution:")
    print(data["Genotype"].value_counts())

    print("\nMissing values:", data.isnull().sum().sum())

    print("\nDuplicate records:", data.duplicated().sum())

    print("\nDataset preparation completed successfully!")


# ==========================================
# 8. MAIN FUNCTION
# ==========================================

def main():

    print("\n======================================")
    print(" TERRASPECTRA DATASET PREPARATION")
    print("======================================")

    vnir, swir, description = load_datasets()

    validate_datasets(vnir, swir, description)

    spectral_data = combine_spectral_data(vnir, swir)

    prepared_data = match_leaf_metadata(
        spectral_data,
        description
    )

    save_prepared_dataset(prepared_data)

    dataset_summary(prepared_data)


if __name__ == "__main__":
    main()
