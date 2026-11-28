
import pandas as pd
from pathlib import Path


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = BASE_DIR / "data" / "processed"

INPUT_DIR = PROCESSED_DIR / "prepared"

OUTPUT_DIR = PROCESSED_DIR / "preprocessed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. LOAD PREPARED DATASET
# ==========================================

def load_dataset():

    print("\n[1/7] Loading prepared dataset...")

    file_path = INPUT_DIR / "terraspectra_prepared.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    data = pd.read_csv(file_path)

    print("Original dataset shape:", data.shape)

    return data


# ==========================================
# 3. IDENTIFY FEATURE COLUMNS
# ==========================================

def identify_columns(data):

    print("\n[2/7] Identifying dataset columns...")

    metadata_columns = [
        "Pot",
        "Side",
        "Leaf",
        "Genotype",
        "Treatment",
        "Confidence"
    ]

    spectral_columns = [
        col for col in data.columns
        if col.startswith(("VNIR_", "SWIR_"))
    ]

    missing_metadata = [
        col for col in metadata_columns
        if col not in data.columns
    ]

    if missing_metadata:
        raise ValueError(
            f"Missing metadata columns: {missing_metadata}"
        )

    if not spectral_columns:
        raise ValueError(
            "No VNIR or SWIR measurement columns found."
        )

    print("Metadata columns:", len(metadata_columns))
    print("Spectral measurement columns:", len(spectral_columns))

    return metadata_columns, spectral_columns


# ==========================================
# 4. CHECK COMPLETELY EMPTY FEATURES
# ==========================================

def identify_empty_features(data, spectral_columns):

    print("\n[3/7] Identifying completely empty features...")

    empty_columns = [
        col for col in spectral_columns
        if data[col].isna().all()
    ]

    usable_columns = [
        col for col in spectral_columns
        if col not in empty_columns
    ]

    print("\nCompletely empty columns:")

    if empty_columns:
        for col in empty_columns:
            print(col)
    else:
        print("None")

    print("\nUsable spectral columns:", len(usable_columns))

    pd.DataFrame({
        "Completely_Empty_Columns": empty_columns
    }).to_csv(
        OUTPUT_DIR / "removed_empty_features.csv",
        index=False
    )

    return empty_columns, usable_columns


# ==========================================
# 5. REMOVE COMPLETELY EMPTY FEATURES
# ==========================================

def remove_empty_features(
    data,
    metadata_columns,
    usable_columns
):

    print("\n[4/7] Removing completely empty features...")

    selected_columns = metadata_columns + usable_columns

    processed_data = data[selected_columns].copy()

    print("Processed dataset shape:", processed_data.shape)

    return processed_data


# ==========================================
# 6. VALIDATE PROCESSED DATASET
# ==========================================

def validate_dataset(data):

    print("\n[5/7] Validating processed dataset...")

    spectral_columns = [
        col for col in data.columns
        if col.startswith(("VNIR_", "SWIR_"))
    ]

    # Check that spectral features are numeric
    for col in spectral_columns:

        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    # Check for completely empty columns
    empty_columns = [
        col for col in spectral_columns
        if data[col].isna().all()
    ]

    # Check duplicate records
    duplicate_count = data.duplicated().sum()

    # Check missing values
    missing_count = data.isna().sum().sum()

    print("Remaining empty spectral columns:", len(empty_columns))
    print("Duplicate records:", duplicate_count)
    print("Remaining missing values:", missing_count)

    if empty_columns:
        raise ValueError(
            f"Empty spectral columns remain: {empty_columns}"
        )

    if duplicate_count > 0:
        print("Warning: Duplicate records detected.")

    print("Dataset validation completed.")

    return data


# ==========================================
# 7. SAVE PREPROCESSED DATASET AND REPORT
# ==========================================

def save_results(data, original_data, removed_columns):

    print("\n[6/7] Saving preprocessed dataset...")

    output_path = (
        OUTPUT_DIR / "terraspectra_preprocessed.csv"
    )

    data.to_csv(
        output_path,
        index=False
    )

    print("Saved dataset:", output_path)

    # Generate missing-value report
    missing_report = pd.DataFrame({
        "Column": data.columns,
        "Missing_Values": data.isna().sum().values,
        "Missing_Percentage": (
            data.isna().mean().values * 100
        )
    })

    missing_report.to_csv(
        OUTPUT_DIR / "preprocessed_missing_report.csv",
        index=False
    )

    # Generate preprocessing summary
    summary = {
        "Original_Rows": len(original_data),
        "Original_Columns": len(original_data.columns),
        "Processed_Rows": len(data),
        "Processed_Columns": len(data.columns),
        "Removed_Empty_Columns": len(removed_columns),
        "Remaining_Missing_Values": int(
            data.isna().sum().sum()
        ),
        "Duplicate_Records": int(
            data.duplicated().sum()
        )
    }

    pd.DataFrame([summary]).to_csv(
        OUTPUT_DIR / "preprocessing_summary.csv",
        index=False
    )

    return summary


# ==========================================
# 8. DISPLAY FINAL SUMMARY
# ==========================================

def display_summary(data, summary):

    print("\n[7/7] FINAL PREPROCESSING SUMMARY")

    print("\nOriginal rows:", summary["Original_Rows"])

    print("Original columns:", summary["Original_Columns"])

    print("Processed rows:", summary["Processed_Rows"])

    print("Processed columns:", summary["Processed_Columns"])

    print(
        "Removed empty columns:",
        summary["Removed_Empty_Columns"]
    )

    print(
        "Remaining missing values:",
        summary["Remaining_Missing_Values"]
    )

    print(
        "Duplicate records:",
        summary["Duplicate_Records"]
    )

    print("\nTreatment distribution:")

    print(data["Treatment"].value_counts())

    print("\nGenotype distribution:")

    print(data["Genotype"].value_counts())

    print("\nResults saved in:")

    print(OUTPUT_DIR)

    print("\nPREPROCESSING COMPLETED SUCCESSFULLY!")


# ==========================================
# 9. MAIN EXECUTION
# ==========================================

def main():

    print("\n========================================")
    print(" TERRASPECTRA DATA PREPROCESSING")
    print("========================================")

    original_data = load_dataset()

    metadata_columns, spectral_columns = identify_columns(
        original_data
    )

    empty_columns, usable_columns = identify_empty_features(
        original_data,
        spectral_columns
    )

    processed_data = remove_empty_features(
        original_data,
        metadata_columns,
        usable_columns
    )

    processed_data = validate_dataset(
        processed_data
    )

    summary = save_results(
        processed_data,
        original_data,
        empty_columns
    )

    display_summary(
        processed_data,
        summary
    )


if __name__ == "__main__":
    main()
