
# ============================================================
# TERRASPECTRA: HYPERSPECTRAL CROP DISEASE FORECASTING
# Exploratory Data Analysis (EDA)
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re

# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed"

OUTPUT_DIR = DATA_DIR / "eda_results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VNIR_PATH = DATA_DIR / "VNIR_cleaned.csv"
SWIR_PATH = DATA_DIR / "SWIR_cleaned.csv"
DESCRIPTION_PATH = DATA_DIR / "Data_description_cleaned.csv"

sns.set_theme(style="whitegrid")


# ============================================================
# 2. LOAD DATASETS
# ============================================================

def load_datasets():

    print("\n[1/7] Loading datasets...")

    for path in [VNIR_PATH, SWIR_PATH, DESCRIPTION_PATH]:

        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {path}"
            )

    vnir = pd.read_csv(VNIR_PATH)
    swir = pd.read_csv(SWIR_PATH)
    description = pd.read_csv(DESCRIPTION_PATH)

    print("VNIR loaded:", vnir.shape)
    print("SWIR loaded:", swir.shape)
    print("Description loaded:", description.shape)

    return vnir, swir, description


# ============================================================
# 3. BASIC DATASET INFORMATION
# ============================================================

def basic_information(vnir, swir, description):

    print("\n[2/7] Basic dataset information...")

    datasets = {
        "VNIR": vnir,
        "SWIR": swir,
        "DESCRIPTION": description
    }

    for name, data in datasets.items():

        print("\n" + "=" * 50)
        print(name, "DATASET")
        print("=" * 50)

        print("\nShape:", data.shape)

        print("\nFirst five records:")
        print(data.head())

        print("\nData types:")
        print(data.dtypes)

        print("\nMissing values:")
        print(data.isnull().sum().sum())

        print("\nDuplicate records:")
        print(data.duplicated().sum())


# ============================================================
# 4. TREATMENT DISTRIBUTION
# ============================================================

def treatment_distribution(description):

    print("\n[3/7] Treatment distribution...")

    counts = description["Treatment"].value_counts()

    print(counts)

    plt.figure(figsize=(8, 5))

    sns.barplot(
        x=counts.index,
        y=counts.values
    )

    plt.title("Infected vs Non-infected Leaf Samples")
    plt.xlabel("Treatment")
    plt.ylabel("Number of Leaf Samples")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "treatment_distribution.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 5. GENOTYPE DISTRIBUTION
# ============================================================

def genotype_distribution(description):

    print("\n[4/7] Genotype distribution...")

    counts = description["Genotype"].value_counts()

    print(counts)

    plt.figure(figsize=(8, 5))

    sns.barplot(
        x=counts.index,
        y=counts.values
    )

    plt.title("Wheat Genotype Distribution")
    plt.xlabel("Genotype")
    plt.ylabel("Number of Leaf Samples")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "genotype_distribution.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 6. INFECTION DISTRIBUTION BY GENOTYPE
# ============================================================

def infection_by_genotype(description):

    print("\n[5/7] Infection distribution by genotype...")

    table = pd.crosstab(
        description["Genotype"],
        description["Treatment"]
    )

    print(table)

    table.to_csv(
        OUTPUT_DIR / "infection_by_genotype.csv"
    )

    table.plot(
        kind="bar",
        figsize=(9, 5)
    )

    plt.title("Infection Distribution by Genotype")
    plt.xlabel("Genotype")
    plt.ylabel("Number of Leaf Samples")

    plt.xticks(rotation=0)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "infection_by_genotype.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 7. CONFIDENCE LEVEL ANALYSIS
# ============================================================

def confidence_analysis(description):

    print("\n[6/7] Confidence level analysis...")

    counts = description[
        "Confidence_Level"
    ].value_counts().sort_index()

    print(counts)

    plt.figure(figsize=(8, 5))

    sns.barplot(
        x=counts.index.astype(str),
        y=counts.values
    )

    plt.title("Infection Confidence Level")
    plt.xlabel("Confidence Level")
    plt.ylabel("Number of Leaf Samples")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "confidence_distribution.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 8. MEASUREMENT AVAILABILITY
# ============================================================

def measurement_availability(data, dataset_name):

    print(f"\nAnalyzing {dataset_name} measurement availability...")

    pattern = re.compile(r"^dpi_(-?\d+)_([GD])$")

    records = []

    for column in data.columns:

        match = pattern.match(column)

        if match:

            date = int(match.group(1))
            side = match.group(2)

            records.append({

                "Date": date,

                "Side": side,

                "Available": data[column].notna().sum(),

                "Missing": data[column].isna().sum()

            })

    if not records:

        print(
            "No valid date columns found. "
            "Check the cleaned dataset column names."
        )

        return

    result = pd.DataFrame(records)

    summary = result.groupby("Date")[
        ["Available", "Missing"]
    ].sum()

    print("\nMeasurement availability:")
    print(summary)

    summary.to_csv(
        OUTPUT_DIR /
        f"{dataset_name.lower()}_availability.csv"
    )

    summary.plot(
        kind="bar",
        figsize=(12, 6)
    )

    plt.title(
        f"{dataset_name} Measurement Availability by Date"
    )

    plt.xlabel("Days Post Inoculation (DPI)")
    plt.ylabel("Number of Measurements")

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        f"{dataset_name.lower()}_availability.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 9. MISSING VALUE ANALYSIS
# ============================================================

def missing_value_analysis(data, dataset_name):

    print(f"\nAnalyzing {dataset_name} missing values...")

    measurement_columns = [
        column for column in data.columns
        if column not in ["Pot", "Genotype", "Treatment"]
    ]

    missing = data[measurement_columns].isnull().sum()

    missing = missing[missing > 0].sort_values(
        ascending=False
    )

    print("\nMissing values:")
    print(missing)

    if missing.empty:

        print("No missing values found.")

        return

    plt.figure(figsize=(13, 6))

    missing.plot(kind="bar")

    plt.title(
        f"{dataset_name} Missing Value Distribution"
    )

    plt.xlabel("Measurement Columns")
    plt.ylabel("Missing Values")

    plt.xticks(rotation=90)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        f"{dataset_name.lower()}_missing_values.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 10. GENOTYPE AND TREATMENT SUMMARY
# ============================================================

def genotype_treatment_summary(description):

    print("\nGenerating genotype-treatment summary...")

    summary = pd.crosstab(
        [description["Genotype"]],
        description["Treatment"],
        margins=True
    )

    print(summary)

    summary.to_csv(
        OUTPUT_DIR / "genotype_treatment_summary.csv"
    )


# ============================================================
# 11. SAVE EDA SUMMARY
# ============================================================

def save_summary(vnir, swir, description):

    print("\nSaving EDA summary...")

    summary = pd.DataFrame({

        "Dataset": [
            "VNIR",
            "SWIR",
            "Description"
        ],

        "Rows": [
            len(vnir),
            len(swir),
            len(description)
        ],

        "Columns": [
            len(vnir.columns),
            len(swir.columns),
            len(description.columns)
        ],

        "Missing_Values": [
            vnir.isnull().sum().sum(),
            swir.isnull().sum().sum(),
            description.isnull().sum().sum()
        ],

        "Duplicate_Rows": [
            vnir.duplicated().sum(),
            swir.duplicated().sum(),
            description.duplicated().sum()
        ]

    })

    print(summary)

    summary.to_csv(
        OUTPUT_DIR / "eda_summary.csv",
        index=False
    )


# ============================================================
# 12. MAIN FUNCTION
# ============================================================

def main():

    print("\n")
    print("=" * 60)
    print("TERRASPECTRA EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    # Load datasets
    vnir, swir, description = load_datasets()

    # Basic information
    basic_information(
        vnir,
        swir,
        description
    )

    # Treatment distribution
    treatment_distribution(description)

    # Genotype distribution
    genotype_distribution(description)

    # Infection by genotype
    infection_by_genotype(description)

    # Confidence analysis
    confidence_analysis(description)

    # Measurement availability
    measurement_availability(vnir, "VNIR")

    measurement_availability(swir, "SWIR")

    # Missing value analysis
    missing_value_analysis(vnir, "VNIR")

    missing_value_analysis(swir, "SWIR")

    # Genotype and treatment summary
    genotype_treatment_summary(description)

    # Save summary
    save_summary(vnir, swir, description)

    print("\n")
    print("=" * 60)
    print("EDA COMPLETED SUCCESSFULLY!")
    print("=" * 60)

    print("\nAll results saved in:")
    print(OUTPUT_DIR)


# ============================================================
# 13. EXECUTION
# ============================================================

if __name__ == "__main__":

    print("TerraSpectra EDA script started!")

    main()
