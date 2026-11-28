
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = BASE_DIR / "data" / "processed"

PREPARED_DIR = PROCESSED_DIR / "prepared"

OUTPUT_DIR = PROCESSED_DIR / "missing_value_results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. LOAD PREPARED DATASET
# ==========================================

def load_dataset():

    print("\n[1/7] Loading prepared dataset...")

    file_path = PREPARED_DIR / "terraspectra_prepared.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    data = pd.read_csv(file_path)

    print("Dataset shape:", data.shape)

    return data


# ==========================================
# 3. BASIC MISSING VALUE ANALYSIS
# ==========================================

def basic_missing_analysis(data):

    print("\n[2/7] Basic missing value analysis...")

    total_missing = data.isnull().sum().sum()

    total_cells = data.size

    missing_percentage = (
        total_missing / total_cells
    ) * 100

    print("Total cells:", total_cells)

    print("Total missing values:", total_missing)

    print(
        f"Missing percentage: {missing_percentage:.2f}%"
    )

    print("\nMissing values by column:")

    missing = data.isnull().sum()

    missing = missing[missing > 0].sort_values(
        ascending=False
    )

    print(missing)

    missing.to_csv(
        OUTPUT_DIR / "missing_values_by_column.csv"
    )

    return missing


# ==========================================
# 4. VNIR AND SWIR MISSING VALUE ANALYSIS
# ==========================================

def spectral_missing_analysis(data):

    print("\n[3/7] Spectral missing value analysis...")

    spectral_columns = [
        col for col in data.columns
        if col.startswith(("VNIR_", "SWIR_"))
    ]

    missing_data = []

    for col in spectral_columns:

        total = len(data)

        missing = data[col].isnull().sum()

        percentage = (missing / total) * 100

        sensor = col.split("_")[0]

        missing_data.append({
            "Sensor": sensor,
            "Column": col,
            "Total_Records": total,
            "Missing_Values": missing,
            "Available_Values": total - missing,
            "Missing_Percentage": percentage
        })

    result = pd.DataFrame(missing_data)

    result.to_csv(
        OUTPUT_DIR / "spectral_missing_analysis.csv",
        index=False
    )

    print("\nMissing values by sensor:")

    print(
        result.groupby("Sensor")[
            ["Missing_Values", "Available_Values"]
        ].sum()
    )

    return result


# ==========================================
# 5. DATE-WISE MISSING VALUE ANALYSIS
# ==========================================

def date_wise_analysis(result):

    print("\n[4/7] Date-wise missing value analysis...")

    result = result.copy()

    result["Days_After_Inoculation"] = result[
        "Column"
    ].str.extract(r"dpi_(-?\d+)")[0]

    result["Days_After_Inoculation"] = pd.to_numeric(
        result["Days_After_Inoculation"],
        errors="coerce"
    )

    result["Side"] = result["Column"].str.extract(
        r"_(G|D)$"
    )[0]

    summary = result.groupby(
        ["Sensor", "Days_After_Inoculation"],
        dropna=False
    )["Missing_Values"].sum().reset_index()

    summary.to_csv(
        OUTPUT_DIR / "date_wise_missing_analysis.csv",
        index=False
    )

    print(summary.to_string(index=False))

    return result


# ==========================================
# 6. MISSING VALUE VISUALIZATION
# ==========================================

def create_visualizations(data, result):

    print("\n[5/7] Creating visualizations...")

    # Missing values per column
    missing = data.isnull().sum()

    missing = missing[missing > 0].sort_values(
        ascending=False
    )

    if not missing.empty:

        plt.figure(figsize=(15, 7))

        missing.plot(kind="bar")

        plt.title("Missing Values by Column")

        plt.xlabel("Columns")

        plt.ylabel("Missing Records")

        plt.xticks(rotation=90)

        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / "missing_values_by_column.png",
            dpi=300
        )

        plt.close()

    # Missing percentage by sensor and date
    result = result.copy()

    result["Date"] = result["Column"].str.extract(
        r"dpi_(-?\d+)"
    )[0]

    result["Sensor_Date"] = (
        result["Sensor"] + "_DPI_" + result["Date"]
    )

    summary = result.groupby(
        "Sensor_Date"
    )["Missing_Percentage"].mean()

    plt.figure(figsize=(14, 6))

    summary.plot(kind="bar")

    plt.title("Missing Value Percentage by Sensor and Date")

    plt.xlabel("Sensor and Date")

    plt.ylabel("Missing Percentage")

    plt.xticks(rotation=90)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "missing_percentage_by_date.png",
        dpi=300
    )

    plt.close()

    # Missing data matrix
    spectral_columns = [
        col for col in data.columns
        if col.startswith(("VNIR_", "SWIR_"))
    ]

    missing_matrix = data[spectral_columns].isnull()

    plt.figure(figsize=(18, 8))

    plt.imshow(
        missing_matrix.T,
        aspect="auto",
        interpolation="nearest"
    )

    plt.title("Spectral Missing Data Matrix")

    plt.xlabel("Leaf Records")

    plt.ylabel("Spectral Measurement Columns")

    plt.yticks(
        range(len(spectral_columns)),
        spectral_columns,
        fontsize=7
    )

    plt.colorbar(
        label="Missing (1) / Available (0)"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "missing_data_matrix.png",
        dpi=300
    )

    plt.close()

    print("Visualizations saved successfully.")


# ==========================================
# 7. CHECK COMPLETELY EMPTY COLUMNS
# ==========================================

def check_empty_columns(data):

    print("\n[6/7] Checking completely empty columns...")

    spectral_columns = [
        col for col in data.columns
        if col.startswith(("VNIR_", "SWIR_"))
    ]

    empty_columns = [
        col for col in spectral_columns
        if data[col].isnull().all()
    ]

    print("\nCompletely empty columns:")

    if empty_columns:

        for col in empty_columns:
            print(col)

    else:

        print("No completely empty spectral columns.")

    pd.Series(
        empty_columns,
        name="Empty_Columns"
    ).to_csv(
        OUTPUT_DIR / "completely_empty_columns.csv",
        index=False
    )

    return empty_columns


# ==========================================
# 8. FINAL REPORT
# ==========================================

def final_report(data, empty_columns):

    print("\n[7/7] Generating final report...")

    total_missing = int(data.isnull().sum().sum())

    report = {
        "Total_Rows": len(data),
        "Total_Columns": len(data.columns),
        "Total_Cells": data.size,
        "Total_Missing_Values": total_missing,
        "Missing_Percentage": (
            total_missing / data.size
        ) * 100,
        "Completely_Empty_Columns": len(empty_columns),
        "Duplicate_Records": int(data.duplicated().sum())
    }

    report_df = pd.DataFrame(
        [report]
    )

    report_df.to_csv(
        OUTPUT_DIR / "missing_value_summary.csv",
        index=False
    )

    print("\nFINAL REPORT")

    for key, value in report.items():

        print(f"{key}: {value}")

    print("\nResults saved in:")

    print(OUTPUT_DIR)


# ==========================================
# 9. MAIN EXECUTION
# ==========================================

def main():

    print("\n========================================")

    print(" TERRASPECTRA MISSING VALUE ANALYSIS")

    print("========================================")

    data = load_dataset()

    missing = basic_missing_analysis(data)

    result = spectral_missing_analysis(data)

    result = date_wise_analysis(result)

    create_visualizations(data, result)

    empty_columns = check_empty_columns(data)

    final_report(data, empty_columns)

    print("\nMISSING VALUE ANALYSIS COMPLETED!")


if __name__ == "__main__":
    main()
