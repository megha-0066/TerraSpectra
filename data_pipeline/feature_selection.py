import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================
# TERRASPECTRA FEATURE SELECTION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = BASE_DIR / "data" / "processed" / "feature_analysis"
DATA_DIR = BASE_DIR / "data" / "processed"

OUTPUT_DIR = DATA_DIR / "feature_selection"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 1. LOAD DATA
# ==========================================

def load_data():

    print("\n[1/5] Loading feature analysis results...")

    file_path = INPUT_DIR / "feature_summary.csv"

    data = pd.read_csv(file_path)

    print("Feature summary shape:", data.shape)

    return data


# ==========================================
# 2. REMOVE INVALID FEATURES
# ==========================================

def remove_invalid_features(data):

    print("\n[2/5] Removing invalid features...")

    data = data.copy()

    data = data.dropna(subset=["Overall_Mean"])

    data = data[data["Overall_Std"].fillna(0) > 0]

    print("Features after removing invalid variables:", len(data))

    return data


# ==========================================
# 3. CORRELATION ANALYSIS
# ==========================================

def correlation_analysis():

    print("\n[3/5] Performing correlation analysis...")

    vnir_path = DATA_DIR / "VNIR_cleaned.csv"
    swir_path = DATA_DIR / "SWIR_cleaned.csv"

    vnir = pd.read_csv(vnir_path)
    swir = pd.read_csv(swir_path)

    metadata = ["Pot", "Genotype", "Treatment"]

    vnir_features = [
        col for col in vnir.columns if col not in metadata
    ]

    swir_features = [
        col for col in swir.columns if col not in metadata
    ]

    vnir = vnir.rename(
        columns={col: f"VNIR_{col}" for col in vnir_features}
    )

    swir = swir.rename(
        columns={col: f"SWIR_{col}" for col in swir_features}
    )

    combined = pd.merge(
        vnir,
        swir,
        on=metadata,
        validate="one_to_one"
    )

    features = combined.drop(columns=metadata)

    features = features.apply(
        pd.to_numeric,
        errors="coerce"
    )

    correlation = features.corr(min_periods=3)

    correlation.to_csv(
        OUTPUT_DIR / "feature_correlation_matrix.csv"
    )

    return correlation


# ==========================================
# 4. IDENTIFY HIGHLY CORRELATED FEATURES
# ==========================================

def identify_correlated_features(correlation):

    print("\n[4/5] Identifying highly correlated features...")

    upper_triangle = correlation.where(
        np.triu(
            np.ones(correlation.shape),
            k=1
        ).astype(bool)
    )

    correlated_pairs = []

    for column in upper_triangle.columns:

        for index in upper_triangle.index:

            value = upper_triangle.loc[index, column]

            if pd.notna(value) and abs(value) >= 0.95:

                correlated_pairs.append({
                    "Feature_1": index,
                    "Feature_2": column,
                    "Correlation": value
                })

    result = pd.DataFrame(correlated_pairs)

    result.to_csv(
        OUTPUT_DIR / "highly_correlated_features.csv",
        index=False
    )

    print("Highly correlated pairs:", len(result))

    return result


# ==========================================
# 5. SAVE FEATURE SUMMARY
# ==========================================

def save_feature_summary(data, correlation, correlated_pairs):

    print("\n[5/5] Saving feature selection results...")

    summary = pd.DataFrame([{
        "Total_Initial_Features": len(
            [c for c in correlation.columns]
        ),
        "Features_With_Variance": len(data),
        "Highly_Correlated_Pairs": len(correlated_pairs),
        "Correlation_Threshold": 0.95
    }])

    summary.to_csv(
        OUTPUT_DIR / "feature_selection_summary.csv",
        index=False
    )

    print("\nFeature selection summary:")
    print(summary.to_string(index=False))

    print("\nResults saved in:")
    print(OUTPUT_DIR)


# ==========================================
# MAIN
# ==========================================

def main():

    print("\n========================================")
    print(" TERRASPECTRA FEATURE SELECTION")
    print("========================================")

    data = load_data()

    data = remove_invalid_features(data)

    correlation = correlation_analysis()

    correlated_pairs = identify_correlated_features(
        correlation
    )

    save_feature_summary(
        data,
        correlation,
        correlated_pairs
    )

    print("\nFEATURE SELECTION ANALYSIS COMPLETED!")


if __name__ == "__main__":
    main()