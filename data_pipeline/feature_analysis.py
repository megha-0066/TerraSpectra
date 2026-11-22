import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import mannwhitneyu

# ==========================================
# TERRASPECTRA FEATURE ANALYSIS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = PROCESSED_DIR / "feature_analysis"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 1. LOAD DATASETS
# ==========================================

def load_datasets():

    print("\n[1/6] Loading VNIR and SWIR datasets...")

    vnir_path = PROCESSED_DIR / "VNIR_cleaned.csv"
    swir_path = PROCESSED_DIR / "SWIR_cleaned.csv"

    vnir = pd.read_csv(vnir_path)
    swir = pd.read_csv(swir_path)

    print("VNIR shape:", vnir.shape)
    print("SWIR shape:", swir.shape)

    return vnir, swir


# ==========================================
# 2. COMBINE DATASETS
# ==========================================

def combine_datasets(vnir, swir):

    print("\n[2/6] Combining VNIR and SWIR datasets...")

    metadata = ["Pot", "Genotype", "Treatment"]

    for column in metadata:
        if column not in vnir.columns or column not in swir.columns:
            raise ValueError(f"Missing metadata column: {column}")

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

    data = pd.merge(
        vnir,
        swir,
        on=metadata,
        how="inner",
        validate="one_to_one"
    )

    print("Combined dataset shape:", data.shape)

    return data


# ==========================================
# 3. IDENTIFY FEATURES
# ==========================================

def identify_features(data):

    print("\n[3/6] Identifying measurement features...")

    metadata = ["Pot", "Genotype", "Treatment"]

    feature_columns = [
        col for col in data.columns
        if col not in metadata
    ]

    for column in feature_columns:
        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        )

    usable_features = [
        col for col in feature_columns
        if data[col].notna().sum() > 0
    ]

    print("Total measurement features:", len(feature_columns))
    print("Usable measurement features:", len(usable_features))

    return usable_features


# ==========================================
# 4. DESCRIPTIVE FEATURE ANALYSIS
# ==========================================

def descriptive_analysis(data, features):

    print("\n[4/6] Performing descriptive analysis...")

    infected = data[data["Treatment"] == "Infected"]
    non_infected = data[data["Treatment"] == "Non-infected"]

    results = []

    for feature in features:

        results.append({
            "Feature": feature,
            "Overall_Mean": data[feature].mean(),
            "Overall_Std": data[feature].std(),
            "Missing_Count": data[feature].isna().sum(),
            "Infected_Mean": infected[feature].mean(),
            "Non_Infected_Mean": non_infected[feature].mean()
        })

    summary = pd.DataFrame(results)

    summary.to_csv(
        OUTPUT_DIR / "feature_summary.csv",
        index=False
    )

    print("Feature summary saved.")

    return summary


# ==========================================
# 5. COMPARE TREATMENT GROUPS
# ==========================================

def compare_groups(data, features):

    print("\n[5/6] Comparing infected and non-infected groups...")

    infected = data[data["Treatment"] == "Infected"]
    non_infected = data[data["Treatment"] == "Non-infected"]

    results = []

    for feature in features:

        infected_values = infected[feature].dropna()
        non_infected_values = non_infected[feature].dropna()

        if len(infected_values) < 2 or len(non_infected_values) < 2:
            continue

        if infected_values.nunique() == 1 and non_infected_values.nunique() == 1:
            if infected_values.iloc[0] == non_infected_values.iloc[0]:
                p_value = 1.0
            else:
                p_value = np.nan
        else:
            try:
                _, p_value = mannwhitneyu(
                    infected_values,
                    non_infected_values,
                    alternative="two-sided"
                )
            except ValueError:
                p_value = np.nan

        infected_mean = infected_values.mean()
        non_infected_mean = non_infected_values.mean()

        pooled_std = np.sqrt(
            (
                infected_values.var() +
                non_infected_values.var()
            ) / 2
        )

        if pd.isna(pooled_std) or pooled_std == 0:
            standardized_difference = np.nan
        else:
            standardized_difference = (
                infected_mean - non_infected_mean
            ) / pooled_std

        results.append({
            "Feature": feature,
            "Infected_Mean": infected_mean,
            "Non_Infected_Mean": non_infected_mean,
            "Mean_Difference": infected_mean - non_infected_mean,
            "Standardized_Difference": standardized_difference,
            "Mann_Whitney_P_Value": p_value
        })

    comparison = pd.DataFrame(results)

    comparison["Absolute_Standardized_Difference"] = (
        comparison["Standardized_Difference"].abs()
    )

    comparison = comparison.sort_values(
        "Absolute_Standardized_Difference",
        ascending=False,
        na_position="last"
    )

    comparison.to_csv(
        OUTPUT_DIR / "treatment_feature_comparison.csv",
        index=False
    )

    print("Treatment comparison saved.")

    return comparison


# ==========================================
# 6. GENERATE VISUALIZATIONS
# ==========================================

def generate_visualizations(data, comparison):

    print("\n[6/6] Generating feature visualizations...")

    # Treatment distribution
    data["Treatment"].value_counts().plot(
        kind="bar",
        rot=0
    )

    plt.title("Treatment Distribution")
    plt.xlabel("Treatment")
    plt.ylabel("Number of Pots")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "treatment_distribution.png",
        dpi=300
    )

    plt.close()

    # Compare top 10 features
    top_features = (
        comparison
        .dropna(subset=["Absolute_Standardized_Difference"])
        .head(10)
    )

    if not top_features.empty:

        plot_data = top_features.set_index("Feature")[
            ["Infected_Mean", "Non_Infected_Mean"]
        ]

        plot_data.plot(kind="bar", figsize=(12, 6))

        plt.title("Top 10 Exploratory Feature Comparisons")
        plt.xlabel("Measurement Feature")
        plt.ylabel("Mean Value")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / "top_feature_comparison.png",
            dpi=300
        )

        plt.close()

    print("\nAll analysis results saved in:")
    print(OUTPUT_DIR)


# ==========================================
# MAIN FUNCTION
# ==========================================

def main():

    print("\n========================================")
    print(" TERRASPECTRA FEATURE ANALYSIS")
    print("========================================")

    vnir, swir = load_datasets()

    data = combine_datasets(vnir, swir)

    features = identify_features(data)

    summary = descriptive_analysis(data, features)

    comparison = compare_groups(data, features)

    generate_visualizations(data, comparison)

    print("\n========================================")
    print(" FEATURE ANALYSIS COMPLETED")
    print("========================================")

    print("\nNumber of pots:", len(data))
    print("Number of features:", len(features))

    print("\nResults saved in:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
    