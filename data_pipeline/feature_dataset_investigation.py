import os
import numpy as np
import pandas as pd

from scipy.stats import mannwhitneyu


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "integrated",
    "vnir_swir_integrated.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "feature_investigation"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

print("\nLoading integrated dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)

required_columns = ["Pot", "Genotype", "Treatment"]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Missing required column: {column}")

df = df.dropna(subset=["Treatment"]).copy()

df["Treatment"] = (
    df["Treatment"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({
        "non-infected": 0,
        "infected": 1
    })
)

df = df.dropna(subset=["Treatment"]).copy()

df["Treatment"] = df["Treatment"].astype(int)

print("\nTreatment distribution:")
print(df["Treatment"].value_counts().sort_index())


# --------------------------------------------------
# 3. IDENTIFY SENSOR FEATURES
# --------------------------------------------------

metadata_columns = [
    "Pot",
    "Genotype",
    "Treatment"
]

feature_columns = [
    column for column in df.columns
    if column.startswith(("VNIR_", "SWIR_"))
]

if not feature_columns:
    raise ValueError("No VNIR or SWIR features found.")

X = df[feature_columns].apply(
    pd.to_numeric,
    errors="coerce"
)

y = df["Treatment"]

print("\nTotal measurement features:", len(feature_columns))
print("VNIR features:", sum(c.startswith("VNIR_") for c in feature_columns))
print("SWIR features:", sum(c.startswith("SWIR_") for c in feature_columns))


# --------------------------------------------------
# 4. DATASET QUALITY SUMMARY
# --------------------------------------------------

print("\nDATASET QUALITY")
print("=" * 50)

quality_summary = pd.DataFrame({
    "Parameter": [
        "Total samples",
        "Total measurement features",
        "Total missing values",
        "Missing percentage",
        "Duplicate pot records",
        "Infected samples",
        "Non-infected samples"
    ],

    "Value": [
        len(df),
        X.shape[1],
        X.isnull().sum().sum(),
        round(X.isnull().sum().sum() / X.size * 100, 2),
        df["Pot"].duplicated().sum(),
        int((y == 1).sum()),
        int((y == 0).sum())
    ]
})

print(quality_summary.to_string(index=False))

quality_summary.to_csv(
    os.path.join(OUTPUT_DIR, "dataset_quality_summary.csv"),
    index=False
)


# --------------------------------------------------
# 5. FEATURE MISSING VALUE ANALYSIS
# --------------------------------------------------

print("\nFEATURE MISSING VALUE ANALYSIS")
print("=" * 50)

missing_summary = pd.DataFrame({
    "Feature": X.columns,
    "Sensor": [
        "VNIR" if c.startswith("VNIR_") else "SWIR"
        for c in X.columns
    ],
    "Missing_Count": X.isnull().sum().values,
    "Missing_Percentage": (
        X.isnull().mean().values * 100
    )
})

missing_summary = missing_summary.sort_values(
    by="Missing_Percentage",
    ascending=False
)

missing_summary.to_csv(
    os.path.join(OUTPUT_DIR, "feature_missing_summary.csv"),
    index=False
)

print(missing_summary.to_string(index=False))


# --------------------------------------------------
# 6. MISSING VALUES BY TREATMENT
# --------------------------------------------------

print("\nMISSING VALUES BY TREATMENT")
print("=" * 50)

missing_by_treatment = []

for feature in X.columns:

    infected_missing = X.loc[y == 1, feature].isnull().mean() * 100

    noninfected_missing = X.loc[y == 0, feature].isnull().mean() * 100

    missing_by_treatment.append({
        "Feature": feature,
        "Infected_Missing_Percentage": infected_missing,
        "Noninfected_Missing_Percentage": noninfected_missing,
        "Difference_Percentage_Points": (
            infected_missing - noninfected_missing
        )
    })

missing_treatment_df = pd.DataFrame(missing_by_treatment)

missing_treatment_df.to_csv(
    os.path.join(OUTPUT_DIR, "missing_values_by_treatment.csv"),
    index=False
)

print(missing_treatment_df.to_string(index=False))


# --------------------------------------------------
# 7. FEATURE VARIANCE ANALYSIS
# --------------------------------------------------

print("\nFEATURE VARIANCE ANALYSIS")
print("=" * 50)

variance_summary = pd.DataFrame({
    "Feature": X.columns,
    "Variance": X.var(),
    "Unique_Values": X.nunique(),
    "Observed_Values": X.notna().sum()
})

variance_summary["Status"] = np.where(
    variance_summary["Observed_Values"] == 0,
    "All Missing",
    np.where(
        variance_summary["Variance"].fillna(0) == 0,
        "Zero Variance",
        "Has Variation"
    )
)

variance_summary = variance_summary.sort_values(
    by="Variance",
    ascending=True,
    na_position="first"
)

variance_summary.to_csv(
    os.path.join(OUTPUT_DIR, "feature_variance_summary.csv"),
    index=False
)

print(variance_summary.to_string(index=False))


# --------------------------------------------------
# 8. INFECTED VS NON-INFECTED FEATURE ANALYSIS
# --------------------------------------------------

print("\nTREATMENT FEATURE COMPARISON")
print("=" * 50)

comparison_results = []

for feature in X.columns:

    infected = X.loc[y == 1, feature].dropna()

    noninfected = X.loc[y == 0, feature].dropna()

    if len(infected) < 2 or len(noninfected) < 2:
        continue

    infected_mean = infected.mean()
    noninfected_mean = noninfected.mean()

    infected_median = infected.median()
    noninfected_median = noninfected.median()

    try:
        statistic, p_value = mannwhitneyu(
            infected,
            noninfected,
            alternative="two-sided"
        )
    except ValueError:
        statistic, p_value = np.nan, np.nan

    comparison_results.append({
        "Feature": feature,
        "Sensor": "VNIR" if feature.startswith("VNIR_") else "SWIR",
        "Infected_Count": len(infected),
        "Noninfected_Count": len(noninfected),
        "Infected_Mean": infected_mean,
        "Noninfected_Mean": noninfected_mean,
        "Infected_Median": infected_median,
        "Noninfected_Median": noninfected_median,
        "Mean_Difference": infected_mean - noninfected_mean,
        "Mann_Whitney_U": statistic,
        "P_Value": p_value
    })

comparison_df = pd.DataFrame(comparison_results)


# --------------------------------------------------
# 9. MULTIPLE TESTING CORRECTION
# --------------------------------------------------

def benjamini_hochberg(p_values):

    p_values = np.asarray(p_values, dtype=float)

    adjusted = np.full(len(p_values), np.nan)

    valid_indices = np.where(np.isfinite(p_values))[0]

    if len(valid_indices) == 0:
        return adjusted

    valid_p = p_values[valid_indices]

    order = np.argsort(valid_p)

    sorted_p = valid_p[order]

    m = len(sorted_p)

    adjusted_sorted = (
        sorted_p * m / np.arange(1, m + 1)
    )

    adjusted_sorted = np.minimum.accumulate(
        adjusted_sorted[::-1]
    )[::-1]

    adjusted_sorted = np.clip(
        adjusted_sorted,
        0,
        1
    )

    adjusted[valid_indices[order]] = adjusted_sorted

    return adjusted


if not comparison_df.empty:

    comparison_df["Adjusted_P_Value"] = benjamini_hochberg(
        comparison_df["P_Value"].values
    )

    comparison_df["Significant_After_FDR"] = (
        comparison_df["Adjusted_P_Value"] < 0.05
    )

    comparison_df = comparison_df.sort_values(
        by="P_Value",
        ascending=True
    )

comparison_df.to_csv(
    os.path.join(OUTPUT_DIR, "treatment_feature_comparison.csv"),
    index=False
)

print(comparison_df.to_string(index=False))


# --------------------------------------------------
# 10. HIGHLY CORRELATED FEATURES
# --------------------------------------------------

print("\nCORRELATION ANALYSIS")
print("=" * 50)

# Remove entirely empty and constant features for correlation.
valid_features = [
    column for column in X.columns
    if X[column].nunique(dropna=True) > 1
]

correlation_matrix = X[valid_features].corr(
    method="spearman"
)

correlation_matrix.to_csv(
    os.path.join(OUTPUT_DIR, "feature_correlation_matrix.csv")
)

correlation_pairs = []

for i in range(len(valid_features)):

    for j in range(i + 1, len(valid_features)):

        feature_1 = valid_features[i]
        feature_2 = valid_features[j]

        correlation = correlation_matrix.loc[
            feature_1,
            feature_2
        ]

        if abs(correlation) >= 0.95:

            correlation_pairs.append({
                "Feature_1": feature_1,
                "Feature_2": feature_2,
                "Spearman_Correlation": correlation
            })

correlation_df = pd.DataFrame(
    correlation_pairs,
    columns=[
        "Feature_1",
        "Feature_2",
        "Spearman_Correlation"
    ]
)

correlation_df.to_csv(
    os.path.join(OUTPUT_DIR, "highly_correlated_features.csv"),
    index=False
)

print("Valid features:", len(valid_features))
print("Highly correlated pairs:", len(correlation_df))

print(correlation_df.to_string(index=False))


# --------------------------------------------------
# 11. SENSOR-LEVEL SUMMARY
# --------------------------------------------------

print("\nSENSOR LEVEL SUMMARY")
print("=" * 50)

sensor_summary = []

for sensor in ["VNIR", "SWIR"]:

    sensor_columns = [
        c for c in X.columns
        if c.startswith(sensor + "_")
    ]

    sensor_data = X[sensor_columns]

    sensor_summary.append({
        "Sensor": sensor,
        "Feature_Count": len(sensor_columns),
        "Missing_Values": int(sensor_data.isnull().sum().sum()),
        "Missing_Percentage": (
            sensor_data.isnull().sum().sum()
            / sensor_data.size * 100
        ),
        "Zero_Variance_Features": int(
            (sensor_data.nunique(dropna=True) <= 1).sum()
        )
    })

sensor_summary_df = pd.DataFrame(sensor_summary)

sensor_summary_df.to_csv(
    os.path.join(OUTPUT_DIR, "sensor_summary.csv"),
    index=False
)

print(sensor_summary_df.to_string(index=False))


# --------------------------------------------------
# 12. FINAL SUMMARY
# --------------------------------------------------

print("\n" + "=" * 55)
print("FEATURE AND DATASET INVESTIGATION COMPLETED")
print("=" * 55)

print("Total samples:", len(df))
print("Total features:", len(feature_columns))
print("Missing values:", X.isnull().sum().sum())
print("Highly correlated pairs:", len(correlation_df))

if not comparison_df.empty:

    significant_count = int(
        comparison_df["Significant_After_FDR"].sum()
    )

    print("Features significant after FDR correction:", significant_count)

print("\nResults saved in:")
print(OUTPUT_DIR)

print("\nIMPORTANT:")
print(
    "Statistical associations do not establish causation "
    "or predictive usefulness."
)

print(
    "This dataset contains measurement counts, not "
    "pixel-level hyperspectral reflectance values."
)