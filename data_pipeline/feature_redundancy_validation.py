
import os
import numpy as np
import pandas as pd


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
    "redundancy_validation"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

print("\nLoading integrated dataset...")

df = pd.read_csv(DATA_PATH)

print("Original dataset shape:", df.shape)

metadata_columns = [
    "Pot",
    "Genotype",
    "Treatment"
]

for column in metadata_columns:
    if column not in df.columns:
        raise ValueError(f"Missing metadata column: {column}")

feature_columns = [
    column for column in df.columns
    if column.startswith(("VNIR_", "SWIR_"))
]

X = df[feature_columns].apply(
    pd.to_numeric,
    errors="coerce"
)

print("Total samples:", len(df))
print("Total measurement features:", len(feature_columns))


# --------------------------------------------------
# 3. CHECK DATA TYPES
# --------------------------------------------------

print("\nDATA TYPE ANALYSIS")
print("=" * 55)

dtype_summary = pd.DataFrame({
    "Feature": X.columns,
    "Data_Type": X.dtypes.astype(str).values,
    "Unique_Values": X.nunique(dropna=True).values,
    "Missing_Values": X.isnull().sum().values
})

dtype_summary.to_csv(
    os.path.join(OUTPUT_DIR, "feature_dtype_summary.csv"),
    index=False
)

print(dtype_summary.to_string(index=False))


# --------------------------------------------------
# 4. IDENTIFY COMPLETELY EMPTY FEATURES
# --------------------------------------------------

print("\nCOMPLETELY EMPTY FEATURES")
print("=" * 55)

empty_features = X.columns[
    X.isnull().all()
].tolist()

print("Completely empty features:", len(empty_features))

for feature in empty_features:
    print(feature)

pd.DataFrame({
    "Completely_Empty_Feature": empty_features
}).to_csv(
    os.path.join(OUTPUT_DIR, "empty_features.csv"),
    index=False
)


# --------------------------------------------------
# 5. IDENTIFY ZERO-VARIANCE FEATURES
# --------------------------------------------------

print("\nZERO-VARIANCE FEATURES")
print("=" * 55)

zero_variance_features = []

for feature in X.columns:

    observed = X[feature].dropna()

    if len(observed) > 0 and observed.nunique() == 1:
        zero_variance_features.append(feature)

print("Zero-variance features:", len(zero_variance_features))

for feature in zero_variance_features:
    print(feature)

pd.DataFrame({
    "Zero_Variance_Feature": zero_variance_features
}).to_csv(
    os.path.join(OUTPUT_DIR, "zero_variance_features.csv"),
    index=False
)


# --------------------------------------------------
# 6. IDENTIFY EXACT DUPLICATE FEATURES
# --------------------------------------------------

print("\nEXACT DUPLICATE FEATURE ANALYSIS")
print("=" * 55)

duplicate_groups = {}

for feature in X.columns:

    values = X[feature]

    if values.isnull().all():
        continue

    matched_feature = None

    for existing_feature in duplicate_groups:

        if values.equals(X[existing_feature]):
            matched_feature = existing_feature
            break

    if matched_feature is not None:

        duplicate_groups[matched_feature].append(feature)

    else:

        duplicate_groups[feature] = []

duplicate_results = []

for original, duplicates in duplicate_groups.items():

    for duplicate in duplicates:

        duplicate_results.append({
            "Original_Feature": original,
            "Duplicate_Feature": duplicate
        })

duplicate_df = pd.DataFrame(
    duplicate_results,
    columns=[
        "Original_Feature",
        "Duplicate_Feature"
    ]
)

print("Exact duplicate feature pairs:", len(duplicate_df))

if not duplicate_df.empty:
    print(duplicate_df.to_string(index=False))
else:
    print("No exact duplicate columns found.")

duplicate_df.to_csv(
    os.path.join(OUTPUT_DIR, "exact_duplicate_features.csv"),
    index=False
)


# --------------------------------------------------
# 7. CHECK VNIR AND SWIR MATCHING FEATURES
# --------------------------------------------------

print("\nVNIR AND SWIR MATCHING FEATURE ANALYSIS")
print("=" * 55)

vnir_features = [
    column for column in X.columns
    if column.startswith("VNIR_")
]

swir_features = [
    column for column in X.columns
    if column.startswith("SWIR_")
]

vnir_suffixes = {
    feature.replace("VNIR_", "", 1): feature
    for feature in vnir_features
}

swir_suffixes = {
    feature.replace("SWIR_", "", 1): feature
    for feature in swir_features
}

common_measurements = sorted(
    set(vnir_suffixes) & set(swir_suffixes)
)

sensor_comparison = []

for measurement in common_measurements:

    vnir_name = vnir_suffixes[measurement]
    swir_name = swir_suffixes[measurement]

    vnir_values = X[vnir_name]
    swir_values = X[swir_name]

    valid = vnir_values.notna() & swir_values.notna()

    valid_vnir = vnir_values[valid]
    valid_swir = swir_values[valid]

    if len(valid_vnir) >= 2:

        correlation = valid_vnir.corr(
            valid_swir,
            method="spearman"
        )

        exact_match_count = int(
            (valid_vnir == valid_swir).sum()
        )

        exact_match_percentage = (
            exact_match_count / len(valid_vnir) * 100
        )

    else:

        correlation = np.nan
        exact_match_count = 0
        exact_match_percentage = np.nan

    sensor_comparison.append({

        "Measurement": measurement,

        "VNIR_Feature": vnir_name,

        "SWIR_Feature": swir_name,

        "Valid_Paired_Values": len(valid_vnir),

        "Spearman_Correlation": correlation,

        "Exact_Matching_Values": exact_match_count,

        "Exact_Matching_Percentage": exact_match_percentage

    })

sensor_comparison_df = pd.DataFrame(sensor_comparison)

sensor_comparison_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "vnir_swir_matching_analysis.csv"
    ),
    index=False
)

print(sensor_comparison_df.to_string(index=False))


# --------------------------------------------------
# 8. HIGHLY CORRELATED FEATURE ANALYSIS
# --------------------------------------------------

print("\nHIGHLY CORRELATED FEATURES")
print("=" * 55)

valid_features = [
    feature for feature in X.columns
    if X[feature].nunique(dropna=True) > 1
]

correlation_matrix = X[valid_features].corr(
    method="spearman",
    min_periods=3
)

correlation_matrix.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "spearman_correlation_matrix.csv"
    )
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

        if pd.notna(correlation) and abs(correlation) >= 0.95:

            correlation_pairs.append({

                "Feature_1": feature_1,

                "Feature_2": feature_2,

                "Correlation": correlation

            })

correlation_df = pd.DataFrame(
    correlation_pairs,
    columns=[
        "Feature_1",
        "Feature_2",
        "Correlation"
    ]
)

correlation_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "high_correlation_pairs.csv"
    ),
    index=False
)

print("Valid features:", len(valid_features))
print("Highly correlated pairs:", len(correlation_df))


# --------------------------------------------------
# 9. FEATURE DISTRIBUTION ANALYSIS
# --------------------------------------------------

print("\nFEATURE DISTRIBUTION ANALYSIS")
print("=" * 55)

distribution_summary = pd.DataFrame({

    "Feature": X.columns,

    "Mean": X.mean().values,

    "Median": X.median().values,

    "Standard_Deviation": X.std().values,

    "Minimum": X.min().values,

    "Maximum": X.max().values,

    "Unique_Values": X.nunique(dropna=True).values,

    "Missing_Percentage": (
        X.isnull().mean().values * 100
    )

})

distribution_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "feature_distribution_summary.csv"
    ),
    index=False
)

print(distribution_summary.to_string(index=False))


# --------------------------------------------------
# 10. CREATE CONSERVATIVE VALIDATED DATASET
# --------------------------------------------------

print("\nCREATING CONSERVATIVE VALIDATED DATASET")
print("=" * 55)

# Remove only completely empty features.
# Do not automatically remove highly correlated features.
# Do not automatically remove zero-variance features here.

validated_features = [
    feature for feature in X.columns
    if feature not in empty_features
]

validated_df = df[
    metadata_columns + validated_features
].copy()

validated_path = os.path.join(
    OUTPUT_DIR,
    "terraspectra_validated_features.csv"
)

validated_df.to_csv(
    validated_path,
    index=False
)

print("Original shape:", df.shape)
print("Validated shape:", validated_df.shape)

print("Removed completely empty features:", len(empty_features))

print(
    "Highly correlated features were retained "
    "for further investigation."
)


# --------------------------------------------------
# 11. FINAL VALIDATION REPORT
# --------------------------------------------------

summary = pd.DataFrame({

    "Parameter": [

        "Total samples",

        "Original measurement features",

        "Completely empty features",

        "Zero variance features",

        "Exact duplicate feature pairs",

        "Highly correlated feature pairs",

        "VNIR features",

        "SWIR features",

        "Validated measurement features"

    ],

    "Value": [

        len(df),

        len(feature_columns),

        len(empty_features),

        len(zero_variance_features),

        len(duplicate_df),

        len(correlation_df),

        len(vnir_features),

        len(swir_features),

        len(validated_features)

    ]

})

summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "validation_summary.csv"
    ),
    index=False
)

print("\n" + "=" * 55)
print("FEATURE REDUNDANCY VALIDATION COMPLETED")
print("=" * 55)

print(summary.to_string(index=False))

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nGenerated files:")

for filename in sorted(os.listdir(OUTPUT_DIR)):
    print(filename)

print("\nIMPORTANT:")
print(
    "The validated dataset only removes completely "
    "empty measurement columns."
)

print(
    "Correlation does not automatically mean that "
    "a feature should be deleted."
)

print(
    "These CSV features are measurement counts, "
    "not hyperspectral reflectance values."
)
