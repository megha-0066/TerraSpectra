
import os
import pandas as pd

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

OUTPUT_DIR = os.path.join(
    DATA_DIR,
    "integrated"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

VNIR_PATH = os.path.join(DATA_DIR, "VNIR_cleaned.csv")
SWIR_PATH = os.path.join(DATA_DIR, "SWIR_cleaned.csv")


# --------------------------------------------------
# 2. LOAD VNIR AND SWIR DATA
# --------------------------------------------------

print("\nLoading VNIR and SWIR datasets...")

vnir = pd.read_csv(VNIR_PATH)
swir = pd.read_csv(SWIR_PATH)

print("VNIR shape:", vnir.shape)
print("SWIR shape:", swir.shape)


# --------------------------------------------------
# 3. CHECK REQUIRED COLUMNS
# --------------------------------------------------

key_columns = ["Pot", "Genotype", "Treatment"]

for column in key_columns:

    if column not in vnir.columns:
        raise ValueError(f"{column} missing from VNIR dataset")

    if column not in swir.columns:
        raise ValueError(f"{column} missing from SWIR dataset")


# --------------------------------------------------
# 4. CHECK DUPLICATES
# --------------------------------------------------

print("\nChecking duplicate pot records...")

if vnir.duplicated(subset=["Pot"]).any():
    raise ValueError("Duplicate Pot records found in VNIR dataset")

if swir.duplicated(subset=["Pot"]).any():
    raise ValueError("Duplicate Pot records found in SWIR dataset")


# --------------------------------------------------
# 5. CHECK POT MATCHING
# --------------------------------------------------

vnir_pots = set(vnir["Pot"])
swir_pots = set(swir["Pot"])

print("VNIR unique pots:", len(vnir_pots))
print("SWIR unique pots:", len(swir_pots))

if vnir_pots != swir_pots:
    raise ValueError(
        "VNIR and SWIR pot IDs do not match. "
        "Check the unmatched pot records before merging."
    )


# --------------------------------------------------
# 6. PREPARE SENSOR FEATURES
# --------------------------------------------------

metadata_columns = [
    "Pot",
    "Genotype",
    "Treatment"
]

vnir_features = [
    column for column in vnir.columns
    if column not in metadata_columns
]

swir_features = [
    column for column in swir.columns
    if column not in metadata_columns
]

# Add prefixes to identify sensor features
vnir = vnir.rename(
    columns={
        column: f"VNIR_{column}"
        for column in vnir_features
    }
)

swir = swir.rename(
    columns={
        column: f"SWIR_{column}"
        for column in swir_features
    }
)


# --------------------------------------------------
# 7. MERGE VNIR AND SWIR
# --------------------------------------------------

print("\nIntegrating VNIR and SWIR features...")

combined = pd.merge(
    vnir,
    swir,
    on=key_columns,
    how="inner",
    validate="one_to_one"
)

print("Combined dataset shape:", combined.shape)


# --------------------------------------------------
# 8. CHECK MERGED DATA
# --------------------------------------------------

print("\nChecking integrated dataset...")

print("Total samples:", len(combined))

print(
    "Total measurement features:",
    combined.drop(columns=metadata_columns).shape[1]
)

print("Duplicate records:", combined.duplicated().sum())

print("\nMissing values:")
print(combined.isnull().sum().sum())

print("\nTreatment distribution:")
print(combined["Treatment"].value_counts())


# --------------------------------------------------
# 9. SAVE INTEGRATED DATASET
# --------------------------------------------------

output_path = os.path.join(
    OUTPUT_DIR,
    "vnir_swir_integrated.csv"
)

combined.to_csv(
    output_path,
    index=False
)

print("\nIntegrated dataset saved successfully!")

print(output_path)


# --------------------------------------------------
# 10. FINAL SUMMARY
# --------------------------------------------------

print("\n" + "=" * 50)
print("VNIR + SWIR INTEGRATION COMPLETED")
print("=" * 50)

print("VNIR samples:", len(vnir))
print("SWIR samples:", len(swir))
print("Integrated samples:", len(combined))
print(
    "Total features:",
    combined.drop(columns=metadata_columns).shape[1]
)

print("\nNext step: Train and compare models using combined features.")
