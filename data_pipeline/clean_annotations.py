import pandas as pd
import os

# Create output directory
os.makedirs("data/processed", exist_ok=True)

# Load VNIR
vnir = pd.read_csv(
    "data/raw/VNIR_data.csv",
    sep=";",
    skiprows=3,
    encoding="cp1252"
)

# Load SWIR
swir = pd.read_csv(
    "data/raw/SWIR_data.csv",
    sep=";",
    skiprows=3,
    encoding="cp1252"
)

# Load disease descriptions
description = pd.read_csv(
    "data/raw/Data_description.csv",
    sep=";",
    skiprows=2,
    encoding="cp1252"
)

# Remove empty and explanatory rows
vnir = vnir.dropna(subset=["Pot [2]"]).copy()
swir = swir.dropna(subset=["Pot [2]"]).copy()

# Remove empty and explanatory columns
description = description.iloc[:, :6].copy()

# Remove unnecessary empty rows
description = description.dropna(
    subset=["Pot [1]", "Treatment [5]"]
).copy()

# Save cleaned datasets
vnir.to_csv(
    "data/processed/VNIR_cleaned.csv",
    index=False
)

swir.to_csv(
    "data/processed/SWIR_cleaned.csv",
    index=False
)

description.to_csv(
    "data/processed/Data_description_cleaned.csv",
    index=False
)

print("========== CLEANING COMPLETED ==========")

print("VNIR shape:", vnir.shape)
print("SWIR shape:", swir.shape)
print("Description shape:", description.shape)

print("\nDisease distribution:")

print(description["Treatment [5]"].value_counts())

print("\nFiles saved successfully!")