import pandas as pd
import os

# Dataset locations
BASE_PATH = "data/raw"

vnir_path = os.path.join(BASE_PATH, "VNIR_data.csv")
swir_path = os.path.join(BASE_PATH, "SWIR_data.csv")
description_path = os.path.join(BASE_PATH, "Data_description.csv")


# Check files
for path in [vnir_path, swir_path, description_path]:

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")


# Load VNIR dataset
vnir = pd.read_csv(
    vnir_path,
    sep=";",
    skiprows=3,
    encoding="cp1252"
)

# Load SWIR dataset
swir = pd.read_csv(
    swir_path,
    sep=";",
    skiprows=3,
    encoding="cp1252"
)

# Load description dataset
description = pd.read_csv(
    description_path,
    sep=";",
    skiprows=2,
    encoding="cp1252"
)


# Display dataset information
print("\n========== VNIR DATASET ==========")
print("Shape:", vnir.shape)
print(vnir.head())

print("\n========== SWIR DATASET ==========")
print("Shape:", swir.shape)
print(swir.head())

print("\n========== DATA DESCRIPTION ==========")
print("Shape:", description.shape)
print(description.head())


# Disease distribution
print("\n========== DISEASE DISTRIBUTION ==========")

print(
    description["Treatment [5]"].value_counts()
)


# Genotype distribution
print("\n========== GENOTYPE DISTRIBUTION ==========")

print(
    description["Genotype [4]"].value_counts()
)


# Missing values
print("\n========== MISSING VALUES ==========")

print("VNIR missing values:", vnir.isnull().sum().sum())

print("SWIR missing values:", swir.isnull().sum().sum())

print("Description missing values:", description.isnull().sum().sum())


print("\nDATA LOADING COMPLETED SUCCESSFULLY!")