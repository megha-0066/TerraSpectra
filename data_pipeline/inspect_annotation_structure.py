
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")

DATES = [
    "-1", "0", "4", "5", "6", "7", "8", "10",
    "11", "12", "13", "14", "15", "16", "17",
    "18", "19", "20"
]


def load_annotation(filename):

    path = RAW_DIR / filename

    df = pd.read_csv(
        path,
        sep=";",
        encoding="cp1252",
        skiprows=3,
        header=None,
        nrows=48
    )

    columns = [
        "pot",
        "genotype",
        "treatment",
        "day_minus_1_G",
        "day_minus_1_D"
    ]

    for day in DATES[1:]:
        columns.extend([
            f"day_{day}_G",
            f"day_{day}_D"
        ])

    if len(df.columns) != len(columns):
        raise ValueError(
            f"Column mismatch in {filename}: "
            f"Expected {len(columns)}, got {len(df.columns)}"
        )

    df.columns = columns

    # Remove accidental header or non-pot rows
    df["pot"] = df["pot"].astype(str).str.strip()

    df = df[
        df["pot"].str.fullmatch(r"\d+", na=False)
    ].copy()

    # Convert pot numbers to integers
    df["pot"] = df["pot"].astype(int)

    return df


for filename in ["VNIR_data.csv", "SWIR_data.csv"]:

    print("\n" + "=" * 60)
    print(filename)
    print("=" * 60)

    df = load_annotation(filename)

    print("\nShape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst five records:")
    print(df.head().to_string(index=False))

    print("\nTreatment distribution:")
    print(df["treatment"].value_counts())

    print("\nMissing values:")
    print(df.isnull().sum().sum())

    print("\nUnique pot count:")
    print(df["pot"].nunique())

    print("\nPot numbers:")
    print(sorted(df["pot"].unique()))