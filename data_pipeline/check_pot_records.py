
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")

for filename in ["VNIR_data.csv", "SWIR_data.csv"]:

    path = RAW_DIR / filename

    print("\n" + "=" * 50)
    print(filename)
    print("=" * 50)

    df = pd.read_csv(
        path,
        sep=";",
        encoding="cp1252",
        skiprows=3,
        header=None
    )

    print("Total rows:", len(df))
    print("\nFirst column values:")

    print(df.iloc[:, 0].tail(15).to_string(index=False))

    pot_values = (
        df.iloc[:, 0]
        .astype(str)
        .str.strip()
    )

    valid_pots = pot_values[
        pot_values.str.fullmatch(r"\d+", na=False)
    ]

    print("\nValid pot records:", len(valid_pots))
    print("Pot numbers:", valid_pots.tolist())

    missing_pots = sorted(
        set(range(1, 49)) - set(valid_pots.astype(int))
    )

    print("Potentially missing pots:", missing_pots)
