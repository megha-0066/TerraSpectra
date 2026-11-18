import pandas as pd
import numpy as np

def validate_vnir_data():
    print("=== 1. VNIR Data Validation ===")
    try:
        # Skip top metadata rows and load using semicolon separator
        vnir_df = pd.read_csv('VNIR_data.csv', skiprows=3, sep=';', encoding='cp1252')
        vnir_df = vnir_df.dropna(how='all', axis=1) # Drop empty columns

        print(f"VNIR Dataset Shape: {vnir_df.shape}")

        # Check missing values
        missing_vals = vnir_df.isnull().sum().sum()
        print(f"Total Missing Values: {missing_vals}")

        # Validate columns & data types
        print(f"Total Columns Checked: {vnir_df.shape[1]}")
        numeric_cols = vnir_df.select_dtypes(include=[np.number]).columns
        print(f"Numeric Columns: {len(numeric_cols)}")

        print("VNIR Data Validation Completed Successfully!\n")
    except Exception as e:
        print(f"VNIR Error: {e}")

def validate_swir_data():
    print("=== 2. SWIR Data Validation  ===")
    try:
        # Skip top metadata rows and load using semicolon separator
        swir_df = pd.read_csv('SWIR_data.csv', skiprows=3, sep=';', encoding='cp1252')
        swir_df = swir_df.dropna(how='all', axis=1) # Drop empty columns

        print(f"SWIR Dataset Shape: {swir_df.shape}")

        # Check missing values
        missing_vals = swir_df.isnull().sum().sum()
        print(f"Total Missing Values: {missing_vals}")

        # Validate columns & data types
        print(f"Total Columns Checked: {swir_df.shape[1]}")
        numeric_cols = swir_df.select_dtypes(include=[np.number]).columns
        print(f"Numeric Columns: {len(numeric_cols)}")

        print("SWIR Data Validation Completed Successfully!")
    except Exception as e:
        print(f"SWIR Error: {e}")

if __name__ == "__main__":
    validate_vnir_data()
    validate_swir_data()    
    
     === 1. VNIR Data Validation (Member 1 Task) ===
VNIR Dataset Shape: (55, 36)
Total Missing Values: 655
Total Columns Checked: 36
Numeric Columns: 33
VNIR Data Validation Completed Successfully!

=== 2. SWIR Data Validation (Member 1 Task) ===
SWIR Dataset Shape: (55, 36)
Total Missing Values: 655
Total Columns Checked: 36
Numeric Columns: 33
SWIR Data Validation Completed Successfully!
                                                                                                                                            
