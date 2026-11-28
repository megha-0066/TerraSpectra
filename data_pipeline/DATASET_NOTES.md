# TerraSpectra: Dataset Documentation

## 1. Dataset Information

**Project:** TerraSpectra: Hyperspectral Crop Disease Forecasting

**Dataset:** Early Detection of Septoria Infection on Wheat Leaves Using Hyperspectral Imaging Data.

**Source:** INRAE Research Data Repository.

## 2. Dataset Files

| Filename | Description |
|---|---|
| VNIR_data.csv | VNIR image annotation and leaf-count information |
| SWIR_data.csv | SWIR image annotation and leaf-count information |
| Data_description.csv | Leaf-level sample metadata |

## 3. Dataset Statistics

- Pot-level records: 48
- Leaf-level records: 279
- Infected pots: 24
- Non-infected pots: 24
- Genotypes: EPO_67 and EPO_68
- Sensors: VNIR and SWIR

## 4. Important Dataset Understanding

The available CSV files contain annotation information and leaf counts. They do not contain the actual hyperspectral reflectance values.

The original dataset contains hyperspectral image data separately.

Therefore, the current machine learning experiments are exploratory tabular baselines, not validated hyperspectral image classification models.

## 5. Current Data Pipeline

1. Data loading
2. CSV cleaning
3. Exploratory data analysis
4. Missing-value analysis
5. Dataset integration
6. Feature investigation
7. Feature redundancy validation
8. Baseline model development
9. Model comparison
10. Error analysis

## 6. Current Limitations

- Actual hyperspectral image cubes have not yet been loaded.
- Current sample size is limited to 48 pots.
- Several features contain missing values.
- Existing baseline models show weak predictive performance.
- Further validation is required before making disease-prediction claims.

## 7. Next Development Steps

1. Obtain actual VNIR hyperspectral image files.
2. Inspect `.hyspex` and `.hdr` files.
3. Load and visualize hyperspectral image cubes.
4. Extract spectral information.
5. Develop preprocessing and normalization techniques.
6. Build and evaluate hyperspectral classification models.
7. Explore 3D-CNN and Vision Transformer architectures.