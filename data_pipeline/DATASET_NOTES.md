# TerraSpectra Dataset Documentation

## 1. Dataset Source
Dataset: Early detection of septoria infection on wheat leaves using hyperspectral imaging data.

Source: INRAE Research Data Repository.

## 2. Available CSV Files

| File | Description |
|---|---|
| VNIR_data.csv | VNIR image annotation and leaf-count information |
| SWIR_data.csv | SWIR image annotation and leaf-count information |
| Data_description.csv | Leaf-level sample metadata |

## 3. Current Dataset Status

- 48 pot-level records.
- 279 leaf-level records.
- 24 infected pots.
- 24 non-infected pots.
- VNIR and SWIR annotation tables are available.
- Actual hyperspectral image cubes have not yet been loaded.

## 4. Important Limitation

The current CSV files contain annotation and leaf-count information. They are not hyperspectral reflectance measurements.

Therefore, existing classification experiments are exploratory tabular baselines.

They must not be presented as validated hyperspectral image classification results.

## 5. Next Development Objectives

1. Obtain actual VNIR hyperspectral image data.
2. Inspect image headers and metadata.
3. Load hyperspectral image cubes.
4. Extract spectral information.
5. Develop a suitable preprocessing pipeline.
6. Train and evaluate hyperspectral classification models.