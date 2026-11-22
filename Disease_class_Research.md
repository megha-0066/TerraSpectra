TerraSpectra – Disease Classes and Sample Metadata Research

1. Research Objective

The objective of this research is to examine the disease treatment labels, wheat genotypes and sample identification information available in the hyperspectral dataset. This information helps identify the target labels and sample metadata relevant to wheat disease classification.

2. Crop and Disease Context

- Crop: Durum wheat (Triticum turgidum subsp. durum)

3. Treatment Classes

The dataset contains two treatment labels:

- Infected
- Non-infected

These treatment labels can be used as target classes for binary classification of wheat leaf infection status.

4. Verified Label Distribution

Treatment| Sample Count
Infected| 139
Non-infected| 140
Total| 279

The dataset contains nearly balanced treatment classes.

5. Genotype Distribution

Genotype| Sample Count
EPO_67| 140
EPO_68| 139
Total| 279

EPO_67 and EPO_68 are two durum wheat genotypes originating from the Evolutionary Pre-breeding population (EPO).


6. Sample Metadata

The Data_description.csv file contains the following fields:

- Pot: Identification number of the pot containing durum wheat plants.
- Side: G represents the left side and D represents the right side of the pot.
- Leaf: One to three leaves per pot side were imaged using hyperspectral imaging.
- Genotype: Wheat genotype identification.
- Treatment: Infected or Non-infected treatment label.
- Confidence level: Confidence of infection identification based on visual symptom evaluation.

7. Confidence Level Distribution

Confidence Level| Meaning| Sample Count
1| High confidence| 246
2| Low confidence| 33
Total| | 279

The confidence level is based on visual evaluation of symptoms. Level 1 indicates high confidence and level 2 indicates low confidence.

8. Relevance to Hyperspectral Disease Classification

The dataset provides treatment labels and hyperspectral measurements that can be investigated to identify spectral differences between infected and non-infected wheat leaves.

VNIR and SWIR spectral measurements may provide complementary information for analysing wheat leaf conditions. The verified treatment labels can be used in a binary classification research workflow.

9. Dataset Limitations

- Treatment labels represent the recorded infection treatment categories.
- Confidence levels represent visual symptom evaluation confidence, not disease severity scores.
- The metadata alone does not establish which wavelengths are most important for classification.
- The dataset does not independently demonstrate future disease forecasting capability.
- Further analysis and model evaluation are required to determine the usefulness of spectral measurements for disease classification.

10. Conclusion

The dataset contains 279 valid sample records with infected and non-infected treatment labels, two durum wheat genotypes and confidence information based on visual symptom evaluation.

The verified metadata provides a foundation for hyperspectral wheat disease classification research using VNIR and SWIR measurements.
