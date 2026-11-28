import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ============================================
# TERRASPECTRA: LEAF SPECTRAL VISUALIZATION
# ============================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR / "data" / "processed" /
    "leaf_spectra" / "13_G_leaf_spectral_summary.csv"
)

OUTPUT_DIR = (
    BASE_DIR / "data" / "processed" /
    "leaf_spectra" / "visualizations"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("TERRASPECTRA: LEAF SPECTRAL VISUALIZATION")
print("=" * 60)

# 1. Load spectral summary
print("\nLoading spectral summary...")

df = pd.read_csv(INPUT_FILE)

print("Total records:", len(df))
print("Available columns:", list(df.columns))

# 2. Separate leaves
leaf1 = df[df["leaf_id"] == "Leaf_1"].sort_values("wavelength_nm")
leaf2 = df[df["leaf_id"] == "Leaf_2"].sort_values("wavelength_nm")

if leaf1.empty or leaf2.empty:
    raise ValueError("Leaf spectral records are missing.")

# Verify spectral bands
if len(leaf1) != 216 or len(leaf2) != 216:
    raise ValueError("Expected 216 spectral bands per leaf.")

wavelengths = leaf1["wavelength_nm"].to_numpy()

if not np.allclose(
    wavelengths,
    leaf2["wavelength_nm"].to_numpy()
):
    raise ValueError("Wavelengths do not match between leaves.")

mean1 = leaf1["mean"].to_numpy()
mean2 = leaf2["mean"].to_numpy()

std1 = leaf1["std"].to_numpy()
std2 = leaf2["std"].to_numpy()

print("\nLeaf 1 bands:", len(mean1))
print("Leaf 2 bands:", len(mean2))

# ============================================
# GRAPH 1: MEAN SPECTRAL SIGNATURE COMPARISON
# ============================================

plt.figure(figsize=(12, 6))

plt.plot(
    wavelengths,
    mean1,
    label="Leaf 1",
    linewidth=2
)

plt.plot(
    wavelengths,
    mean2,
    label="Leaf 2",
    linewidth=2
)

plt.xlabel("Wavelength (nm)")
plt.ylabel("Mean Sensor Intensity")

plt.title(
    "TerraSpectra: Mean VNIR Spectral Signatures"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "mean_spectral_comparison.png",
    dpi=300
)

plt.close()

# ============================================
# GRAPH 2: SPECTRAL VARIABILITY
# ============================================

plt.figure(figsize=(12, 6))

plt.plot(
    wavelengths,
    std1,
    label="Leaf 1",
    linewidth=2
)

plt.plot(
    wavelengths,
    std2,
    label="Leaf 2",
    linewidth=2
)

plt.xlabel("Wavelength (nm)")
plt.ylabel("Standard Deviation")

plt.title(
    "TerraSpectra: Spectral Variability Comparison"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "spectral_variability.png",
    dpi=300
)

plt.close()

# ============================================
# GRAPH 3: MEAN WITH STANDARD DEVIATION
# ============================================

plt.figure(figsize=(12, 6))

plt.plot(
    wavelengths,
    mean1,
    label="Leaf 1",
    linewidth=2
)

plt.fill_between(
    wavelengths,
    mean1 - std1,
    mean1 + std1,
    alpha=0.2
)

plt.plot(
    wavelengths,
    mean2,
    label="Leaf 2",
    linewidth=2
)

plt.fill_between(
    wavelengths,
    mean2 - std2,
    mean2 + std2,
    alpha=0.2
)

plt.xlabel("Wavelength (nm)")
plt.ylabel("Mean Sensor Intensity")

plt.title(
    "TerraSpectra: Mean Spectral Signatures "
    "with Standard Deviation"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "spectral_signature_variability.png",
    dpi=300
)

plt.close()

# ============================================
# COMPLETION
# ============================================

print("\nVisualization completed successfully!")

print("\nGenerated files:")

for file in OUTPUT_DIR.glob("*.png"):
    print(file)

print("\nAll three spectral plots have been generated.")