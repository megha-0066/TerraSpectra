
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.model_selection import StratifiedKFold, cross_val_predict

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score
)


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "integrated",
    "vnir_swir_integrated.csv"
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "error_analysis"
)

os.makedirs(RESULT_DIR, exist_ok=True)


# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

print("\nLoading integrated dataset...")

df = pd.read_csv(DATA_PATH)

df = df.dropna(subset=["Treatment"]).copy()

df["Treatment"] = (
    df["Treatment"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({
        "non-infected": 0,
        "infected": 1
    })
)

df = df.dropna(subset=["Treatment"]).copy()

y = df["Treatment"].astype(int)

print("Dataset shape:", df.shape)
print("Samples:", len(df))


# --------------------------------------------------
# 3. SELECT SWIR FEATURES
# --------------------------------------------------

swir_columns = [
    column for column in df.columns
    if column.startswith("SWIR_")
]

if not swir_columns:
    raise ValueError("No SWIR features found.")

X = df[swir_columns].apply(
    pd.to_numeric,
    errors="coerce"
)

print("SWIR feature count:", X.shape[1])


# --------------------------------------------------
# 4. DEFINE RANDOM FOREST PIPELINE
# --------------------------------------------------

model = Pipeline([

    (
        "imputer",
        SimpleImputer(
            strategy="median",
            keep_empty_features=True
        )
    ),

    (
        "variance_filter",
        VarianceThreshold(threshold=0)
    ),

    (
        "scaler",
        StandardScaler()
    ),

    (
        "feature_selection",
        SelectKBest(
            score_func=f_classif,
            k=10
        )
    ),

    (
        "classifier",
        RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42
        )
    )

])


# --------------------------------------------------
# 5. OUT-OF-FOLD PREDICTIONS
# --------------------------------------------------

print("\nGenerating out-of-fold predictions...")

cv = StratifiedKFold(
    n_splits=4,
    shuffle=True,
    random_state=42
)

y_pred = cross_val_predict(
    model,
    X,
    y,
    cv=cv,
    method="predict"
)

y_probability = cross_val_predict(
    model,
    X,
    y,
    cv=cv,
    method="predict_proba"
)[:, 1]

print("Out-of-fold predictions generated.")


# --------------------------------------------------
# 6. PERFORMANCE METRICS
# --------------------------------------------------

accuracy = accuracy_score(y, y_pred)

precision = precision_score(
    y,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y,
    y_pred,
    zero_division=0
)

auc = roc_auc_score(y, y_probability)

print("\nOUT-OF-FOLD PERFORMANCE")
print("=" * 45)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC-AUC  : {auc:.4f}")


# --------------------------------------------------
# 7. CLASSIFICATION REPORT
# --------------------------------------------------

report = classification_report(
    y,
    y_pred,
    labels=[0, 1],
    target_names=["Non-infected", "Infected"],
    zero_division=0
)

print("\nCLASSIFICATION REPORT")
print(report)

with open(
    os.path.join(RESULT_DIR, "classification_report.txt"),
    "w",
    encoding="utf-8"
) as file:
    file.write(report)


# --------------------------------------------------
# 8. CONFUSION MATRIX
# --------------------------------------------------

cm = confusion_matrix(
    y,
    y_pred,
    labels=[0, 1]
)

print("\nCONFUSION MATRIX")
print(cm)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Non-infected", "Infected"]
)

display.plot()

plt.title("SWIR Random Forest Out-of-Fold Confusion Matrix")
plt.tight_layout()

plt.savefig(
    os.path.join(RESULT_DIR, "confusion_matrix.png"),
    dpi=300
)

plt.show()


# --------------------------------------------------
# 9. ROC CURVE
# --------------------------------------------------

fpr, tpr, thresholds = roc_curve(
    y,
    y_probability
)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"SWIR Random Forest (AUC = {auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("SWIR Random Forest Out-of-Fold ROC Curve")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(RESULT_DIR, "roc_curve.png"),
    dpi=300
)

plt.show()


# --------------------------------------------------
# 10. SAVE SAMPLE-LEVEL PREDICTIONS
# --------------------------------------------------

prediction_results = pd.DataFrame({

    "Pot": df["Pot"].values,

    "Actual": y.values,

    "Predicted": y_pred,

    "Infected_Probability": y_probability

})

prediction_results["Actual_Label"] = (
    prediction_results["Actual"].map({
        0: "Non-infected",
        1: "Infected"
    })
)

prediction_results["Predicted_Label"] = (
    prediction_results["Predicted"].map({
        0: "Non-infected",
        1: "Infected"
    })
)

prediction_results["Result"] = (
    prediction_results["Actual"] ==
    prediction_results["Predicted"]
).map({
    True: "Correct",
    False: "Incorrect"
})

prediction_results.to_csv(
    os.path.join(RESULT_DIR, "out_of_fold_predictions.csv"),
    index=False
)


# --------------------------------------------------
# 11. IDENTIFY MISCLASSIFIED SAMPLES
# --------------------------------------------------

errors = prediction_results[
    prediction_results["Result"] == "Incorrect"
]

errors.to_csv(
    os.path.join(RESULT_DIR, "misclassified_samples.csv"),
    index=False
)

print("\nMISCLASSIFICATION SUMMARY")
print("=" * 45)

print("Correct predictions:", len(prediction_results) - len(errors))
print("Incorrect predictions:", len(errors))

print("\nMisclassified samples:")
print(errors.to_string(index=False))


# --------------------------------------------------
# 12. SAVE METRICS
# --------------------------------------------------

metrics = pd.DataFrame([{
    "Dataset": "SWIR Only",
    "Model": "Random Forest",
    "Evaluation": "4-fold Out-of-Fold",
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1_Score": f1,
    "ROC_AUC": auc
}])

metrics.to_csv(
    os.path.join(RESULT_DIR, "evaluation_metrics.csv"),
    index=False
)


# --------------------------------------------------
# 13. FINAL SUMMARY
# --------------------------------------------------

print("\n" + "=" * 50)
print("ERROR ANALYSIS COMPLETED")
print("=" * 50)

print("Results saved in:")
print(RESULT_DIR)

print("\nGenerated files:")
print("1. classification_report.txt")
print("2. confusion_matrix.png")
print("3. roc_curve.png")
print("4. out_of_fold_predictions.csv")
print("5. misclassified_samples.csv")
print("6. evaluation_metrics.csv")

print(
    "\nThese are out-of-fold predictions. "
    "Each sample was predicted without being used "
    "to train the model for its fold."
)
