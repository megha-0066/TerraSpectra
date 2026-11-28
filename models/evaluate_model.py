
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

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
    "VNIR_cleaned.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "saved_models",
    "baseline_model.pkl"
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "model_evaluation"
)

os.makedirs(RESULT_DIR, exist_ok=True)


# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

print("\nLoading dataset...")

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

X = df.drop(
    columns=["Pot", "Genotype", "Treatment"],
    errors="ignore"
)

X = X.apply(pd.to_numeric, errors="coerce")

# Match the features used during model training
X = X.dropna(axis=1, how="all")

print("Dataset shape:", df.shape)
print("Feature shape:", X.shape)


# --------------------------------------------------
# 3. LOAD TRAINED MODEL
# --------------------------------------------------

print("\nLoading trained SVM model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully!")


# --------------------------------------------------
# 4. MAKE PREDICTIONS
# --------------------------------------------------

print("\nGenerating predictions...")

y_pred = model.predict(X)

print("Predictions generated successfully!")


# --------------------------------------------------
# 5. CLASSIFICATION METRICS
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

print("\nMODEL PERFORMANCE")
print("=" * 45)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")


# --------------------------------------------------
# 6. CLASSIFICATION REPORT
# --------------------------------------------------

print("\nCLASSIFICATION REPORT")
print("=" * 45)

report = classification_report(
    y,
    y_pred,
    target_names=["Non-infected", "Infected"],
    zero_division=0
)

print(report)

report_path = os.path.join(
    RESULT_DIR,
    "classification_report.txt"
)

with open(report_path, "w", encoding="utf-8") as file:
    file.write(report)


# --------------------------------------------------
# 7. CONFUSION MATRIX
# --------------------------------------------------

print("\nGenerating confusion matrix...")

cm = confusion_matrix(
    y,
    y_pred,
    labels=[0, 1]
)

print(cm)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Non-infected", "Infected"]
)

display.plot()

plt.title("TerraSpectra SVM Confusion Matrix")
plt.tight_layout()

plt.savefig(
    os.path.join(RESULT_DIR, "confusion_matrix.png"),
    dpi=300
)

plt.show()


# --------------------------------------------------
# 8. ROC CURVE AND ROC-AUC
# --------------------------------------------------

print("\nGenerating ROC curve...")

if len(set(y)) == 2:

    y_scores = model.decision_function(X)

    auc_score = roc_auc_score(y, y_scores)

    fpr, tpr, thresholds = roc_curve(y, y_scores)

    print(f"ROC-AUC Score: {auc_score:.4f}")

    plt.figure(figsize=(7, 5))

    plt.plot(
        fpr,
        tpr,
        label=f"SVM (AUC = {auc_score:.4f})"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title("TerraSpectra ROC Curve")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        os.path.join(RESULT_DIR, "roc_curve.png"),
        dpi=300
    )

    plt.show()

else:
    auc_score = None
    print("ROC-AUC cannot be calculated because only one class is present.")


# --------------------------------------------------
# 9. SAVE PREDICTIONS
# --------------------------------------------------

prediction_results = pd.DataFrame({
    "Actual": y.values,
    "Predicted": y_pred
})

prediction_results["Actual_Label"] = prediction_results["Actual"].map({
    0: "Non-infected",
    1: "Infected"
})

prediction_results["Predicted_Label"] = prediction_results["Predicted"].map({
    0: "Non-infected",
    1: "Infected"
})

prediction_results.to_csv(
    os.path.join(RESULT_DIR, "prediction_results.csv"),
    index=False
)


# --------------------------------------------------
# 10. SAVE METRICS
# --------------------------------------------------

metrics = pd.DataFrame([{
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1_Score": f1,
    "ROC_AUC": auc_score
}])

metrics.to_csv(
    os.path.join(RESULT_DIR, "evaluation_metrics.csv"),
    index=False
)


# --------------------------------------------------
# 11. FINAL SUMMARY
# --------------------------------------------------

print("\n" + "=" * 45)
print("MODEL EVALUATION COMPLETED")
print("=" * 45)

print("Evaluation results saved in:")
print(RESULT_DIR)

print("\nGenerated files:")
print("1. classification_report.txt")
print("2. confusion_matrix.png")
print("3. roc_curve.png")
print("4. prediction_results.csv")
print("5. evaluation_metrics.csv")

print("\nIMPORTANT:")
print(
    "These predictions are on the same 48 samples used "
    "to fit the final saved model. They are training-set "
    "diagnostics, NOT independent test-set performance."
)
