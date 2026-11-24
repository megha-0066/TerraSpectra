
import os
import warnings
import joblib
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import make_scorer, precision_score


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

MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
RESULT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "model_results"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

print("\nLoading VNIR dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# --------------------------------------------------
# 3. PREPARE FEATURES AND TARGET
# --------------------------------------------------

TARGET_COLUMN = "Treatment"

if TARGET_COLUMN not in df.columns:
    raise ValueError(
        f"Target column '{TARGET_COLUMN}' not found. "
        f"Available columns: {list(df.columns)}"
    )

# Remove rows with missing target values
df = df.dropna(subset=[TARGET_COLUMN]).copy()

# Convert treatment labels into numerical values
df[TARGET_COLUMN] = (
    df[TARGET_COLUMN]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({
        "non-infected": 0,
        "infected": 1
    })
)

# Remove rows whose treatment labels could not be mapped
df = df.dropna(subset=[TARGET_COLUMN]).copy()

y = df[TARGET_COLUMN].astype(int)

# Exclude metadata and target columns
excluded_columns = [
    "Pot",
    "Genotype",
    "Treatment"
]

X = df.drop(
    columns=excluded_columns,
    errors="ignore"
)

# Convert all features to numeric
X = X.apply(pd.to_numeric, errors="coerce")

# Remove columns containing no observed values
# The imputer will handle missing values in remaining columns.
X = X.dropna(axis=1, how="all")

print("\nFeature matrix shape:", X.shape)
print("Target distribution:")
print(y.value_counts().sort_index())

print("\nClass mapping:")
print("0 = Non-infected")
print("1 = Infected")


# --------------------------------------------------
# 4. DEFINE CROSS-VALIDATION
# --------------------------------------------------

class_counts = y.value_counts()

minimum_class_count = class_counts.min()

if minimum_class_count < 2:
    raise ValueError(
        "At least two samples per class are required for "
        "stratified cross-validation."
    )

n_splits = min(4, minimum_class_count)

cv = StratifiedKFold(
    n_splits=n_splits,
    shuffle=True,
    random_state=42
)

print(f"\nUsing {n_splits}-fold stratified cross-validation.")


# --------------------------------------------------
# 5. DEFINE MODELS
# --------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    ),

    "SVM": SVC(
        kernel="rbf",
        class_weight="balanced",
        random_state=42
    )
}


# --------------------------------------------------
# 6. DEFINE EVALUATION METRICS
# --------------------------------------------------

scoring = {
    "accuracy": "accuracy",

    "precision": make_scorer(
        precision_score,
        zero_division=0
    ),

    "recall": "recall",

    "f1": "f1"
}


# --------------------------------------------------
# 7. MODEL TRAINING AND EVALUATION
# --------------------------------------------------

results = []
trained_pipelines = {}

print("\nStarting model evaluation...")

for model_name, classifier in models.items():

    print("\n" + "=" * 50)
    print("Evaluating:", model_name)
    print("=" * 50)

    pipeline = Pipeline([
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
                k=min(10, X.shape[1])
            )
        ),

        (
            "classifier",
            classifier
        )
    ])

    try:

        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            error_score="raise"
        )

        result = {
            "Model": model_name,

            "Accuracy": scores["test_accuracy"].mean(),

            "Accuracy_Std": scores["test_accuracy"].std(),

            "Precision": scores["test_precision"].mean(),

            "Recall": scores["test_recall"].mean(),

            "F1_Score": scores["test_f1"].mean()
        }

        results.append(result)

        trained_pipelines[model_name] = pipeline

        print(f"Accuracy : {result['Accuracy']:.4f}")
        print(f"Precision: {result['Precision']:.4f}")
        print(f"Recall   : {result['Recall']:.4f}")
        print(f"F1-score : {result['F1_Score']:.4f}")

    except Exception as error:

        print(f"Error evaluating {model_name}: {error}")


# --------------------------------------------------
# 8. SAVE MODEL COMPARISON RESULTS
# --------------------------------------------------

if not results:
    raise RuntimeError(
        "No model completed evaluation successfully."
    )

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1_Score",
    ascending=False
)

comparison_path = os.path.join(
    RESULT_DIR,
    "model_comparison.csv"
)

results_df.to_csv(
    comparison_path,
    index=False
)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))

print("\nComparison saved at:")
print(comparison_path)


# --------------------------------------------------
# 9. SELECT BEST MODEL
# --------------------------------------------------

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_pipelines[best_model_name]

print("\nBest model based on mean cross-validation F1-score:")
print(best_model_name)


# --------------------------------------------------
# 10. TRAIN BEST MODEL ON FULL DATASET
# --------------------------------------------------

print("\nTraining selected model on all available samples...")

best_model.fit(X, y)


# --------------------------------------------------
# 11. SAVE TRAINED MODEL
# --------------------------------------------------

model_path = os.path.join(
    MODEL_DIR,
    "baseline_model.pkl"
)

joblib.dump(
    best_model,
    model_path
)

print("\nTrained model saved successfully:")
print(model_path)


# --------------------------------------------------
# 12. FINAL SUMMARY
# --------------------------------------------------

print("\n" + "=" * 50)
print("BASELINE MODEL PIPELINE COMPLETED")
print("=" * 50)

print("Dataset samples:", len(X))
print("Input features:", X.shape[1])
print("Cross-validation folds:", n_splits)
print("Best model:", best_model_name)

print("\nImportant:")
print(
    "This is an exploratory tabular baseline using VNIR "
    "CSV measurement counts, not a validated hyperspectral "
    "image classification model."
)

print(
    "The dataset is small, so the cross-validation results "
    "should be interpreted cautiously."
)
