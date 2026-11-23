import os
import joblib
import pandas as pd

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
    "integrated",
    "vnir_swir_integrated.csv"
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "combined_model_results"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "saved_models"
)

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# --------------------------------------------------
# 2. LOAD INTEGRATED DATASET
# --------------------------------------------------

print("\nLoading integrated VNIR + SWIR dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# --------------------------------------------------
# 3. PREPARE TARGET
# --------------------------------------------------

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

print("\nTreatment distribution:")
print(y.value_counts().sort_index())


# --------------------------------------------------
# 4. IDENTIFY SENSOR FEATURES
# --------------------------------------------------

metadata_columns = [
    "Pot",
    "Genotype",
    "Treatment"
]

vnir_columns = [
    col for col in df.columns
    if col.startswith("VNIR_")
]

swir_columns = [
    col for col in df.columns
    if col.startswith("SWIR_")
]

if not vnir_columns:
    raise ValueError("No VNIR features found.")

if not swir_columns:
    raise ValueError("No SWIR features found.")

# Convert sensor measurements to numeric values
df[vnir_columns + swir_columns] = (
    df[vnir_columns + swir_columns]
    .apply(pd.to_numeric, errors="coerce")
)

# Three separate experiments
datasets = {
    "VNIR Only": df[vnir_columns],
    "SWIR Only": df[swir_columns],
    "VNIR + SWIR": df[vnir_columns + swir_columns]
}

print("\nFeature counts:")

for name, features in datasets.items():
    print(f"{name}: {features.shape[1]} features")


# --------------------------------------------------
# 5. CROSS-VALIDATION
# --------------------------------------------------

class_counts = y.value_counts()

if class_counts.min() < 2:
    raise ValueError(
        "At least two samples per class are required."
    )

n_splits = min(4, class_counts.min())

cv = StratifiedKFold(
    n_splits=n_splits,
    shuffle=True,
    random_state=42
)

# Create shared splits for fair comparison
cv_splits = list(cv.split(df, y))

print(f"\nUsing {n_splits}-fold stratified cross-validation.")


# --------------------------------------------------
# 6. DEFINE CLASSIFIERS
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
# 7. EVALUATION METRICS
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
# 8. TRAIN AND COMPARE ALL EXPERIMENTS
# --------------------------------------------------

results = []
pipelines = {}

print("\nStarting model comparison...")

for dataset_name, X in datasets.items():

    print("\n" + "=" * 55)
    print("DATASET:", dataset_name)
    print("=" * 55)

    for model_name, classifier in models.items():

        print("\nEvaluating:", model_name)

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
                    k=10
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
                cv=cv_splits,
                scoring=scoring,
                return_train_score=False,
                error_score="raise"
            )

            result = {
                "Dataset": dataset_name,
                "Model": model_name,

                "Accuracy": scores["test_accuracy"].mean(),
                "Accuracy_Std": scores["test_accuracy"].std(),

                "Precision": scores["test_precision"].mean(),
                "Recall": scores["test_recall"].mean(),
                "F1_Score": scores["test_f1"].mean()
            }

            results.append(result)

            pipelines[(dataset_name, model_name)] = (
                pipeline,
                X.columns.tolist()
            )

            print(f"Accuracy : {result['Accuracy']:.4f}")
            print(f"Precision: {result['Precision']:.4f}")
            print(f"Recall   : {result['Recall']:.4f}")
            print(f"F1-score : {result['F1_Score']:.4f}")

        except Exception as error:

            print(f"Evaluation failed: {error}")


# --------------------------------------------------
# 9. SAVE COMPARISON RESULTS
# --------------------------------------------------

if not results:
    raise RuntimeError("No experiment completed successfully.")

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1_Score",
    ascending=False
)

comparison_path = os.path.join(
    RESULT_DIR,
    "vnir_swir_model_comparison.csv"
)

results_df.to_csv(
    comparison_path,
    index=False
)

print("\nFINAL MODEL COMPARISON")
print("=" * 75)

print(results_df.to_string(index=False))

print("\nResults saved at:")
print(comparison_path)


# --------------------------------------------------
# 10. SELECT BEST EXPERIMENT
# --------------------------------------------------

best_result = results_df.iloc[0]

best_dataset = best_result["Dataset"]
best_model_name = best_result["Model"]

print("\nBEST EXPERIMENT")
print("Dataset:", best_dataset)
print("Model:", best_model_name)
print("F1-score:", round(best_result["F1_Score"], 4))


# --------------------------------------------------
# 11. TRAIN BEST PIPELINE ON FULL DATA
# --------------------------------------------------

best_pipeline, best_features = pipelines[
    (best_dataset, best_model_name)
]

X_best = datasets[best_dataset]

print("\nTraining best pipeline on all 48 samples...")

best_pipeline.fit(X_best, y)


# --------------------------------------------------
# 12. SAVE BEST MODEL
# --------------------------------------------------

model_path = os.path.join(
    MODEL_DIR,
    "best_vnir_swir_model.pkl"
)

joblib.dump(
    {
        "model": best_pipeline,
        "dataset": best_dataset,
        "model_name": best_model_name,
        "feature_columns": best_features
    },
    model_path
)

print("\nBest model saved at:")
print(model_path)


# --------------------------------------------------
# 13. FINAL SUMMARY
# --------------------------------------------------

print("\n" + "=" * 55)
print("COMBINED MODEL TRAINING COMPLETED")
print("=" * 55)

print("Total samples:", len(y))
print("VNIR features:", len(vnir_columns))
print("SWIR features:", len(swir_columns))
print("Combined features:", len(vnir_columns) + len(swir_columns))

print("Best dataset:", best_dataset)
print("Best algorithm:", best_model_name)

print("\nIMPORTANT:")
print(
    "The reported scores are cross-validation estimates. "
    "The saved model is subsequently fitted on all available samples."
)

print(
    "This remains an exploratory tabular ML experiment, "
    "not a validated hyperspectral image classifier."
)