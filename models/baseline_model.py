import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# TERRASPECTRA BASELINE ML MODEL
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR / "data" / "processed" / "VNIR_cleaned.csv"
)

MODEL_DIR = BASE_DIR / "saved_models"
RESULT_DIR = BASE_DIR / "data" / "processed" / "model_results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 1. LOAD DATASET
# ==========================================

def load_dataset():

    print("\n[1/6] Loading dataset...")

    data = pd.read_csv(DATA_PATH)

    print("Dataset shape:", data.shape)

    print("\nTreatment distribution:")
    print(data["Treatment"].value_counts())

    return data


# ==========================================
# 2. PREPARE FEATURES AND TARGET
# ==========================================

def prepare_data(data):

    print("\n[2/6] Preparing features and target...")

    metadata_columns = [
        "Pot",
        "Genotype",
        "Treatment"
    ]

    X = data.drop(columns=metadata_columns)

    X = X.apply(pd.to_numeric, errors="coerce")

    y = data["Treatment"].map({
        "Non-infected": 0,
        "Infected": 1
    })

    if y.isna().any():
        raise ValueError("Unexpected treatment labels found.")

    print("Number of samples:", len(X))
    print("Number of features:", X.shape[1])

    return X, y


# ==========================================
# 3. DEFINE MODELS
# ==========================================

def create_models():

    print("\n[3/6] Creating ML models...")

    models = {

        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=3,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42
        ),

        "SVM": SVC(
            kernel="linear",
            class_weight="balanced"
        )
    }

    return models


# ==========================================
# 4. BUILD LEAKAGE-SAFE PIPELINE
# ==========================================

def create_pipeline(model):

    return Pipeline([

        (
            "imputer",
            SimpleImputer(strategy="median")
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
            SelectKBest(score_func=f_classif, k=10)
        ),

        (
            "classifier",
            model
        )

    ])


# ==========================================
# 5. CROSS-VALIDATION
# ==========================================

def evaluate_models(X, y, models):

    print("\n[4/6] Evaluating models using cross-validation...")

    cv = StratifiedKFold(
        n_splits=4,
        shuffle=True,
        random_state=42
    )

    scoring = [
        "accuracy",
        "precision",
        "recall",
        "f1"
    ]

    results = []

    for name, model in models.items():

        print(f"\nEvaluating {name}...")

        pipeline = create_pipeline(model)

        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=scoring,
            error_score="raise"
        )

        results.append({

            "Model": name,

            "Accuracy_Mean": scores["test_accuracy"].mean(),

            "Accuracy_Std": scores["test_accuracy"].std(),

            "Precision_Mean": scores["test_precision"].mean(),

            "Recall_Mean": scores["test_recall"].mean(),

            "F1_Mean": scores["test_f1"].mean()

        })

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        RESULT_DIR / "model_comparison.csv",
        index=False
    )

    print("\nCross-validation results:")
    print(results_df.to_string(index=False))

    return results_df, cv


# ==========================================
# 6. TRAIN FINAL EXPERIMENTAL MODEL
# ==========================================

def train_final_model(X, y, models, results_df):

    print("\n[5/6] Training final experimental model...")

    best_name = results_df.sort_values(
        "F1_Mean",
        ascending=False
    ).iloc[0]["Model"]

    best_model = models[best_name]

    pipeline = create_pipeline(best_model)

    pipeline.fit(X, y)

    model_path = MODEL_DIR / "baseline_model.pkl"

    joblib.dump(pipeline, model_path)

    print("\nSelected model:", best_name)
    print("Saved model:", model_path)

    return pipeline, best_name


# ==========================================
# MAIN FUNCTION
# ==========================================

def main():

    print("\n========================================")
    print(" TERRASPECTRA BASELINE ML MODEL")
    print("========================================")

    data = load_dataset()

    X, y = prepare_data(data)

    models = create_models()

    results_df, cv = evaluate_models(
        X,
        y,
        models
    )

    pipeline, best_name = train_final_model(
        X,
        y,
        models,
        results_df
    )

    print("\n[6/6] MODEL DEVELOPMENT COMPLETED")

    print("\nBest experimental model:", best_name)

    print("\nResults saved in:")
    print(RESULT_DIR)

    print("\n========================================")


if __name__ == "__main__":
    main()