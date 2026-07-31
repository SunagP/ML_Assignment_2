"""
Train and evaluate all 5 classification models on the Breast Cancer Wisconsin dataset.

Models: Logistic Regression, Decision Tree, KNN, Naive Bayes (Gaussian), Random Forest
Metrics: Accuracy, AUC, Precision, Recall, F1, MCC

Outputs:
  - model/trained_models.pkl   (dict of fitted models)
  - model/metrics.pkl          (dict of per-model metric dicts)
  - model/scaler.pkl           (fitted StandardScaler)
  - model/feature_names.pkl    (list of feature names)
  - test_data.csv              (the held-out test split with a 'target' column)
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
MODEL_DIR = BASE_DIR


def load_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    return df, list(data.feature_names), list(data.target_names)


def compute_metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "AUC": round(roc_auc_score(y_true, y_prob), 4),
        "Precision": round(precision_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "F1": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "MCC": round(matthews_corrcoef(y_true, y_pred), 4),
    }


def main():
    print("=" * 60)
    print("  ML Classification — Training Pipeline")
    print("=" * 60)

    df, feature_names, target_names = load_data()
    print(f"\nDataset shape : {df.shape}")
    print(f"Features      : {len(feature_names)}")
    print(f"Classes       : {target_names}")

    X = df[feature_names]
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df["target"] = y_test.values
    test_csv_path = os.path.join(PROJECT_DIR, "test_data.csv")
    test_df.to_csv(test_csv_path, index=False)
    print(f"\nTest data saved -> {test_csv_path}  ({len(test_df)} rows)")

    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=10000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    trained_models = {}
    all_metrics = {}

    print("\n" + "-" * 60)
    for name, clf in classifiers.items():
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        y_prob = clf.predict_proba(X_test_scaled)[:, 1]
        metrics = compute_metrics(y_test.values, y_pred, y_prob)
        trained_models[name] = clf
        all_metrics[name] = metrics
        print(f"\n{name}")
        for k, v in metrics.items():
            print(f"  {k:12s}: {v}")
    print("-" * 60)

    model_filenames = {
        "Logistic Regression": "logistic_regression.pkl",
        "Decision Tree": "decision_tree.pkl",
        "KNN": "knn.pkl",
        "Naive Bayes": "naive_bayes.pkl",
        "Random Forest": "random_forest.pkl",
    }
    for name, clf in trained_models.items():
        fpath = os.path.join(MODEL_DIR, model_filenames[name])
        with open(fpath, "wb") as f:
            pickle.dump(clf, f)
        print(f"  Saved {model_filenames[name]}")

    with open(os.path.join(MODEL_DIR, "metrics.pkl"), "wb") as f:
        pickle.dump(all_metrics, f)
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(MODEL_DIR, "feature_names.pkl"), "wb") as f:
        pickle.dump(feature_names, f)

    print("\nAll artefacts saved to model/ directory.")
    print("Done.")


if __name__ == "__main__":
    main()
