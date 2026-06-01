from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_baseline_model(feature_csv: Path, label_col: str, test_size: float = 0.2):
    df = pd.read_csv(feature_csv)
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in {feature_csv}")

    feature_cols = [c for c in df.columns if c not in {"source_id", "epoch_idx", label_col}]
    X = df[feature_cols]
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[("num", num_pipe, feature_cols)],
        remainder="drop",
    )

    clf = Pipeline(
        steps=[
            ("pre", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    multi_class="auto",
                ),
            ),
        ]
    )

    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    report = classification_report(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    return clf, report, cm
