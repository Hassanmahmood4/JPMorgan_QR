"""Loan default prediction and expected loss calculation for Task 3."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Union

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = Path(__file__).parent / "Task 3 and 4 Loan Data.csv"
RECOVERY_RATE = 0.10
FEATURE_COLUMNS = [
    "credit_lines_outstanding",
    "loan_amt_outstanding",
    "total_debt_outstanding",
    "income",
    "years_employed",
    "fico_score",
]

_model = None


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    return df[FEATURE_COLUMNS].copy()


def compare_models(df: pd.DataFrame | None = None, test_size: float = 0.2) -> pd.DataFrame:
    if df is None:
        df = load_data()
    X = _prepare_features(df)
    y = df["default"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
    }
    rows = []
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        rows.append({
            "model": name,
            "accuracy": accuracy_score(y_test, model.predict(X_test)),
            "roc_auc": roc_auc_score(y_test, proba),
        })
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)


class LoanRiskModel:
    def __init__(self, model=None):
        self.model = model or LogisticRegression(max_iter=1000, random_state=42)
        self.recovery_rate = RECOVERY_RATE

    def fit(self, df: pd.DataFrame) -> "LoanRiskModel":
        self.model.fit(_prepare_features(df), df["default"])
        return self

    def predict_pd(self, loan: Union[Mapping, pd.Series, pd.DataFrame]) -> float:
        frame = loan[FEATURE_COLUMNS] if isinstance(loan, pd.DataFrame) else pd.DataFrame([pd.Series(loan)])[FEATURE_COLUMNS]
        return float(self.model.predict_proba(frame)[0, 1])

    def expected_loss(self, loan: Union[Mapping, pd.Series, pd.DataFrame]) -> float:
        s = loan if isinstance(loan, pd.Series) else pd.Series(loan)
        return float(self.predict_pd(s) * float(s["loan_amt_outstanding"]) * (1.0 - self.recovery_rate))


def get_model() -> LoanRiskModel:
    global _model
    if _model is None:
        _model = LoanRiskModel().fit(load_data())
    return _model


def get_probability_of_default(loan) -> float:
    return get_model().predict_pd(loan)


def calculate_expected_loss(loan) -> float:
    return get_model().expected_loss(loan)
