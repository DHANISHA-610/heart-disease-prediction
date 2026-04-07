from typing import Tuple

from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def build_ensemble(base_models: dict) -> VotingClassifier:
    """Build a soft-voting ensemble classifier from base models."""
    estimators = [(name, model) for name, model in base_models.items()]
    ensemble = VotingClassifier(estimators=estimators, voting="soft")
    return ensemble


def evaluate_model(model, X, y) -> dict:
    """Evaluate a binary classifier and return common metrics."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1_score": f1_score(y, y_pred),
        "roc_auc": roc_auc_score(y, y_proba),
    }
