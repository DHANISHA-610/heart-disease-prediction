from typing import Dict

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier


def build_base_models() -> Dict[str, object]:
    """Return a set of base classifiers for the ensemble."""
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, solver="liblinear"),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42),
        "xgboost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42, verbosity=0),
    }
