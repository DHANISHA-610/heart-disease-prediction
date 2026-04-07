import os

from joblib import dump, load


def save_pipeline(path: str, artifacts: dict) -> None:
    """Save a model pipeline dictionary to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dump(artifacts, path)


def load_pipeline(path: str):
    """Load a saved model pipeline dictionary from disk."""
    return load(path)


def pipeline_exists(path: str) -> bool:
    """Check whether a saved pipeline file exists."""
    return os.path.exists(path)
