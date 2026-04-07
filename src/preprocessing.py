import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
]
TARGET_COLUMN = "target"

COLUMN_MAPPING = {
    "Age": "age",
    "Sex": "sex",
    "ChestPainType": "cp",
    "RestingBP": "trestbps",
    "Cholesterol": "chol",
    "FastingBS": "fbs",
    "RestingECG": "restecg",
    "MaxHR": "thalach",
    "ExerciseAngina": "exang",
    "Oldpeak": "oldpeak",
    "HeartDisease": "target",
}

CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang"]
NUMERIC_FEATURES = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_FEATURES]


def load_data(path: str) -> pd.DataFrame:
    """Load the heart disease dataset from a CSV file and normalize column names."""
    df = pd.read_csv(path)
    df = df.rename(columns=COLUMN_MAPPING)
    df = validate_data(df)
    return df


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate input columns and normalize invalid values."""
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Dataset must include a '{TARGET_COLUMN}' column after renaming")
    missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_features:
        raise ValueError(f"Dataset is missing expected columns: {missing_features}")

    # Treat zero cholesterol as missing data and interpolate later.
    if "chol" in df.columns:
        df["chol"] = df["chol"].replace(0, np.nan)

    # Convert target to binary 0/1
    df[TARGET_COLUMN] = (df[TARGET_COLUMN] > 0).astype(int)
    if df[TARGET_COLUMN].nunique() < 2:
        raise ValueError(
            "Dataset must contain both classes 0 and 1 in the target column. "
            "Your current dataset contains only one class."
        )

    missing_counts = df[FEATURE_COLUMNS].isna().sum()
    if missing_counts.any():
        missing_report = missing_counts[missing_counts > 0].to_dict()
        print(f"Warning: missing values found in features: {missing_report}")
    return df


def build_preprocessor() -> ColumnTransformer:
    """Create a preprocessing pipeline for numeric and categorical features."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor


def split_dataset(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Split the dataset into training and test sets."""
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def get_encoded_feature_names(preprocessor) -> list:
    """Return the feature names after preprocessing for SHAP and reporting."""
    numeric_names = NUMERIC_FEATURES
    onehot = preprocessor.named_transformers_["cat"]["onehot"]
    categorical_names = onehot.get_feature_names_out(CATEGORICAL_FEATURES)
    return list(numeric_names) + list(categorical_names)
