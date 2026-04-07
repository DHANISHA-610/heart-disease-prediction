from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import shap


def compute_shap_explanation(model, X, feature_names: List[str], sample_size: int = 100):
    """Compute SHAP values for a fitted classifier and dataset (on a sample for speed)."""
    if X.shape[0] > sample_size:
        indices = np.random.choice(X.shape[0], size=sample_size, replace=False)
        X_sample = X[indices]
    else:
        X_sample = X

    def predict_fn(X_input):
        return model.predict_proba(X_input)[:, 1]

    try:
        explainer = shap.explainers.Permutation(predict_fn, X_sample[:50], seed=42)
        shap_values = explainer(X_sample, silent=True, max_evals=50)
    except Exception:
        explainer = shap.KernelExplainer(predict_fn, X_sample[:20])
        shap_values = explainer(X_sample, silent=True)

    # Use the provided feature names instead of generic feature_0, feature_1, etc.
    shap_df = pd.DataFrame(shap_values.values, columns=feature_names)
    return shap_df, shap_values


def get_shap_importance(shap_df: pd.DataFrame) -> pd.Series:
    """Return mean absolute SHAP importance for each encoded feature."""
    return shap_df.abs().mean().sort_values(ascending=False)


def get_top_shap_features_with_direction(shap_df: pd.DataFrame, top_n: int = 5) -> List[dict]:
    """Return the top features with their direction of impact."""
    mean_importance = shap_df.mean().sort_values(ascending=False)
    mean_importance_abs = shap_df.abs().mean().sort_values(ascending=False)
    top_features = []

    for feature in mean_importance_abs.head(top_n).index:
        impact = mean_importance[feature]
        direction = "increased risk" if impact > 0 else "decreased risk"
        top_features.append({
            'feature': feature,
            'impact': abs(impact),
            'direction': direction
        })
    return top_features


def normalize_encoded_name(encoded_feature: str) -> str:
    """Normalize encoded feature names by stripping common prefixes."""
    if encoded_feature.startswith("cat__"):
        return encoded_feature[len("cat__"):]
    if encoded_feature.startswith("num__"):
        return encoded_feature[len("num__"):]
    return encoded_feature


def map_encoded_feature_to_name(encoded_feature: str) -> str:
    """Map encoded feature names to human-readable names."""
    normalized = normalize_encoded_name(encoded_feature)
    feature_map = {
        'age': 'Age',
        'trestbps': 'Resting Blood Pressure',
        'chol': 'Cholesterol',
        'thalach': 'Max Heart Rate',
        'oldpeak': 'ST Depression (Oldpeak)',
        'sex_F': 'Sex: Female',
        'sex_M': 'Sex: Male',
        'cp_ASY': 'Chest Pain: Asymptomatic',
        'cp_ATA': 'Chest Pain: Atypical Angina',
        'cp_NAP': 'Chest Pain: Non-Anginal Pain',
        'cp_TA': 'Chest Pain: Typical Angina',
        'fbs_0': 'Fasting Blood Sugar: Normal',
        'fbs_1': 'Fasting Blood Sugar: High',
        'restecg_LVH': 'Resting ECG: Left Ventricular Hypertrophy',
        'restecg_Normal': 'Resting ECG: Normal',
        'restecg_ST': 'Resting ECG: ST-T Wave Abnormality',
        'exang_N': 'Exercise Angina: No',
        'exang_Y': 'Exercise Angina: Yes',
    }
    return feature_map.get(normalized, encoded_feature)


def map_encoded_feature_to_feature(encoded_feature: str) -> str:
    """Map an encoded feature to the original input feature key for recommendations."""
    normalized = normalize_encoded_name(encoded_feature)
    feature_key_map = {
        'age': 'age',
        'trestbps': 'trestbps',
        'chol': 'chol',
        'thalach': 'thalach',
        'oldpeak': 'oldpeak',
        'sex_F': 'sex',
        'sex_M': 'sex',
        'cp_ASY': 'cp',
        'cp_ATA': 'cp',
        'cp_NAP': 'cp',
        'cp_TA': 'cp',
        'fbs_0': 'fbs',
        'fbs_1': 'fbs',
        'restecg_LVH': 'restecg',
        'restecg_Normal': 'restecg',
        'restecg_ST': 'restecg',
        'exang_N': 'exang',
        'exang_Y': 'exang',
    }
    return feature_key_map.get(normalized, normalized)


def get_patient_feature_value(patient_data: pd.Series, encoded_feature: str) -> Tuple[str, any]:
    """
    Get the actual patient value for an encoded feature.

    Returns:
        Tuple of (original_feature_name, patient_value)
    """
    normalized = normalize_encoded_name(encoded_feature)

    # Map encoded features back to original features
    feature_mapping = {
        'age': ('age', patient_data.get('age', 'Unknown')),
        'trestbps': ('trestbps', patient_data.get('trestbps', 'Unknown')),
        'chol': ('chol', patient_data.get('chol', 'Unknown')),
        'thalach': ('thalach', patient_data.get('thalach', 'Unknown')),
        'oldpeak': ('oldpeak', patient_data.get('oldpeak', 'Unknown')),
        'sex_F': ('sex', 'F' if normalized == 'sex_F' else 'M'),
        'sex_M': ('sex', 'M' if normalized == 'sex_M' else 'F'),
        'cp_ASY': ('cp', 'ASY'),
        'cp_ATA': ('cp', 'ATA'),
        'cp_NAP': ('cp', 'NAP'),
        'cp_TA': ('cp', 'TA'),
        'fbs_0': ('fbs', 0),
        'fbs_1': ('fbs', 1),
        'restecg_LVH': ('restecg', 'LVH'),
        'restecg_Normal': ('restecg', 'Normal'),
        'restecg_ST': ('restecg', 'ST'),
        'exang_N': ('exang', 'N'),
        'exang_Y': ('exang', 'Y'),
    }

    if normalized in feature_mapping:
        original_feature, expected_value = feature_mapping[normalized]
        actual_value = patient_data.get(original_feature, 'Unknown')
        return original_feature, actual_value

    return normalized, 'Unknown'


def explain_shap_feature_with_patient_data(
    encoded_feature: str,
    shap_value: float,
    patient_data: pd.Series
) -> str:
    """
    Generate a model-based SHAP explanation without medical assumptions.

    Only uses SHAP value sign to determine impact direction.
    Does not hardcode medical knowledge about what "should" increase/decrease risk.

    Args:
        encoded_feature: The encoded feature name (e.g., 'cp_ASY')
        shap_value: The SHAP contribution value
        patient_data: The original patient input data

    Returns:
        Explanation in format: Feature -> Impact -> Reason
    """
    normalized = normalize_encoded_name(encoded_feature)
    original_feature, actual_value = get_patient_feature_value(patient_data, encoded_feature)

    # Get readable feature name
    readable_name = map_encoded_feature_to_name(encoded_feature)

    # Determine impact direction from SHAP value only
    if shap_value > 0:
        impact_direction = "increases risk"
        reason = "The model learned that this feature contributes to higher heart disease risk"
    else:
        impact_direction = "decreases risk"
        reason = "The model learned that this feature contributes to lower heart disease risk"

    # Include actual value for numeric features
    if normalized in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']:
        feature_display = f"{readable_name} ({actual_value})"
    else:
        feature_display = readable_name

    return f"{feature_display} -> {impact_direction} -> {reason}"


def generate_patient_specific_shap_explanation(
    shap_contributions: pd.Series,
    patient_data: pd.Series,
    top_n: int = 8
) -> List[str]:
    """
    Generate patient-specific SHAP explanations based on actual input data.

    This function filters explanations to avoid contradictory statements by:
    1. For categorical features, only explaining the encoded feature that matches the patient's actual value
    2. For numeric features, explaining their actual contribution
    3. Prioritizing the most important features by absolute SHAP value

    Args:
        shap_contributions: Series of SHAP values for encoded features
        patient_data: Original patient input data
        top_n: Number of top features to explain

    Returns:
        List of human-readable explanation sentences
    """
    explanations = []

    # Group features by their original categorical feature to avoid contradictions
    categorical_groups = {}
    numeric_features = []

    for encoded_feature in shap_contributions.index:
        normalized = normalize_encoded_name(encoded_feature)

        if normalized in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']:
            # Numeric features
            numeric_features.append(encoded_feature)
        else:
            # Categorical features - group by original feature
            original_feature, _ = get_patient_feature_value(patient_data, encoded_feature)
            if original_feature not in categorical_groups:
                categorical_groups[original_feature] = []
            categorical_groups[original_feature].append(encoded_feature)

    # For categorical features, only keep the encoded feature that matches patient's actual value
    selected_features = []

    # Add numeric features
    selected_features.extend(numeric_features)

    # Add categorical features (only the matching one per group)
    for original_feature, encoded_features in categorical_groups.items():
        # Find which encoded feature matches the patient's actual value
        patient_value = patient_data.get(original_feature, 'Unknown')

        matching_feature = None
        for encoded_feature in encoded_features:
            normalized = normalize_encoded_name(encoded_feature)
            if normalized.startswith('sex_'):
                expected_gender = 'F' if normalized == 'sex_F' else 'M'
                if patient_value == expected_gender:
                    matching_feature = encoded_feature
                    break
            elif normalized.startswith('cp_'):
                expected_cp = normalized.split('_')[1]
                if patient_value == expected_cp:
                    matching_feature = encoded_feature
                    break
            elif normalized.startswith('fbs_'):
                expected_fbs = 1 if normalized == 'fbs_1' else 0
                if patient_value == expected_fbs:
                    matching_feature = encoded_feature
                    break
            elif normalized.startswith('restecg_'):
                expected_ecg = normalized.split('_')[1]
                if patient_value == expected_ecg:
                    matching_feature = encoded_feature
                    break
            elif normalized.startswith('exang_'):
                expected_exang = 'Y' if normalized == 'exang_Y' else 'N'
                if patient_value == expected_exang:
                    matching_feature = encoded_feature
                    break

        if matching_feature:
            selected_features.append(matching_feature)
        else:
            # If no exact match, add the one with highest absolute SHAP value
            best_feature = max(encoded_features, key=lambda x: abs(shap_contributions.get(x, 0)))
            selected_features.append(best_feature)

    # Sort selected features by absolute SHAP value and take top_n
    selected_contributions = shap_contributions[selected_features]
    top_features = selected_contributions.abs().sort_values(ascending=False).head(top_n)

    # Generate explanations for top features
    for encoded_feature in top_features.index:
        shap_value = shap_contributions[encoded_feature]
        explanation = explain_shap_feature_with_patient_data(
            encoded_feature, shap_value, patient_data
        )
        explanations.append(explanation)

    return explanations


def format_shap_for_display(shap_values, feature_names: List[str], sample_index: int = 0):
    """Format SHAP contributions for a single sample."""
    contributions = pd.Series(shap_values.values[sample_index], index=feature_names)
    contributions = contributions.sort_values(ascending=False)
    return contributions
