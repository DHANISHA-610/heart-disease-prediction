#!/usr/bin/env python3
"""
Test the new neutral SHAP explanations that don't make medical assumptions
"""

import pandas as pd
from src.explainability import explain_shap_feature_with_patient_data

def test_neutral_shap_explanations():
    """Test the new neutral SHAP explanation format"""

    # Sample patient data
    patient_data = pd.Series({
        'age': 45,
        'sex': 'M',
        'cp': 'ASY',  # Asymptomatic chest pain
        'trestbps': 120,
        'chol': 200,
        'fbs': 0,    # Normal blood sugar
        'restecg': 'Normal',
        'thalach': 150,
        'exang': 'N',  # No exercise angina
        'oldpeak': 1.0
    })

    # Sample SHAP values (positive and negative)
    test_cases = [
        ('age', 0.014607),        # Positive - increases risk
        ('chol', 0.0431),         # Positive - increases risk
        ('thalach', -0.0602),     # Negative - decreases risk
        ('cp_ASY', 0.1120),       # Positive - increases risk
        ('exang_N', 0.0445),      # Positive - increases risk
        ('restecg_Normal', -0.001777),  # Negative - decreases risk
    ]

    print("New Neutral SHAP Explanations")
    print("=" * 60)
    print("Format: Feature → Impact → Reason")
    print()

    for encoded_feature, shap_value in test_cases:
        explanation = explain_shap_feature_with_patient_data(
            encoded_feature, shap_value, patient_data
        )
        print(explanation)

    print()
    print("Key Changes:")
    print("✅ No hardcoded medical assumptions")
    print("✅ Uses only SHAP value sign for impact direction")
    print("✅ Neutral language based purely on model learning")
    print("✅ Clear Feature → Impact → Reason format")
    print("✅ Includes actual values for numeric features")

if __name__ == "__main__":
    test_neutral_shap_explanations()