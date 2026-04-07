#!/usr/bin/env python3
"""
Test script for the new patient-specific SHAP explanations
"""

import pandas as pd
from src.explainability import explain_shap_feature_with_patient_data, generate_patient_specific_shap_explanation

def test_shap_explanations():
    """Test the new SHAP explanation functionality"""

    # Sample patient data
    patient_data = pd.Series({
        'age': 45,
        'sex': 'M',
        'cp': 'ASY',
        'trestbps': 120,
        'chol': 200,
        'fbs': 0,
        'restecg': 'Normal',
        'thalach': 150,
        'exang': 'N',
        'oldpeak': 1.0
    })

    # Sample SHAP contributions (simulated)
    shap_contributions = pd.Series({
        'sex_F': 0.015231,
        'age': 0.014607,
        'sex_M': 0.005255,
        'trestbps': 0.004143,
        'cp_NAP': 0.002013,
        'cp_TA': 0.000131,
        'restecg_ST': -0.004431,
        'restecg_LVH': -0.007482,
        'cp_ASY': -0.001000,  # This should explain absence of ASY
        'fbs_0': -0.002000,   # Normal blood sugar
    })

    print("Testing Patient-Specific SHAP Explanations")
    print("=" * 50)
    print(f"Patient Data: {patient_data.to_dict()}")
    print()

    # Test individual explanations
    print("Individual Feature Explanations:")
    print("-" * 30)
    for feature, shap_value in shap_contributions.items():
        explanation = explain_shap_feature_with_patient_data(feature, shap_value, patient_data)
        print(f"{feature}: {explanation}")
        print()

    # Test top explanations
    print("Top 8 Explanations:")
    print("-" * 20)
    explanations = generate_patient_specific_shap_explanation(shap_contributions, patient_data, top_n=8)
    for i, explanation in enumerate(explanations, 1):
        print(f"{i}. {explanation}")

if __name__ == "__main__":
    test_shap_explanations()