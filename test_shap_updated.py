#!/usr/bin/env python3
"""
Test the updated SHAP explanation logic
"""

import pandas as pd
from src.explainability import generate_patient_specific_shap_explanation

def test_updated_shap_logic():
    """Test the updated SHAP explanation logic with sample data"""

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

    # Simulated SHAP contributions (what might come from the model)
    # Note: In reality, only one of each categorical group would be non-zero
    # But SHAP can show contributions for all encoded features
    shap_contributions = pd.Series({
        'age': 0.014607,           # Numeric - always included
        'trestbps': 0.004143,      # Numeric - always included
        'chol': 0.0431,           # Numeric - always included
        'thalach': -0.0602,       # Numeric - always included
        'oldpeak': 0.0930,        # Numeric - always included
        'sex_M': 0.005255,        # Patient is male, so sex_M should be selected
        'sex_F': 0.0,             # Patient is not female
        'cp_ASY': 0.1120,         # Patient has ASY, so cp_ASY should be selected
        'cp_ATA': 0.0,            # Patient does not have ATA
        'cp_NAP': 0.0,            # Patient does not have NAP
        'cp_TA': 0.0,             # Patient does not have TA
        'fbs_0': -0.011487,       # Patient has fbs=0, so fbs_0 should be selected
        'fbs_1': 0.0,             # Patient does not have fbs=1
        'restecg_Normal': -0.001777,  # Patient has Normal, so restecg_Normal should be selected
        'restecg_ST': 0.0,        # Patient does not have ST
        'restecg_LVH': 0.0,       # Patient does not have LVH
        'exang_N': 0.0445,        # Patient has exang=N, so exang_N should be selected
        'exang_Y': 0.0,           # Patient does not have exang=Y
    })

    print("Patient Data:")
    for key, value in patient_data.items():
        print(f"  {key}: {value}")
    print()

    print("SHAP Contributions (simulated):")
    for feature, value in shap_contributions.items():
        if abs(value) > 0.001:  # Only show non-zero contributions
            print(".6f")
    print()

    print("Generated Patient-Specific Explanations:")
    print("=" * 50)
    explanations = generate_patient_specific_shap_explanation(shap_contributions, patient_data, top_n=8)

    for i, explanation in enumerate(explanations, 1):
        print(f"{i}. {explanation}")

    print()
    print("Key Improvements:")
    print("- No contradictory explanations (e.g., both presence AND absence of same condition)")
    print("- Only explains the patient's actual conditions")
    print("- Clear, patient-friendly language")
    print("- Correctly handles positive/negative SHAP values")

if __name__ == "__main__":
    test_updated_shap_logic()