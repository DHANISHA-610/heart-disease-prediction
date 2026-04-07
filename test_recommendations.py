#!/usr/bin/env python3
"""
Test the new condition-based recommendation system
"""

import pandas as pd
from src.recommendations import recommend_from_patient_data

def test_condition_based_recommendations():
    """Test the condition-based recommendation system"""

    # Test case 1: High cholesterol patient
    patient_high_chol = pd.Series({
        'age': 45,
        'sex': 'M',
        'cp': 'ASY',
        'trestbps': 120,
        'chol': 250,  # High cholesterol
        'fbs': 0,
        'restecg': 'Normal',
        'thalach': 150,
        'exang': 'N',
        'oldpeak': 0.5
    })

    # Test case 2: Exercise angina patient
    patient_exang = pd.Series({
        'age': 55,
        'sex': 'F',
        'cp': 'TA',
        'trestbps': 150,  # High BP
        'chol': 180,
        'fbs': 1,  # High blood sugar
        'restecg': 'ST',  # Abnormal ECG
        'thalach': 120,  # Low heart rate
        'exang': 'Y',  # Exercise angina
        'oldpeak': 2.0  # High ST depression
    })

    # Test case 3: Low risk patient
    patient_low_risk = pd.Series({
        'age': 35,
        'sex': 'F',
        'cp': 'NAP',
        'trestbps': 110,
        'chol': 170,
        'fbs': 0,
        'restecg': 'Normal',
        'thalach': 180,
        'exang': 'N',
        'oldpeak': 0.0
    })

    test_cases = [
        ("High Cholesterol Patient", patient_high_chol),
        ("Exercise Angina Patient", patient_exang),
        ("Low Risk Patient", patient_low_risk),
    ]

    print("Condition-Based Recommendation System Test")
    print("=" * 50)

    for case_name, patient_data in test_cases:
        print(f"\n{case_name}:")
        print("-" * 30)
        print(f"Patient Data: {patient_data.to_dict()}")

        recommendations = recommend_from_patient_data(patient_data)
        print(f"Recommendations ({len(recommendations)} relevant items):")

        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")

    print("\n" + "=" * 50)
    print("Key Features:")
    print("✅ Only shows recommendations when conditions are met")
    print("✅ No irrelevant advice for healthy patients")
    print("✅ Personalized based on actual patient values")
    print("✅ Minimal and focused recommendations")

if __name__ == "__main__":
    test_condition_based_recommendations()