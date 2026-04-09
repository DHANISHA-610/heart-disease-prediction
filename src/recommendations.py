from typing import List
import pandas as pd


def recommend_from_patient_data(patient_data: pd.Series) -> List[str]:
    """
    Generate personalized recommendations based on patient data.
    Only returns recommendations when specific conditions are met.
    """
    recommendations = []

    # High cholesterol (>240)
    if patient_data.get('chol', 0) > 240:
        recommendations.append(
            "Work with your doctor to manage high cholesterol through medication and diet."
        )

    # Low max heart rate (<120 bpm, adjusted for age)
    age = patient_data.get('age', 50)
    expected_max_hr = 220 - age
    actual_max_hr = patient_data.get('thalach', 150)

    if actual_max_hr < expected_max_hr * 0.7:  # More than 30% below expected
        recommendations.append(
            "Improve cardiovascular fitness through regular exercise to raise maximum heart rate safely."
        )

    # Exercise-induced angina present
    if patient_data.get('exang', 'N') == 'Y':
        recommendations.append(
            "If you feel chest pain during exercise, follow a doctor-guided safe exercise plan (exercise-induced angina present)."
        )

    # ST depression > 2.0 mm
    if patient_data.get('oldpeak', 0) > 2.0:
        recommendations.append(
            "Manage stress and exercise habits to reduce ST depression during exertion."
        )

    # Asymptomatic chest pain (highest risk)
    if patient_data.get('cp', '') == 'ASY':
        recommendations.append(
            "Report chest pain symptoms promptly and follow medical advice for angina management."
        )

    # Abnormal resting ECG
    if patient_data.get('restecg', 'Normal') in ['ST', 'LVH']:
        recommendations.append(
            "Follow up on abnormal resting ECG results with your cardiologist."
        )

    # Fasting blood sugar > 120
    if patient_data.get('fbs', 0) == 1:
        recommendations.append(
            "Monitor and manage blood sugar levels through diet and exercise."
        )

    return recommendations