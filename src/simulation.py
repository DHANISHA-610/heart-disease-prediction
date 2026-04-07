from typing import Dict, List

import pandas as pd


def risk_category(probability: float) -> str:
    """Convert predicted probability into a risk category."""
    if probability < 0.30:
        return "Low Risk"
    if probability < 0.70:
        return "Moderate Risk"
    return "High Risk"


def apply_realistic_lifestyle_changes(
    patient_data: pd.Series,
    scenario_type: str
) -> pd.Series:
    """
    Apply realistic lifestyle changes based on scenario type.

    Ensures changes are medically plausible and health-improving.
    """
    modified_patient = patient_data.copy()

    if scenario_type == "Reduce cholesterol to 200":
        # Realistic cholesterol reduction through diet (typically 20-30% reduction)
        current_chol = patient_data.get('chol', 200)
        # Reduce by realistic amount (not to an arbitrary value)
        reduction = min(50, current_chol * 0.25)  # 25% reduction, max 50 points
        modified_patient['chol'] = max(120, current_chol - reduction)  # Don't go below 120

    elif scenario_type == "Increase exercise":
        # Realistic exercise improvement effects
        current_thalach = patient_data.get('thalach', 150)
        current_oldpeak = patient_data.get('oldpeak', 1.0)
        current_exang = patient_data.get('exang', 'N')

        # Increase max heart rate (better fitness)
        modified_patient['thalach'] = min(200, current_thalach + 15)  # +15 bpm realistic improvement

        # Reduce ST depression (better cardiac conditioning)
        modified_patient['oldpeak'] = max(0, current_oldpeak - 0.5)  # Reduce by 0.5

        # If they had exercise angina, it might improve with conditioning
        if current_exang == 'Y':
            # Small chance of improvement, but keep realistic
            modified_patient['exang'] = 'N'  # Assume supervised exercise program helps

    elif scenario_type == "Improve both":
        # Combine both cholesterol reduction and exercise improvement
        modified_patient = apply_realistic_lifestyle_changes(modified_patient, "Reduce cholesterol to 200")
        modified_patient = apply_realistic_lifestyle_changes(modified_patient, "Increase exercise")

    return modified_patient


def simulate_lifestyle_changes(
    patient_row: pd.Series,
    preprocessor,
    model,
    scenarios: Dict[str, Dict[str, float]],
) -> List[Dict[str, object]]:
    """
    Simulate realistic lifestyle changes and return revised risk estimates.

    Shows clear before/after comparisons with realistic health improvements.
    """
    results = []
    original_probability = float(model.predict_proba(
        preprocessor.transform(pd.DataFrame([patient_row]))
    )[:, 1][0])

    for description, adjustments in scenarios.items():
        # Apply realistic changes instead of arbitrary adjustments
        modified_patient = apply_realistic_lifestyle_changes(patient_row, description)

        # Calculate new risk
        X_modified = preprocessor.transform(pd.DataFrame([modified_patient]))
        new_probability = float(model.predict_proba(X_modified)[:, 1][0])

        # Get original values for comparison
        original_values = {}
        new_values = {}

        if "cholesterol" in description.lower():
            original_values['chol'] = patient_row.get('chol', 'Unknown')
            new_values['chol'] = modified_patient.get('chol', 'Unknown')

        if "exercise" in description.lower():
            original_values['thalach'] = patient_row.get('thalach', 'Unknown')
            original_values['oldpeak'] = patient_row.get('oldpeak', 'Unknown')
            original_values['exang'] = patient_row.get('exang', 'Unknown')
            new_values['thalach'] = modified_patient.get('thalach', 'Unknown')
            new_values['oldpeak'] = modified_patient.get('oldpeak', 'Unknown')
            new_values['exang'] = modified_patient.get('exang', 'Unknown')

        results.append({
            "scenario": description,
            "original_probability": original_probability,
            "new_probability": new_probability,
            "risk_level": risk_category(new_probability),
            "original_values": original_values,
            "new_values": new_values,
            "improvement": original_probability - new_probability  # Positive = improvement
        })

    return results


def format_lifestyle_simulation_results(results: List[Dict]) -> List[str]:
    """
    Format lifestyle simulation results into clear, readable statements.
    """
    formatted_results = []

    for result in results:
        scenario = result['scenario']
        orig_prob = result['original_probability']
        new_prob = result['new_probability']
        improvement = result['improvement']

        # Create the main comparison statement
        statement = f"If {scenario.lower()} -> risk changes from {orig_prob:.2f} to {new_prob:.2f} ({result['risk_level']})"

        # Add specific value changes if available
        details = []
        if result['original_values'] and result['new_values']:
            for key in result['original_values']:
                if key in result['new_values']:
                    orig_val = result['original_values'][key]
                    new_val = result['new_values'][key]
                    if orig_val != new_val and orig_val != 'Unknown':
                        if key == 'chol':
                            details.append(f"cholesterol: {orig_val} -> {new_val}")
                        elif key == 'thalach':
                            details.append(f"max heart rate: {orig_val} -> {new_val} bpm")
                        elif key == 'oldpeak':
                            details.append(f"ST depression: {orig_val} -> {new_val}")
                        elif key == 'exang':
                            if orig_val == 'Y' and new_val == 'N':
                                details.append("exercise angina: present -> absent")

        if details:
            statement += f" (changes: {', '.join(details)})"

        formatted_results.append(statement)

    return formatted_results
