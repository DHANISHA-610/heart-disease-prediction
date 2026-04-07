"""
Module to convert SHAP feature attributes into readable, understandable format
for patients and healthcare professionals.
"""

# Feature name mapping: Technical -> Readable
FEATURE_READABLE_NAMES = {
    # Age and Gender
    'age': 'Patient Age (Years)',
    'sex_M': 'Male Gender',
    'sex_F': 'Female Gender',
    
    # Blood Pressure
    'trestbps': 'Resting Blood Pressure (mmHg)',
    
    # Cholesterol
    'chol': 'Cholesterol Level (mg/dL)',
    
    # Chest Pain Types
    'cp_ASY': 'Asymptomatic (No Chest Pain)',
    'cp_ATA': 'Atypical Angina (Unusual Chest Pain)',
    'cp_NAP': 'Non-Anginal Pain (Not Heart-Related Pain)',
    'cp_TA': 'Typical Angina (Classic Heart Pain)',
    
    # Blood Sugar
    'fbs': 'Fasting Blood Sugar',
    'fbs_0': 'Normal Blood Sugar (≤120 mg/dl)',
    'fbs_1': 'High Blood Sugar (>120 mg/dl)',
    
    # ECG Results
    'restecg_Normal': 'Normal Resting ECG',
    'restecg_ST': 'ST-T Wave Abnormality on ECG',
    'restecg_LVH': 'Left Ventricular Hypertrophy (Enlarged Heart Chamber)',
    
    # Heart Rate
    'thalach': 'Maximum Heart Rate Achieved',
    
    # Exercise Effects
    'exang_N': 'No Exercise-Induced Chest Pain',
    'exang_Y': 'Exercise-Induced Chest Pain Present',
    
    # ST Depression (Exercise stress indicator)
    'oldpeak': 'ST Depression During Exercise',
}

# Impact interpretation
IMPACT_DESCRIPTIONS = {
    'increases_risk_very_strong': {
        'symbol': '⬆️ INCREASES RISK',
        'color': 'RED',
        'meaning': 'This factor significantly raises heart disease risk',
        'action': 'Requires immediate medical attention and lifestyle changes'
    },
    'increases_risk_strong': {
        'symbol': '⬆️ INCREASES RISK',
        'color': 'ORANGE',
        'meaning': 'This factor notably increases heart disease risk',
        'action': 'Should be managed and monitored'
    },
    'increases_risk_moderate': {
        'symbol': '⬆️ Slightly Increases Risk',
        'color': 'YELLOW',
        'meaning': 'This factor mildly increases heart disease risk',
        'action': 'Should be included in prevention plan'
    },
    'increases_risk_minimal': {
        'symbol': '➡️ Minimal Effect',
        'color': 'LIGHT GRAY',
        'meaning': 'This factor has very little impact on risk',
        'action': 'Monitor but not a priority'
    },
    'decreases_risk_strong': {
        'symbol': '⬇️ DECREASES RISK',
        'color': 'GREEN',
        'meaning': 'This is a positive protective factor',
        'action': 'Continue maintaining this condition'
    },
    'decreases_risk_moderate': {
        'symbol': '⬇️ Decreases Risk',
        'color': 'LIGHT GREEN',
        'meaning': 'This is somewhat protective against heart disease',
        'action': 'Good sign; maintain current status'
    },
}

# Clinical translations
CLINICAL_TRANSLATIONS = {
    'sex_F': {
        'readable': 'Female Gender',
        'simple_explanation': 'Being female affects heart disease risk patterns',
        'clinical_detail': 'Women often have atypical heart disease symptoms and different risk presentations',
        'patient_action': 'Women should be especially vigilant about any chest discomfort'
    },
    'age': {
        'readable': 'Patient Age',
        'simple_explanation': 'Older age increases heart disease risk',
        'clinical_detail': 'Each year of age adds cumulative risk due to arterial aging and disease progression',
        'patient_action': 'Regular health screenings become more important as you age'
    },
    'sex_M': {
        'readable': 'Male Gender',
        'simple_explanation': 'Being male affects heart disease risk',
        'clinical_detail': 'Men typically have higher baseline heart disease risk, especially before age 60',
        'patient_action': 'Men should maintain regular check-ups and healthy lifestyle'
    },
    'trestbps': {
        'readable': 'Resting Blood Pressure',
        'simple_explanation': 'Higher blood pressure increases heart strain',
        'clinical_detail': 'Sustained high BP (hypertension) damages blood vessels and increases heart workload',
        'patient_action': 'Monitor blood pressure regularly; manage with medication/lifestyle if elevated'
    },
    'cp_NAP': {
        'readable': 'Non-Anginal Chest Pain',
        'simple_explanation': 'Chest pain not caused by the heart',
        'clinical_detail': 'Could be musculoskeletal, gastrointestinal, or anxiety-related',
        'patient_action': 'Still requires evaluation to rule out cardiac causes'
    },
    'cp_TA': {
        'readable': 'Typical Angina (Classic Heart Pain)',
        'simple_explanation': 'Chest pain triggered by exertion, relieved by rest',
        'clinical_detail': 'Classic presentation of reduced blood flow to heart during stress',
        'patient_action': 'Requires immediate medical evaluation and possible stress testing'
    },
    'restecg_ST': {
        'readable': 'ST-T Wave Abnormality',
        'simple_explanation': 'ECG shows electrical changes in the heart',
        'clinical_detail': 'May indicate past heart damage, current ischemia, or other cardiac stress',
        'patient_action': 'Requires cardiology consultation and further testing'
    },
    'restecg_LVH': {
        'readable': 'Left Ventricular Hypertrophy',
        'simple_explanation': 'Enlarged main heart pumping chamber',
        'clinical_detail': 'Usually caused by chronic high blood pressure; indicates heart muscle thickening',
        'patient_action': 'Requires blood pressure management and cardiac follow-up'
    },
    'fbs_1': {
        'readable': 'High Fasting Blood Sugar',
        'simple_explanation': 'Blood sugar elevated when fasting (12+ hours)',
        'clinical_detail': 'Indicates prediabetes or diabetes; major cardiovascular risk factor',
        'patient_action': 'Diet modification, exercise, and possible medication needed'
    },
    'cp_ATA': {
        'readable': 'Atypical Angina',
        'simple_explanation': 'Chest pain that is unusual or not classic in pattern',
        'clinical_detail': 'May be harder to diagnose but can still indicate serious cardiac issues',
        'patient_action': 'Requires careful evaluation to determine if cardiac in origin'
    },
}


def format_shap_attribute_readable(feature_name: str, shap_value: float) -> dict:
    """
    Convert a technical SHAP feature into readable format.
    
    Returns a dictionary with:
    - readable_name: Patient-friendly feature name
    - shap_value: Numeric contribution
    - impact_direction: Whether it increases or decreases risk
    - interpretation: Simple explanation of what this means
    - clinical_context: Medical explanation
    - recommendation: What patient should do
    """
    
    # Get readable name
    readable_name = FEATURE_READABLE_NAMES.get(feature_name, feature_name)
    
    # Determine impact direction and strength
    abs_value = abs(shap_value)
    
    if shap_value > 0:  # Increases risk
        if abs_value > 0.010:
            impact = 'increases_risk_very_strong'
            strength_level = 'VERY STRONG'
        elif abs_value > 0.005:
            impact = 'increases_risk_strong'
            strength_level = 'STRONG'
        elif abs_value > 0.001:
            impact = 'increases_risk_moderate'
            strength_level = 'MODERATE'
        else:
            impact = 'increases_risk_minimal'
            strength_level = 'MINIMAL'
    else:  # Decreases risk
        if abs_value > 0.005:
            impact = 'decreases_risk_strong'
            strength_level = 'STRONG'
        else:
            impact = 'decreases_risk_moderate'
            strength_level = 'MODERATE'
    
    impact_info = IMPACT_DESCRIPTIONS[impact]
    clinical_info = CLINICAL_TRANSLATIONS.get(
        feature_name,
        {'readable': readable_name, 'simple_explanation': '', 'clinical_detail': '', 'patient_action': ''}
    )
    
    return {
        'feature_code': feature_name,
        'readable_name': readable_name,
        'shap_value': shap_value,
        'absolute_impact': abs_value,
        'impact_symbol': impact_info['symbol'],
        'impact_strength': strength_level,
        'impact_meaning': impact_info['meaning'],
        'recommended_action': impact_info['action'],
        'simple_explanation': clinical_info.get('simple_explanation', ''),
        'clinical_detail': clinical_info.get('clinical_detail', ''),
        'patient_action': clinical_info.get('patient_action', ''),
    }


def print_readable_shap_analysis(features_dict: dict):
    """
    Print SHAP analysis in a highly readable format for patients and professionals.
    
    Args:
        features_dict: Dictionary mapping feature codes to SHAP values
    """
    print("\n" + "="*80)
    print("HEART DISEASE RISK ANALYSIS - READABLE FORMAT")
    print("="*80 + "\n")
    
    # Separate into risk-increasing and risk-decreasing
    risk_increasing = []
    risk_decreasing = []
    
    for feature_name, shap_value in features_dict.items():
        readable = format_shap_attribute_readable(feature_name, shap_value)
        if shap_value > 0:
            risk_increasing.append(readable)
        else:
            risk_decreasing.append(readable)
    
    # Sort by absolute impact
    risk_increasing.sort(key=lambda x: x['absolute_impact'], reverse=True)
    risk_decreasing.sort(key=lambda x: x['absolute_impact'], reverse=True)
    
    # Print risk-increasing factors
    if risk_increasing:
        print("🔴 FACTORS THAT INCREASE HEART DISEASE RISK:")
        print("-" * 80)
        for i, item in enumerate(risk_increasing, 1):
            print(f"\n{i}. {item['impact_symbol']} {item['readable_name']}")
            print(f"   Strength: {item['impact_strength']} | Impact: {item['shap_value']:.6f}")
            print(f"   ➜ {item['simple_explanation']}")
            print(f"   🏥 {item['clinical_detail']}")
            print(f"   ⚕️ {item['patient_action']}")
    
    # Print risk-decreasing factors
    if risk_decreasing:
        print("\n" + "="*80)
        print("🟢 FACTORS THAT DECREASE HEART DISEASE RISK (PROTECTIVE):")
        print("-" * 80)
        for i, item in enumerate(risk_decreasing, 1):
            print(f"\n{i}. {item['impact_symbol']} {item['readable_name']}")
            print(f"   Strength: {item['impact_strength']} | Impact: {item['shap_value']:.6f}")
            print(f"   ➜ {item['simple_explanation']}")
            print(f"   🏥 {item['clinical_detail']}")
            print(f"   ⚕️ {item['patient_action']}")
    
    print("\n" + "="*80)


# Example usage
if __name__ == "__main__":
    # Example SHAP values
    example_data = {
        'sex_F': 0.015231,
        'age': 0.014607,
        'sex_M': 0.005255,
        'trestbps': 0.004143,
        'cp_NAP': 0.002013,
        'cp_TA': 0.000131,
        'restecg_ST': -0.004431,
        'restecg_LVH': -0.007482,
    }
    
    print_readable_shap_analysis(example_data)
    
    # Print individual analysis
    print("\n\n" + "="*80)
    print("DETAILED FEATURE ANALYSIS")
    print("="*80)
    for feature, value in example_data.items():
        result = format_shap_attribute_readable(feature, value)
        print(f"\n{result['readable_name']}:")
        print(f"  Technical Code: {result['feature_code']}")
        print(f"  SHAP Value: {result['shap_value']:.6f}")
        print(f"  {result['impact_symbol']} ({result['impact_strength']})")
        print(f"  Meaning: {result['impact_meaning']}")
        print(f"  Action: {result['recommended_action']}")
