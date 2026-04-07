#!/usr/bin/env python3
"""
Quick test of the heart disease prediction system
"""

import sys
import os
os.chdir('src')
sys.path.append('.')

import pandas as pd
from preprocessing import load_data, split_dataset
from persistence import load_pipeline, pipeline_exists
from explainability import compute_shap_explanation
from recommendations import recommend_from_patient_data
from simulation import simulate_lifestyle_changes, format_lifestyle_simulation_results

def quick_test():
    """Quick test of the system components"""
    print("Testing Heart Disease Prediction System...")
    print("="*50)

    # Load data
    data_path = "../data/heart (1).csv"
    df = load_data(data_path)
    X_train, X_test, y_train, y_test = split_dataset(df)

    # Load model
    pipeline_path = "saved_models/ensemble_pipeline.joblib"
    if not pipeline_exists(pipeline_path):
        print("ERROR: Model pipeline not found. Please run training first.")
        return

    persisted = load_pipeline(pipeline_path)
    preprocessor = persisted["preprocessor"]
    ensemble_model = persisted["ensemble_model"]

    # Test with sample patient
    sample = X_test.iloc[0]
    X_sample = preprocessor.transform(pd.DataFrame([sample]))

    probability = float(ensemble_model.predict_proba(X_sample)[:, 1][0])
    risk_level = "High Risk" if probability > 0.7 else "Moderate Risk" if probability > 0.3 else "Low Risk"

    print(f"Sample Patient Risk: {probability:.1%} ({risk_level})")

    # Test SHAP
    feature_names = []
    for name, transformer, columns in preprocessor.transformers_:
        names = transformer[-1].get_feature_names_out(columns)
        feature_names.extend(names)

    shap_df, _ = compute_shap_explanation(ensemble_model, preprocessor.transform(X_train), feature_names)
    print(f"SHAP Analysis: Generated {len(shap_df)} explanations")

    # Test recommendations
    recommendations = recommend_from_patient_data(sample)
    print(f"Recommendations: {len(recommendations)} condition-based suggestions")

    # Test simulation
    scenarios = {"Test": {"chol": 200}}
    results = simulate_lifestyle_changes(sample, preprocessor, ensemble_model, scenarios)
    print(f"Lifestyle Simulation: {len(results)} scenarios tested")

    print("="*50)
    print("SUCCESS: All system components working correctly!")
    print("You can now run: cd src && python predict.py")

if __name__ == "__main__":
    quick_test()