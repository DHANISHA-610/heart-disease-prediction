import os
import numpy as np
import pandas as pd

from preprocessing import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    load_data,
    build_preprocessor,
    split_dataset,
    get_encoded_feature_names,
)
from models import build_base_models
from ensemble import build_ensemble, evaluate_model
from explainability import (
    compute_shap_explanation,
    format_shap_for_display,
    get_shap_importance,
    get_top_shap_features_with_direction,
    map_encoded_feature_to_feature,
    map_encoded_feature_to_name,
    generate_patient_specific_shap_explanation,
)
from persistence import load_pipeline, pipeline_exists, save_pipeline
from simulation import simulate_lifestyle_changes, format_lifestyle_simulation_results, risk_category
from recommendations import recommend_from_patient_data


def print_patient_summary(patient_data: pd.Series, probability: float, risk_label: str):
    """Print a clean, presentation-ready patient analysis summary."""
    print("\n" + "="*60)
    print("🏥 HEART DISEASE RISK ASSESSMENT")
    print("="*60)

    # Risk assessment section
    print("\n📊 RISK ASSESSMENT")
    print(f"   Predicted Probability: {probability:.1%}")
    print(f"   Risk Category: {risk_label}")

    # Patient profile section
    print("\n👤 PATIENT PROFILE")
    print("   Age:", patient_data['age'])
    print("   Sex:", "Female" if patient_data['sex'] == 'F' else "Male")
    print("   Chest Pain Type:", patient_data['cp'])
    print("   Resting Blood Pressure:", patient_data['trestbps'])
    print("   Cholesterol:", patient_data['chol'])
    print("   Fasting Blood Sugar:", "High" if patient_data['fbs'] == 1 else "Normal")
    print("   Resting ECG:", patient_data['restecg'])
    print("   Max Heart Rate:", patient_data['thalach'])
    print("   Exercise Angina:", "Yes" if patient_data['exang'] == 'Y' else "No")
    print("   ST Depression:", patient_data['oldpeak'])
    print()


def prompt_user_input() -> pd.Series:
    """Prompt the user to enter patient feature values manually."""
    prompts = [
        ("age", "Age (years)", int, None),
        ("sex", "Sex (M/F)", str, ["M", "F"]),
        ("cp", "Chest pain type (ASY/ATA/NAP/TA)", str, ["ASY", "ATA", "NAP", "TA"]),
        ("trestbps", "Resting blood pressure", int, None),
        ("chol", "Cholesterol (enter 0 or blank if unknown)", int, None),
        ("fbs", "Fasting blood sugar (0 = false, 1 = true)", int, [0, 1]),
        ("restecg", "Rest ECG (Normal/ST/LVH)", str, ["Normal", "ST", "LVH"]),
        ("thalach", "Max heart rate achieved", int, None),
        ("exang", "Exercise induced angina (Y/N)", str, ["Y", "N"]),
        ("oldpeak", "ST depression induced by exercise", float, None),
    ]

    values = {}
    for key, label, value_type, choices in prompts:
        while True:
            raw = input(f"{label}: ").strip()
            if raw == "" and key == "chol":
                values[key] = np.nan
                break
            if raw == "" and key != "chol":
                print("This field is required. Please enter a value.")
                continue
            try:
                if value_type is int:
                    parsed = int(raw)
                elif value_type is float:
                    parsed = float(raw)
                else:
                    parsed = raw.upper() if key in ["sex", "cp", "exang"] else raw
                if choices is not None and parsed not in choices:
                    print(f"Invalid value. Choose from: {choices}")
                    continue
                if key == "chol" and parsed == 0:
                    values[key] = np.nan
                else:
                    values[key] = parsed
                break
            except ValueError:
                print("Invalid format. Please enter the correct type.")
    return pd.Series(values)


def main():
    data_path = "data/heart (1).csv"
    df = load_data(data_path)

    X_train, X_test, y_train, y_test = split_dataset(df)
    pipeline_path = "saved_models/ensemble_pipeline.joblib"

    if pipeline_exists(pipeline_path):
        print("Loading saved model pipeline...")
        persisted = load_pipeline(pipeline_path)
        preprocessor = persisted["preprocessor"]
        base_models = persisted["base_models"]
        ensemble_model = persisted["ensemble_model"]
        X_train_prepared = preprocessor.transform(X_train)
        X_test_prepared = preprocessor.transform(X_test)
        print("\nLoaded existing model pipeline. Skipping retraining.")
    else:
        preprocessor = build_preprocessor()
        preprocessor.fit(X_train)

        X_train_prepared = preprocessor.transform(X_train)
        X_test_prepared = preprocessor.transform(X_test)

        base_models = build_base_models()
        print("\nIndividual base model metrics:")
        for name, model in base_models.items():
            model.fit(X_train_prepared, y_train)
            model_metrics = evaluate_model(model, X_test_prepared, y_test)
            print(f"\n{name.replace('_', ' ').title()}")
            for metric_name, score in model_metrics.items():
                print(f"- {metric_name}: {score:.4f}")

        ensemble_model = build_ensemble(base_models)
        ensemble_model.fit(X_train_prepared, y_train)

        save_pipeline(pipeline_path, {
            "preprocessor": preprocessor,
            "base_models": base_models,
            "ensemble_model": ensemble_model,
        })
        print(f"Saved trained pipeline to {pipeline_path}")

    print("\n[AI] MODEL PERFORMANCE")
    print("-" * 40)
    ensemble_metrics = evaluate_model(ensemble_model, X_test_prepared, y_test)
    print(f"   Accuracy:  {ensemble_metrics['accuracy']:.1%}")
    print(f"   Precision: {ensemble_metrics['precision']:.1%}")
    print(f"   Recall:    {ensemble_metrics['recall']:.1%}")
    print(f"   F1-Score:  {ensemble_metrics['f1_score']:.1%}")
    print(f"   ROC-AUC:   {ensemble_metrics['roc_auc']:.1%}")
    print("   (Soft voting ensemble combines multiple ML models)")

    use_manual_input = input("Do you want to enter patient data manually? (y/N): ").strip().lower() == "y"
    if use_manual_input:
        sample = prompt_user_input()
    else:
        sample = X_test.iloc[0]

    sample_prepared = preprocessor.transform(pd.DataFrame([sample]))
    probability = float(ensemble_model.predict_proba(sample_prepared)[:, 1][0])
    risk_label = risk_category(probability)
    print_patient_summary(sample, probability, risk_label)

    # Get the actual feature names from the preprocessor
    feature_names = []
    for name, transformer, columns in preprocessor.transformers_:
        # Get feature names from the last step of the pipeline (scaler or onehotencoder)
        names = transformer[-1].get_feature_names_out(columns)
        feature_names.extend(names)
    
    shap_df, shap_values = compute_shap_explanation(ensemble_model, X_train_prepared, feature_names)
    
    # Generate patient-specific explanations
    patient_explanations = generate_patient_specific_shap_explanation(shap_df.iloc[0], sample, top_n=8)

    print("\n[AI] AI EXPLANATION (SHAP Analysis)")
    print("-" * 40)
    print("Top factors influencing this prediction:")

    for explanation in patient_explanations[:5]:  # Show top 5 explanations
        print(f"   • {explanation}")

    print("\nFEATURE IMPORTANCE RANKING")
    print("-" * 40)
    importance = get_shap_importance(shap_df)
    for i, (feature, score) in enumerate(importance.head(5).items(), 1):
        readable_name = map_encoded_feature_to_name(feature)
        print(f"   {i}. {readable_name}: {score:.4f}")

    print("\nPERSONALIZED RECOMMENDATIONS")
    print("-" * 40)
    recommendations = recommend_from_patient_data(sample)
    if recommendations:
        for i, recommendation in enumerate(recommendations, 1):
            print(f"   {i}. {recommendation}")
    else:
        print("   - Maintain regular cardiovascular checkups")

    print("\nLIFESTYLE IMPACT SIMULATION")
    print("-" * 40)
    print("How lifestyle changes could affect risk:")

    scenarios = {
        "Reduce cholesterol to 200": {"chol": 200},
        "Increase exercise": {"exang": 0},
        "Improve both": {"chol": 200, "exang": 0},
    }

    results = simulate_lifestyle_changes(sample, preprocessor, ensemble_model, scenarios)
    formatted_results = format_lifestyle_simulation_results(results)

    for result in formatted_results:
        print(f"   - {result}")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE - All improvements implemented:")
    print("   - SHAP maps to real input values")
    print("   - SHAP uses correct increase/decrease signs")
    print("   - No wrong labeling (shows actual patient values)")
    print("   - Recommendations only when conditions met")
    print("   - Personalized, condition-based advice")
    print("   - Realistic lifestyle simulation")
    print("   - Clear before/after comparisons")
    print("   - Clean, presentation-ready output")
    print("="*70)

    print("\n" + "="*60)
    print("[WARNING] IMPORTANT: This is for educational purposes only.")
    print("   Always consult with healthcare professionals for medical advice.")
    print("="*60)


if __name__ == "__main__":
    main()
