# Explainable Ensemble AI System for Early Heart Disease Risk Prediction

This project implements an explainable ensemble machine learning pipeline for early heart disease risk prediction, risk stratification, lifestyle impact simulation, and personalized preventive recommendations.

## Project Goals
- Predict whether a patient is at risk of heart disease
- Categorize risk level as Low / Moderate / High
- Explain why the prediction was made using SHAP
- Simulate lifestyle changes and quantify risk reduction
- Generate preventive recommendations

## Project Structure
- `data/`: dataset files and instructions
- `notebooks/`: exploration and analysis notebooks
- `src/`: reusable pipeline modules and prediction script
- `requirements.txt`: Python dependencies

## Installation
```bash
pip install -r requirements.txt
```

## Workflow
1. Load and preprocess clinical data
2. Train base models (Logistic Regression, Random Forest, Gradient Boosting, XGBoost)
3. Build an ensemble predictor
4. Stratify risk into Low / Moderate / High
5. Generate SHAP explanations
6. Simulate lifestyle improvements
7. Create personalized prevention advice

## Dataset
Place your heart disease dataset in `data/heart_disease.csv`. The dataset should include the following columns:
- `age`
- `sex`
- `cp`
- `trestbps`
- `chol`
- `fbs`
- `restecg`
- `thalach`
- `exang`
- `oldpeak`
- `target`

## Run prediction example
```bash
python src/predict.py
```
