import gradio as gr
import pandas as pd
import pickle
import joblib
from pickle import UnpicklingError
from sklearn.pipeline import Pipeline  # ensure Pipeline class is available for unpickling

# ----------------------------------------
# Configuration / Paths
# ----------------------------------------
MODEL_PATH = "wine_model.pkl"  # ensure model is placed at project root

# ----------------------------------------
# Load model artifact
# ----------------------------------------
def load_model():
    try:
        with open(MODEL_PATH, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict) and 'model' in obj:
            return obj['model']
        return obj
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found at '{MODEL_PATH}'.")
    except (UnpicklingError, ModuleNotFoundError) as e:
        try:
            m = joblib.load(MODEL_PATH)
            if isinstance(m, dict) and 'model' in m:
                return m['model']
            return m
        except Exception:
            raise ImportError(
                f"Failed to load model with pickle ({e}).\n"
                "Re-save your trained pipeline with joblib or cloudpickle to capture all components."
            )
    except ImportError as e:
        raise ImportError(
            f"Dependency import failed when loading the model: {e}.\n"
            "Install identical package versions to those used during training."
        )

model = load_model()

# ----------------------------------------
# Prediction function using DataFrame
# ----------------------------------------
FEATURE_ORDER = [
    'fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
    'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide',
    'density', 'pH', 'sulphates', 'alcohol'
]

def predict_quality(
    fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
    chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
    density, pH, sulphates, alcohol
):
    # Build a DataFrame so ColumnTransformer picks correct features
    data = {
        'fixed_acidity': fixed_acidity,
        'volatile_acidity': volatile_acidity,
        'citric_acid': citric_acid,
        'residual_sugar': residual_sugar,
        'chlorides': chlorides,
        'free_sulfur_dioxide': free_sulfur_dioxide,
        'total_sulfur_dioxide': total_sulfur_dioxide,
        'density': density,
        'pH': pH,
        'sulphates': sulphates,
        'alcohol': alcohol
    }
    df = pd.DataFrame([data], columns=FEATURE_ORDER)
    try:
        pred = model.predict(df)[0]
        proba = model.predict_proba(df)[0]
    except Exception as e:
        return f"Error during prediction: {e}", None

    label = 'Good Quality' if pred == 1 else 'Not Good'
    confidence = proba[pred]
    return label, f"{confidence:.2%}"

# ----------------------------------------
# Gradio Interface
# ----------------------------------------
title = "Boutique Winery Wine Quality Predictor"
description = (
    "Enter the chemical properties of a red wine sample to predict if it's 'Good Quality' (rating ≥7) or 'Not Good' (<7)."
)
inputs = [
    gr.Number(value=7.4, label="Fixed Acidity"),
    gr.Number(value=0.70, label="Volatile Acidity"),
    gr.Number(value=0.00, label="Citric Acid"),
    gr.Number(value=1.9, label="Residual Sugar"),
    gr.Number(value=0.076, label="Chlorides"),
    gr.Number(value=11.0, label="Free Sulfur Dioxide"),
    gr.Number(value=34.0, label="Total Sulfur Dioxide"),
    gr.Number(value=0.9978, label="Density"),
    gr.Number(value=3.51, label="pH"),
    gr.Number(value=0.56, label="Sulphates"),
    gr.Number(value=9.4, label="Alcohol")
]
outputs = [
    gr.Textbox(label="Prediction Result"),
    gr.Textbox(label="Confidence Score")
]
# Example formatted to show values in table
examples = [
    [7.4, 0.70, 0.00, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]
]

demo = gr.Interface(
    fn=predict_quality,
    inputs=inputs,
    outputs=outputs,
    title=title,
    description=description,
    examples=examples,
    examples_per_page=1
)

if __name__ == "__main__":
    demo.launch()