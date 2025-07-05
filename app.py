import gradio as gr
import pandas as pd
import numpy as np
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
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found at '{MODEL_PATH}'.")
    except (UnpicklingError, ModuleNotFoundError) as e:
        try:
            model = joblib.load(MODEL_PATH)
            return model
        except Exception:
            raise ImportError(
                f"Failed to load model with pickle ({e}).\n"
                "Try re-saving your trained model using joblib or cloudpickle to capture custom Pipeline objects."
            )
    except ImportError as e:
        raise ImportError(
            f"Dependency import failed when loading the model: {e}.\n"
            "Ensure the same package versions used during training are installed."
        )

model = load_model()

# ----------------------------------------
# Prediction function
# ----------------------------------------
def predict_quality(
    fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
    chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
    density, pH, sulphates, alcohol
):
    inputs = np.array([[
        fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
        chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
        density, pH, sulphates, alcohol
    ]])
    try:
        pred = model.predict(inputs)[0]
        proba = model.predict_proba(inputs)[0]
    except Exception as e:
        return f"Error during prediction: {e}", None

    label = 'Good Quality' if pred == 1 else 'Not Good'
    confidence = np.max(proba)
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

demo = gr.Interface(
    fn=predict_quality,
    inputs=inputs,
    outputs=outputs,
    title=title,
    description=description,
    examples=[
        [7.4, 0.70, 0.00, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]
    ]
)

if __name__ == "__main__":
    demo.launch()