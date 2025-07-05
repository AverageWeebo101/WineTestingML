import gradio as gr
import pandas as pd
import numpy as np
import pickle
import joblib
from pickle import UnpicklingError
from sklearn.pipeline import Pipeline  

MODEL_PATH = "wine_model.pkl"  

def load_model():

    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found at '{MODEL_PATH}'.")
    except UnpicklingError:

        try:
            model = joblib.load(MODEL_PATH)
            return model
        except Exception:
            raise ValueError(
                "Invalid model file format. Ensure it's a valid pickle or joblib file without special characters in its name."
            )
    except ImportError as e:
        raise ImportError(
            f"Dependency import failed when loading the model: {e}.\n"
            "Ensure the same package versions used during training (e.g. scikit-learn==1.6.1) are installed."
        )

model = load_model()

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
        return f"Error: {e}", None

    label = 'Good Quality' if pred == 1 else 'Not Good'
    confidence = np.max(proba)
    return label, f"{confidence:.2%}"

title = "Boutique Winery Wine Quality Predictor"
description = (
    "Enter the chemical properties of a red wine sample to predict if it's 'Good Quality' (rating ≥7) or 'Not Good' (<7)."
)

demo = gr.Interface(
    fn=predict_quality,
    inputs=[
        gr.inputs.Number(default=7.4, label="Fixed Acidity"),
        gr.inputs.Number(default=0.70, label="Volatile Acidity"),
        gr.inputs.Number(default=0.00, label="Citric Acid"),
        gr.inputs.Number(default=1.9, label="Residual Sugar"),
        gr.inputs.Number(default=0.076, label="Chlorides"),
        gr.inputs.Number(default=11.0, label="Free Sulfur Dioxide"),
        gr.inputs.Number(default=34.0, label="Total Sulfur Dioxide"),
        gr.inputs.Number(default=0.9978, label="Density"),
        gr.inputs.Number(default=3.51, label="pH"),
        gr.inputs.Number(default=0.56, label="Sulphates"),
        gr.inputs.Number(default=9.4, label="Alcohol")
    ],
    outputs=[
        gr.outputs.Textbox(label="Prediction Result"),
        gr.outputs.Textbox(label="Confidence Score")
    ],
    title=title,
    description=description,
    examples=[
        [7.4, 0.70, 0.00, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]
    ]
)

if __name__ == "__main__":
    demo.launch()
