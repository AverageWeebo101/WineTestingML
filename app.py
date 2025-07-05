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
            obj = pickle.load(f)

        if isinstance(obj, dict) and 'model' in obj:
            model = obj['model']
        else:
            model = obj
        return model
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found at '{MODEL_PATH}'.")
    except (UnpicklingError, ModuleNotFoundError) as e:
        try:
            model = joblib.load(MODEL_PATH)
            if isinstance(model, dict) and 'model' in model:
                return model['model']
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

def predict_quality(**kwargs):

    feature_order = [
        'fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
        'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide',
        'density', 'pH', 'sulphates', 'alcohol'
    ]
    inputs = np.array([[kwargs[feat] for feat in feature_order]])
    try:
        pred = model.predict(inputs)[0]
        proba = model.predict_proba(inputs)[0]
    except Exception as e:
        return f"Error during prediction: {e}", None

    label = 'Good Quality' if pred == 1 else 'Not Good'
    confidence = np.max(proba)
    return label, f"{confidence:.2%}"

title = "Boutique Winery Wine Quality Predictor"
description = (
    "Enter the chemical properties of a red wine sample to predict if it's 'Good Quality' (rating ≥7) or 'Not Good' (<7)."
)


inputs = [
    gr.Number(value=7.4, label="Fixed Acidity", name="fixed_acidity"),
    gr.Number(value=0.70, label="Volatile Acidity", name="volatile_acidity"),
    gr.Number(value=0.00, label="Citric Acid", name="citric_acid"),
    gr.Number(value=1.9, label="Residual Sugar", name="residual_sugar"),
    gr.Number(value=0.076, label="Chlorides", name="chlorides"),
    gr.Number(value=11.0, label="Free Sulfur Dioxide", name="free_sulfur_dioxide"),
    gr.Number(value=34.0, label="Total Sulfur Dioxide", name="total_sulfur_dioxide"),
    gr.Number(value=0.9978, label="Density", name="density"),
    gr.Number(value=3.51, label="pH", name="pH"),
    gr.Number(value=0.56, label="Sulphates", name="sulphates"),
    gr.Number(value=9.4, label="Alcohol", name="alcohol")
]
outputs = [
    gr.Textbox(label="Prediction Result"),
    gr.Textbox(label="Confidence Score")
]

examples = [
    {
        "fixed_acidity": 7.4, "volatile_acidity": 0.70, "citric_acid": 0.00,
        "residual_sugar": 1.9, "chlorides": 0.076, "free_sulfur_dioxide": 11.0,
        "total_sulfur_dioxide": 34.0, "density": 0.9978, "pH": 3.51,
        "sulphates": 0.56, "alcohol": 9.4
    }
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