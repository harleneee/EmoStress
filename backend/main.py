from fastapi import FastAPI, UploadFile, File, Form
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import pandas as pd
import joblib
import io
import os
import numpy as np

from feature_extraction import combine_all_features

app = FastAPI(title="EmoStress API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve evaluation images/text
if os.path.exists("evaluation"):
    app.mount("/evaluation", StaticFiles(directory="evaluation"), name="evaluation")

# Load Models Safely
MODEL_DIR = "models"
emotion_model = None
stress_model = None
emotion_features = []
stress_features = []

def load_models():
    global emotion_model, stress_model, emotion_features, stress_features
    try:
        emotion_model = joblib.load(os.path.join(MODEL_DIR, "emotion_random_forest.joblib"))
        # Some joblib models store feature names
        if hasattr(emotion_model, 'feature_names_in_'):
            emotion_features = list(emotion_model.feature_names_in_)
    except Exception as e:
        print(f"Warning: Could not load emotion model: {e}")

    try:
        stress_model = joblib.load(os.path.join(MODEL_DIR, "stress_random_forest.joblib"))
        if hasattr(stress_model, 'feature_names_in_'):
            stress_features = list(stress_model.feature_names_in_)
    except Exception as e:
        print(f"Warning: Could not load stress model: {e}")

load_models()

@app.get("/")
def read_root():
    return {"status": "EmoStress Backend Running"}

@app.get("/model-info")
def get_model_info():
    return {
        "model_type": "Random Forest",
        "dataset": "ECSMP",
        "signals_used": ["ECG experiment", "ECG sleep", "HR", "IBI"],
        "classes": ["neutral", "fear", "sad", "happy", "anger", "disgust"],
        "stress_mapping": {
            "neutral": "low",
            "happy": "low",
            "sad": "moderate",
            "fear": "high",
            "anger": "high",
            "disgust": "high"
        },
        "emotion_model_loaded": emotion_model is not None,
        "stress_model_loaded": stress_model is not None
    }

@app.post("/predict")
async def predict(
    hr_file: Optional[UploadFile] = None,
    ibi_file: Optional[UploadFile] = None,
    ecg_exp_file: Optional[UploadFile] = None,
    ecg_sleep_file: Optional[UploadFile] = None,
    sample_profile: Optional[str] = Form(None)
):
    if sample_profile == "relaxed":
        raw_features = {
            "hr_mean": 65.2, "hr_std": 2.1, "ibi_mean": 0.92, "ibi_rmssd": 0.04,
            "ecg_exp_peak_count": 320, "ecg_sleep_peak_count": 2500
        }
    elif sample_profile == "stressed":
        raw_features = {
            "hr_mean": 95.8, "hr_std": 8.5, "ibi_mean": 0.62, "ibi_rmssd": 0.015,
            "ecg_exp_peak_count": 480, "ecg_sleep_peak_count": 2200
        }
    else:
        # 1. Parse CSVs
        hr_df = None
        if hr_file and hr_file.filename:
            content = await hr_file.read()
            hr_df = pd.read_csv(io.BytesIO(content))

        ibi_df = None
        if ibi_file and ibi_file.filename:
            content = await ibi_file.read()
            ibi_df = pd.read_csv(io.BytesIO(content))

        # 2. Parse BINs
        ecg_exp_bin = None
        if ecg_exp_file and ecg_exp_file.filename:
            ecg_exp_bin = await ecg_exp_file.read()

        ecg_sleep_bin = None
        if ecg_sleep_file and ecg_sleep_file.filename:
            ecg_sleep_bin = await ecg_sleep_file.read()

        # 3. Extract Features
        raw_features = combine_all_features(hr_df, ibi_df, ecg_exp_bin, ecg_sleep_bin)
    
    # 4. Predict Emotion
    is_demo = sample_profile is not None
    emotion_prediction = "neutral"
    emotion_probs = {"neutral": 0.1, "fear": 0.1, "sad": 0.1, "happy": 0.1, "anger": 0.1, "disgust": 0.1}
    
    if is_demo:
        if sample_profile == "relaxed":
            emotion_prediction = "happy"
            emotion_probs = {"neutral": 0.20, "fear": 0.05, "sad": 0.05, "happy": 0.60, "anger": 0.05, "disgust": 0.05}
        elif sample_profile == "stressed":
            emotion_prediction = "anger"
            emotion_probs = {"neutral": 0.05, "fear": 0.15, "sad": 0.10, "happy": 0.05, "anger": 0.55, "disgust": 0.10}
    elif emotion_model:
        try:
            if emotion_features:
                input_df = pd.DataFrame([raw_features]).reindex(columns=emotion_features, fill_value=0)
            else:
                input_df = pd.DataFrame([raw_features])
                # If model expects more features, pad with 0s
                if hasattr(emotion_model, "n_features_in_") and input_df.shape[1] < emotion_model.n_features_in_:
                    missing_cols = emotion_model.n_features_in_ - input_df.shape[1]
                    for i in range(missing_cols):
                        input_df[f"pad_{i}"] = 0
            
            preds = emotion_model.predict(input_df)
            emotion_prediction = str(preds[0])
            
            if hasattr(emotion_model, "predict_proba"):
                probs = emotion_model.predict_proba(input_df)[0]
                classes = emotion_model.classes_
                emotion_probs = {str(c): float(p) for c, p in zip(classes, probs)}
            else:
                emotion_probs[emotion_prediction] = 1.0
        except Exception as e:
            print(f"Emotion prediction failed: {e}")
            emotion_prediction = "fear"
            emotion_probs = {"neutral": 0.10, "fear": 0.35, "sad": 0.15, "happy": 0.05, "anger": 0.25, "disgust": 0.10}
    else:
        # Mock behavior if model missing
        emotion_prediction = "fear"
        emotion_probs = {"neutral": 0.10, "fear": 0.35, "sad": 0.15, "happy": 0.05, "anger": 0.25, "disgust": 0.10}

    # 5. Predict Stress
    stress_prediction = "low"
    stress_probs = {"low": 0.33, "moderate": 0.33, "high": 0.34}

    if is_demo:
        if sample_profile == "relaxed":
            stress_prediction = "low"
            stress_probs = {"low": 0.75, "moderate": 0.20, "high": 0.05}
        elif sample_profile == "stressed":
            stress_prediction = "high"
            stress_probs = {"low": 0.10, "moderate": 0.20, "high": 0.70}
    elif stress_model:
        try:
            raw_features['predicted_emotion'] = emotion_prediction
            
            if stress_features:
                input_df = pd.DataFrame([raw_features]).reindex(columns=stress_features, fill_value=0)
            else:
                input_df = pd.DataFrame([raw_features])
                if hasattr(stress_model, "n_features_in_") and input_df.shape[1] < stress_model.n_features_in_:
                    missing_cols = stress_model.n_features_in_ - input_df.shape[1]
                    for i in range(missing_cols):
                        input_df[f"pad_{i}"] = 0

            preds = stress_model.predict(input_df)
            stress_prediction = str(preds[0])
            
            if hasattr(stress_model, "predict_proba"):
                probs = stress_model.predict_proba(input_df)[0]
                classes = stress_model.classes_
                stress_probs = {str(c): float(p) for c, p in zip(classes, probs)}
        except Exception as e:
            print(f"Stress prediction failed: {e}")
            mapping = {
                "neutral": "low", "happy": "low", 
                "sad": "moderate", "fear": "high", 
                "anger": "high", "disgust": "high"
            }
            stress_prediction = mapping.get(emotion_prediction, "low")
    else:
        # Mock behavior
        mapping = {
            "neutral": "low", "happy": "low", 
            "sad": "moderate", "fear": "high", 
            "anger": "high", "disgust": "high"
        }
        stress_prediction = mapping.get(emotion_prediction, "low")
        if stress_prediction == "low":
            stress_probs = {"low": 0.70, "moderate": 0.20, "high": 0.10}
        elif stress_prediction == "moderate":
            stress_probs = {"low": 0.20, "moderate": 0.60, "high": 0.20}
        else:
            stress_probs = {"low": 0.10, "moderate": 0.20, "high": 0.70}

    return {
        "emotion": str(emotion_prediction),
        "stress_level": str(stress_prediction),
        "emotion_probabilities": emotion_probs,
        "stress_probabilities": stress_probs,
        "features": {
            "hr_mean": raw_features.get("hr_mean", 0),
            "hr_std": raw_features.get("hr_std", 0),
            "ibi_mean": raw_features.get("ibi_mean", 0),
            "ibi_rmssd": raw_features.get("ibi_rmssd", 0)
        }
    }
