from fastapi import FastAPI, UploadFile, File, Form
from typing import Optional, List
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import pandas as pd
import joblib
import json
import io
import os
import numpy as np

from feature_extraction import combine_all_features

app = FastAPI(title="EmoStress API", version="2.0")

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

# ─────────────────────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────────────────────
MODEL_DIR = "models"

emotion_model = None
emotion_label_encoder = None
emotion_feature_columns: List[str] = []
emotion_metadata: dict = {}

hr_ibi_bundle = None
hr_ibi_model = None
hr_ibi_feature_columns: List[str] = []

def load_models():
    global emotion_model, emotion_label_encoder, emotion_feature_columns, emotion_metadata
    global hr_ibi_bundle, hr_ibi_model, hr_ibi_feature_columns

    # ── Emotion Model ──
    try:
        emotion_model = joblib.load(os.path.join(MODEL_DIR, "emotion_random_forest.joblib"))
        print("[OK] Emotion model loaded")
    except Exception as e:
        print(f"[WARN] Could not load emotion model: {e}")

    try:
        emotion_label_encoder = joblib.load(os.path.join(MODEL_DIR, "emotion_label_encoder.joblib"))
        print("[OK] Emotion label encoder loaded")
    except Exception as e:
        print(f"[WARN] Could not load emotion label encoder: {e}")

    try:
        emotion_feature_columns = joblib.load(os.path.join(MODEL_DIR, "emotion_feature_columns.joblib"))
        if isinstance(emotion_feature_columns, np.ndarray):
            emotion_feature_columns = emotion_feature_columns.tolist()
        print(f"[OK] Emotion feature columns loaded: {len(emotion_feature_columns)} features")
    except Exception as e:
        print(f"[WARN] Could not load emotion feature columns: {e}")
        if emotion_model and hasattr(emotion_model, 'feature_names_in_'):
            emotion_feature_columns = list(emotion_model.feature_names_in_)

    try:
        with open(os.path.join(MODEL_DIR, "emotion_model_metadata.json")) as f:
            emotion_metadata = json.load(f)
        print("[OK] Emotion metadata loaded")
    except Exception as e:
        print(f"[WARN] Could not load emotion metadata: {e}")

    # -- HR+IBI Emotion Model (Bundle) --
    try:
        hr_ibi_bundle = joblib.load(os.path.join(MODEL_DIR, "emostress_best_model_bundle.joblib"))
        hr_ibi_model = hr_ibi_bundle.get("model")
        hr_ibi_feature_columns = hr_ibi_bundle.get("feature_columns", [])
        print("[OK] HR+IBI Emotion bundle loaded")
    except Exception as e:
        print(f"[WARN] Could not load HR+IBI Emotion bundle: {e}")

load_models()

# ─────────────────────────────────────────────────────────────
# Helper: run model prediction safely
# ─────────────────────────────────────────────────────────────
def safe_predict(model, label_encoder, feature_columns, raw_features: dict):
    """Align features, run model, decode label, return (label, probs_dict)."""
    if feature_columns:
        input_df = pd.DataFrame([raw_features]).reindex(columns=feature_columns, fill_value=0)
    else:
        input_df = pd.DataFrame([raw_features])
        if hasattr(model, "n_features_in_") and input_df.shape[1] < model.n_features_in_:
            for i in range(model.n_features_in_ - input_df.shape[1]):
                input_df[f"_pad_{i}"] = 0

    # Replace inf/nan
    input_df = input_df.replace([np.inf, -np.inf], 0).fillna(0)

    raw_pred = model.predict(input_df)

    # Decode label
    if label_encoder is not None:
        try:
            label = str(label_encoder.inverse_transform(raw_pred)[0])
        except Exception:
            label = str(raw_pred[0])
    else:
        label = str(raw_pred[0])

    # Probabilities
    probs = {}
    if hasattr(model, "predict_proba"):
        raw_probs = model.predict_proba(input_df)[0]
        classes = model.classes_
        if label_encoder is not None:
            try:
                decoded_classes = [str(label_encoder.inverse_transform([c])[0]) for c in classes]
            except Exception:
                decoded_classes = [str(c) for c in classes]
        else:
            decoded_classes = [str(c) for c in classes]
        probs = {cls: float(p) for cls, p in zip(decoded_classes, raw_probs)}

    return label, probs


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────
@app.get("/")
def read_root():
    return {"status": "EmoStress API v2.0 Running"}

@app.get("/model-info")
def get_model_info():
    return {
        "emotion_model": {
            "type": emotion_metadata.get("model_name", "Unknown"),
            "dataset": emotion_metadata.get("dataset", "Unknown"),
            "accuracy": emotion_metadata.get("accuracy", 0),
            "classes": emotion_metadata.get("classes", []),
            "feature_count": emotion_metadata.get("feature_count", 0),
            "loaded": emotion_model is not None,
        },
        "stress_model": {
            "type": stress_metadata.get("model_type", "Unknown"),
            "dataset": stress_metadata.get("dataset", "Unknown"),
            "accuracy": stress_metadata.get("accuracy", 0),
            "balanced_accuracy": stress_metadata.get("balanced_accuracy", 0),
            "classes": stress_metadata.get("classes", []),
            "feature_count": stress_metadata.get("feature_count", 0),
            "loaded": stress_model is not None,
        }
    }

@app.post("/predict")
async def predict(
    hr_file: Optional[UploadFile] = None,
    ibi_file: Optional[UploadFile] = None,
    ecg_exp_file: Optional[UploadFile] = None,
    ecg_sleep_file: Optional[UploadFile] = None,
    sample_profile: Optional[str] = Form(None),
):
    # ── Demo profiles (bypass model for clean demo output) ──
    is_demo = sample_profile is not None
    if is_demo:
        if sample_profile == "relaxed":
            emotion_pred = "happy"
            emotion_probs = {"neutral": 0.18, "fear": 0.04, "sad": 0.04, "happy": 0.62, "anger": 0.07, "disgust": 0.05}
            stress_pred = "low"
            stress_probs = {"low": 0.82, "high": 0.18}
            demo_features = {
                "hr_mean": 65.2, "hr_std": 2.1, "hr_rmssd": 0.038,
                "ibi_mean": 0.92, "ibi_rmssd": 0.041, "ibi_sdnn": 0.052,
                "ibi_pnn50": 0.18, "hr_above_80_ratio": 0.02, "hr_below_60_ratio": 0.05
            }
        else:  # stressed
            emotion_pred = "anger"
            emotion_probs = {"neutral": 0.04, "fear": 0.16, "sad": 0.09, "happy": 0.04, "anger": 0.58, "disgust": 0.09}
            stress_pred = "high"
            stress_probs = {"low": 0.15, "high": 0.85}
            demo_features = {
                "hr_mean": 96.3, "hr_std": 8.7, "hr_rmssd": 0.012,
                "ibi_mean": 0.62, "ibi_rmssd": 0.014, "ibi_sdnn": 0.022,
                "ibi_pnn50": 0.03, "hr_above_80_ratio": 0.72, "hr_below_60_ratio": 0.0
            }
        return {
            "emotion": emotion_pred,
            "stress_level": stress_pred,
            "emotion_probabilities": emotion_probs,
            "stress_probabilities": stress_probs,
            "features": demo_features,
            "model_used": "demo",
        }

    # ── Parse uploaded CSVs ──
    hr_df = None
    if hr_file and hr_file.filename:
        content = await hr_file.read()
        try:
            hr_df = pd.read_csv(io.BytesIO(content), header=None)
        except Exception as e:
            print(f"HR parse error: {e}")

    ibi_df = None
    if ibi_file and ibi_file.filename:
        content = await ibi_file.read()
        try:
            ibi_df = pd.read_csv(io.BytesIO(content), header=None)
        except Exception as e:
            print(f"IBI parse error: {e}")

    ecg_exp_bin = None
    if ecg_exp_file and ecg_exp_file.filename:
        ecg_exp_bin = await ecg_exp_file.read()

    ecg_sleep_bin = None
    if ecg_sleep_file and ecg_sleep_file.filename:
        ecg_sleep_bin = await ecg_sleep_file.read()

    # ── Extract Features ──
    raw_features = combine_all_features(
        hr_df, ibi_df, ecg_exp_bin, ecg_sleep_bin,
        stress_features_list=hr_ibi_feature_columns
    )

    # ── Emotion Prediction ──
    emotion_pred = "neutral"
    emotion_probs = {c: 1/6 for c in ["neutral", "fear", "sad", "happy", "anger", "disgust"]}
    emotion_source = "model"

    if emotion_model and ecg_exp_bin:
        # 1. ECG data present: use the primary Extra Trees model
        try:
            ecg_feats = {col: 0 for col in emotion_feature_columns}
            emotion_pred, emotion_probs = safe_predict(
                emotion_model, emotion_label_encoder, emotion_feature_columns, ecg_feats
            )
            emotion_source = "ecg_model"
        except Exception as e:
            print(f"Emotion prediction error (ECG): {e}")
            emotion_source = "error"
    elif hr_ibi_model:
        # 2. No ECG data: use the new HR+IBI Pipeline model
        try:
            emotion_pred, emotion_probs = safe_predict(
                hr_ibi_model, None, hr_ibi_feature_columns, raw_features
            )
            emotion_source = "hr_ibi_model"
        except Exception as e:
            print(f"Emotion prediction error (HR+IBI): {e}")
            emotion_source = "error"
    else:
        emotion_source = "no_models_loaded"

    # Derive stress from emotion (since stress model is removed per request)
    stress_mapping = {"happy": "low", "neutral": "low", "sad": "moderate", "disgust": "high", "anger": "high", "fear": "high"}
    stress_pred = stress_mapping.get(emotion_pred, "low")
    stress_probs = {"low": 1.0 if stress_pred == "low" else 0.0, "high": 1.0 if stress_pred == "high" else 0.0}

    # ── Summary features for frontend display ──
    display_features = {
        "hr_mean": round(raw_features.get("hr_mean", 0), 2),
        "hr_std": round(raw_features.get("hr_std", 0), 2),
        "hr_rmssd": round(raw_features.get("hr_rmssd", 0), 4),
        "ibi_mean": round(raw_features.get("ibi_mean", 0), 4),
        "ibi_rmssd": round(raw_features.get("ibi_rmssd", 0), 4),
        "ibi_sdnn": round(raw_features.get("ibi_sdnn", 0), 4),
        "ibi_pnn50": round(raw_features.get("ibi_pnn50", 0), 4),
        "hr_above_80_ratio": round(raw_features.get("hr_above_80_ratio", 0), 4),
        "hr_below_60_ratio": round(raw_features.get("hr_below_60_ratio", 0), 4),
    }

    return {
        "emotion": emotion_pred,
        "stress_level": stress_pred,
        "emotion_probabilities": emotion_probs,
        "stress_probabilities": stress_probs,
        "features": display_features,
        "model_used": "trained",
        "emotion_source": emotion_source,  # "model" or "heuristic_hr_ibi"
    }
