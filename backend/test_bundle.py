import joblib
import pandas as pd
import numpy as np

bundle = joblib.load('models/emostress_best_model_bundle.joblib')
model = bundle['model']
fc = bundle['feature_columns']

# Create dummy input
dummy = pd.DataFrame([np.zeros(len(fc))], columns=fc)

pred = model.predict(dummy)[0]
probs = model.predict_proba(dummy)[0]

print("Prediction:", pred)
print("Type:", type(pred))
print("Classes:", model.classes_)
print("Probs:", probs)
