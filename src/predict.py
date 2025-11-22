import joblib
import numpy as np

def load_model_and_scaler(disease):
    model = joblib.load(f'models/{disease}_model.pkl')
    scaler = joblib.load(f'models/{disease}_scaler.pkl')
    return model, scaler

def predict(disease, features):
    model, scaler = load_model_and_scaler(disease)
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)[0]
    
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(features_scaled)[0]
    else:
        proba = [0.5, 0.5]
    
    return {
        'prediction': int(prediction),
        'probability': {
            'negative': float(proba[0]),
            'positive': float(proba[1])
        }
    }