import json
import pickle
import pandas as pd
import numpy as np

# ---------------- FEATURE COLUMNS (must match training order) ---------------- #

FEATURE_COLS = [
    "heart_rate",
    "systolic_bp",
    "diastolic_bp",
    "mean_bp",
    "shock_index",
    "pulse_pressure",
    "respiratory_rate",
    "spo2",
    "gender",
    "anchor_age"
]

# ---------------- LOAD MODEL, SCALER, TRAIN MEANS (cached) ---------------- #

_model  = None
_scaler = None
_means  = None
_cfg    = None

def load_model():
    global _model, _scaler, _means, _cfg
    if _model is None or _scaler is None:
        with open("src/models/saved/xgboost_model.pkl", "rb") as f:
            _model = pickle.load(f)
        with open("src/models/saved/scaler.pkl", "rb") as f:
            _scaler = pickle.load(f)
        # training-set means, saved by train_app_model.py, used to fill any missing input
        with open("src/models/saved/train_means.pkl", "rb") as f:
            _means = pickle.load(f)
        # calibration + risk cut-offs chosen on held-out patients by train_app_model.py
        with open("src/models/saved/thresholds.json") as f:
            _cfg = json.load(f)
    return _model, _scaler, _means, _cfg


# ---------------- PREDICT FUNCTION ---------------- #

def predict_sepsis(
    heart_rate: float,
    systolic_bp: float,
    diastolic_bp: float,
    mean_bp: float,
    respiratory_rate: float,
    spo2: float,
    gender: int,      # 1 = Male, 0 = Female
    age: float
) -> dict:
    """
    Predicts sepsis risk for a patient.

    Returns dict:
        prediction          : 0 or 1  (model's binary output)
        probability         : float 0.0-1.0
        probability_percent : float 0.0-100.0
        risk_level          : "Low" / "Moderate" / "High"
        relative_risk       : probability divided by the average risk in the ICU population

    probability = calibrated chance of Sepsis-3 onset within the next 6 hours.
    It is small by design: only ~0.6% of ICU hours precede sepsis onset.
    """

    model, scaler, means, cfg = load_model()

    # engineered features - must match train_app_model.py exactly
    shock_index    = heart_rate / (systolic_bp + 1)
    pulse_pressure = systolic_bp - diastolic_bp

    features = pd.DataFrame([[
        heart_rate,
        systolic_bp,
        diastolic_bp,
        mean_bp,
        shock_index,
        pulse_pressure,
        respiratory_rate,
        spo2,
        gender,
        age
    ]], columns=FEATURE_COLS)

    # fill missing values with TRAINING means
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(means)

    # scale
    features_scaled = scaler.transform(features)

    # predict + calibrate (Platt scaling fitted on held-out patients)
    raw = float(model.predict_proba(features_scaled)[0][1])
    raw = min(max(raw, 1e-6), 1 - 1e-6)
    z = cfg["platt_a"] * np.log(raw / (1 - raw)) + cfg["platt_b"]
    probability = float(1 / (1 + np.exp(-z)))

    # ---------------- RISK LEVELS ---------------- #
    # cut-offs are the top-5% (High) and top-20% (Moderate) score levels on held-out calibration patients
    if probability >= cfg["high_thr"]:
        risk_level = "High"
    elif probability >= cfg["moderate_thr"]:
        risk_level = "Moderate"
    else:
        risk_level = "Low"
    prediction = 1 if risk_level == "High" else 0

    return {
        "prediction":          prediction,
        "probability":         round(probability, 5),
        "probability_percent": round(probability * 100, 2),
        "risk_level":          risk_level,
        "relative_risk":       round(probability / cfg["base_rate"], 2)
    }


# ---------------- TEST WITH REAL DATA VALUES ---------------- #

if __name__ == "__main__":
    print("=== Testing with MIMIC-IV representative values ===\n")

    tests = [
        ("Non-sepsis typical (Female, 48)",
         dict(heart_rate=78,  systolic_bp=122, diastolic_bp=74, mean_bp=90,
              respiratory_rate=16, spo2=98, gender=0, age=48)),

        ("Borderline (Male, 67)",
         dict(heart_rate=92,  systolic_bp=108, diastolic_bp=65, mean_bp=79,
              respiratory_rate=20, spo2=96, gender=1, age=67)),

        ("High risk (Male, 76)",
         dict(heart_rate=106, systolic_bp=96,  diastolic_bp=58, mean_bp=71,
              respiratory_rate=24, spo2=94, gender=1, age=76)),

        ("Sepsis mean values (Male, 65)",
         dict(heart_rate=90,  systolic_bp=113, diastolic_bp=67, mean_bp=82,
              respiratory_rate=19, spo2=97, gender=1, age=65)),
    ]

    for label, kwargs in tests:
        result = predict_sepsis(**kwargs)
        print(f"{label}:")
        print(f"  Probability  : {result['probability_percent']}%")
        print(f"  Risk Level   : {result['risk_level']}")
        print(f"  Prediction   : {'Sepsis' if result['prediction'] == 1 else 'No Sepsis'}")
        print()