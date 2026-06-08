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

# ---------------- LOAD MODEL & SCALER (cached) ---------------- #

_model  = None
_scaler = None

def load_model():
    global _model, _scaler
    if _model is None or _scaler is None:
        with open("src/models/saved/xgboost_model.pkl", "rb") as f:
            _model = pickle.load(f)
        with open("src/models/saved/scaler.pkl", "rb") as f:
            _scaler = pickle.load(f)
    return _model, _scaler


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
        probability         : float 0.0–1.0
        probability_percent : float 0.0–100.0
        risk_level          : "Low" / "Moderate" / "High"
    """

    model, scaler = load_model()

    # engineered features — must match train_and_save.py exactly
    shock_index    = heart_rate / (systolic_bp + 1)
    pulse_pressure = systolic_bp - diastolic_bp

    # build DataFrame with correct column names — fixes sklearn feature-name warning
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

    # replace inf/nan safely
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.mean())

    # scale
    features_scaled = scaler.transform(features)

    # predict
    prediction  = int(model.predict(features_scaled)[0])
    probability = float(model.predict_proba(features_scaled)[0][1])

    # ---------------- RISK THRESHOLDS ---------------- #
    # Based on actual MIMIC-IV data analysis:
    # - Sepsis and non-sepsis vitals overlap heavily
    # - Model probability distribution is compressed (most values 0.1–0.5)
    # - Thresholds tuned to reflect clinical reality of this dataset
    if probability < 0.25:
        risk_level = "Low"
    elif probability < 0.45:
        risk_level = "Moderate"
    else:
        risk_level = "High"

    return {
        "prediction":          prediction,
        "probability":         round(probability, 4),
        "probability_percent": round(probability * 100, 1),
        "risk_level":          risk_level
    }


# ---------------- TEST WITH REAL DATA VALUES ---------------- #

if __name__ == "__main__":
    print("=== Testing with MIMIC-IV representative values ===\n")

    tests = [
        # non-sepsis mean: HR=84, SBP=119, Age=62
        ("Non-sepsis typical (Female, 48)",
         dict(heart_rate=78,  systolic_bp=122, diastolic_bp=74, mean_bp=90,
              respiratory_rate=16, spo2=98, gender=0, age=48)),

        # borderline: older male near sepsis 50th percentile
        ("Borderline (Male, 67)",
         dict(heart_rate=92,  systolic_bp=108, diastolic_bp=65, mean_bp=79,
              respiratory_rate=20, spo2=96, gender=1, age=67)),

        # sepsis 75th pct HR, 25th pct SBP, older male
        ("High risk (Male, 76)",
         dict(heart_rate=106, systolic_bp=96,  diastolic_bp=58, mean_bp=71,
              respiratory_rate=24, spo2=94, gender=1, age=76)),

        # sepsis mean values directly
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