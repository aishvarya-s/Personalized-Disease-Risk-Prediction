"""Research demo web app for the sepsis early-warning model (NOT a medical device).

Local run (college demo):   python app.py            -> http://127.0.0.1:5000
Shared on the college LAN:  HOST=0.0.0.0 python app.py
Production-style:           gunicorn -w 2 -b 0.0.0.0:5000 app:app
"""
import logging
import math
import os

from flask import Flask, request, jsonify, render_template

from predict import predict_sepsis

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("sepsis_demo")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024        # requests are tiny; reject anything large

# Plausible ranges = the same ones used to clean the training data (preprocess_sepsis3.py)
RANGES = {
    "heart_rate": (20, 250), "systolic_bp": (30, 300), "diastolic_bp": (10, 200), "mean_bp": (20, 250),
    "respiratory_rate": (4, 60), "spo2": (50, 100), "age": (18, 120),
}


def parse_inputs(data):
    """Return (clean_values, None) or (None, error_message)."""
    if not isinstance(data, dict):
        return None, "Request body must be a JSON object."
    out = {}
    for name, (lo, hi) in RANGES.items():
        if name not in data:
            return None, f"Missing field: {name}"
        try:
            v = float(data[name])
        except (TypeError, ValueError):
            return None, f"{name} must be a number."
        if not math.isfinite(v) or not (lo <= v <= hi):
            return None, f"{name} must be between {lo} and {hi}."
        out[name] = v
    try:
        g = int(data["gender"])
    except (KeyError, TypeError, ValueError):
        return None, "gender must be 0 (female) or 1 (male)."
    if g not in (0, 1):
        return None, "gender must be 0 (female) or 1 (male)."
    out["gender"] = g
    if out["systolic_bp"] < out["diastolic_bp"]:
        return None, "systolic_bp cannot be lower than diastolic_bp."
    return out, None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    values, err = parse_inputs(request.get_json(silent=True))
    if err:
        return jsonify({"success": False, "error": err}), 400
    try:
        return jsonify({"success": True, "result": predict_sepsis(**values)})
    except Exception:
        log.exception("prediction failed")                       # details go to the server log only
        return jsonify({"success": False, "error": "Prediction failed. Please try again."}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    # debug stays OFF: Flask's debug mode can let visitors run code on the server
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "5000")), debug=False)
