from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from predict import predict_sepsis

app = Flask(__name__)
CORS(app)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        result = predict_sepsis(
            heart_rate=float(data["heart_rate"]),
            systolic_bp=float(data["systolic_bp"]),
            diastolic_bp=float(data["diastolic_bp"]),
            mean_bp=float(data["mean_bp"]),
            respiratory_rate=float(data["respiratory_rate"]),
            spo2=float(data["spo2"]),
            gender=int(data["gender"]),
            age=float(data["age"])
        )

        return jsonify({"success": True, "result": result})

    except KeyError as e:
        return jsonify({"success": False, "error": f"Missing field: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)