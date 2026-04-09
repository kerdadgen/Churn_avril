import pickle
from flask import Flask, request, jsonify
import pandas as pd

MODEL_PATH = "churn_regression_log.pkl"
FEATURE_NAMES = ["Age", "Total_Purchase", "Years", "Num_Sites"]

app = Flask(__name__)

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    model = None
    print(f"Warning: model file '{MODEL_PATH}' not found. Please exécutez train.py d'abord.")


def validate_input(json_data):
    if not isinstance(json_data, dict):
        return False, "Le payload doit être un objet JSON."

    missing = [name for name in FEATURE_NAMES if name not in json_data]
    if missing:
        return False, f"Champs manquants: {', '.join(missing)}"

    values = []
    for name in FEATURE_NAMES:
        value = json_data[name]
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            return False, f"La valeur pour '{name}' doit être numérique."

    return True, values


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "model_loaded": model is not None})


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Modèle non chargé. Exécutez train.py pour générer churn_regression_log.pkl."}), 500

    request_data = request.get_json(silent=True)
    is_valid, result = validate_input(request_data)
    if not is_valid:
        return jsonify({"error": result}), 400

    feature_values = [result]
    prediction = model.predict(feature_values)
    probability = model.predict_proba(feature_values)[0][1] if hasattr(model, "predict_proba") else None

    return jsonify({
        "prediction": int(prediction[0]),
        "churn_probability": float(probability) if probability is not None else None,
        "features": dict(zip(FEATURE_NAMES, result)),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
