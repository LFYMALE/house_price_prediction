import pickle
import pandas as pd
from flask import Flask, jsonify, request

MODEL_PATH = "model.bin"

with open(MODEL_PATH, "rb") as f_in:
    artifact = pickle.load(f_in)

model = artifact["model"]
FEATURES = artifact["features"]

app = Flask("house-price-prediction")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model": "Lasso", "alpha": artifact["alpha"]}), 200


@app.route("/predict", methods=["POST"])
def predict():
    house_data = request.get_json(silent=True)
    if not house_data:
        return jsonify({"error": "Invalid or missing JSON payload."}), 400

    # Check for missing required features
    missing_features = [feat for feat in FEATURES if feat not in house_data]
    if missing_features:
        return jsonify({
            "error": "Missing required features.",
            "missing_features": missing_features,
        }), 400

    try:
        # Build single-row DataFrame preserving exact feature column names and order
        input_df = pd.DataFrame([{feat: float(house_data[feat]) for feat in FEATURES}])
    except (ValueError, TypeError) as exc:
        return jsonify({"error": f"Invalid numeric feature value: {exc}"}), 400

    prediction = float(model.predict(input_df)[0])

    result = {
        "predicted_house_price": round(prediction, 2),
        "currency": "USD",
    }
    return jsonify(result), 200


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)