from flask import Flask, request, jsonify
import pandas as pd
import pickle

app = Flask("ChurnShield AI")

# Load model + preprocessors with error handling
try:
    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("encoder.pkl", "rb") as f:
        encoders = pickle.load(f)
    print("Model and preprocessors loaded successfully")
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    model = None
    scaler = None
    encoders = {}

features = [
    'age',
    'last_login_days',
    'monthly_fee',
    'number_of_profiles',
    'avg_watch_time_per_day'
]

# ---------------- RETENTION LOGIC ----------------
def retention_strategy(probability):
    if probability >= 0.90:
        return (
            "Critical Risk: Contact the customer immediately, "
            "offer a premium retention package or significant discount, "
            "and assign a dedicated support representative."
        )
    elif probability >= 0.75:
        return (
            "High Risk: Offer personalized discounts, free premium features, "
            "or a subscription renewal incentive."
        )
    elif probability >= 0.50:
        return (
            "Moderate Risk: Send targeted recommendations, "
            "engagement campaigns, and loyalty rewards."
        )
    elif probability >= 0.25:
        return (
            "Low Risk: Encourage continued engagement through "
            "regular communications and feature updates."
        )
    else:
        return (
            "Very Low Risk: Continue the current customer experience "
            "and monitor behavior periodically."
        )
# ---------------- HOME ROUTE ----------------
@app.route("/")
def home():
    return "Churn Prediction API is Running"

# ---------------- HEALTH CHECK ----------------
@app.route("/health", methods=["GET"])
def health():
    model_status = "loaded" if model is not None else "not loaded"
    return jsonify({
        "status": "API is running",
        "model_status": model_status
    }), 200

# ---------------- PREDICTION ROUTE ----------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500
        
        data = request.json
        if data is None:
            return jsonify({"error": "Invalid JSON format"}), 400
        
        df = pd.DataFrame([data])
        
        # Get model's expected columns
        expected_cols = model.feature_names_in_
        
        # Ensure all expected columns exist
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0
        
        # Encode categorical columns
        for col in expected_cols:
            if col in encoders and col in df.columns:
                val = df[col].iloc[0]
                if not isinstance(val, (int, float)):
                    try:
                        df[col] = encoders[col].transform(df[col])
                    except ValueError:
                        # Use first class value as fallback
                        df[col] = encoders[col].classes_[0]
                        df[col] = encoders[col].transform(df[col])
        
        # Scale numerical features
        numerical_features = [col for col in features if col in expected_cols]
        if numerical_features:
            df[numerical_features] = scaler.transform(df[numerical_features])
        
        # Make prediction using only expected columns
        X_pred = df[expected_cols].astype(float)
        prob = model.predict_proba(X_pred)[0, 1]
        pred = model.predict(X_pred)[0]
        
        churn_status = "Churn" if pred == 1 else "No Churn"
        
        return jsonify({
            "Prediction": churn_status,
            "Churn Probability": round(float(prob), 4),
            "churn_probability": round(float(prob), 4),
            "churn_prediction": int(pred),
            "Retention Strategy": retention_strategy(prob)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------- RUN SERVER ----------------
if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)