# Import Flask to create the web server and handle HTTP requests
from flask import Flask, request, jsonify

# Import CORS to allow the frontend HTML page to talk to this API
from flask_cors import CORS

# Import pickle to load our pre-trained ML model from the .pkl file
import pickle

# Import pandas to create named feature arrays that match training data format
import pandas as pd

# Create the Flask application instance
app = Flask(__name__)

# Enable CORS so our frontend dashboard can make requests to this API
CORS(app)

# Load the trained model from disk when the API starts up
# This runs once at startup so every request can use the same model
with open("models/anomaly_model.pkl", "rb") as f:
    model = pickle.load(f)

# Health check endpoint — used to verify the API is running
# Example: GET http://localhost:5000/health
@app.route("/health", methods=["GET"])
def health():
    # Return a simple JSON response confirming the service is alive
    return jsonify({
        "status": "ok",
        "model": "anomaly_detector_v1"
    })

# Single transaction prediction endpoint
# Example: POST http://localhost:5000/predict
@app.route("/predict", methods=["POST"])
def predict():
    # Get the JSON data sent in the request body
    data = request.get_json()

    # Define which fields are required in every request
    required_fields = ["amount", "hour", "merchant_known", "is_weekend"]

    # Check that all required fields are present — input validation (OWASP A03)
    if not all(field in data for field in required_fields):
        # Return 400 Bad Request if any field is missing
        return jsonify({
            "error": "Missing required fields",
            "required": required_fields
        }), 400

    # Try to convert input values to correct types to prevent type errors
    try:
        # Wrap features in a DataFrame with column names to match how model was trained
        features = pd.DataFrame([[
            float(data["amount"]),
            int(data["hour"]),
            int(data["merchant_known"]),
            int(data["is_weekend"])
        ]], columns=["amount", "hour", "merchant_known", "is_weekend"])
    except (ValueError, TypeError):
        # Return 422 if values exist but are wrong type (e.g. text instead of number)
        return jsonify({"error": "Invalid field types — all fields must be numbers"}), 422

    # Run the transaction features through the ML model to get a prediction
    prediction = model.predict(features)[0]

    # Get the probability score (0.0 to 1.0) that this transaction is an anomaly
    probability = model.predict_proba(features)[0][1]

    # Assign a human-readable risk level based on the probability score
    if probability > 0.7:
        risk_level = "HIGH"
    elif probability > 0.4:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    # Return the prediction results as JSON
    return jsonify({
        "is_anomaly": bool(prediction),           # True or False
        "confidence": round(float(probability), 4), # e.g. 0.9231
        "risk_level": risk_level                   # HIGH / MEDIUM / LOW
    })

# Batch prediction endpoint — check multiple transactions at once
# Example: POST http://localhost:5000/batch-predict
@app.route("/batch-predict", methods=["POST"])
def batch_predict():
    # Get the list of transactions from the request body
    transactions = request.get_json().get("transactions", [])

    # Return error if no transactions were provided
    if not transactions:
        return jsonify({"error": "No transactions provided"}), 400

    # Store results for each transaction
    results = []

    # Loop through each transaction and run the same prediction logic
    for txn in transactions:
        try:
            # Wrap features in a DataFrame with column names to match how model was trained
            features = pd.DataFrame([[
                float(txn["amount"]),
                int(txn["hour"]),
                int(txn["merchant_known"]),
                int(txn["is_weekend"])
            ]], columns=["amount", "hour", "merchant_known", "is_weekend"])

            # Get prediction and probability
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0][1]

            # Append result for this transaction
            results.append({
                "is_anomaly": bool(prediction),
                "confidence": round(float(probability), 4),
                "risk_level": "HIGH" if probability > 0.7 else "MEDIUM" if probability > 0.4 else "LOW"
            })
        except Exception as e:
            # If one transaction fails, record the error and continue
            results.append({"error": str(e)})

    # Return all results with a total count
    return jsonify({
        "results": results,
        "total_transactions": len(results),
        "anomalies_found": sum(1 for r in results if r.get("is_anomaly"))
    })

# Start the Flask development server when this file is run directly
if __name__ == "__main__":
    # debug=True means the server auto-restarts when you save changes
    app.run(debug=True)
