# Import pytest - the testing framework we use to run all our tests
import pytest

# Import sys and os to help Python find our api folder
import sys
import os

# Add the root project folder to Python's path so we can import app.py
# Without this, Python wouldn't know where to find our Flask app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import our Flask app so we can test it directly without running the server
from api.app import app

# This is a pytest fixture — it creates a test client before each test runs
# The test client lets us send fake HTTP requests without starting a real server
@pytest.fixture
def client():
    # Enable testing mode so Flask shows errors clearly
    app.config["TESTING"] = True
    # Return the test client to be used in each test function
    return app.test_client()

# Test 1 — Check that the health endpoint returns 200 OK
def test_health_check(client):
    # Send a GET request to /health
    res = client.get("/health")
    # Assert the status code is 200 (success)
    assert res.status_code == 200
    # Assert the response contains status: ok
    assert res.json["status"] == "ok"

# Test 2 — Check that a clearly fraudulent transaction is flagged correctly
def test_anomaly_detected(client):
    # Send a suspicious transaction: high amount, odd hour, unknown merchant
    res = client.post("/predict", json={
        "amount": 8500,
        "hour": 3,
        "merchant_known": 0,
        "is_weekend": 0
    })
    # Assert the request was successful
    assert res.status_code == 200
    # Assert the model correctly identified it as an anomaly
    assert res.json["is_anomaly"] == True
    # Assert the risk level is HIGH for this obvious fraud
    assert res.json["risk_level"] == "HIGH"

# Test 3 — Check that a normal everyday transaction is not flagged
def test_normal_transaction(client):
    # Send a normal transaction: small amount, daytime, known merchant
    res = client.post("/predict", json={
        "amount": 45.50,
        "hour": 14,
        "merchant_known": 1,
        "is_weekend": 0
    })
    # Assert the request was successful
    assert res.status_code == 200
    # Assert the model correctly identified it as normal
    assert res.json["is_anomaly"] == False

# Test 4 — Check that missing fields return a 400 Bad Request error
def test_missing_fields(client):
    # Send an incomplete request with only amount
    res = client.post("/predict", json={"amount": 100})
    # Assert the API correctly rejects incomplete input
    assert res.status_code == 400
    # Assert the error message is present
    assert "error" in res.json

# Test 5 — Check that wrong data types return a 422 error
def test_invalid_types(client):
    # Send text strings instead of numbers
    res = client.post("/predict", json={
        "amount": "abc",
        "hour": "x",
        "merchant_known": 1,
        "is_weekend": 0
    })
    # Assert the API correctly rejects invalid data types
    assert res.status_code == 422

# Test 6 — Check that risk_level is always one of the three valid values
def test_risk_level_valid(client):
    # Send a normal transaction
    res = client.post("/predict", json={
        "amount": 45.50,
        "hour": 14,
        "merchant_known": 1,
        "is_weekend": 0
    })
    # Assert risk_level is one of the three expected values
    assert res.json["risk_level"] in ["LOW", "MEDIUM", "HIGH"]

# Test 7 — Check that confidence score is between 0 and 1
def test_confidence_range(client):
    # Send any valid transaction
    res = client.post("/predict", json={
        "amount": 200,
        "hour": 10,
        "merchant_known": 1,
        "is_weekend": 1
    })
    # Assert confidence is a valid probability between 0.0 and 1.0
    assert 0.0 <= res.json["confidence"] <= 1.0

# Test 8 — Check that batch predict works with multiple transactions
def test_batch_predict(client):
    # Send two transactions at once
    res = client.post("/batch-predict", json={
        "transactions": [
            {"amount": 8500, "hour": 3, "merchant_known": 0, "is_weekend": 0},
            {"amount": 45.50, "hour": 14, "merchant_known": 1, "is_weekend": 0}
        ]
    })
    # Assert the request was successful
    assert res.status_code == 200
    # Assert we got results back for both transactions
    assert res.json["total_transactions"] == 2
    # Assert at least one anomaly was found (the first transaction)
    assert res.json["anomalies_found"] >= 1