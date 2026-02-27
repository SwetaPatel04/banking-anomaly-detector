# 🏦 Banking Transaction Anomaly Detector

ML-powered fraud detection system with a Flask REST API and interactive 
dashboard — built with security-first engineering principles aligned with 
OWASP standards.

![Python](https://img.shields.io/badge/Python-3.14-blue?style=flat-square)
![Flask](https://img.shields.io/badge/Flask-3.0-lightgrey?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange?style=flat-square)
![Tests](https://img.shields.io/badge/Tests-8%20Passing-brightgreen?style=flat-square)
![Accuracy](https://img.shields.io/badge/Accuracy-100%25-brightgreen?style=flat-square)

---

## 📌 Project Overview

This project simulates a real-world banking fraud detection pipeline. It 
generates synthetic transaction data, trains a Random Forest classifier to 
identify anomalous transactions, exposes predictions through a REST API, 
and visualises results in an interactive dashboard.

Built to demonstrate:
- Security-focused backend engineering (OWASP A03, A07)
- ML model development and evaluation
- REST API design with input validation and error handling
- Comprehensive unit testing with pytest
- Full-stack integration (Python backend + HTML/JS frontend)

---

## 🗂️ Project Structure
```
banking-anomaly-detector/
├── data/
│   └── generate_data.py      # Synthetic transaction data generator
├── models/
│   ├── train_model.py        # ML model training and evaluation
│   └── anomaly_model.pkl     # Saved trained model
├── api/
│   └── app.py                # Flask REST API (3 endpoints)
├── static/
│   └── index.html            # Interactive frontend dashboard
├── tests/
│   └── test_api.py           # Unit tests (8 passing)
├── requirements.txt          # Python dependencies
└── README.md
```

---

## ⚙️ Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/SwetaPatel04/banking-anomaly-detector.git
cd banking-anomaly-detector
```

### 2. Create and activate virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Generate data and train the model
```bash
python data/generate_data.py
python models/train_model.py
```

### 5. Start the API
```bash
python api/app.py
```

### 6. Open the dashboard
Open `static/index.html` in your browser.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Service health check |
| POST | `/predict` | Single transaction prediction |
| POST | `/batch-predict` | Bulk transaction screening |

### Example Request
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"amount": 8500, "hour": 3, "merchant_known": 0, "is_weekend": 0}'
```

### Example Response
```json
{
  "is_anomaly": true,
  "confidence": 1.0,
  "risk_level": "HIGH"
}
```

---

## 🧪 Running Tests
```bash
pytest tests/ -v
```

Expected output:
```
test_health_check        PASSED ✅
test_anomaly_detected    PASSED ✅
test_normal_transaction  PASSED ✅
test_missing_fields      PASSED ✅
test_invalid_types       PASSED ✅
test_risk_level_valid    PASSED ✅
test_confidence_range    PASSED ✅
test_batch_predict       PASSED ✅

8 passed in 2.33s
```

---

## 🔐 Security Practices

| Practice | Implementation |
|----------|---------------|
| Input Validation | All fields validated before model inference |
| Type Safety | Explicit type casting with error handling |
| OWASP A03 | No raw queries — feature arrays only |
| OWASP A07 | Error messages never expose stack traces |
| CORS | Configured for controlled frontend access |

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 100% |
| Precision | 1.00 |
| Recall | 1.00 |
| F1 Score | 1.00 |
| Test Samples | 479 |

> High accuracy reflects clearly separated synthetic data patterns.
> Real-world deployment would use production transaction data with
> additional feature engineering.

---

## 🛠️ Tech Stack

- **Backend:** Python, Flask, Flask-CORS
- **ML:** scikit-learn (Random Forest), pandas, NumPy
- **Testing:** pytest
- **Frontend:** HTML, CSS, JavaScript (vanilla)
- **Dev Tools:** Git, VS Code, Thunder Client

---

## 👩‍💻 Author

**Sweta Patel** — Software Engineer | Python Developer | AI/ML Specialist

[![LinkedIn](https://img.shields.io/badge/LinkedIn-sweta--patel-blue?style=flat-square)](https://linkedin.com/in/sweta-patel)
[![GitHub](https://img.shields.io/badge/GitHub-SwetaPatel04-black?style=flat-square)](https://github.com/SwetaPatel04)
