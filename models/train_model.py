# Import pandas to load our CSV transaction data into a DataFrame
import pandas as pd

# Import pickle to save the trained model as a file so the API can load it later
import pickle

# Import RandomForestClassifier - our main ML algorithm for detecting anomalies
from sklearn.ensemble import RandomForestClassifier

# Import train_test_split to divide data into training set and testing set
from sklearn.model_selection import train_test_split

# Import classification_report to print accuracy, precision, recall metrics
from sklearn.metrics import classification_report

# Load the CSV file we generated in the previous step
df = pd.read_csv("transactions.csv")

# Define which columns the model will use to make predictions
# These are the "features" - the inputs to our model
FEATURES = ["amount", "hour", "merchant_known", "is_weekend"]

# X = the input features (what the model looks at to make a decision)
X = df[FEATURES]

# y = the labels (the correct answer: 1 = anomaly, 0 = normal)
y = df["is_anomaly"]

# Split data: 80% for training the model, 20% for testing how accurate it is
# random_state=42 ensures we get the same split every time we run this
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the Random Forest model with 100 decision trees
# More trees = more accurate but slower to train
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model using our training data
# It learns patterns like: high amount + odd hour + unknown merchant = anomaly
model.fit(X_train, y_train)

# Use the trained model to predict on the test data it has never seen before
predictions = model.predict(X_test)

# Print a full report showing accuracy, precision and recall for each class
print("Model Performance Report:")
print(classification_report(y_test, predictions))

# Save the trained model to a .pkl file so our Flask API can load and use it
with open("models/anomaly_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Confirm the model was saved successfully
print("Model saved to models/anomaly_model.pkl")
