# train_model.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score

# Load preprocessed data
DATA_PATH = "data/processed_data.csv"
MODEL_PATH = "models/readmission_model.joblib"
SCALER_PATH = "models/preprocessing_pipeline.joblib"

print("🔹 Loading preprocessed data...")
df = pd.read_csv(DATA_PATH)

# Features and target
X = df.drop(columns=["readmitted_within_30_days"])
y = df["readmitted_within_30_days"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
conf_matrix = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("✅ Model training complete!")
print("\nConfusion Matrix:\n", conf_matrix)
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")

# Save model and scaler
import os
os.makedirs("models", exist_ok=True)
joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print(f"✅ Model saved to {MODEL_PATH}")
print(f"✅ Scaler saved to {SCALER_PATH}")
