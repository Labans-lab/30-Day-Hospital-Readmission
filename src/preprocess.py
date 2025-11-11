# preprocess.py
import pandas as pd
import numpy as np
import os

# Input and output paths
DATA_PATH = "data/diabetic_data.csv"
OUTPUT_PATH = "data/processed_data.csv"

def preprocess_data():
    print("🔹 Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    # Drop irrelevant columns
    df.drop(columns=["encounter_id", "patient_nbr"], inplace=True)

    # Replace '?' with NaN
    df.replace("?", np.nan, inplace=True)

    # Handle missing values
    df["race"].fillna("Unknown", inplace=True)

    # Simplify the target variable
    # readmitted: <30, >30, NO → 1 if <30, else 0
    df["readmitted_within_30_days"] = df["readmitted"].apply(lambda x: 1 if x == "<30" else 0)
    df.drop(columns=["readmitted"], inplace=True)

    # Convert categorical variables using one-hot encoding
    categorical_cols = df.select_dtypes(include=["object"]).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Save processed dataset
    os.makedirs("data", exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Preprocessing complete! Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    preprocess_data()
