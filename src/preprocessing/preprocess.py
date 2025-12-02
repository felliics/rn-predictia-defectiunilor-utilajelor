import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import joblib

RAW_PATH = os.path.join("data", "raw", "raw_dataset.csv")
PROCESSED_PATH = os.path.join("data", "processed", "processed_dataset.csv")
SCALER_PATH = os.path.join("models", "scaler.joblib")

def preprocess():
    # 1. Load raw data
    df = pd.read_csv(RAW_PATH)

    # 2. Drop duplicates
    df.drop_duplicates(inplace=True)

    # 3. Remove obvious outliers
    df = df[(df["temperature"] <= 150) & (df["vibration"] <= 5)]

    # 4. Scale numeric features
    feature_cols = ["temperature", "vibration", "runtime_hours", "noise_level"]
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # 5. Save processed dataset
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)

    # 6. Save scaler for later use (predictie)
    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)

    print("[INFO] Preprocessing complete. Saved to:", PROCESSED_PATH)
    print("[INFO] Scaler salvat în:", SCALER_PATH)

if __name__ == "__main__":
    preprocess()
