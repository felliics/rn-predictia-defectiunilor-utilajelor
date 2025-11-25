import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

RAW_PATH = os.path.join("data", "raw", "raw_dataset.csv")
PROCESSED_PATH = os.path.join("data", "processed", "processed_dataset.csv")

def preprocess():
    df = pd.read_csv(RAW_PATH)

    df.drop_duplicates(inplace=True)

    df = df[(df["temperature"] <= 150) & (df["vibration"] <= 5)]

    feature_cols = ["temperature", "vibration", "runtime_hours", "noise_level"]
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)

    print("[INFO] Preprocessing complete. Saved to:", PROCESSED_PATH)

if __name__ == "__main__":
    preprocess()
