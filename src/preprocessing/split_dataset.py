import pandas as pd
from sklearn.model_selection import train_test_split
import os

PROCESSED_PATH = os.path.join("data", "processed", "processed_dataset.csv")
TRAIN_PATH = os.path.join("data", "train", "train.csv")
VAL_PATH = os.path.join("data", "validation", "validation.csv")
TEST_PATH = os.path.join("data", "test", "test.csv")

def split_dataset():
    df = pd.read_csv(PROCESSED_PATH)

    y = df["status"]

    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=42, stratify=y
    )

    y_temp = temp_df["status"]
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=y_temp
    )

    os.makedirs(os.path.dirname(TRAIN_PATH), exist_ok=True)
    train_df.to_csv(TRAIN_PATH, index=False)
    val_df.to_csv(VAL_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    print("[INFO] Split complete.")
    print("Train:", train_df.shape[0], "samples")
    print("Validation:", val_df.shape[0], "samples")
    print("Test:", test_df.shape[0], "samples")

if __name__ == "__main__":
    split_dataset()
