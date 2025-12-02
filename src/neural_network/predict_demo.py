import os
import joblib
import numpy as np
import pandas as pd

MODEL_PATH = os.path.join("models", "nn_model.joblib")
SCALER_PATH = os.path.join("models", "scaler.joblib")

FEATURE_COLS = ["temperature", "vibration", "runtime_hours", "noise_level"]


def load_model_and_scaler():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modelul nu a fost găsit la {MODEL_PATH}. Rulează întâi train_model.py")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scalerul nu a fost găsit la {SCALER_PATH}. Rulează preprocess.py")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def main():
    model, scaler = load_model_and_scaler()
    print("=== Demo Predicție Defecțiuni Utilaj Industrial ===")

    try:
        temp = float(input("Temperatură [°C]: "))
        vib = float(input("Vibrație [g/Hz]: "))
        hours = float(input("Ore de funcționare: "))
        noise = float(input("Zgomot [dB]: "))
    except ValueError:
        print("Valori introduse greșit. Repornește programul.")
        return

    # Construim DataFrame cu feature-urile brute
    X_input = pd.DataFrame([{
        "temperature": temp,
        "vibration": vib,
        "runtime_hours": hours,
        "noise_level": noise
    }])

    # Aplicăm ACELAȘI scaler ca la antrenare
    X_scaled = scaler.transform(X_input)

    # Predicție
    pred = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0][1]  # probabilitatea clasei 1 (defect)

    print("\n=== Rezultat ===")
    if pred == 1:
        print("⚠️  STATUS: DEFECT IMINENT")
    else:
        print("✅ STATUS: UTILAJ OK")

    print(f"Probabilitate defect: {prob * 100:.2f}%")

if __name__ == "__main__":
    main()
