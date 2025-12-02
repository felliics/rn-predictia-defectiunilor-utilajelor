import os
import time
import random
import joblib
import pandas as pd

MODEL_PATH = os.path.join("models", "nn_model.joblib")

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Modelul nu există. Rulează train_model.py mai întâi.")
    return joblib.load(MODEL_PATH)

model = load_model()

def generate_random_sample():
    temp = random.uniform(60, 120)       # °C
    vib = random.uniform(0.3, 3.5)       # g/Hz
    hours = random.uniform(500, 12000)   # ore
    noise = random.uniform(50, 90)       # dB
    return temp, vib, hours, noise

def main():
    print("=== Simulare live senzori utilaj + RN ===")
    try:
        while True:
            temp, vib, hours, noise = generate_random_sample()
            X = pd.DataFrame([{
                "temperature": temp,
                "vibration": vib,
                "runtime_hours": hours,
                "noise_level": noise
            }])
            pred = model.predict(X)[0]
            prob = model.predict_proba(X)[0][1]

            status = "DEFECT" if pred == 1 else "OK"
            print(f"T={temp:.1f}°C | V={vib:.2f} | H={hours:.0f} | N={noise:.1f}dB "
                  f"→ {status} (P_defect={prob*100:.1f}%)")

            time.sleep(1.5)
    except KeyboardInterrupt:
        print("\nSimulare oprită de utilizator.")

if __name__ == "__main__":
    main()
