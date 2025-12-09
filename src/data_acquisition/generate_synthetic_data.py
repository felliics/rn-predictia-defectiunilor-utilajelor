import pandas as pd
import numpy as np
import os

OUTPUT_PATH = os.path.join("data", "generated", "generated_utilaj_sintetic.csv")

def generate_data(n_samples=50, random_state=42):
    rng = np.random.default_rng(random_state)

    temperature = rng.uniform(50, 130, size=n_samples)
    vibration = rng.uniform(0.2, 4.5, size=n_samples)
    runtime_hours = rng.uniform(500, 15000, size=n_samples)
    noise_level = rng.uniform(50, 95, size=n_samples)

    status = []
    for t, v, h, n in zip(temperature, vibration, runtime_hours, noise_level):
        risk_score = 0
        if t > 100: risk_score += 1
        if v > 3: risk_score += 1
        if h > 10000: risk_score += 1
        if n > 80: risk_score += 1
        status.append(1 if risk_score >= 2 else 0)

    df = pd.DataFrame({
        "temperature": temperature,
        "vibration": vibration,
        "runtime_hours": runtime_hours,
        "noise_level": noise_level,
        "status": status,
    })

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"[INFO] Am generat {n_samples} mostre în {OUTPUT_PATH}")

if __name__ == "__main__":
    generate_data()
