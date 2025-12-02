import os
import joblib
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import pandas as pd
from datetime import datetime
import csv
from fpdf import FPDF

MODEL_PATH = os.path.join("models", "nn_model.joblib")
SCALER_PATH = os.path.join("models", "scaler.joblib")
LOG_PATH = os.path.join("logs", "predictions_log.csv")
FEATURE_COLS = ["temperature", "vibration", "runtime_hours", "noise_level"]
FONT_PATH = os.path.join("src", "neural_network", "fonts", "DejaVuSans.ttf")


# ======================================================
#  LOAD MODEL + SCALER
# ======================================================

def load_model_and_scaler():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Modelul nu există. Rulează train_model.py mai întâi.")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("Scalerul nu există. Rulează preprocess.py mai întâi.")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


model, scaler = load_model_and_scaler()


# ======================================================
#  LOG PREDICȚII
# ======================================================

def log_prediction(temp, vib, hours, noise, pred, prob):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    file_exists = os.path.exists(LOG_PATH)

    with open(LOG_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "temperature", "vibration",
                             "runtime_hours", "noise_level",
                             "predicted_status", "probability_defect"])
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            temp, vib, hours, noise,
            int(pred),
            round(prob, 4)
        ])


# ======================================================
#  PDF REPORT (cu suport pentru diacritice)
# ======================================================

def generate_pdf_report(temp, vib, hours, noise, pred, prob):
    os.makedirs("reports", exist_ok=True)

    pdf = FPDF()
    pdf.add_page()

    if os.path.exists(FONT_PATH):
        pdf.add_font("DejaVu", "", FONT_PATH, uni=True)
        pdf.add_font("DejaVu", "B", FONT_PATH, uni=True)
        title_font = ("DejaVu", "B", 16)
        text_font = ("DejaVu", "", 12)
    else:
        # fallback dacă nu găsește fontul
        title_font = ("Arial", "B", 16)
        text_font = ("Arial", "", 12)

    pdf.set_font(*title_font)
    pdf.cell(0, 10, "Raport Predicție Utilaj Industrial", ln=True)

    pdf.set_font(*text_font)
    status_text = "DEFECT IMINENT" if pred == 1 else "UTILAJ OK"

    pdf.ln(3)
    pdf.cell(0, 8, f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(0, 8, f"Temperatură: {temp} °C", ln=True)
    pdf.cell(0, 8, f"Vibrație: {vib} g/Hz", ln=True)
    pdf.cell(0, 8, f"Ore de funcționare: {hours}", ln=True)
    pdf.cell(0, 8, f"Zgomot: {noise} dB", ln=True)

    pdf.ln(3)
    pdf.cell(0, 8, f"STATUS PREZIS: {status_text}", ln=True)
    pdf.cell(0, 8, f"Probabilitate defect: {prob * 100:.2f}%", ln=True)

    filename = datetime.now().strftime("reports/raport_%Y%m%d_%H%M%S.pdf")
    pdf.output(filename)
    print(f"[INFO] Raport PDF salvat: {filename}")


# ======================================================
#  FUNCȚIE DE PREDICȚIE
# ======================================================

def predict():
    try:
        temp = float(entry_temp.get())
        vib = float(entry_vib.get())
        hours = float(entry_hours.get())
        noise = float(entry_noise.get())
    except ValueError:
        messagebox.showerror("Eroare", "Te rog introdu doar valori numerice.")
        return

    # DataFrame cu feature-urile brute
    X_input = pd.DataFrame([{
        "temperature": temp,
        "vibration": vib,
        "runtime_hours": hours,
        "noise_level": noise
    }])

    # Aplicăm același scaler ca la antrenare
    X_scaled_np = scaler.transform(X_input)
    X_scaled = pd.DataFrame(X_scaled_np, columns=FEATURE_COLS)

    # Predicție
    pred = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0][1]

    # Log + PDF
    log_prediction(temp, vib, hours, noise, pred, prob)
    generate_pdf_report(temp, vib, hours, noise, pred, prob)

    if pred == 1:
        status_text = "⚠️ DEFECT IMINENT"
    else:
        status_text = "✅ UTILAJ OK"

    label_result.config(text=f"{status_text}\nProbabilitate defect: {prob * 100:.2f}%")


# ======================================================
#  GUI – DESIGN RESPONSIVE
# ======================================================

root = tk.Tk()
root.title("Predicția Defecțiunilor Utilajelor Industriale")
root.geometry("520x320")
root.minsize(480, 280)

# Culori & stil
root.configure(bg="#1f2933")  # dark blue-grey

style = ttk.Style()
style.theme_use("clam")

style.configure("TLabel", background="#1f2933", foreground="#e5e7eb", font=("Segoe UI", 10))
style.configure("Title.TLabel", background="#1f2933", foreground="#f9fafb", font=("Segoe UI", 14, "bold"))
style.configure("Card.TFrame", background="#111827")
style.configure("TButton", font=("Segoe UI", 10, "bold"))

# Grid principal – responsive
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=0)  # title
root.rowconfigure(1, weight=1)  # form
root.rowconfigure(2, weight=0)  # result

# Titlu sus
title_label = ttk.Label(root, text="Sistem RN – Predicția defecțiunilor utilajelor", style="Title.TLabel")
title_label.grid(row=0, column=0, pady=(10, 5), padx=10, sticky="n")

# Card central cu input-uri
card = ttk.Frame(root, style="Card.TFrame", padding=15)
card.grid(row=1, column=0, padx=15, pady=10, sticky="nsew")

# Card responsive pe orizontală
card.columnconfigure(0, weight=0)
card.columnconfigure(1, weight=1)

# Labels & Entries
lbl_temp = ttk.Label(card, text="Temperatură [°C]:")
lbl_vib = ttk.Label(card, text="Vibrație [g/Hz]:")
lbl_hours = ttk.Label(card, text="Ore de funcționare:")
lbl_noise = ttk.Label(card, text="Zgomot [dB]:")

entry_temp = ttk.Entry(card)
entry_vib = ttk.Entry(card)
entry_hours = ttk.Entry(card)
entry_noise = ttk.Entry(card)

lbl_temp.grid(row=0, column=0, sticky="e", pady=3, padx=(0, 8))
lbl_vib.grid(row=1, column=0, sticky="e", pady=3, padx=(0, 8))
lbl_hours.grid(row=2, column=0, sticky="e", pady=3, padx=(0, 8))
lbl_noise.grid(row=3, column=0, sticky="e", pady=3, padx=(0, 8))

entry_temp.grid(row=0, column=1, sticky="ew", pady=3)
entry_vib.grid(row=1, column=1, sticky="ew", pady=3)
entry_hours.grid(row=2, column=1, sticky="ew", pady=3)
entry_noise.grid(row=3, column=1, sticky="ew", pady=3)

# Buton
btn_predict = ttk.Button(card, text="Prezice starea utilajului", command=predict)
btn_predict.grid(row=4, column=0, columnspan=2, pady=(12, 4), sticky="ew")

# Label rezultat jos
label_result = ttk.Label(root, text="Introduceți valorile și apăsați butonul.", anchor="center")
label_result.grid(row=2, column=0, padx=10, pady=(5, 12), sticky="ew")

root.mainloop()
