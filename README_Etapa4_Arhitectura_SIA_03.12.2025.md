📘 README – Etapa 4: Arhitectura Completă a Aplicației SIA bazată pe Rețele Neuronale

Disciplina: Rețele Neuronale
Instituție: POLITEHNICA București – FIIR
Student: Ionescu Mihai Felix
Link Repository GitHub: https://github.com/felliics/rn-predictia-defectiunilor-utilajelor

Data: 09.12.2025

Scopul Etapei 4

Această etapă corespunde punctului 5. Dezvoltarea arhitecturii aplicației software bazată pe RN din lista celor 9 etape din documentul RN Specificații Proiect.

În această etapă se livrează scheletul complet și funcțional al Sistemului cu Inteligență Artificială (SIA):

Toate modulele pornesc fără erori

Pipeline-ul rulează end-to-end

Modelul RN este definit și compilat

UI/GUI funcțional, capabil să preia input și să afișeze output

Se asigură structura completă a aplicației software

Nu este necesar un model antrenat bine sau metrici ridicate.
Scopul este arhitectura, nu performanța.

1. Tabel: Nevoie reală → Soluție SIA → Modul software
Nevoie reală concretă	Cum o rezolvă SIA-ul	Modul software responsabil
Detectarea timpurie a defectării unui utilaj industrial	RN clasifică utilajul în OK sau DEFECT IMINENT, reducând opririle neplanificate	Neural Network Module + Preprocessing
Mentenanță preventivă bazată pe date	Sistemul analizează temperatură, vibrații, zgomot și ore de funcționare → oferă alertă dacă există risc	Data Acquisition + RN Inference + UI
Necesitatea unei interfețe rapide pentru tehnicieni	GUI simplu pentru introducerea valorilor și generarea raportului PDF	Web Service / UI Module
2. Contribuția originală la setul de date (100% original)
Total observații finale: 66
Observații originale: 66 (100%)
Tip contribuție: Date generate prin simulare fizică realistă

temperatură utilaj

vibrație pe ansamblu rotativ

zgomot industrial

ore de funcționare

Descriere detaliată:

Dataset-ul a fost generat programatic pentru a simula comportamentul unui utilaj industrial real aflat în diferite stări de uzură.
Datele au fost generate folosind modele simple inspirate din literatură tehnică:

temperatură ridicată + vibrații crescute → risc de avarie

zgomot crescut → dezechilibru sau frecare excesivă

ore multe de funcționare → uzură mecanică acumulată

Pentru fiecare observație, un scor de risc a fost calculat pe baza combinației de factori.
Dacă scorul ≥ 2 → utilajul a fost etichetat ca DEFECT IMINENT.

Locație în repo:
Cod generare date: src/data_acquisition/generate_synthetic_data.py
Date originale: data/generated/generated_utilaj_sintetic.csv

3. Diagrama State Machine a Întregului Sistem (versiune complexă)
                         ┌───────────────┐
                         │     IDLE       │
                         └───────┬───────┘
                                 │ Start
                                 ▼
                     ┌────────────────────────┐
                     │   DATA_ACQUISITION     │
                     └──────────┬─────────────┘
                                │ CSV generat
                                ▼
                     ┌────────────────────────┐
                     │     VALIDATE_DATA      │
                     └───────┬───────┬────────┘
                         ok   │       │ invalid
                              ▼       ▼
                     ┌─────────────────────┐
                     │     PREPROCESS      │
                     └──────────┬──────────┘
                                │ scaled features
                                ▼
                     ┌────────────────────────┐
                     │      LOAD_MODEL        │
                     └──────────┬─────────────┘
                                │ model + scaler
                                ▼
                     ┌────────────────────────┐
                     │     RUN_INFERENCE      │
                     └──────┬─────────┬──────┘
                       OK    │         │ defect
                             ▼         ▼
                     ┌────────────────────────┐
                     │       LOG_RESULT        │
                     └──────────┬─────────────┘
                                │
                                ▼
                   ┌──────────────────────────────┐
                   │        GENERATE_REPORT        │
                   └──────────┬───────────────────┘
                                │
                             End/Loop to IDLE

Justificare:

Am ales un State Machine complex deoarece fluxul proiectului NU este liniar și simplu, ci implică:

Achiziție sau generare de date (State: DATA_ACQUISITION)

Validare pentru a preveni inferențe eronate pe date incomplete

Preprocesare (scalare, filtrare)

Încărcare model în memorie pentru inferență rapidă

Inferență RN → clasificare OK/DEFECT

Logare rezultate + generare raport PDF

Buclă de feedback – sistemul revine în IDLE

Starea ERROR a fost integrată pentru situații reale:

fișier corupt

valori lipsă

imposibilitatea încărcării modelului

Acest tip de arhitectură reflectă comportamentul sistemelor industriale reale folosite în predictive maintenance.

4. Scheletul complet al celor 3 module
Modul 1 – Data Logging / Acquisition

Folder: src/data_acquisition/

Conține:

generate_synthetic_data.py – generează date originale

produce CSV în data/generated/

rulează fără erori

Rezultat minim livrabil → OK

Modul 2 – Neural Network Module

Folder: src/neural_network/

Conține:

train_model.py – definește arhitectura RN și o compilează

predict_demo.py – inferență în terminal

gui_app.py – interfața grafică

Arhitectura RN:

MLPClassifier

input: 4 features

hidden layers: (16, 8)

activare: ReLU

solver: Adam

Modelul este definit, compilat și salvat (weights inițiali).

Modul 3 – Web Service / UI

Folder: src/app/ (sau integrat în neural_network pentru versiunea ta)

Conține:

gui_app.py – Tkinter UI

Input utilizator: temperatură, vibrație, ore funcționare, zgomot

Output: OK / DEFECT IMINENT + probabilitate

Generare PDF cu font Unicode

Screenshot inclus în
docs/screenshots/ui_demo.png

5. Structura finală a repository-ului (Etapa 4)
rn-predictia-defectiunilor-utilajelor/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── generated/
│   ├── train/
│   ├── validation/
│   └── test/
│
├── src/
│   ├── data_acquisition/
│   │     └── generate_synthetic_data.py
│   ├── preprocessing/
│   ├── neural_network/
│   │     ├── train_model.py
│   │     ├── predict_demo.py
│   │     ├── gui_app.py
│   │     └── fonts/
│   └── app/   (opțional)
│
├── docs/
│   ├── state_machine.png
│   └── screenshots/ui_demo.png
│
├── models/
│   ├── nn_model.joblib
│   └── scaler.joblib
│
├── config/
│
├── README.md
├── README_Etapa3.md
└── README_Etapa4_Arhitectura_SIA.md   ← acest fișier

6. Checklist final

 Tabel Nevoie → Soluție → Modul

 Contribuție originală 100%

 Date generate & salvate

 State Machine complet + justificare

 Modul Data Acquisition funcțional

 Modul RN funcțional

 UI funcțional + screenshot

 Structura repo completă