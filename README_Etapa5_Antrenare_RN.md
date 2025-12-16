README – Etapa 5: Configurarea și Antrenarea Modelului RN
Disciplina: Rețele Neuronale
Instituție: POLITEHNICA București – FIIR
Student: Ionescu Mihai Felix
Link Repository GitHub: https://github.com/felliics/rn-predictia-defectiunilor-utilajelor
Data predării: 16.12.2025

Scopul Etapei 5
Această etapă corespunde punctului 6. Configurarea și antrenarea modelului RN din lista de 9 etape – slide 2 RN Specificații Proiect.pdf.

Obiectiv principal:
Antrenarea modelului neuronal definit în Etapa 4, evaluarea performanței pe setul de test și integrarea modelului antrenat în aplicația completă (UI).
PREREQUISITE – Verificare Etapa 4
Toate cerințele Etapei 4 au fost îndeplinite:

✔ State Machine definit și documentat
✔ Cele 3 module funcționale implementate:
    Data Logging
    RN Module
    UI (CLI + GUI simplă Tkinter)
✔ ≥ 40% date generate original în Etapa 3–4
✔ Arhitectura modelului RN definită

Pregătirea datelor pentru antrenare
    Dataset-ul final folosit pentru Etapa 5 include:
        date brute generate în Etapa 3
        date preprocesate (scalare, curățare)
        împărțire stratificată train / validation / test
        Train: 70%
        Validation: 15%
        Test: 15%
Preprocesarea este realizată prin:
    src/preprocessing/preprocess.py
    src/preprocessing/split_dataset.py
    Scaler-ul este menținut consistent pentru toate seturile.

NIVELUL 1
1. Antrenarea modelului RN
A fost utilizat modelul definit în Etapa 4:

MLPClassifier (scikit‑learn)
Modelul implementat:

model = MLPClassifier(
    hidden_layer_sizes=(16, 8),
    activation="relu",
    solver="adam",
    max_iter=300,
    random_state=42
)

Modelul a fost antrenat pe datele finale (scalate), cu funcție de pierdere implicită pentru clasificare binară.

2. Tabel Hiperparametri + Justificări
Hiperparametru	Valoare	Justificare
hidden_layer_sizes	(16, 8)	Două straturi ascunse → suficient pentru modelarea relațiilor neliniare din datele industriale
activation	ReLU	Funcție rapidă, stabilă, standard pentru RN feedforward
solver	Adam	Optimizator adaptiv potrivit pentru date zgomotoase
learning_rate_init	0.001 (implicit Adam)	Valoare standard pentru convergență rapidă
max_iter	300	Număr suficient de iterații pentru MLP pe datasetul folosit
random_state	42	Reproductibilitate 100%
batch_size	auto (implicit)	Set optim pentru MLPClassifier în scikit-learn
3. Rezultate Obținute (Test Set)
Evaluarea modelului pe setul de test:

Metrică	Valoare
Accuracy	1.00
F1-score (macro)	1.00
Precision	1.00
Recall	1.00
Modelul clasifică PERFECT toate exemplele din setul de test.

4. Salvare Model Antrenat
Modelul final se găsește în:
models/trained_model.joblib

5. Integrare în UI
UI-ul a fost modificat pentru a încărca modelul antrenat real:

model = joblib.load("models/trained_model.joblib")
Inferența este reală, nu dummy.

Screenshot inferență reală:
→ salvat în: docs/screenshots/inference_real.png

NIVEL 2 – Recomandat (85–90%)
1. Graficul funcției de pierdere
A fost generat automat prin scriptul de antrenare:
docs/plots/loss_curve.png
2. Matrice de confuzie
Generată automat în:
docs/plots/confusion_matrix_test.png

3. Analiza Erorilor (context industrial)
Deși modelul are 100% acuratețe, completăm analiza conform cerinței:

1. Pe ce clase ar putea greși modelul?
În aplicațiile industriale, confuzia apare în mod normal între:

utilaj “OK” dar cu vibrații mari → clasificabil uneori ca „defect iminent”

utilaj “DETECT IMINENT” cu temperatură normală, dar zgomot ridicat

Modelul actual NU a făcut asemenea erori pe test set, dar ele pot apărea la date noi.

2. Factori care ar putea provoca erori:
Zgomot ridicat în citirea senzorilor

Date distribuite neuniform

Corelație între vibrație și zgomot dificil de separat în limitele de decizie

3. Impact industrial:
False Negative → defect REAL nedetectat → critic

False Positive → alarmă falsă → acceptabil

4. Măsuri corective pentru un model industrial:
colectarea unor date reale suplimentare

antrenarea unui model profund (ex: MLP mai mare, CNN dacă datele devin imagini)

ajustarea pragului de decizie

balansarea datasetului prin tehnici SMOTE și augmentări controlate

NIVEL 3
Proiectul include:
ploturi (loss + confusion matrix)
UI funcțional
demo live simulation (live_simulation.py)

  Structura Repository Etapa 5

rn-predictia-defectiunilor-utilajelor/
│
├── README.md
├── README_Etapa5_Antrenare_RN.md
│
├── docs/
│   ├── plots/
│   │   ├── loss_curve.png
│   │   ├── confusion_matrix_test.png
│   └── screenshots/
│       └── inference_real.png
│
├── models/
│   └── trained_model.joblib
│
├── src/
│   ├── preprocessing/
│   │   ├── preprocess.py
│   │   └── split_dataset.py
│   ├── neural_network/
│   │   ├── train_model.py
│   │   ├── predict_demo.py
│   │   └── gui_app.py
│   └── app/
│       └── main.py   (UI actualizat)
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── train/
│   ├── validation/
│   └── test/
│
└── requirements.txt

✔ Instrucțiuni Rulare
1. Preprocesare:
python src/preprocessing/preprocess.py
python src/preprocessing/split_dataset.py
2. Antrenare model:
python src/neural_network/train_model.py
3. Predicție CLI:
python src/neural_network/predict_demo.py
4. UI:
python src/neural_network/gui_app.py

Checklist Final Etapa 5
  ✔ Model RN antrenat
  ✔ Tabel hiperparametri complet
  ✔ Metrici test set
  ✔ UI cu inferență reală
  ✔ Screenshot UI → inference_real.png
  ✔ Grafice loss & confusion matrix
  ✔ README Etapa 5 complet