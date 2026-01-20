README – Etapa 6: Analiza Performanței, Optimizare și Concluzii
Disciplina: Rețele Neuronale
Instituție: POLITEHNICA București – FIIR
Student: Ionescu Mihai Felix
Link Repository GitHub: https://github.com/felliics/rn-predictia-defectiunilor-utilajelor
Data predării: Ianuarie 2026

Scopul Etapei 6
Această etapă corespunde finalizării ciclului de dezvoltare al Sistemului cu Inteligență Artificială (SIA) pentru predicția defecțiunilor utilajelor industriale.

Obiective principale:
Optimizarea modelului RN antrenat în Etapa 5
Analiza detaliată a performanței (confusion matrix, exemple greșite)
Evaluarea implicațiilor industriale ale erorilor
Finalizarea aplicației software complete (UI + inferență reală)

Înainte de Etapa 6, proiectul îndeplinește următoarele condiții:
 Model RN antrenat (Accuracy ≥ 65%, F1 ≥ 0.60)
 UI funcțional cu inferență reală
 Confusion matrix disponibilă
 Loss curve disponibilă
 Model salvat și încărcat din fișier (.joblib)
Descriere Generală a Aplicației Finale

Aplicația software dezvoltată este un sistem de predicție a defecțiunilor utilajelor industriale, care:
Primește date numerice reale:
Temperatură
Vibrație
Ore de funcționare
Nivel de zgomot
Aplică același preprocessing ca în etapa de antrenare
Utilizează un model RN optimizat

Returnează:
Starea utilajului (OK / Defect iminent)
Probabilitatea defectului

Salvează:
Log CSV cu predicții
Raport PDF automat pentru fiecare inferență
Model Optimizat – Etapa 6
Model utilizat

Tip model: Rețea neuronală (clasificare binară)
Framework: Scikit-learn
Fișier model:models/optimized_model.joblib
Scaler utilizat:models/scaler.joblib
Modelul optimizat este încărcat direct în aplicația GUI și folosit pentru inferență reală.

Experimente de Optimizare 
Tabelul de mai jos sintetizează experimentele realizate pentru optimizarea performanței modelului:

ID	Modificare aplicată	      Accuracy	F1-score	Observații
E1	Creștere număr estimatori	0.74	0.71	     Overfitting redus
E2	Ajustare max_depth	        0.76	0.73	   Generalizare mai bună
E3	Class weights           	0.78    0.75	  Reducere false negatives
E4	Threshold probabilitate 0.4	0.77	0.76	    Alarmare mai sigură

Toate experimentele sunt documentate în:
results/optimization_experiments.csv
Analiza Confusion Matrix (Model Optimizat)

Fișier:
docs/confusion_matrix_optimized.png

Observații principale:
Modelul identifică corect majoritatea cazurilor de defect iminent

False positives sunt acceptabile în context industrial

False negatives sunt reduse față de Etapa 5

Analiza a 5 Exemple Greșite
Exemplul 1
Temperatură normală, vibrație ridicată

Modelul a prezis „OK”

Cauză: semnal ambiguu, vibrație temporară

Exemplul 2
Zgomot ridicat, restul parametrilor normali
Predicție greșită
Cauză: lipsă date similare în dataset

Exemplul 3
Ore de funcționare foarte mari
Modelul a subestimat riscul
Necesită date suplimentare pentru uzură avansată

Exemplul 4
Temperatură + vibrație crescute
Predicție corectă după optimizare

Exemplul 5
Valori limită pentru toate feature-urile
Modelul oscilează între clase

Implicații Industriale ale Erorilor
False Negatives: CRITICE
→ pot conduce la defectarea utilajului și oprirea producției

False Positives: ACCEPTABILE
→ duc la verificări suplimentare, dar nu produc avarii

Prioritate: Minimizarea false negatives
Soluție aplicată: Ajustarea pragului de probabilitate și class weights

Modificări Aplicație Software față de Etapa 5
Componentă	Etapa 5	Etapa 6
Model	trained_model.joblib	optimized_model.joblib
UI	Predicție simplă	Predicție + PDF + log
Logging	Nu	CSV automat
Raportare	Nu	PDF generat automat
UX	Minimal	UI modern, feedback clar
Demonstrație Funcțională
Screenshot inferență reală:

docs/screenshots/inference_optimized.png
Aplicația demonstrează funcționare completă end-to-end:
Date reale → RN optimizat → decizie → raport

Concluzii Finale
În urma parcurgerii tuturor etapelor proiectului:
A fost dezvoltat un Sistem cu Inteligență Artificială funcțional

Modelul RN este:
Antrenat
Optimizat
Integrat într-o aplicație reală

Aplicația poate fi utilizată ca:
Sistem de avertizare timpurie
Instrument de suport decizional industrial
Lecții Învățate
Importanța calității datelor
Impactul direct al false negatives în industrie
Necesitatea analizelor post-antrenare
Diferența dintre un model academic și unul utilizabil real

Structura Finală a Repository-ului (Etapa 6)
models/
 ├── optimized_model.joblib
 ├── scaler.joblib
results/
 ├── optimization_experiments.csv
 ├── final_metrics.json
docs/
 ├── confusion_matrix_optimized.png
 ├── screenshots/inference_optimized.png

Livrabile Finale
 README Etapa 6 complet

 Model optimizat

 Experimente documentate

 Confusion matrix analizată

 UI final funcțional