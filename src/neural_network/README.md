\# Modul Neural Network



Acest modul conține:



\- `train\_model.py` – definește și antrenează rețeaua neuronală MLP

\- `predict\_demo.py` – demo în linie de comandă pentru testarea modelului

\- `gui\_app.py` – interfață grafică Tkinter pentru predicția stării utilajului

\- `fonts/DejaVuSans.ttf` – font pentru generarea rapoartelor PDF cu diacritice



Modelul MLP folosește:

\- 4 features de intrare: temperature, vibration, runtime\_hours, noise\_level

\- două straturi ascunse: (16, 8)

\- activare ReLU

\- solver Adam



