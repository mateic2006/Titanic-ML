Clasificare folosind Rețele Neuronale
1. Sarcina de învățare
Sarcina de învățare abordată în acest proiect este o problemă de clasificare binară. Obiectivul este de a construi un model capabil să prezică șansele de supraviețuire ale pasagerilor de pe nava Titanic, în funcție de caracteristicile lor.

2. Baza de date
Baza de date folosită este fișierul “train.csv” obținut de pe platforma Kaggle (link competiție). Acesta conține date despre supraviețuitorii din dezastrul Titanicului și este utilizat ca set de antrenare într-o competiție de predicție a supraviețuirii pasagerilor pe baza anumitor caracteristici precum vârsta, sexul și clasa.

Fișierul train.csv conține 891 de exemple, din care am folosit 4 caracteristici relevante: Vârstă, Sex, Clasa (Pclass) și un label binar Supraviețuit (0 sau 1). Fișierul conține și alte informații precum numele pasagerilor, dar acestea nu au fost utilizate în modelul meu. Datele au fost împărțite în set de antrenare (80%) și set de testare (20%) pentru evaluarea modelului.

3. Algoritmi de optimizare implementați
În cadrul proiectului, au fost implementați și evaluați doi algoritmi de optimizare: Descendent pe Pantă (SGD Clasic) și Adaptive Moment Estimation (ADAM). Ambii algoritmi au fost utilizați pentru antrenarea unei rețele neuronale cu două straturi ascunse, pentru rezolvarea sarcinii de clasificare.

3.1. SGD Clasic
Algoritmul Descendent pe Pantă este o metodă iterativă de optimizare care actualizează parametrii modelului în direcția inversă a gradientului funcției obiectiv. În cazul nostru, funcția obiectiv este funcția de pierdere (loss function) care măsoară eroarea de predicție a modelului.

3.2. ADAM
ADAM este un algoritm de optimizare bazat pe gradient, similar cu SGD, dar care utilizează medii mobile ale gradientului și pătratului gradientului pentru a accelera convergența și a oferi o performanță mai bună în cazuri complexe.

4. Rezultate și comentarii
Rezultatele obținute pentru fiecare algoritm de optimizare sunt prezentate mai jos:
| Metrica                | SGD    | ADAM   |
|------------------------|--------|--------|
| Timp de antrenare (sec)| 108.48 | 139.48 |
| Acuratețe              | 0.79   | 0.81   |
| Precizie               | 0.85   | 0.82   |
| Recall                 | 0.81   | 0.91   |
| Scor F1                | 0.835  | 0.864  |
Observăm că algoritmul ADAM a obținut rezultate superioare în termeni de acuratețe, precizie, recall și scor F1, în comparație cu SGD. Cu toate acestea, SGD a avut un timp de antrenare mai rapid.

De asemenea, scăderea funcțiilor de pierdere a fost reprezentată grafic pentru ambii algoritmi, oferind o perspectivă vizuală asupra procesului de optimizare.

În general, rezultatele obținute sunt promițătoare și demonstrează capacitatea algoritmilor de optimizare de a antrena modele de inteligență artificială pentru sarcini de clasificare.

