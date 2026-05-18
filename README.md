# Deep Learning Abgabe

## Abgabe von Alan Issa und Hagen Zastrow im Modul Deep Learning.

Hinweis zu WSL:
Das Projekt wurde mit WSL ausgeführt, damit die GPU für das Training genutzt werden konnte.
Deshalb müssen die Pfade im Code angepasst werden.


## Skriptbeschreibung

- `train.py`  
  Trainiert und speichert das ausgewählte Modell.

- `EvaluateTestModel.py`  
  Lädt das finale Modell und testet es auf den Testdaten.

- `Experiment/ExperimentRunner.py`  
  Führt die verschiedenen Testphasen für die Modelloptimierung aus.

- `Experiment/ModelFactory.py`  
  Erstellt die unterschiedlichen Modellvarianten.

- `tools/makeCleanDataset.py`  
  Erstellt den bereinigten Datensatz und teilt ihn in train, val und test.

- `tools/label.py`  
  Liest die Labels ein und ordnet sie den Bildern zu.

- `tools/labelDistribution.py`  
  Erstellt ein Diagramm zur Verteilung der Labels.

- `tools/plot.py`  
  Erstellt Diagramme aus den gespeicherten Ergebnissen.

## Hinweis zum Datensatz

Die Ordner Dataset und CleanDataset wurden nicht mit auf GitHub hochgeladen, da diese zu groß für ein normales GitHub-Repository sind.

Der ursprüngliche Datensatz kann über folgenden Link heruntergeladen werden:

https://www.kaggle.com/datasets/msalman97/dataset-for-traffic-sign-master-app

Nach dem Herunterladen muss der Datensatz als Ordner Dataset im Projekt liegen.

Danach kann mit folgendem Skript der bereinigte Datensatz erstellt werden:

```bash
python tools/makeCleanDataset.py
```

## Hinweis zu den Modellen

Die trainierten `.pth`-Modelldateien wurden nicht mit auf GitHub hochgeladen, da diese zu groß für ein normales GitHub-Repository sind.

Das finale Modell kann mit folgendem Skript neu trainiert und gespeichert werden:

```bash
python train.py
```