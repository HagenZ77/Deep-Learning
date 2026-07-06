# Deep Learning Abgabe

## Abgabe von Alan Issa und Hagen Zastrow im Modul Deep Learning

Hinweis zu WSL:  
Das Projekt wurde mit WSL ausgeführt, damit die GPU für das Training genutzt werden konnte.  
Deshalb müssen die Pfade im Code angepasst werden.

## Skriptbeschreibung

### Allgemeine Tools

- `tools/Dataset/makeCleanDataset.py`  
  Erstellt den bereinigten Datensatz und teilt ihn in train, val und test.

- `tools/Dataset/label.py`  
  Liest die Labels ein und ordnet sie den Bildern zu.

- `tools/Dataset/labelDistribution.py`  
  Erstellt ein Diagramm zur Verteilung der Labels.

- `tools/plot.py`  
  Erstellt Diagramme aus den gespeicherten Ergebnissen.

- `tools/train.py`  
  Enthält die gemeinsame Trainingslogik für die drei Themenbereiche und vermeidet doppelte Trainingsmethoden.

- `tools/test.py`  
  Enthält die gemeinsame Testlogik für die drei Themenbereiche und vermeidet doppelte Auswertungsmethoden.

### Eigenes CNN-Modell

- `architectureDesign/train.py`  
  Trainiert und speichert das ausgewählte eigene CNN-Modell.

- `architectureDesign/test.py`  
  Lädt das finale eigene CNN-Modell und testet es auf den Testdaten.

- `architectureDesign/experiment/ExperimentRunner.py`  
  Führt die verschiedenen Testphasen für die Modelloptimierung des eigenen CNN-Modells aus.

- `architectureDesign/experiment/ModelFactory.py`  
  Erstellt die unterschiedlichen Modellvarianten des eigenen CNN-Modells.

### Transfer Learning

- `transferLearning/transferModelFactory.py`  
  Erstellt die vortrainierten Transfer-Learning-Modelle für den Modellvergleich.

- `transferLearning/transferLearningRunner.py`  
  Führt den Transfer-Learning-Modellvergleich aus.  
  Dabei werden mehrere vortrainierte Modelle geladen, der Feature Extractor eingefroren und nur der jeweilige Classifier bzw. die FC-Schicht trainiert.

- `transferLearning/fineTuningRunner.py`  
  Führt das Fine-Tuning für VGG13 durch.  
  Dabei bleibt der Classifier trainierbar und zusätzlich werden schrittweise die letzten 1 bis 5 Convolutional Layer freigegeben.  
  Jede Variante wird mit den Lernraten 0.0001 und 0.00001 getestet.

- `transferLearning/test.py`  
  Lädt das beste gespeicherte VGG13-Fine-Tuning-Modell und testet es auf den Testdaten.

### Knowledge Distillation

- `knowledgeDistillation/knowledgeDistillation.py`  
  Trainiert GoogLeNet als Student-Modell. Als Teacher wird das trainierte VGG13-Modell aus dem Transfer-Learning-Teil verwendet.

- `knowledgeDistillation/test.py`  
  Lädt das trainierte Knowledge-Distillation-Student-Modell und testet es auf den Testdaten.

- `knowledgeDistillation/withoutKnowledgeDistillation/train.py`  
  Trainiert GoogLeNet ohne Teacher als Vergleichsmodell zur Knowledge Distillation.

- `knowledgeDistillation/withoutKnowledgeDistillation/test.py`  
  Lädt das GoogLeNet-Modell ohne Teacher und testet es auf den Testdaten.

## Hinweis zum Datensatz

Die Ordner `Dataset` und `CleanDataset` wurden nicht mit auf GitHub hochgeladen, da diese zu groß für ein normales GitHub-Repository sind.

Der ursprüngliche Datensatz kann über folgenden Link heruntergeladen werden:

https://www.kaggle.com/datasets/msalman97/dataset-for-traffic-sign-master-app

Nach dem Herunterladen muss der Datensatz als Ordner `Dataset` im Projekt liegen.

Danach kann mit folgendem Skript der bereinigte Datensatz erstellt werden:

```bash
python tools/Dataset/makeCleanDataset.py
```

## Hinweis zu den Ergebnissen

Die trainierten `.pth`-Modelldateien wurden nicht mit auf GitHub hochgeladen, da diese zu groß für ein normales GitHub-Repository sind.

Alle erzeugten Ergebnisse befinden sich im jeweiligen Themenordner unter `results`.

Die Struktur ist dabei jeweils gleich aufgebaut:

```text
results/json
results/diagrams
results/models
```

In `results/json` liegen die gespeicherten Ergebnisdaten.  
In `results/diagrams` liegen Diagramme und Heatmaps.  
In `results/models` liegen die trainierten Modelle.