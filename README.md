# Deep Learning Abgabe

## Abgabe von Alan Issa und Hagen Zastrow im Modul Deep Learning

Hinweis zu WSL:  
Das Projekt wurde mit WSL ausgeführt, damit die GPU für das Training genutzt werden konnte.  
Deshalb müssen die Pfade im Code angepasst werden.

## Skriptbeschreibung

### Eigenes CNN-Modell

- `train.py`  
  Trainiert und speichert das ausgewählte eigene CNN-Modell.

- `EvaluateTestModel.py`  
  Lädt das finale eigene CNN-Modell und testet es auf den Testdaten.

- `Experiment/ExperimentRunner.py`  
  Führt die verschiedenen Testphasen für die Modelloptimierung des eigenen CNN-Modells aus.

- `Experiment/ModelFactory.py`  
  Erstellt die unterschiedlichen Modellvarianten des eigenen CNN-Modells.

### Transfer Learning

- `transferModelFactory.py`  
  Erstellt die vortrainierten Transfer-Learning-Modelle für den Modellvergleich.

- `transferLearningRunner.py`  
  Führt den Transfer-Learning-Modellvergleich aus.  
  Dabei werden mehrere vortrainierte Modelle geladen, der Feature Extractor eingefroren und nur der jeweilige Classifier bzw. die FC-Schicht trainiert.

- `fineTuningRunner.py`  
  Führt das Fine-Tuning für VGG13 durch.  
  Dabei bleibt der Classifier trainierbar und zusätzlich werden schrittweise die letzten 1 bis 5 Convolutional Layer freigegeben.  
  Jede Variante wird mit den Lernraten 0.0001 und 0.00001 getestet.

- `EvaluateTestModelVgg13.py`  
  Lädt das beste gespeicherte VGG13-Fine-Tuning-Modell und testet es auf den Testdaten.

### Allgemeine Tools

- `tools/makeCleanDataset.py`  
  Erstellt den bereinigten Datensatz und teilt ihn in train, val und test.

- `tools/label.py`  
  Liest die Labels ein und ordnet sie den Bildern zu.

- `tools/labelDistribution.py`  
  Erstellt ein Diagramm zur Verteilung der Labels.

- `tools/plot.py`  
  Erstellt Diagramme aus den gespeicherten Ergebnissen.

## Hinweis zum Datensatz

Die Ordner `Dataset` und `CleanDataset` wurden nicht mit auf GitHub hochgeladen, da diese zu groß für ein normales GitHub-Repository sind.

Der ursprüngliche Datensatz kann über folgenden Link heruntergeladen werden:

https://www.kaggle.com/datasets/msalman97/dataset-for-traffic-sign-master-app

Nach dem Herunterladen muss der Datensatz als Ordner `Dataset` im Projekt liegen.

Danach kann mit folgendem Skript der bereinigte Datensatz erstellt werden:

```bash
python tools/makeCleanDataset.py
```

## Hinweis zu den Modellen

Die trainierten `.pth`-Modelldateien wurden nicht mit auf GitHub hochgeladen, da diese zu groß für ein normales GitHub-Repository sind.

Das selbst entwickelte CNN-Modell kann mit folgendem Skript neu trainiert und gespeichert werden:

```bash
python train.py
```

Das VGG13-Transfer-Learning-Modell mit der besten Fine-Tuning-Variante kann mit folgendem Skript neu trainiert und gespeichert werden:

```bash
python fineTuningRunner.py
```

Dabei werden mehrere VGG13-Fine-Tuning-Varianten trainiert. In jeder Variante bleibt der Classifier trainierbar. Zusätzlich werden schrittweise die letzten 1 bis 5 Convolutional Layer freigegeben. Jede dieser Varianten wird mit den Lernraten `0.0001` und `0.00001` getestet.

Die in der Präsentation verwendete beste Variante ist:

- VGG13
- letzte 5 Convolutional Layer freigegeben
- Lernrate `0.00001`

Das erzeugte Modell wird unter folgendem Pfad gespeichert:

```text
TransferLearningResults/vgg13/models/VGG13_finetune_last_5_conv_layers_lr_1e-05.pth
```