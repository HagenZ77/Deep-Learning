import os
import torch.nn.functional as F
from enum import Enum
import torch
import torch.nn as nn
from torchvision.models import googlenet
from torch.utils.data import DataLoader, Dataset
from tools.plot import Plot
from tools.Dataset.label import Label
from tools.Dataset.makeCleanDataset import getAllImages
import json
from datetime import datetime


# Gibt an, in welcher Form das Modell seine Ausgabe liefert
class LossMode(Enum):
    LOGITS = "logits"
    PROBABILITIES = "probabilities"


# Klasse fuer das Erstellen, Trainieren und Speichern des CNN-Modells
class Train:
    EPOCHS = 40
    BATCH_SIZE = 32
    NUM_WORKERS = 8
    PATIENCE = 2
    CLASSES = 62

    def __init__(self, lossMode: LossMode):
        self.lossMode = lossMode


    # Erstellt das CNN-Grundmodell fuer die Klassifikation der Verkehrsschilder.
    # Damit haben wir anfangs ein paar versuche gestartet
    def createModel(self):
        model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1),

            nn.Flatten(),
            nn.Dropout(0.3),

            nn.Linear(16384, 64),
            nn.ReLU(),
            nn.Linear(64, 62),
            nn.Softmax(dim=1)
        )

        return model


    # Erstellt GoogLeNet fuer die Klassifikation der Verkehrsschilder
    def getGoogLeNet(self):
        model = googlenet(weights=None, aux_logits=False)
        model.fc = nn.Linear(model.fc.in_features, self.CLASSES)
        return model


    # Waehlt automatisch die GPU aus, falls CUDA verfuegbar ist, sonst wird die CPU genutzt.
    def createDevice(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Erstellt den Adam-Optimizer mit der angegebenen Lernrate.
    def createOptimizer(self, model, learningRate):
        return torch.optim.Adam(model.parameters(), lr=learningRate)


    # Erstellt die Loss-Funktion fuer die Klassifikation.
    def createLossFunction(self):
        return torch.nn.NLLLoss()


    # Laedt Trainings- und Validierungsbilder. Der Testdatensatz wird separat in EvaluateTestModel.py verwendet.
    def loadFiles(self):
        trainFiles = getAllImages("/mnt/c/Users/hagen/Desktop/Semester6/DeepLearning/CleanDataset/train")
        valFiles = getAllImages("/mnt/c/Users/hagen/Desktop/Semester6/DeepLearning/CleanDataset/val")
        labelFile = Label("/mnt/c/Users/hagen/Desktop/Semester6/DeepLearning/CleanDataset/labels.txt")

        return trainFiles, valFiles, labelFile


    # Erstellt DataLoader fuer Training und Validierung.
    def createDataLoaders(self, trainDataset: Dataset, valDataset: Dataset):
        trainLoader = DataLoader(
            trainDataset,
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            num_workers=self.NUM_WORKERS,
            pin_memory=True,
            persistent_workers=True
        )

        valLoader = DataLoader(
            valDataset,
            batch_size=self.BATCH_SIZE,
            shuffle=False,
            num_workers=self.NUM_WORKERS,
            pin_memory=True,
            persistent_workers=True
        )

        return trainLoader, valLoader


    # Erstellt die History-Struktur, in der Loss und Accuracy pro Epoche gespeichert werden
    def createHistory(self):
        return {
            "loss": [],
            "accuracy": [],
            "val_loss": [],
            "val_accuracy": []
        }


    def trainOneEpoch(self, model, trainLoader, optimizer, device):
        model.train()

        trainLossSum = 0
        trainCorrect = 0
        trainTotal = 0

        # Trainingsdaten batchweise durchlaufen
        for batchX, batchY in trainLoader:
            batchX = batchX.to(device, non_blocking=True)
            batchY = batchY.to(device, non_blocking=True)

            optimizer.zero_grad()

            # Vorhersage berechnen und Loss fuer den aktuellen Batch bestimmen.
            output = model(batchX)

            # Je nach Modell-Ausgabe muss der Loss unterschiedlich berechnet werden
            if self.lossMode == LossMode.LOGITS:
                loss = F.cross_entropy(output, batchY)
            elif self.lossMode == LossMode.PROBABILITIES:
                loss = F.nll_loss(torch.log(output + 1e-8), batchY)
            else:
                raise ValueError("Unbekannter Loss: " + str(self.lossMode))

            # Gradienten berechnen und Modellgewichte aktualisieren.
            loss.backward()
            optimizer.step()

            # Loss und korrekt klassifizierte Bilder aufsummieren.
            trainLossSum += loss.item() * batchX.size(0)
            trainTotal += batchX.size(0)
            trainCorrect += (torch.argmax(output, dim=1) == batchY).sum().item()

        # Durchschnittlichen Loss und Accuracy ueber alle Trainingsbilder berechnen.
        trainLoss = trainLossSum / trainTotal
        trainAccuracy = trainCorrect / trainTotal

        return trainLoss, trainAccuracy


    # Bewertet das Modell auf den Validierungsdaten ohne Gradientenberechnung
    def evaluate(self, model, valLoader, device):
        # Modell in den Auswertungsmodus setzen, damit z.B. Dropout deaktiviert wird.
        model.eval()

        valLossSum = 0
        valCorrect = 0
        valTotal = 0

        # Bei der Validierung werden keine Gradienten benoetigt.
        with torch.no_grad():
            for batchX, batchY in valLoader:
                batchX = batchX.to(device, non_blocking=True)
                batchY = batchY.to(device, non_blocking=True)

                # Vorhersage berechnen und Loss fuer den aktuellen Batch bestimmen.
                output = model(batchX)

                # Je nach Modell-Ausgabe muss der Loss unterschiedlich berechnet werden
                if self.lossMode == LossMode.LOGITS:
                    loss = F.cross_entropy(output, batchY)
                elif self.lossMode == LossMode.PROBABILITIES:
                    loss = F.nll_loss(torch.log(output + 1e-8), batchY)
                else:
                    raise ValueError("Unbekannter Loss: " + str(self.lossMode))

                # Loss und korrekt klassifizierte Bilder aufsummieren.
                valLossSum += loss.item() * batchX.size(0)
                valTotal += batchX.size(0)
                valCorrect += (torch.argmax(output, dim=1) == batchY).sum().item()

        # Durchschnittlichen Loss und Accuracy ueber alle Validierungsbilder berechnen.
        valLoss = valLossSum / valTotal
        valAccuracy = valCorrect / valTotal

        return valLoss, valAccuracy


    # Kopiert den aktuellen Modellzustand, damit spaeter das beste Modell wiederhergestellt werden kann
    def copyModelState(self, model):
        return {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }


    # Fuehrt das komplette Training mit Validierung, Live-Plot und Early Stopping aus
    def fit(self, model, trainLoader, valLoader, optimizer, device):
        # Sammelstruktur erstellen zum speichern des Verlauf fuer Loss und Accuracy ueber alle Epochen
        history = self.createHistory()

        # Variablen fuer Early Stopping und die beste gefundene Epoche vorbereiten
        bestValLoss = None
        bestValAccuracy = None
        bestTrainLoss = None
        bestTrainAccuracy = None
        bestEpoch = None
        bestModelState = None
        patienceCounter = 0
        executedEpochs = 0
        earlyStopped = False

        livePlot = Plot()

        # Training bis zur maximalen Epochenanzahl ausfuehren
        for epoch in range(self.EPOCHS):
            # Eine Epoche auf den Trainingsdaten trainieren
            trainLoss, trainAccuracy = self.trainOneEpoch(
                model,
                trainLoader,
                optimizer,
                device
            )

            # Eine Epoche auf den Validierungsdaten bewerten
            valLoss, valAccuracy = self.evaluate(
                model,
                valLoader,
                device
            )

            # Werte der aktuellen Epoche in der History speichern
            history["loss"].append(trainLoss)
            history["accuracy"].append(trainAccuracy)
            history["val_loss"].append(valLoss)
            history["val_accuracy"].append(valAccuracy)

            livePlot.update(
                trainLoss,
                valLoss,
                trainAccuracy,
                valAccuracy
            )

            currentEpoch = epoch + 1
            executedEpochs = currentEpoch

            print(currentEpoch, trainLoss, trainAccuracy, valLoss, valAccuracy)

            # Wenn der Validierung-Loss besser ist, alle Werte dieser Epoche als bisher bestes Ergebnis speichern
            if bestValLoss is None or valLoss < bestValLoss:
                bestValLoss = valLoss
                bestValAccuracy = valAccuracy
                bestTrainLoss = trainLoss
                bestTrainAccuracy = trainAccuracy
                bestEpoch = currentEpoch
                bestModelState = self.copyModelState(model)
                patienceCounter = 0
            else:
                # Keine Verbesserung erreicht, dann Early-Stopping-Zaehler erhoehen.
                patienceCounter += 1

            # Wenn zulange keine Verbesserung, dann stoppen
            if patienceCounter >= self.PATIENCE:
                earlyStopped = True
                break

        # Bestes Modell wiederherstellen
        if bestModelState is not None:
            model.load_state_dict(bestModelState)

        # Informationen zum Trainingslauf fuer die JSON-Datei sammeln
        fitInfo = {
            "planned_epochs": self.EPOCHS,
            "executed_epochs": executedEpochs,
            "early_stopped": earlyStopped,
            "best_epoch": {
                "epoch": bestEpoch,
                "train_loss": bestTrainLoss,
                "train_accuracy": bestTrainAccuracy,
                "val_loss": bestValLoss,
                "val_accuracy": bestValAccuracy
            }
        }

        return history, fitInfo


    # Erstellt die Ergebnisdaten fuer die JSON-Datei mit Datensatzinfos, Trainingsparametern, Modellaufbau und Verlauf
    def createResultData(self, model, history, fitInfo, trainFiles, valFiles, learningRate):
        if self.lossMode == LossMode.LOGITS:
            lossName = "cross_entropy"
        elif self.lossMode == LossMode.PROBABILITIES:
            lossName = "nll_loss_with_softmax_probabilities"
        else:
            raise ValueError("Unbekannter Loss: " + str(self.lossMode))

        return {
            "created_at": datetime.now().isoformat(),
            "dataset": {
                "train_count": len(trainFiles),
                "val_count": len(valFiles),
                "classes": self.CLASSES
            },
            "training": {
                "planned_epochs": fitInfo["planned_epochs"],
                "executed_epochs": fitInfo["executed_epochs"],
                "early_stopped": fitInfo["early_stopped"],
                "batch_size": self.BATCH_SIZE,
                "optimizer": "adam",
                "learning_rate": learningRate,
                "loss": lossName
            },
            "best_epoch": fitInfo["best_epoch"],
            "model_architecture": str(model),
            "history": history
        }


    # Erstellt den Ordner fuer eine Ausgabedatei, falls dieser noch nicht existiert
    def createFolderForFile(self, filePath):
        folder = os.path.dirname(filePath)

        if folder != "":
            os.makedirs(folder, exist_ok=True)


    # Speichert die Ergebnisdaten als JSON-Datei und dazu auch ein Diagramm
    def saveJsonAndDiagram(self, resultData, jsonOutputPath, diagramOutputPath):
        self.createFolderForFile(jsonOutputPath)

        with open(jsonOutputPath, "w", encoding="utf-8") as file:
            json.dump(resultData, file, indent=4)

        self.createFolderForFile(diagramOutputPath)

        plot = Plot()
        plot.createTrainingHistoryDiagram(jsonOutputPath, diagramOutputPath)


    # Speichert das trainierte Modell als Datei
    def saveModel(self, model, outputPath):
        self.createFolderForFile(outputPath)
        torch.save(model, outputPath)


    # Sucht aus einer Liste von Modellvarianten das Modell mit dem angegebenen Namen heraus
    def getModelByName(self, models, targetName):
        for model, name in models:
            if name == targetName:
                return model

        raise ValueError("Modell nicht gefunden: " + targetName)