import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from Experiment.ModelFactory import ModelFactory
from tools.plot import Plot
from tools.label import Label
from tools.makeCleanDataset import getAllImages
import json
from datetime import datetime


# Eigenes PyTorch-Dataset, das Bildpfade einliest und passende Labels aus dem Dateinamen bestimmt.
class ImageDataset(Dataset):
    def __init__(self, files, labelFile):
        self.files = files
        self.labelFile = labelFile


    def __len__(self):
        return len(self.files)


    # Laedt ein Bild, wandelt es in RGB um, skaliert es auf 128x128,
    # normalisiert die Pixelwerte auf [0, 1] und bringt die Achsen in PyTorch-Format CxHxW.
    def loadImage(self, path):
        img = Image.open(path).convert("RGB")
        img = img.resize((128, 128))
        img = np.array(img).astype("float32")
        img = img / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img


    # Gibt ein einzelnes Bild mit zugehoerigem Klassenindex fuer den DataLoader zurueck.
    def __getitem__(self, index):
        path = self.files[index]

        x = self.loadImage(path)[0]
        y = self.labelFile.getInfoFromImagePath(path).label

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        return x, y



# Klasse fuer das Erstellen, Trainieren und Speichern des CNN-Modells
class Train:
    # Speichert grundlegende Modell und Datensatzinformationen
    def __init__(self):
        self.image_size = (128, 128)
        self.classes = 62


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
        trainFiles = getAllImages("/mnt/c/Users/hagen/Desktop/DeepLearning/CleanDataset/train")
        valFiles = getAllImages("/mnt/c/Users/hagen/Desktop/DeepLearning/CleanDataset/val")
        labelFile = Label("/mnt/c/Users/hagen/Desktop/DeepLearning/CleanDataset/labels.txt")

        return trainFiles, valFiles, labelFile


    # Erstellt DataLoader fuer Training und Validierung.
    def createDataLoaders(self, trainFiles, valFiles, labelFile, batch_size):
        trainDataset = ImageDataset(trainFiles, labelFile)
        valDataset = ImageDataset(valFiles, labelFile)

        trainLoader = DataLoader(
            trainDataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True
        )

        valLoader = DataLoader(
            valDataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True
        )

        return trainLoader, valLoader


    # Erstellt die History-Struktur, in der Loss und Accuracy pro Epoche gespeichert werden.
    def createHistory(self):
        return {
            "loss": [],
            "accuracy": [],
            "val_loss": [],
            "val_accuracy": []
        }


    def trainOneEpoch(self, model, trainLoader, optimizer, lossFunction, device):
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
            loss = lossFunction(torch.log(output + 1e-8), batchY)

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
    def evaluate(self, model, valLoader, lossFunction, device):
        # Modell in den Auswertungsmodus setzen, damit z.B. Dropout deaktiviert wird.
        model.eval()

        valLossSum = 0
        valCorrect = 0
        valTotal = 0

        # Bei der Validierung werden keine Gradienten benoetigt.
        with torch.no_grad():
            for batch_x, batch_y in valLoader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)

                # Vorhersage berechnen und Loss fuer den aktuellen Batch bestimmen.
                output = model(batch_x)
                loss = lossFunction(torch.log(output + 1e-8), batch_y)

                # Loss und korrekt klassifizierte Bilder aufsummieren.
                valLossSum += loss.item() * batch_x.size(0)
                valTotal += batch_x.size(0)
                valCorrect += (torch.argmax(output, dim=1) == batch_y).sum().item()

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
    def fit(self, model, trainLoader, valLoader, optimizer, lossFunction, device, epochs, patience):
        history = self.createHistory()

        bestValLoss = None
        bestModelState = None
        patienceCounter = 0

        livePlot = Plot()

        for epoch in range(epochs):
            trainLoss, trainAccuracy = self.trainOneEpoch(
                model,
                trainLoader,
                optimizer,
                lossFunction,
                device
            )

            valLoss, valAccuracy = self.evaluate(
                model,
                valLoader,
                lossFunction,
                device
            )

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

            print(epoch + 1, trainLoss, trainAccuracy, valLoss, valAccuracy)

            if bestValLoss is None or valLoss < bestValLoss:
                bestValLoss = valLoss
                bestModelState = self.copyModelState(model)
                patienceCounter = 0
            else:
                patienceCounter += 1

            if patienceCounter >= patience:
                break

        if bestModelState is not None:
            model.load_state_dict(bestModelState)

        return history, bestValLoss


    # Erstellt die Ergebnisdaten fuer die JSON-Datei mit Datensatzinfos, Trainingsparametern, Modellaufbau und Verlauf
    def createResultData(self, model, history, trainFiles, valFiles, epochs, batchSize, learningRate):
        return {
            "created_at": datetime.now().isoformat(),
            "dataset": {
                "train_count": len(trainFiles),
                "val_count": len(valFiles),
                "image_size": [128, 128, 3],
                "classes": self.classes
            },
            "training": {
                "epochs": epochs,
                "batch_size": batchSize,
                "optimizer": "adam",
                "learning_rate": learningRate,
                "loss": "nll_loss_with_log_softmax_output",
                "metrics": ["accuracy"]
            },
            "model_architecture": str(model),
            "history": history
        }


    # Speichert die Ergebnisdaten als JSON-Datei
    def saveResultJson(self, resultData, filename):
        with open(filename + ".json", "w", encoding="utf-8") as f:
            json.dump(resultData, f, indent=4)


    # Speichert das trainierte Modell als Datei
    def saveModel(self, model, filename):
        torch.save(model, filename)


    # Sucht aus einer Liste von Modellvarianten das Modell mit dem angegebenen Namen heraus
    def getModelByName(self, models, targetName):
        for model, name in models:
            if name == targetName:
                return model

        raise ValueError("Modell nicht gefunden: " + targetName)


if __name__ == "__main__":
    trainer = Train()

    device = trainer.createDevice()

    epochs = 40
    batchSize = 32
    patience = 2
    learningRate = 0.0003

    # Dies sind die Architekturen die in ihrer Test-Phase am besten waren.
    # Damit bilden wir unser Modell, um die bestmoegliche Klassifizierung zu bekommen
    factory = ModelFactory()
    modelList = factory.getConvolutionLayerModels()
    currentModel = trainer.getModelByName(modelList, "32/64/128/256/512/1024")
    modelList = factory.getConvolutionKernelSizeModels(currentModel)
    currentModel = trainer.getModelByName(modelList, "(5, 5)")
    modelList = factory.getPoolingMethodModels(currentModel)
    currentModel = trainer.getModelByName(modelList, "MaxPool")
    modelList = factory.getFullyConnectedLayerModels(currentModel)
    currentModel = trainer.getModelByName(modelList, "fc_512_128")
    modelList = factory.getDropoutPositionsModels(currentModel)
    currentModel = trainer.getModelByName(modelList, "dropout_before_flatten")
    modelList = factory.getDropoutValueModels(currentModel)
    model = trainer.getModelByName(modelList, "0.6")
    model = model.to(device)

    # Optimizer und Loss Funktion fuer das finale Training erstellen
    optimizer = trainer.createOptimizer(model, learningRate)
    lossFunction = trainer.createLossFunction()

    # Trainings- und Validierungsdaten laden.
    trainFiles, valFiles, labelFile = trainer.loadFiles()

    # DataLoader erstellen, damit die Bilder batchweise geladen werden
    trainLoader, valLoader = trainer.createDataLoaders(
        trainFiles,
        valFiles,
        labelFile,
        batchSize
    )

    # Finales Modell trainieren und dabei den besten Validierungs Loss speichern
    history, bestValLoss = trainer.fit(
        model,
        trainLoader,
        valLoader,
        optimizer,
        lossFunction,
        device,
        epochs,
        patience
    )

    plt.show(block=True)

    print(history["loss"])
    print(history["accuracy"])
    print(history["val_loss"])
    print(history["val_accuracy"])

    # Ergebnisse speichern
    resultData = trainer.createResultData(
        model,
        history,
        trainFiles,
        valFiles,
        epochs,
        batchSize,
        learningRate
    )
    trainer.saveResultJson(resultData, "finalModel")

    # Modell+Gewichte speichern
    trainer.saveModel(model, "finalModel.pth")