import gc
import os
import copy
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tools.train import Train
from tools.ImageDataset224 import ImageDataset224
from transferModelFactory import TransferModelFactory


# Dieser Runner vergleicht mehrere vortrainierte Modelle.
# Der Feature Extractor bleibt eingefroren.
# Trainiert wird nur der jeweilige Classifier bzw. die FC-Schicht.
# Alle Modelle werden mit derselben Lernrate von 0.0001 trainiert.
# Ziel ist die Auswahl der geeignetsten Architektur.
class TransferLearningRunner:
    def __init__(self):
        self.trainer = Train()
        self.device = self.trainer.createDevice()

        self.epochs = 40
        self.batchSize = 32
        self.patience = 2
        self.learningRate = 0.0001

        self.resultRoot = "/mnt/c/Users/hagen/Desktop/Semester6/DeepLearning/TransferLearningResults"
        self.jsonFolder = os.path.join(self.resultRoot, "json")
        self.modelFolder = os.path.join(self.resultRoot, "models")
        self.diagramFolder = os.path.join(self.resultRoot, "diagrams")

        self.lossFunction = nn.CrossEntropyLoss()

        self.createFolders()


    def createFolders(self):
        # Erstellt die Ordner fuer JSON-Dateien, Modelle und Diagramme,
        # falls diese noch nicht existieren.
        os.makedirs(self.jsonFolder, exist_ok=True)
        os.makedirs(self.modelFolder, exist_ok=True)
        os.makedirs(self.diagramFolder, exist_ok=True)


    def getJsonPathWithoutExtension(self, modelName):
        # Gibt den Speicherpfad fuer die JSON-Datei ohne Dateiendung zurueck.
        # Das wird benoetigt, weil saveResultJson die Endung selbst anhaengt.
        return os.path.join(self.jsonFolder, modelName)


    def getJsonPath(self, modelName):
        # Gibt den vollstaendigen Speicherpfad der JSON-Datei zurueck.
        return os.path.join(self.jsonFolder, modelName + ".json")


    def getModelPath(self, modelName):
        # Gibt den Speicherpfad fuer das trainierte Modell zurueck.
        # Das Modell wird als .pth Datei gespeichert.
        return os.path.join(self.modelFolder, modelName + ".pth")


    def getDiagramPath(self, modelName):
        # Gibt den Speicherpfad fuer das Trainingsdiagramm zurueck.
        # Im Diagramm werden Loss und Accuracy pro Epoche dargestellt.
        return os.path.join(self.diagramFolder, modelName + "_history.png")


    def createOptimizer(self, trainableParameters):
        # Erstellt den Adam-Optimizer fuer alle trainierbaren Parameter.
        # Eingefrorene Parameter werden hier nicht uebergeben und daher nicht angepasst.
        # Die Lernrate kommt aus self.learning_rate.
        return torch.optim.Adam(
            trainableParameters,
            lr=self.learningRate
        )

    def createResultData(self, model, history, trainFiles, valFiles, modelName, bestValLoss):
        # Erstellt die Grunddaten fuer die JSON-Datei.
        # Die Methode aus Train speichert allgemeine Infos wie Architektur,
        # History, Datensatzgroessen, Epochen, Batch Size und Lernrate.
        resultData = self.trainer.createResultData(
            model,
            history,
            trainFiles,
            valFiles,
            self.epochs,
            self.batchSize,
            self.learningRate
        )

        # Speichert den Namen des aktuellen Modells in der JSON.
        resultData["name"] = modelName

        # Kennzeichnet, zu welcher Experiment-Phase dieses Ergebnis gehoert.
        resultData["phase"] = "transfer_learning_pretrained_classifier_train"

        # Ueberschreibt die Loss-Beschreibung, weil beim Transfer Learning
        # CrossEntropyLoss verwendet wird.
        resultData["training"]["loss"] = "cross_entropy_loss"

        # Beschreibt, welcher Teil des Modells trainiert wurde.
        # Hier ist der Feature Extractor eingefroren und nur der Classifier trainierbar.
        resultData["training"]["trained_part"] = "feature_extractor_frozen_classifier_trainable"

        # Speichert die wichtigsten Bewertungswerte fuer den Modellvergleich.
        # Best = bester Wert waehrend des Trainings.
        # Final = Wert aus der letzten ausgefuehrten Epoche.
        resultData["evaluation"] = {
            "best_val_loss": bestValLoss,
            "best_val_accuracy": max(history["val_accuracy"]),
            "final_val_loss": history["val_loss"][-1],
            "final_val_accuracy": history["val_accuracy"][-1]
        }

        return resultData


    def saveHistoryDiagram(self, history, outputPath, modelName):
        # Werte aus der History holen.
        loss = history["loss"]
        valLoss = history["val_loss"]
        accuracy = history["accuracy"]
        valAccuracy = history["val_accuracy"]

        # Erstellt die Epochen-Achse passend zur Anzahl der gespeicherten Werte.
        epochs = list(range(1, len(loss) + 1))

        # Erstellt eine Grafik mit zwei Diagrammen nebeneinander.
        fig, (axLoss, axAccuracy) = plt.subplots(1, 2, figsize=(12, 5))

        # Loss-Verlauf fuer Training und Validierung zeichnen.
        axLoss.plot(epochs, loss, label="loss")
        axLoss.plot(epochs, valLoss, label="val_loss")
        axLoss.set_title("Loss - " + modelName)
        axLoss.set_xlabel("Epoch")
        axLoss.set_ylabel("Loss")
        axLoss.legend()

        # Accuracy-Verlauf fuer Training und Validierung zeichnen.
        axAccuracy.plot(epochs, accuracy, label="accuracy")
        axAccuracy.plot(epochs, valAccuracy, label="val_accuracy")
        axAccuracy.set_title("Accuracy - " + modelName)
        axAccuracy.set_xlabel("Epoch")
        axAccuracy.set_ylabel("Accuracy")
        axAccuracy.legend()

        # Layout anpassen, Grafik speichern und Figur wieder schliessen.
        fig.tight_layout()
        fig.savefig(outputPath, dpi=300)
        plt.close(fig)


    def saveAll(self, model, history, trainFiles, valFiles, modelName, bestValLoss):
        # Speicherpfade fuer JSON, Modell und Diagramm erstellen.
        jsonPathWithoutExtension = self.getJsonPathWithoutExtension(modelName)
        jsonPath = self.getJsonPath(modelName)
        modelPath = self.getModelPath(modelName)
        diagramPath = self.getDiagramPath(modelName)

        # Ergebnisdaten fuer die JSON-Datei vorbereiten.
        resultData = self.createResultData(
            model,
            history,
            trainFiles,
            valFiles,
            modelName,
            bestValLoss
        )

        # JSON-Datei, Modell und Trainingsdiagramm speichern.
        self.trainer.saveResultJson(resultData, jsonPathWithoutExtension)
        self.trainer.saveModel(model, modelPath)
        self.saveHistoryDiagram(history, diagramPath, modelName)

        # Speicherorte in der Konsole ausgeben.
        print("Gespeichert:")
        print("  JSON:    " + jsonPath)
        print("  Modell:  " + modelPath)
        print("  Diagramm:" + diagramPath)


    def cleanup(self, model, optimizer):
        # Modell und Optimizer aus dem Speicher entfernen.
        del model
        del optimizer

        # CUDA-Speicher der GPU freigeben.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Python-Speicher aufraeumen und offene Diagramme schliessen.
        gc.collect()
        plt.close("all")


    def createTransferDataLoaders(self, trainFiles, valFiles, labelFile):
        # Erstellt die Dataset-Objekte fuer Training und Validierung.
        trainDataset = ImageDataset224(trainFiles, labelFile)
        valDataset = ImageDataset224(valFiles, labelFile)

        # DataLoader fuer das Training.
        # shuffle=True, damit die Trainingsbilder pro Epoche gemischt werden.
        trainLoader = DataLoader(
            trainDataset,
            batch_size=self.batchSize,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True
        )

        # DataLoader fuer die Validierung.
        # shuffle=False, damit die Validierung immer in gleicher Reihenfolge laeuft.
        valLoader = DataLoader(
            valDataset,
            batch_size=self.batchSize,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True
        )

        # Gibt beide DataLoader fuer das Training zurueck.
        return trainLoader, valLoader


    def trainOneEpoch(self, model, trainLoader, optimizer):
        # Modell in den Trainingsmodus setzen.
        model.train()

        lossSum = 0.0
        correct = 0
        total = 0

        for batchX, batchY in trainLoader:
            batchX = batchX.to(self.device, non_blocking=True)
            batchY = batchY.to(self.device, non_blocking=True)

            # Alte Gradienten loeschen.
            optimizer.zero_grad()

            output = model(batchX)
            loss = self.lossFunction(output, batchY)

            # Gewichte anhand des Fehlers anpassen.
            loss.backward()
            optimizer.step()

            # Loss und Accuracy fuer diese Epoche aufsummieren.
            lossSum += loss.item() * batchX.size(0)
            total += batchX.size(0)
            correct += (torch.argmax(output, dim=1) == batchY).sum().item()

        return lossSum / total, correct / total


    def evaluate(self, model, valLoader):
        # Modell in den Auswertungsmodus setzen.
        model.eval()

        lossSum = 0.0
        correct = 0
        total = 0

        # Keine Gradienten berechnen, da hier nicht trainiert wird.
        with torch.no_grad():
            for batchX, batchY in valLoader:
                batchX = batchX.to(self.device, non_blocking=True)
                batchY = batchY.to(self.device, non_blocking=True)

                output = model(batchX)
                loss = self.lossFunction(output, batchY)

                # Loss und Accuracy fuer die Validierung aufsummieren.
                lossSum += loss.item() * batchX.size(0)
                total += batchX.size(0)
                correct += (torch.argmax(output, dim=1) == batchY).sum().item()

        return lossSum / total, correct / total


    def fit(self, model, trainLoader, valLoader):
        # Speichert den Verlauf von Training und Validierung.
        history = {
            "loss": [],
            "accuracy": [],
            "val_loss": [],
            "val_accuracy": []
        }

        bestValLoss = None
        bestModelState = None
        patienceCounter = 0

        # Optimizer nur fuer trainierbare Parameter erstellen.
        optimizer = self.createOptimizer(
            filter(lambda param: param.requires_grad, model.parameters())
        )

        for epoch in range(self.epochs):
            # Eine Epoche trainieren.
            trainLoss, trainAccuracy = self.trainOneEpoch(
                model,
                trainLoader,
                optimizer
            )

            # Modell auf Validierungsdaten testen.
            valLoss, valAccuracy = self.evaluate(
                model,
                valLoader
            )

            # Werte der aktuellen Epoche speichern.
            history["loss"].append(trainLoss)
            history["accuracy"].append(trainAccuracy)
            history["val_loss"].append(valLoss)
            history["val_accuracy"].append(valAccuracy)

            print(epoch + 1, trainLoss, trainAccuracy, valLoss, valAccuracy)

            # Bestes Modell anhand des kleinsten Val-Loss merken.
            if bestValLoss is None or valLoss < bestValLoss:
                bestValLoss = valLoss
                bestModelState = copy.deepcopy(model.state_dict())
                patienceCounter = 0
            else:
                patienceCounter += 1

            # Training abbrechen, wenn sich Val-Loss nicht mehr verbessert.
            if patienceCounter >= self.patience:
                break

        # Bestes gespeichertes Modell wiederherstellen.
        if bestModelState is not None:
            model.load_state_dict(bestModelState)

        return history, bestValLoss, optimizer


    def runOneModel(self, createModelFunction, trainLoader, valLoader, trainFiles, valFiles):
        model, _, modelName = createModelFunction()

        print("")
        print("============================================================")
        print("Starte Modell:  " + modelName)
        print("============================================================")

        model = model.to(self.device)

        history, bestValLoss, optimizer = self.fit(
            model,
            trainLoader,
            valLoader
        )

        self.saveAll(
            model,
            history,
            trainFiles,
            valFiles,
            modelName,
            bestValLoss
        )

        self.cleanup(model, optimizer)

        print("Fertig: " + modelName)

    def run(self):
        # Dateien laden und in Train-/Val-Split aufteilen.
        trainFiles, valFiles, labelFile = self.trainer.loadFiles()

        # DataLoader fuer Training und Validierung erstellen.
        trainLoader, valLoader = self.createTransferDataLoaders(
            trainFiles,
            valFiles,
            labelFile
        )

        # Factory fuer die Transfer-Learning-Modelle erstellen.
        factory = TransferModelFactory(self.trainer.classes)

        # Alle Modelle nacheinander trainieren und speichern.
        for model in factory.getModels():
            self.runOneModel(
                model,
                trainLoader,
                valLoader,
                trainFiles,
                valFiles
            )


if __name__ == "__main__":
    runner = TransferLearningRunner()
    runner.run()