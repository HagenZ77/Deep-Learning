import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tools.Dataset.label import Label
from tools.Dataset.makeCleanDataset import getAllImages
from tools.plot import Plot
import json
from datetime import datetime


class EvaluateTestModel:
    BATCH_SIZE = 32
    NUM_WORKERS = 8
    CLASSES = 62


    def createFolderForFile(self, filePath):
        folder = os.path.dirname(filePath)

        if folder != "":
            os.makedirs(folder, exist_ok=True)


    def loadFullModel(self, modelPath, device):
        # Laedt das gespeicherte finale Modell und setzt es in den Auswertungsmodus
        model = torch.load(modelPath, map_location=device, weights_only=False)
        model = model.to(device)
        model.eval()
        return model


    def createTestLoader(self, testDir, labelPath, datasetClass):
        # Labeldatei einlesen und alle Testbilder sammeln.
        labelFile = Label(labelPath)
        testFiles = getAllImages(testDir)

        testDataset = datasetClass(testFiles, labelFile)

        # DataLoader fuer die batchweise Testauswertung erstellen
        testLoader = DataLoader(
            testDataset,
            batch_size=self.BATCH_SIZE,
            shuffle=False,
            num_workers=self.NUM_WORKERS,
            pin_memory=True,
            persistent_workers=True
        )

        return testLoader, labelFile, testFiles


    def evaluateModel(self, model, testLoader, device, numClasses):
        # Leere Matrix erstellen
        confusion = torch.zeros(numClasses, numClasses, dtype=torch.int64)

        correct = 0
        total = 0

        confidenceSum = 0
        correctConfidenceSum = 0
        wrongConfidenceSum = 0

        correctConfidenceCount = 0
        wrongConfidenceCount = 0

        predictionDetails = []

        # Modell in den Auswertungsmodus setzen
        model.eval()

        # Testdaten ohne Gradientenberechnung durchlaufen
        with torch.no_grad():
            for batchX, batchY in testLoader:
                batchX = batchX.to(device)
                batchY = batchY.to(device)

                # Vorhersage berechnen und die Klasse mit dem hoechsten Wert bestimmen
                output = model(batchX)

                probabilities = torch.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, dim=1)

                # Korrekte Vorhersagen und Gesamtanzahl summieren
                correct += (predicted == batchY).sum().item()
                total += batchY.size(0)

                confidenceSum += confidence.sum().item()

                # Matrix fuellen
                for trueLabel, predLabel, predictionConfidence in zip(batchY.cpu(), predicted.cpu(), confidence.cpu()):
                    trueLabel = int(trueLabel)
                    predLabel = int(predLabel)
                    predictionConfidence = float(predictionConfidence)

                    confusion[trueLabel, predLabel] += 1

                    isCorrect = trueLabel == predLabel

                    if isCorrect:
                        correctConfidenceSum += predictionConfidence
                        correctConfidenceCount += 1
                    else:
                        wrongConfidenceSum += predictionConfidence
                        wrongConfidenceCount += 1

                    predictionDetails.append({
                        "trueLabelIndex": trueLabel,
                        "predictedLabelIndex": predLabel,
                        "confidence": predictionConfidence,
                        "correct": isCorrect
                    })

        # Finale Testgenauigkeit berechnen
        accuracy = correct / total

        confidenceData = {
            "averageConfidence": confidenceSum / total,
            "averageCorrectConfidence": correctConfidenceSum / correctConfidenceCount if correctConfidenceCount > 0 else 0,
            "averageWrongConfidence": wrongConfidenceSum / wrongConfidenceCount if wrongConfidenceCount > 0 else 0,
            "predictionDetails": predictionDetails
        }

        return confusion.numpy(), accuracy, confidenceData


    def createConfusionData(self, confusion, labelFile):
        confusedLabels = []         # Speichert, welche Klasse mit welcher verwechselt wurden und dessen Anzahl an verwechslungen
        wrongByTrueLabel = []       # Speichert pro echter Klasse, wie viele Bilder insgesamt falsch erkannt wurden
        correctByTrueLabel = []     # Speichert pro echter Klasse, wie viele Bilder insgesamt richtig erkannt wurden

        # Durch alle echten Klassen gehen, also durch die Zeilen der Confusion-Matrix
        for trueIndex in range(confusion.shape[0]):
            trueLabel = labelFile.getLabelByIndex(trueIndex)

            correctCount = int(confusion[trueIndex, trueIndex])
            wrongCount = 0

            # Fuer diese echte Klasse durch alle vorhergesagten Klassen gehen, also durch die Spalten
            for predictedIndex in range(confusion.shape[1]):
                predictedLabel = labelFile.getLabelByIndex(predictedIndex)
                count = int(confusion[trueIndex, predictedIndex])

                if trueIndex == predictedIndex:
                    continue

                if count > 0:
                    wrongCount += count

                    confusedLabels.append({
                        "trueLabelIndex": trueIndex,
                        "trueLabel": trueLabel,
                        "predictedLabelIndex": predictedIndex,
                        "predictedLabel": predictedLabel,
                        "count": count
                    })

            correctByTrueLabel.append({
                "labelIndex": trueIndex,
                "label": trueLabel,
                "correctCount": correctCount
            })

            wrongByTrueLabel.append({
                "labelIndex": trueIndex,
                "label": trueLabel,
                "wrongCount": wrongCount
            })

        # Haeufigste Verwechslungen zuerst
        confusedLabels.sort(
            key=lambda item: item["count"],
            reverse=True
        )

        # Klassen mit den meisten Verwechslungen zuerst
        wrongByTrueLabel.sort(
            key=lambda item: item["wrongCount"],
            reverse=True
        )

        return confusedLabels, wrongByTrueLabel, correctByTrueLabel


    def createTestResultData(self, modelPath, testDir, labelPath, heatmapSavePath, confusion, accuracy, confidenceData, labelFile, testFiles, modelName=None):
        # Confusion-Matrix Daten speichern fuer das spaetere analysieren
        confusedLabels, wrongByTrueLabel, correctByTrueLabel = self.createConfusionData(
            confusion,
            labelFile
        )

        testCount = len(testFiles)
        correctPredictions = int(np.trace(confusion))
        wrongPredictions = int(testCount - correctPredictions)

        if modelName is None:
            modelName = os.path.basename(modelPath)

        # Alle Testdaten fuer die JSON-Datei sammeln
        resultData = {
            "createdAt": datetime.now().isoformat(),
            "model": {
                "name": modelName,
                "modelPath": modelPath
            },
            "dataset": {
                "testDir": testDir,
                "labelPath": labelPath,
                "testCount": testCount,
                "classes": self.CLASSES
            },
            "test": {
                "accuracy": accuracy,
                "errorRate": 1.0 - accuracy,
                "correctPredictions": correctPredictions,
                "wrongPredictions": wrongPredictions,
                "averageConfidence": confidenceData["averageConfidence"],
                "averageCorrectConfidence": confidenceData["averageCorrectConfidence"],
                "averageWrongConfidence": confidenceData["averageWrongConfidence"]
            },
            "files": {
                "heatmapSavePath": heatmapSavePath
            },
            "mistakes": {
                "wrongPredictions": wrongPredictions,
                "confusedLabels": confusedLabels,
                "wrongByTrueLabel": wrongByTrueLabel,
                "correctByTrueLabel": correctByTrueLabel
            },
            "confidence": {
                "averageConfidence": confidenceData["averageConfidence"],
                "averageCorrectConfidence": confidenceData["averageCorrectConfidence"],
                "averageWrongConfidence": confidenceData["averageWrongConfidence"],
                "predictionDetails": confidenceData["predictionDetails"]
            },
            "confusionMatrix": confusion.tolist()
        }

        return resultData


    def saveTestResultJson(self, resultData, jsonSavePath):
        self.createFolderForFile(jsonSavePath)

        with open(jsonSavePath, "w") as file:
            json.dump(resultData, file, indent=4)


    def runTest(self, modelPath, testDir, labelPath, datasetClass, heatmapSavePath, jsonSavePath=None, modelName=None):
        # Versuchen mit der GPU zu arbeiten sonst CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        testLoader, labelFile, testFiles = self.createTestLoader(
            testDir,
            labelPath,
            datasetClass
        )

        model = self.loadFullModel(modelPath, device)

        # Modell auf dem Testdatensatz auswerten
        confusion, accuracy, confidenceData = self.evaluateModel(
            model,
            testLoader,
            device,
            self.CLASSES
        )

        # Confusion-Matrix als Heatmap speichern und anzeigen
        Plot().createConfusionHeatmap(
            confusion,
            labelFile,
            savePath=heatmapSavePath
        )


        # Testdaten in JSON-Format bringen
        resultData = self.createTestResultData(
            modelPath,
            testDir,
            labelPath,
            heatmapSavePath,
            confusion,
            accuracy,
            confidenceData,
            labelFile,
            testFiles,
            modelName
        )

        # Testdaten speichern
        if jsonSavePath is not None:
            self.saveTestResultJson(
                resultData,
                jsonSavePath
            )

        return confusion, accuracy, confidenceData