import os
import json
from datetime import datetime

import torch

from train import Train
from ModelFactory import ModelFactory


class ExperimentRunner:
    def __init__(self):
        # Ausgabeordner fuer JSON-Ergebnisse und gespeicherte Modelle festlegen
        self.outputDir = "ExperimentResults"
        self.jsonDir = os.path.join(self.outputDir, "json")
        self.modelDir = os.path.join(self.outputDir, "models")

        # Ausgabeordner erstellen, falls sie noch nicht existieren
        os.makedirs(self.jsonDir, exist_ok=True)
        os.makedirs(self.modelDir, exist_ok=True)

        # Trainingsklasse und Modellerzeugung vorbereiten
        self.trainer = Train()
        self.factory = ModelFactory()

        # GPU verwenden
        self.device = self.trainer.createDevice()

        # Trainingsparameter
        self.epochs = 40
        self.batch_size = 32
        self.patience = 2
        self.learning_rate = 0.0003

        # Platzhalter fuer Dateien und DataLoader
        self.trainFiles = None
        self.testFiles = None
        self.labelFile = None

        self.train_loader = None
        self.test_loader = None


    def setupData(self):
        # Trainings- und Validierungsdateien laden
        self.trainFiles, self.testFiles, self.labelFile = self.trainer.loadFiles()

        # DataLoader fuer Training und Validierung erstellen
        self.train_loader, self.test_loader = self.trainer.createDataLoaders(
            self.trainFiles,
            self.testFiles,
            self.labelFile,
            self.batch_size
        )


    def evaluateHistory(self, history):
        # Beste Epoche anhand der hoechsten Validierungsgenauigkeit bestimmen
        bestEpochIndex = max(
            range(len(history["val_accuracy"])),
            key=lambda index: history["val_accuracy"][index]
        )

        # Trainingswerte der besten Epoche auslesen
        trainLoss = history["loss"][bestEpochIndex]
        trainAccuracy = history["accuracy"][bestEpochIndex]

        # Validierungswerte der besten Epoche auslesen
        valLoss = history["val_loss"][bestEpochIndex]
        valAccuracy = history["val_accuracy"][bestEpochIndex]

        # Abstand zwischen Training und Validierung berechnen
        accuracyGap = trainAccuracy - valAccuracy
        lossGap = valLoss - trainLoss

        # Score berechnen und starkes Overfitting bestrafen
        overfittingPenalty = max(0.0, accuracyGap - 0.10) * 0.5
        score = valAccuracy - overfittingPenalty

        return {
            "best_epoch": bestEpochIndex + 1,

            "train_loss": trainLoss,
            "train_accuracy": trainAccuracy,

            "val_loss": valLoss,
            "val_accuracy": valAccuracy,

            "accuracy_gap": accuracyGap,
            "loss_gap": lossGap,

            "overfitting_penalty": overfittingPenalty,
            "score": score
        }


    def saveResult(self, phase, name, model, history, evaluation):
        # Dateinamen aus Phase und Modellname erzeugen
        filename = self.sanitizeFilename(phase + "_" + name)

        # Ergebnisdaten mit Modellaufbau, History und Trainingsparametern erstellen
        resultData = self.trainer.createResultData(
            model,
            history,
            self.trainFiles,
            self.testFiles,
            self.epochs,
            self.batch_size,
            self.learning_rate
        )

        # Zusatzinformationen zur Experimentphase ergaenzen
        resultData["phase"] = phase
        resultData["name"] = name
        resultData["evaluation"] = evaluation

        # Speicherpfade fuer JSON-Datei und Modell erzeugen
        jsonPath = os.path.join(self.jsonDir, filename + ".json")
        modelPath = os.path.join(self.modelDir, filename + ".pth")

        # Ergebnisdaten als JSON speichern
        with open(jsonPath, "w", encoding="utf-8") as f:
            json.dump(resultData, f, indent=4)

        # Trainiertes Modell+Gewichte speichern
        torch.save(model, modelPath)

        return jsonPath, modelPath


    def resetModelWeights(self, model):
        # Modellgewichte vor jedem Versuch neu initialisieren.
        # Dadurch startet jedes Modell fair ohne bereits gelernte Gewichte aus einem vorherigen Training
        for layer in model.modules():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()


    def trainModel(self, phase, name, model):
        print()
        print("========================================")
        print("PHASE:", phase)
        print("MODEL:", name)
        print("========================================")

        # Gewichte vor dem Training zuruecksetzen, damit jeder Versuch fair startet
        self.resetModelWeights(model)
        model = model.to(self.device)

        # Optimizer und Loss-Funktion fuer dieses Modell erstellen
        optimizer = self.trainer.createOptimizer(
            model,
            self.learning_rate
        )

        lossFunction = self.trainer.createLossFunction()

        # Modell trainieren und Trainingsverlauf speichern
        history, bestValLoss = self.trainer.fit(
            model,
            self.train_loader,
            self.test_loader,
            optimizer,
            lossFunction,
            self.device,
            self.epochs,
            self.patience
        )

        # Trainingsverlauf auswerten und Score berechnen
        evaluation = self.evaluateHistory(history)

        # Ergebnisdaten und Modell speichern
        jsonPath, modelPath = self.saveResult(
            phase,
            name,
            model,
            history,
            evaluation
        )

        return {
            "phase": phase,
            "name": name,
            "model": model,
            "history": history,
            "evaluation": evaluation,
            "json_path": jsonPath,
            "model_path": modelPath
        }


    def getModelsForPhase(self, phase, currentModel):
        # Gibt je nach aktueller Experimentphase die passenden Modellvarianten zurueck
        # So kann der Runner jede Phase gleich behandeln, obwohl jede Phase andere Modellanpassungen testet

        if phase == "ConvolutionLayer":
            return self.factory.getConvolutionLayerModels()

        if phase == "ConvolutionKernelSize":
            return self.factory.getConvolutionKernelSizeModels(currentModel)

        if phase == "PoolingMethod":
            return self.factory.getPoolingMethodModels(currentModel)

        if phase == "FullyConnectedLayer":
            return self.factory.getFullyConnectedLayerModels(currentModel)

        if phase == "DropoutPositions":
            return self.factory.getDropoutPositionsModels(currentModel)

        if phase == "DropoutValue":
            return self.factory.getDropoutValueModels(currentModel)

        raise ValueError("Unbekannte Phase: " + phase)


    def runAllPhases(self):
        # Daten einmal vorbereiten, damit alle Experimente denselben Train und Val Split verwenden
        self.setupData()

        # Reihenfolge der Phasen festlegen
        phases = [
            "ConvolutionLayer",
            "ConvolutionKernelSize",
            "PoolingMethod",
            "FullyConnectedLayer",
            "DropoutPositions",
            "DropoutValue"
        ]

        currentModel = None
        allPhaseResults = []

        # Jede Phase testet mehrere Varianten und uebergibt die beste Variante an die naechste Phase
        for phase in phases:
            testingModels = self.getModelsForPhase(phase, currentModel)

            phaseResults = []

            # Alle Modellvarianten der aktuellen Phase trainieren
            for model, name in testingModels:
                result = self.trainModel(
                    phase,
                    name,
                    model
                )

                phaseResults.append(result)

            # Bestes Modell der Phase anhand des berechneten Scores bestimmen
            bestResult = max(
                phaseResults,
                key=lambda result: result["evaluation"]["score"]
            )

            # Gewinner der aktuellen Phase wird Grundlage fuer die naechste Phase
            currentModel = bestResult["model"]

            # Zusammenfassung der aktuellen Phase speichern
            allPhaseResults.append({
                "phase": phase,
                "best_name": bestResult["name"],
                "best_score": bestResult["evaluation"]["score"],
                "best_json_path": bestResult["json_path"],
                "best_model_path": bestResult["model_path"],
                "results": [
                    {
                        "name": result["name"],
                        "score": result["evaluation"]["score"],
                        "val_accuracy": result["evaluation"]["val_accuracy"],
                        "val_loss": result["evaluation"]["val_loss"],
                        "json_path": result["json_path"],
                        "model_path": result["model_path"]
                    }
                    for result in phaseResults
                ]
            })

            print()
            print("BEST IN PHASE:", phase)
            print("Name:", bestResult["name"])
            print("Score:", bestResult["evaluation"]["score"])
            print("Val Accuracy:", bestResult["evaluation"]["val_accuracy"])
            print("Val Loss:", bestResult["evaluation"]["val_loss"])

        # Gesamte Phasenzusammenfassung als JSON speichern
        self.saveFinalSummary(allPhaseResults)


    def saveFinalSummary(self, allPhaseResults):
        # Pfad fuer die finale Zusammenfassung erstellen
        path = os.path.join(self.outputDir, "final_summary.json")

        # Alle Phasenergebnisse als JSON speichern
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "created_at": datetime.now().isoformat(),
                "phases": allPhaseResults
            }, f, indent=4)


    def sanitizeFilename(self, text):
        # Ungueltige Zeichen fuer Dateinamen durch Unterstriche ersetzen (Sicher ist Sicher nicht das es nach 8h einen crash gibt)
        invalidChars = [" ", "/", "\\", ":", "*", "?", "\"", "<", ">", "|", "[", "]", "(", ")", ","]

        for char in invalidChars:
            text = text.replace(char, "_")

        return text


if __name__ == "__main__":
    runner = ExperimentRunner()
    runner.runAllPhases()