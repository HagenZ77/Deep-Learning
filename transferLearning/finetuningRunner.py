import copy
import os
import torch.nn as nn
from torchvision.models import vgg13, VGG13_Weights
from train import Train
from transferLearningRunner import TransferLearningRunner

# Fuehrt das Fine-Tuning fuer VGG13 durch.
# Der Classifier bleibt trainierbar.
# Zusaetzlich werden die letzten 1 bis 5 Conv-Layer freigegeben.
# Jede Variante wird mit 0.0001 und 0.00001 getestet.
class VGG13FineTuningRunner(TransferLearningRunner):
    def __init__(self):
        self.trainer = Train()
        self.device = self.trainer.createDevice()

        self.epochs = 40
        self.batchSize = 32
        self.patience = 2
        self.lossFunction = nn.CrossEntropyLoss()

        self.learningRates = [0.0001, 0.00001]
        self.unfreezeLayerCounts = [1, 2, 3, 4, 5]

        self.resultRoot = "/mnt/c/Users/hagen/Desktop/DeepLearning/TransferLearningResults/vgg13"
        self.jsonFolder = os.path.join(self.resultRoot, "json")
        self.modelFolder = os.path.join(self.resultRoot, "models")
        self.diagramFolder = os.path.join(self.resultRoot, "diagrams")

        self.createFolders()


    def freezeAllParameters(self, model):
        for param in model.parameters():
            param.requires_grad = False


    def unfreezeClassifier(self, model):
        for param in model.classifier.parameters():
            param.requires_grad = True


    def getConvLayerIndices(self, model):
        # Sammelt die Indices aller Conv2d-Layer aus dem VGG13-Feature-Extractor.
        convLayerIndices = []

        for index, layer in enumerate(model.features):
            if isinstance(layer, nn.Conv2d):
                convLayerIndices.append(index)

        # Wird spaeter genutzt, um die letzten Conv-Layer gezielt freizugeben.
        return convLayerIndices


    def unfreezeLastConvLayers(self, model, layerCount):
        # Waehlt die letzten layerCount Conv-Layer aus.
        convLayerIndices = self.getConvLayerIndices(model)
        selectedIndices = convLayerIndices[-layerCount:]

        # Gibt diese Conv-Layer fuer das Fine-Tuning frei.
        for index in selectedIndices:
            for param in model.features[index].parameters():
                param.requires_grad = True

        return selectedIndices


    def createModel(self, layerCount):
        # Laedt VGG13 mit ImageNet-Gewichten.
        weights = VGG13_Weights.DEFAULT
        model = vgg13(weights=weights)

        # Ersetzt die letzte Classifier-Schicht passend zu unseren 62 Klassen.
        oldClassifierLayer = model.classifier[6]
        model.classifier[6] = nn.Linear(
            oldClassifierLayer.in_features,
            self.trainer.classes
        )

        # Friert zuerst alles ein und gibt dann Classifier + letzte Conv-Layer frei.
        self.freezeAllParameters(model)
        self.unfreezeClassifier(model)
        unfrozenFeatureIndices = self.unfreezeLastConvLayers(model, layerCount)

        return model, unfrozenFeatureIndices


    def fit(self, model, trainLoader, valLoader, learningRate):
        history = {
            "loss": [],
            "accuracy": [],
            "val_loss": [],
            "val_accuracy": []
        }

        bestValLoss = None
        bestValAccuracy = None
        bestEpoch = None
        bestModelState = None
        patienceCounter = 0

        self.learningRate = learningRate

        optimizer = self.createOptimizer(
            filter(lambda param: param.requires_grad, model.parameters())
        )

        for epoch in range(self.epochs):
            trainLoss, trainAccuracy = self.trainOneEpoch(
                model,
                trainLoader,
                optimizer
            )

            valLoss, valAccuracy = self.evaluate(
                model,
                valLoader
            )

            history["loss"].append(trainLoss)
            history["accuracy"].append(trainAccuracy)
            history["val_loss"].append(valLoss)
            history["val_accuracy"].append(valAccuracy)

            print(epoch + 1, trainLoss, trainAccuracy, valLoss, valAccuracy)

            if bestValLoss is None or valLoss < bestValLoss:
                bestValLoss = valLoss
                bestValAccuracy = valAccuracy
                bestEpoch = epoch + 1
                bestModelState = copy.deepcopy(model.state_dict())
                patienceCounter = 0
            else:
                patienceCounter += 1

            if patienceCounter >= self.patience:
                break

        if bestModelState is not None:
            model.load_state_dict(bestModelState)

        return history, bestValLoss, bestValAccuracy, bestEpoch, optimizer


    def createResultData(self, model, history, trainFiles, valFiles, modelName, learningRate, layerCount, unfrozenFeatureIndices, bestValLoss, bestValAccuracy, bestEpoch):
        resultData = self.trainer.createResultData(
            model,
            history,
            trainFiles,
            valFiles,
            self.epochs,
            self.batchSize,
            learningRate
        )

        resultData["name"] = modelName
        resultData["phase"] = "vgg13_fine_tuning"

        resultData["training"]["loss"] = "cross_entropy_loss"
        resultData["training"]["trained_part"] = "classifier_and_last_vgg13_feature_layers"
        resultData["training"]["unfrozen_conv_layer_count"] = layerCount
        resultData["training"]["unfrozen_feature_indices"] = unfrozenFeatureIndices

        resultData["evaluation"] = {
            "best_epoch": bestEpoch,
            "best_val_loss": bestValLoss,
            "best_val_accuracy": bestValAccuracy,
            "max_val_accuracy": max(history["val_accuracy"]),
            "final_val_loss": history["val_loss"][-1],
            "final_val_accuracy": history["val_accuracy"][-1]
        }

        return resultData


    def saveAll(self, model, history, trainFiles, valFiles, modelName, learningRate, layerCount, unfrozenFeatureIndices, bestValLoss, bestValAccuracy, bestEpoch):
        jsonPathWithoutExtension = self.getJsonPathWithoutExtension(modelName)
        jsonPath = self.getJsonPath(modelName)
        modelPath = self.getModelPath(modelName)
        diagramPath = self.getDiagramPath(modelName)

        resultData = self.createResultData(
            model,
            history,
            trainFiles,
            valFiles,
            modelName,
            learningRate,
            layerCount,
            unfrozenFeatureIndices,
            bestValLoss,
            bestValAccuracy,
            bestEpoch
        )

        self.trainer.saveResultJson(resultData, jsonPathWithoutExtension)
        self.trainer.saveModel(model, modelPath)
        self.saveHistoryDiagram(history, diagramPath, modelName)

        print("Gespeichert:")
        print("  JSON:    " + jsonPath)
        print("  Modell:  " + modelPath)
        print("  Diagramm:" + diagramPath)


    def runOneTest(self, layerCount, learningRate, trainLoader, valLoader, trainFiles, valFiles):
        # Erstellt Modellnamen passend zur aktuellen Layer-/Lernraten-Kombination.
        lrName = str(learningRate).replace(".", "_")
        modelName = "VGG13_finetune_last_" + str(layerCount) + "_conv_layers_lr_" + lrName

        print("")
        print("============================================================ ")
        print("Starte Modell: " + modelName)
        print(" ============================================================")

        # Erstellt VGG13 mit den freigegebenen Conv-Layern.
        model, unfrozenFeatureIndices = self.createModel(layerCount)
        model = model.to(self.device)

        print("Freigegebene VGG13 feature indices: " + str(unfrozenFeatureIndices))

        # Trainiert und validiert diese Fine-Tuning-Variante.
        history, bestValLoss, bestValAccuracy, bestEpoch, optimizer = self.fit(
            model,
            trainLoader,
            valLoader,
            learningRate
        )

        # Speichert Modell, JSON und Diagramm.
        self.saveAll(
            model,
            history,
            trainFiles,
            valFiles,
            modelName,
            learningRate,
            layerCount,
            unfrozenFeatureIndices,
            bestValLoss,
            bestValAccuracy,
            bestEpoch
        )

        self.cleanup(model, optimizer)

        print("Fertig: " + modelName)


    def run(self):
        # Trainings- und Validierungsdaten laden.
        trainFiles, valFiles, labelFile = self.trainer.loadFiles()

        # DataLoader fuer Training und Validierung erstellen.
        trainLoader, valLoader = self.createTransferDataLoaders(
            trainFiles,
            valFiles,
            labelFile
        )

        # Alle Kombinationen aus Lernrate und freigegebenen Conv-Layern testen.
        for learningRate in self.learningRates:
            for layerCount in self.unfreezeLayerCounts:
                self.runOneTest(
                    layerCount,
                    learningRate,
                    trainLoader,
                    valLoader,
                    trainFiles,
                    valFiles
                )


if __name__ == "__main__":
    runner = VGG13FineTuningRunner()
    runner.run()