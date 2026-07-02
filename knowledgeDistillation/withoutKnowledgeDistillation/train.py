from matplotlib import pyplot as plt
from tools.train import Train, LossMode
from tools.ImageDataset128 import ImageDataset128

if __name__ == "__main__":
    # Trainer fuer GoogLeNet ohne Teacher erstellen
    trainer = Train(LossMode.LOGITS)
    device = trainer.createDevice()

    # Einstellungen und Ausgabepfade setzen
    learningRate = 0.0001
    modelName = "googlenet_without_teacher"
    jsonOutputPath = "/mnt/c/Users/hagen/Desktop/Semester6/DeepLearning/knowledgeDistillation/withoutKnowledgeDistillation/results/json/googlenet_without_teacher.json"
    diagramOutputPath = "/mnt/c/Users/hagen/Desktop/Semester6/DeepLearning/knowledgeDistillation/withoutKnowledgeDistillation/results/diagrams/googlenet_without_teacher.png"
    modelOutputPath = "/mnt/c/Users/hagen/Desktop/Semester6/DeepLearning/knowledgeDistillation/withoutKnowledgeDistillation/results/models/googlenet_without_teacher.pth"

    # GoogLeNet-Modell ohne Teacher erstellen
    model = trainer.getGoogLeNet()
    model = model.to(device)

    # Optimizer fuer das Modell erstellen
    optimizer = trainer.createOptimizer(model, learningRate)

    # Trainings- und Validierungsdaten laden
    trainFiles, valFiles, labelFile = trainer.loadFiles()

    # Datasets mit 128x128 Bildern fuer GoogLeNet erstellen
    trainDataset = ImageDataset128(trainFiles, labelFile)
    valDataset = ImageDataset128(valFiles, labelFile)

    # DataLoader fuer Training und Validierung erstellen
    trainLoader, valLoader = trainer.createDataLoaders(
        trainDataset,
        valDataset
    )

    # Modell normal ohne Knowledge-Distillation trainieren
    history, fitInfo = trainer.fit(
        model,
        trainLoader,
        valLoader,
        optimizer,
        device
    )

    # Ergebnisdaten fuer die JSON-Datei erstellen
    resultData = trainer.createResultData(
        model,
        history,
        fitInfo,
        trainFiles,
        valFiles,
        learningRate
    )

    # Modellname in den Ergebnisdaten speichern
    resultData["training"]["model"] = modelName

    # JSON-Datei und Trainingsdiagramm speichern
    trainer.saveJsonAndDiagram(
        resultData,
        jsonOutputPath,
        diagramOutputPath
    )

    # Trainiertes Modell speichern
    trainer.saveModel(model, modelOutputPath)

    plt.show(block=True)