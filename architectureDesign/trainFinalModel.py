from matplotlib import pyplot as plt
from experiment.ModelFactory import ModelFactory
from tools.train import Train, LossMode
from tools.ImageDataset128 import ImageDataset128

if __name__ == "__main__":
    trainer = Train(LossMode.PROBABILITIES)

    device = trainer.createDevice()

    epochs = 40
    batchSize = 32
    patience = 2
    learningRate = 0.0003

    trainer.EPOCHS = epochs
    trainer.BATCH_SIZE = batchSize
    trainer.PATIENCE = patience

    jsonOutputPath = "/mnt/c/Users/hagen/Desktop/Semester6/DeepLearning/architectureDesign/results/json/finalModel.json"
    diagramOutputPath = "/mnt/c/Users/hagen/Desktop/Semester6/DeepLearning/architectureDesign/results/diagrams/finalModel.png"
    modelOutputPath = "/mnt/c/Users/hagen/Desktop/Semester6/DeepLearning/architectureDesign/results/models/finalModel.pth"

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

    # Optimizer fuer das finale Training erstellen
    optimizer = trainer.createOptimizer(model, learningRate)

    # Trainings- und Validierungsdaten laden.
    trainFiles, valFiles, labelFile = trainer.loadFiles()

    trainDataset = ImageDataset128(trainFiles, labelFile)
    valDataset = ImageDataset128(valFiles, labelFile)

    # DataLoader erstellen, damit die Bilder batchweise geladen werden
    trainLoader, valLoader = trainer.createDataLoaders(
        trainDataset,
        valDataset
    )

    # Finales Modell trainieren und dabei die besten Validierungswerte speichern
    history, fitInfo = trainer.fit(
        model,
        trainLoader,
        valLoader,
        optimizer,
        device
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
        fitInfo,
        trainFiles,
        valFiles,
        learningRate
    )

    resultData["training"]["model"] = "finalModel"

    trainer.saveJsonAndDiagram(
        resultData,
        jsonOutputPath,
        diagramOutputPath
    )

    # Modell+Gewichte speichern
    trainer.saveModel(model, modelOutputPath)