import matplotlib
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tools.label import Label
from tools.makeCleanDataset import getAllImages
from train import ImageDataset


def loadFullModel(modelPath, device):
    # Laedt das gespeicherte finale Modell und setzt es in den Auswertungsmodus
    model = torch.load(modelPath, map_location=device, weights_only=False)
    model = model.to(device)
    model.eval()
    return model


def evaluateModel(model, testLoader, device, numClasses):
    # Leere Matrix erstellen.
    confusion = torch.zeros(numClasses, numClasses, dtype=torch.int64)

    correct = 0
    total = 0

    # Modell in den Auswertungsmodus setzen.
    model.eval()

    # Testdaten ohne Gradientenberechnung durchlaufen
    with torch.no_grad():
        for batchX, batchY in testLoader:
            batchX = batchX.to(device)
            batchY = batchY.to(device)

            # Vorhersage berechnen und die Klasse mit dem hoechsten Wert bestimmen
            output = model(batchX)
            predicted = torch.argmax(output, dim=1)

            # Korrekte Vorhersagen und Gesamtanzahl summieren
            correct += (predicted == batchY).sum().item()
            total += batchY.size(0)

            # Matrix fuellen
            for trueLabel, predLabel in zip(batchY.cpu(), predicted.cpu()):
                confusion[trueLabel, predLabel] += 1

    # Finale Testgenauigkeit berechnen
    accuracy = correct / total

    return confusion.numpy(), accuracy


def plotConfusionHeatmap(confusion, labelFile, savePath=None):
    matplotlib.use('TkAgg')
    fig, ax = plt.subplots(figsize=(18, 16))

    image = ax.imshow(confusion)

    labels = [labelFile.getLabelByIndex(i) for i in range(confusion.shape[0])]

    ax.set_title("Heatmap der Modell-Aussagen auf den Testdaten")
    ax.set_xlabel("Vorhergesagte Klasse")
    ax.set_ylabel("Echte Klasse")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))

    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)

    fig.colorbar(image, ax=ax, label="Anzahl Bilder")

    plt.tight_layout()

    if savePath is not None:
        plt.savefig(savePath, dpi=300)

    plt.show()



def main():
    # Pfade zu Testdaten, Labeldatei und final gespeichertem Modell. Muessen angepasst werden
    testDir = "/mnt/c/Users/hagen/Desktop/DeepLearning/CleanDataset/test"
    labelPath = "/mnt/c/Users/hagen/Desktop/DeepLearning/CleanDataset/labels.txt"
    modelPath = "/mnt/c/Users/hagen/Desktop/DeepLearning/finalModel.pth"

    # Versuchen mit der GPU zu arbeiten sonst CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Labeldatei einlesen und alle Testbilder sammeln.
    labelFile = Label(labelPath)
    testFiles = getAllImages(testDir)

    testDataset = ImageDataset(testFiles, labelFile)

    # DataLoader fuer die batchweise Testauswertung erstellen
    testLoader = DataLoader(
        testDataset,
        batch_size=32,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )

    model = loadFullModel(modelPath, device)

    # Modellaufbau zur Kontrolle ausgeben.
    print("======================")
    print("     Modellaufbau")
    print("======================")
    for index, layer in enumerate(model):
        print(layer)

    # Modell auf dem Testdatensatz auswerten
    confusion, accuracy = evaluateModel(
        model,
        testLoader,
        device,
        62
    )

    print("Test Accuracy:", accuracy)

    # Confusion-Matrix als Heatmap speichern und anzeigen
    plotConfusionHeatmap(
        confusion,
        labelFile,
        savePath="images/test_confusion_heatmap.png"
    )


if __name__ == "__main__":
    main()
