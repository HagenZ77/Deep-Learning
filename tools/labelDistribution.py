import os
from collections import Counter
import matplotlib.pyplot as plt
from label import Label
from tools.makeCleanDataset import getAllImages


def createLabelDistributionDiagram(imageFolder, labelFile, outputPath):
    # Labeldatei laden
    labels = Label(labelFile)
    imageFiles = getAllImages(imageFolder)

    labelCounts = Counter()
    invalidFiles = []

    # Fuer jedes Bild das Label aus dem Dateinamen lesen und zaehlen
    for imagePath in imageFiles:
        info = labels.getInfoFromImagePath(imagePath)

        if info is False:
            # Ungueltige Dateinamen merken, damit sie spaeter kontrolliert werden koennen
            invalidFiles.append(imagePath)
            continue

        labelCounts[info.label] += 1

    x_labels = []
    y_values = []

    # Achsen-Beschriftungen und Werte fuer das Diagramm vorbereiten
    for labelIndex in labels.indices:
        labelName = labels.getLabelByIndex(labelIndex)
        count = labelCounts.get(labelIndex, 0)

        x_labels.append(f"{labelIndex}: {labelName}")
        y_values.append(count)

    # Ungueltige Bilder ausgeben
    if invalidFiles:
        print("\nUngueltige bilder:")
        for path in invalidFiles:
            print(path)


    # Balkendiagramm erstellen und beschriften
    plt.figure(figsize=(18, 8))
    plt.bar(range(len(y_values)), y_values)

    plt.title("Anzahl der Bilder pro Label")
    plt.xlabel("Label")
    plt.ylabel("Anzahl Bilder")

    plt.xticks(
        range(len(x_labels)),
        x_labels,
        rotation=90,
        fontsize=7
    )

    plt.tight_layout()
    plt.savefig(outputPath, dpi=300)
    plt.show()


if __name__ == "__main__":
    imageFolder = "/mnt/c/Users/hagen/Desktop/DeepLearning/CleanDataset/train"
    labelFile = "/mnt/c/Users/hagen/Desktop/DeepLearning/CleanDataset/labels.txt"
    outputPath = "labelDistribution.png"

    createLabelDistributionDiagram(imageFolder, labelFile, outputPath)
