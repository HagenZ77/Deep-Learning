import hashlib
import shutil
import re
import os
import random
from tools.label import Label
from collections import defaultdict


def getAllImages(basePath):
    fileList = []

    # Sammelt rekursiv alle Bilder
    for nextPath in os.listdir(basePath):
        nextPath = os.path.join(basePath, nextPath)

        if (os.path.isdir(nextPath)):
            fileList.extend(getAllImages(nextPath))
        elif nextPath.endswith(".png") or nextPath.endswith(".jpg"):
            fileList.append(nextPath)

    return fileList


def checkingDuplicatesImages(pathList, debug=False):
    hashList = []
    duplicates = []
    counter = 0

    for checkingPath in pathList:
        # Bild binaer lesen
        file = open(checkingPath, "rb")
        content = file.read()
        file.close()

        # Prueft Bilder per MD5-Hash auf doppelte Dateien
        hash = hashlib.md5(content).hexdigest()
        if hash in hashList:
            duplicates.append(checkingPath)

            index = hashList.index(hash)
            if debug:
                print(pathList[index] + " == " + checkingPath)
                counter += 1

        hashList.append(hash)

    if debug:
        print(counter, " Duplicates")

    return duplicates


def createOutputDir(name):

    if os.path.exists(name):
        shutil.rmtree(name)

    return os.mkdir(name)


def copyLabelFile(csv, outputPath):
    shutil.copy2(csv, outputPath)


def isValidImage(imageBasename):
    pattern = r"^(\d{5})_(\d{5})_(\d{5})$"
    match = re.match(pattern, imageBasename)

    # Pruefen, ob der Dateiname dem Format Label_Batch_Index entspricht
    if not match:
        return False

    label = match.group(1)

    # Label 6 wird ausgeschlossen, weil dies eine Schnittmenge ist mit Label 32
    if label == "00006":
        return False

    return True


def removeDuplicates(validImages):
    duplicates = checkingDuplicatesImages(validImages, True)

    # Alle Bilder die mehrfach vorkommen entfernen
    for duplicate in duplicates:
        os.remove(duplicate)


def labelCounter(labelFile, dir):
    # Labeldatei laden und Zaehler fuer jedes Label vorbereiten
    l = Label(labelFile)
    counter = [0 for i in range(0, len(l.indices))]

    # Nur Bilder direkt im angegebenen Ordner zaehlen
    images = [
        imagePath for imagePath in getAllImages(dir)
        if os.path.dirname(imagePath) == dir
    ]

    # Label aus jedem gueltigen Dateinamen lesen und hochzaehlen
    for imagePath in images:
        pattern = r"^(\d{5})_(\d{5})_(\d{5})$"
        imageBasename = os.path.basename(imagePath).split(".")[0]
        match = re.match(pattern, imageBasename)

        if not match:
            continue

        label = int(match.group(1))

        # Label 6 haben wir wegen der Schnittmenge (Label 32) entfernt
        if label == 6:
            continue

        # Labels nach dem entfernten Label 6 um eins nach unten verschieben
        if label > 6:
            label -= 1

        counter[label] += 1

    # Anzahl pro Label ausgeben
    for index in l.indices:
        print(index, l.getLabelByIndex(index), counter[index])

    print("Total number of Images: " + str(len(images)))

    # Ergebnis als Liste fuer die spaetere Aufteilung zurueckgeben
    outputList = [[l.getLabelByIndex(i), counter[i]] for i in range(0, len(counter))]
    return outputList


def extractTestFolder(outputDir):
    # Test-CSV einlesen, weil die Testbilder ihr Label nicht im Dateinamen haben
    file = open("../Dataset/Test.csv")
    content = file.readlines()[1:] # Erste Zeile sind Spaltennamen
    file.close()

    # Alle Bilder aus dem offiziellen Testordner sammeln
    images = getAllImages("../Dataset/Test")
    counter = 0

    for imagePath in images:
        basename = os.path.basename(imagePath)
        label = False

        # Passenden CSV-Eintrag zum Bild suchen und daraus das Label lesen
        for labelData in content:
            if basename in labelData:
                label = int(labelData.split(",")[-2])

        if not label: continue

        # Testbild in das einheitliche Format Label_Batch_Index umbenennen
        # Batch 99999 markiert dabei Bilder aus dem Testordner
        counter += 1
        formatted_name = "{:05d}_{:05d}_{:05d}".format(label, 99999, counter)
        outputFile = os.path.join(outputDir, formatted_name+"."+basename.split(".")[1])

        # Nur gueltige Bilder kopieren
        if isValidImage(formatted_name):
            shutil.copy2(imagePath, outputFile)


def splitDataToTrainValTest(outputDir, countedLabels, valSplit=0.15, testSplit=0.15):
    # Zielordner fuer die Aufteilung festlegen
    trainPath = os.path.join(outputDir, "train")
    valPath = os.path.join(outputDir, "val")
    testPath = os.path.join(outputDir, "test")

    labels = Label(os.path.join(outputDir, "labels.txt"))

    # Bilder holen
    images = [
        imagePath for imagePath in getAllImages(outputDir)
        if os.path.dirname(imagePath) == outputDir
    ]

    os.makedirs(trainPath, exist_ok=True)
    os.makedirs(valPath, exist_ok=True)
    os.makedirs(testPath, exist_ok=True)

    pattern = r"^(\d{5})_(\d{5})_(\d{5})$"

    # Fuer jedes Label separat aufteilen, damit alle Klassen in allen Splits vorkommen koennen
    for labelData in countedLabels:
        labelName = labelData[0]
        labelNumber = labels.getIndexByLabel(labelName)

        batches = defaultdict(list)

        for imagePath in images:
            imageBasename = os.path.splitext(os.path.basename(imagePath))[0]
            match = re.match(pattern, imageBasename)

            if not match:
                continue

            imageLabel = int(match.group(1))
            imageBatch = match.group(2)

            # Labels nach dem entfernten Label 6 um eins nach unten verschieben (Label 32 Schnittmenge)
            if imageLabel == 6:
                continue
            if imageLabel > 6:
                imageLabel -= 1

            if imageLabel == labelNumber:
                batches[imageBatch].append(imagePath)

        # Batchgruppen zufaellig mischen
        batchGroups = list(batches.values())
        random.shuffle(batchGroups)

        numberOfBatches = len(batchGroups)

        # Anzahl der Val- und Test-Batches bestimmen
        numberOfValBatches = int(numberOfBatches * valSplit)
        numberOfTestBatches = int(numberOfBatches * testSplit)

        # Bei ausreichend Daten mindestens einen Batch fuer Val und Test verwenden
        if numberOfBatches >= 3:
            if numberOfValBatches == 0:
                numberOfValBatches = 1

            if numberOfTestBatches == 0:
                numberOfTestBatches = 1

        valEnd = numberOfValBatches
        testEnd = numberOfValBatches + numberOfTestBatches


        # Ganze Batchgruppen in train, val oder test kopieren
        for index, batchImages in enumerate(batchGroups):
            if index < valEnd:
                targetPath = valPath
            elif index < testEnd:
                targetPath = testPath
            else:
                targetPath = trainPath

            for imagePath in batchImages:
                shutil.copy2(
                    imagePath,
                    os.path.join(targetPath, os.path.basename(imagePath))
                )


def cleanupRootImages(outputDir):

    images = [
        imagePath for imagePath in getAllImages(outputDir)
        if os.path.dirname(imagePath) == outputDir
    ]

    for imagePath in images:
        os.remove(imagePath)


def run():
    outputDir = "../CleanDataset"

    # Ausgabeordner neu erstellen, Labeldatei kopieren und Testdaten vorbereiten
    createOutputDir(outputDir)
    copyLabelFile("../Dataset/labels.csv", outputDir+"/labels.txt")
    extractTestFolder(outputDir)

    # Alle Bilder aus dem Originaldatensatz sammeln
    images = getAllImages("../Dataset")

    # Gueltige Bilder in den CleanDataset kopieren
    for imagePath in images:
        imageBasename = os.path.basename(imagePath).split(".")[0]

        if not isValidImage(imageBasename):
            continue

        parent_folder = os.path.basename(os.path.dirname(imagePath))
        basename = os.path.basename(imagePath)

        # Im Ordner 22 waren Bilder falsch mit 00021 gelabelt
        # Deshalb wird fuer diese Bilder das Label aus dem Ordnernamen uebernommen
        if parent_folder == "22" and basename.startswith("00021_"):
            basename = "00022_" + basename[len("00021_"):]

        output = os.path.join(outputDir, basename)
        shutil.copy2(imagePath, output)

    # Doppelte Bilder entfernen
    removeDuplicates(getAllImages(outputDir))

    # Bilder pro Label zaehlen, in train/val/test aufteilen und Root-Bilder entfernen
    countedLabels = labelCounter(outputDir+"/labels.txt", outputDir)
    splitDataToTrainValTest(outputDir, countedLabels)
    cleanupRootImages(outputDir)


if __name__ == "__main__":
    run()