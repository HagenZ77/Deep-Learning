import torch
from tools.label import Label
from tools.makeCleanDataset import getAllImages
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from EvaluateTestModel import plotConfusionHeatmap, evaluateModel, loadFullModel
from train import ImageDataset


class TransferImageDataset(ImageDataset):
    def loadImage(self, path):
        img = Image.open(path).convert("RGB")
        img = img.resize((224, 224))
        img = np.array(img).astype("float32") / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype="float32")
        std = np.array([0.229, 0.224, 0.225], dtype="float32")

        img = (img - mean) / std
        img = np.transpose(img, (2, 0, 1))

        return img

    def __getitem__(self, index):
        path = self.files[index]

        x = self.loadImage(path)
        y = self.labelFile.getInfoFromImagePath(path).label

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        return x, y


def main():
    # Pfade zu Testdaten, Labeldatei und final gespeichertem Modell. Muessen angepasst werden
    testDir = "/mnt/c/Users/hagen/Desktop/Semester6/DeepLearning/CleanDataset/test"
    labelPath = "/mnt/c/Users/hagen/Desktop/Semester6/DeepLearning/CleanDataset/labels.txt"
    modelPath = "/mnt/c/Users/hagen/Desktop/Semester6/DeepLearning/TransferLearningResults/vgg13/models/VGG13_finetune_last_5_conv_layers_lr_1e-05.pth"

    # Versuchen mit der GPU zu arbeiten sonst CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Labeldatei einlesen und alle Testbilder sammeln.
    labelFile = Label(labelPath)
    testFiles = getAllImages(testDir)
    testDataset = TransferImageDataset(testFiles, labelFile)

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
        savePath="images/VGG13_test_confusion_heatmap.png"
    )


if __name__ == "__main__":
    main()
