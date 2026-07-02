from tools.test import EvaluateTestModel
from tools.ImageDataset224 import ImageDataset224


def main():
    # Pfade zu Testdaten, Labeldatei und final gespeichertem Modell. Muessen angepasst werden
    testDir = "/mnt/c/Users/hagen/Desktop/Semester6/DeepLearning/CleanDataset/test"
    labelPath = "/mnt/c/Users/hagen/Desktop/Semester6/DeepLearning/CleanDataset/labels.txt"
    modelPath = "/mnt/c/Users/hagen/Desktop/Semester6/DeepLearning/transferLearning/results/vgg13/models/VGG13_finetune_last_5_conv_layers_lr_1e-05.pth"
    heatmapSavePath = "/mnt/c/Users/hagen/Desktop/Semester6/DeepLearning/transferLearning/results/vgg13/diagrams/VGG13_test_confusion_heatmap.png"
    jsonSavePath = "/mnt/c/Users/hagen/Desktop/Semester6/DeepLearning/transferLearning/results/vgg13/json/VGG13_test.json"

    evaluator = EvaluateTestModel()

    evaluator.runTest(
        modelPath,
        testDir,
        labelPath,
        ImageDataset224,
        heatmapSavePath,
        jsonSavePath,
        "VGG13_finetune_last_5_conv_layers_lr_1e-05"
    )


if __name__ == "__main__":
    main()