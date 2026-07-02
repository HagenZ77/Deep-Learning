from tools.test import EvaluateTestModel
from tools.ImageDataset128 import ImageDataset128


def main():
    # Pfade zu Testdaten, Labeldatei und final gespeichertem Modell. Muessen angepasst werden
    testDir = "/mnt/c/Users/hagen/Desktop/Semester6/DeepLearning/CleanDataset/test"
    labelPath = "/mnt/c/Users/hagen/Desktop/Semester6/DeepLearning/CleanDataset/labels.txt"
    modelPath = "/mnt/c/Users/hagen/Desktop/Semester6/DeepLearning/knowledgeDistillation/withoutKnowledgeDistillation/results/models/googlenet_without_teacher.pth"
    jsonOutputPath = "/mnt/c/Users/hagen/Desktop/Semester6/DeepLearning/knowledgeDistillation/withoutKnowledgeDistillation/results/json/googlenet_without_teacher_test.json"
    heatmapSavePath = "/mnt/c/Users/hagen/Desktop/Semester6/DeepLearning/knowledgeDistillation/withoutKnowledgeDistillation/results/diagrams/googlenet_without_teacher_heatmap.png"

    # Trainiertes GoogLeNet ohne Teacher auf den Testdaten bewerten
    EvaluateTestModel().runTest(
        modelPath,
        testDir,
        labelPath,
        ImageDataset128,
        heatmapSavePath,
        jsonOutputPath,
        "googlenet_without_teacher_heatmap"

    )


if __name__ == "__main__":
    main()