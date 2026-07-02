from tools.test import EvaluateTestModel
from tools.ImageDataset128 import ImageDataset128


def main():
    # Pfade zu Testdaten, Labeldatei und final gespeichertem Modell. Muessen angepasst werden
    testDir = "/mnt/c/Users/hagen/Desktop/Semester6/DeepLearning/CleanDataset/test"
    labelPath = "/mnt/c/Users/hagen/Desktop/Semester6/DeepLearning/CleanDataset/labels.txt"
    modelPath = "/mnt/c/Users/hagen/Desktop/Semester6/DeepLearning/knowledgeDistillation/results/models/googlenet_distillation_student.pth"
    heatmapSavePath = "/mnt/c/Users/hagen/Desktop/Semester6/DeepLearning/knowledgeDistillation/results/diagrams/googlenet_distillation_student_heatmap.png"
    jsonSavePath = "/mnt/c/Users/hagen/Desktop/Semester6/DeepLearning/knowledgeDistillation/results/json/googlenet_distillation_student_test.json"

    evaluator = EvaluateTestModel()

    evaluator.runTest(
        modelPath,
        testDir,
        labelPath,
        ImageDataset128,
        heatmapSavePath,
        jsonSavePath,
        "googlenet_distillation_student"
    )


if __name__ == "__main__":
    main()