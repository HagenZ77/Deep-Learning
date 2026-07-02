import os
import json
import csv


class CreateTestConfusionTable:
    def createFolderForFile(self, filePath):
        folder = os.path.dirname(filePath)

        if folder != "":
            os.makedirs(folder, exist_ok=True)


    def loadJson(self, jsonPath):
        with open(jsonPath, "r", encoding="utf-8") as file:
            return json.load(file)


    def createRows(self, jsonFiles, maxRowsPerModel):
        rows = []

        for jsonPath in jsonFiles:
            resultData = self.loadJson(jsonPath)

            modelName = resultData["model"]["name"]
            accuracy = resultData["test"]["accuracy"]
            testCount = resultData["dataset"]["testCount"]
            wrongPredictions = resultData["test"]["wrongPredictions"]

            confusedLabels = resultData["mistakes"]["confusedLabels"]
            confusedLabels = confusedLabels[:maxRowsPerModel]

            for confusedLabel in confusedLabels:
                count = confusedLabel["count"]

                if wrongPredictions > 0:
                    percentOfErrors = count / wrongPredictions
                else:
                    percentOfErrors = 0

                row = {
                    "modelName": modelName,
                    "accuracy": accuracy,
                    "testCount": testCount,
                    "wrongPredictions": wrongPredictions,
                    "trueLabel": confusedLabel["trueLabel"],
                    "predictedLabel": confusedLabel["predictedLabel"],
                    "count": count,
                    "percentOfErrors": percentOfErrors
                }

                rows.append(row)

        return rows


    def saveCsvTable(self, rows, outputPath):
        self.createFolderForFile(outputPath)

        fieldNames = [
            "modelName",
            "accuracy",
            "testCount",
            "wrongPredictions",
            "trueLabel",
            "predictedLabel",
            "count",
            "percentOfErrors"
        ]

        with open(outputPath, "w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldNames,
                delimiter=";"
            )

            writer.writeheader()

            for row in rows:
                writer.writerow(row)


def main():
    jsonFiles = [
        "/mnt/c/Users/hagen/Desktop/Semester6/DeepLearning/architectureDesign/results/json/finalModel_test.json",
        "/mnt/c/Users/hagen/Desktop/Semester6/DeepLearning/transferLearning/results/vgg13/json/VGG13_test.json",
        "/mnt/c/Users/hagen/Desktop/Semester6/DeepLearning/knowledgeDistillation/results/json/knowledge_distillation_test.json",
        "/mnt/c/Users/hagen/Desktop/Semester6/DeepLearning/knowledgeDistillation/withoutKnowledgeDistillation/results/json/googlenet_without_teacher_test.json"
    ]

    outputPath = "/mnt/c/Users/hagen/Desktop/Semester6/DeepLearning/results/tables/test_confusion_table.csv"

    tableCreator = CreateTestConfusionTable()

    rows = tableCreator.createRows(
        jsonFiles,
        maxRowsPerModel=10
    )

    tableCreator.saveCsvTable(
        rows,
        outputPath
    )

    print("CSV gespeichert:", outputPath)


if __name__ == "__main__":
    main()