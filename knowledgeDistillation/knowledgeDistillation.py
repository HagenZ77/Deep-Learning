import torch
import torch.nn.functional as F
from tools.plot import Plot
from tools.train import Train, LossMode
from tools.ImageDataset224 import ImageDataset224
from tools.ImageDataset128 import ImageDataset128

# Diese Hilfsklasse ist wichtig, weil für das Knowledge-Distillation zwei modelle haben mit
# unterschiedlicher Eingabegrössen. Somit bekommen wir einmal GoogLeNet-Student(128x128) und einmal VGG13-Teacher(224x224)
class DistillationImageDataset(ImageDataset128):
    def __init__(self, files, labelFile):
        super().__init__(files, labelFile)
        self.teacherDataset = ImageDataset224(files, labelFile)

    def getStudentX(self, path):
        # normales ImageDataset: 128x128, / 255.0
        return ImageDataset128.loadImage(self, path)[0]


    def getTeacherX(self, path):
        # VGG13-Loader: 224x224, ImageNet normalisiert
        return self.teacherDataset.loadImage(path)


    def getY(self, path):
        return self.labelFile.getInfoFromImagePath(path).label


    def __getitem__(self, index):
        path = self.files[index]

        studentX = self.getStudentX(path)
        teacherX = self.getTeacherX(path)
        y = self.getY(path)

        studentX = torch.tensor(studentX, dtype=torch.float32)
        teacherX = torch.tensor(teacherX, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        return studentX, teacherX, y


class KnowledgeDistillation:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trainer = Train(LossMode.LOGITS)

        # Training
        self.epochs = 40
        self.batchSize = 32
        self.learningRate = 0.0001
        self.patience = 2

        # Knowledge-Distillation
        self.temperature = 4.0
        self.alpha = 0.5

        self.trainer.EPOCHS = self.epochs
        self.trainer.BATCH_SIZE = self.batchSize
        self.trainer.PATIENCE = self.patience


    # Zaehlt die trainierbaren und nicht trainierbaren Parameter eines Modells
    def countParameters(self, model):
        return sum(param.numel() for param in model.parameters())


    def getVGG13(self):
        modelPath = "/mnt/c/Users/hagen/Desktop/Semester6/DeepLearning/transferLearning/results/vgg13/models/VGG13_finetune_last_5_conv_layers_lr_1e-05.pth"

        model = torch.load(modelPath, map_location=self.device, weights_only=False)
        model = model.to(self.device)
        model.eval()

        for param in model.parameters():
            param.requires_grad = False

        return model


    def getGoogLeNet(self):
        model = self.trainer.getGoogLeNet()
        model = model.to(self.device)

        return model


    def distillationLoss(self, studentOutput, teacherOutput, labels):
        # Loss berechnen mit Student gegen echte Labels
        studentLoss = F.cross_entropy(studentOutput, labels)

        # Teacher und Student Ausgaben mit Temperatur weicher machen bzw. die werte näher an ein ander holen
        softStudent = F.log_softmax(studentOutput / self.temperature, dim=1)
        softTeacher = F.softmax(teacherOutput / self.temperature, dim=1)

        # Divergenz zwischen Teacher-Verteilung und Student-Verteilung berechnen
        distillationLoss = F.kl_div(
            softStudent,
            softTeacher,
            reduction="batchmean"
        ) * (self.temperature * self.temperature)

        # Student Loss und Distillation Loss mit Alpha gewichten
        totalLoss = self.alpha * studentLoss + (1 - self.alpha) * distillationLoss

        return totalLoss


    def trainOneEpoch(self, studentModel, teacherModel, trainLoader, optimizer):
        studentModel.train()
        teacherModel.eval()

        trainLossSum = 0
        trainCorrect = 0
        trainTotal = 0

        # trainLoader enthaelt einmal die Bilder fuer den Student, einmal fuer den Teacher und die echten Labels
        for studentX, teacherX, batchY in trainLoader:
            studentX = studentX.to(self.device, non_blocking=True)
            teacherX = teacherX.to(self.device, non_blocking=True)
            batchY = batchY.to(self.device, non_blocking=True)

            optimizer.zero_grad()

            # Student berechnet seine Vorhersage mit den 128x128 Bildern
            studentOutput = studentModel(studentX)

            # Teacher berechnet seine Ausgabe mit den 224x224 Bildern, wird aber nicht trainiert
            teacherOutput = teacherModel(teacherX)

            # Loss aus echten Labels und Teacherausgabe berechnen
            loss = self.distillationLoss(
                studentOutput,
                teacherOutput,
                batchY
            )

            # Nur die Gewichte vom Student aktualisieren
            loss.backward()
            optimizer.step()

            # Loss und richtige Vorhersagen fuer die Epoche summieren
            trainLossSum += loss.item() * studentX.size(0)
            trainTotal += studentX.size(0)
            trainCorrect += (torch.argmax(studentOutput, dim=1) == batchY).sum().item()

        # Durchschnittlichen Loss und Accuracy ueber alle Trainingsbilder berechnen
        trainLoss = trainLossSum / trainTotal
        trainAccuracy = trainCorrect / trainTotal

        return trainLoss, trainAccuracy


    def evaluate(self, studentModel, valLoader):
        studentModel.eval()

        valLossSum = 0
        valCorrect = 0
        valTotal = 0


        # Validierung ohne Gradienten, weil keine Gewichte aktualisiert werden
        with torch.no_grad():
            for batchX, batchY in valLoader:
                batchX = batchX.to(self.device, non_blocking=True)
                batchY = batchY.to(self.device, non_blocking=True)

                # Nur den Studenten mit den echten Labels bewertet
                output = studentModel(batchX)
                loss = F.cross_entropy(output, batchY)

                # Loss und richtige Vorhersagen fuer die Validierung summieren
                valLossSum += loss.item() * batchX.size(0)
                valTotal += batchX.size(0)
                valCorrect += (torch.argmax(output, dim=1) == batchY).sum().item()

        # Durchschnittlichen Loss und Accuracy ueber alle Bilder berechnen
        valLoss = valLossSum / valTotal
        valAccuracy = valCorrect / valTotal
        return valLoss, valAccuracy


    def createFitInfo(self, executedEpochs, earlyStopped, bestEpoch, bestTrainLoss, bestTrainAccuracy, bestValLoss, bestValAccuracy):
        fitInfo = {
            "planned_epochs": self.epochs,
            "executed_epochs": executedEpochs,
            "early_stopped": earlyStopped,
            "best_epoch": {
                "epoch": bestEpoch,
                "train_loss": bestTrainLoss,
                "train_accuracy": bestTrainAccuracy,
                "val_loss": bestValLoss,
                "val_accuracy": bestValAccuracy
            }
        }

        return fitInfo


    def createDataLoaders(self, trainFiles, valFiles, labelFile):
        trainDataset = DistillationImageDataset(trainFiles, labelFile)
        valDataset = ImageDataset128(valFiles, labelFile)

        trainLoader = torch.utils.data.DataLoader(
            trainDataset,
            batch_size=self.batchSize,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True
        )

        valLoader = torch.utils.data.DataLoader(
            valDataset,
            batch_size=self.batchSize,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True
        )

        return trainLoader, valLoader


    def fit(self, studentModel, teacherModel, trainLoader, valLoader, optimizer):
        # Sammelstruktur erstellen zum speichern des Verlauf fuer Loss und Accuracy ueber alle Epochen
        history = self.trainer.createHistory()

        # Variablen fuer Early Stopping und die beste gefundene Epoche vorbereiten
        bestValLoss = None
        bestValAccuracy = None
        bestTrainLoss = None
        bestTrainAccuracy = None
        bestEpoch = None
        bestModelState = None
        patienceCounter = 0
        executedEpochs = 0
        earlyStopped = False

        livePlot = Plot()

        # Training bis zur maximalen Epochenanzahl ausfuehren
        for epoch in range(self.epochs):
            # Student eine Epoche mit den Trainingsdaten und dem Teacher trainieren
            trainLoss, trainAccuracy = self.trainOneEpoch(
                studentModel,
                teacherModel,
                trainLoader,
                optimizer
            )

            # Student nach einer Epoche auf den Validierungsdaten bewerten
            valLoss, valAccuracy = self.evaluate(
                studentModel,
                valLoader
            )

            # Werte der aktuellen Epoche in der History speichern
            history["loss"].append(trainLoss)
            history["accuracy"].append(trainAccuracy)
            history["val_loss"].append(valLoss)
            history["val_accuracy"].append(valAccuracy)

            livePlot.update(
                trainLoss,
                valLoss,
                trainAccuracy,
                valAccuracy
            )

            currentEpoch = epoch + 1
            executedEpochs = currentEpoch

            print(currentEpoch, trainLoss, trainAccuracy, valLoss, valAccuracy)

            # Wenn der Validierung-Loss besser ist, alle Werte dieser Epoche als bisher bestes Ergebnis speichern
            if bestValLoss is None or valLoss < bestValLoss:
                bestValLoss = valLoss
                bestValAccuracy = valAccuracy
                bestTrainLoss = trainLoss
                bestTrainAccuracy = trainAccuracy
                bestEpoch = currentEpoch

                bestModelState = self.trainer.copyModelState(studentModel)
                patienceCounter = 0
            else:
                # Keine Verbesserung erreicht, dann Early-Stopping-Zaehler erhoehen.
                patienceCounter += 1

            # Wenn zulange keine Verbesserung, dann stoppen
            if patienceCounter >= self.patience:
                earlyStopped = True
                break

        # Bestes Modell wiederherstellen
        if bestModelState is not None:
            studentModel.load_state_dict(bestModelState)

        # Informationen zum Trainingslauf fuer die JSON-Datei sammeln
        fitInfo = self.createFitInfo(executedEpochs, earlyStopped, bestEpoch, bestTrainLoss, bestTrainAccuracy, bestValLoss, bestValAccuracy)

        return history, fitInfo


    def saveResults(self, studentModel, teacherModel, history, fitInfo, trainFiles, valFiles):
        resultData = self.trainer.createResultData(studentModel, history, fitInfo, trainFiles, valFiles,self.learningRate)

        resultData["training"]["model"] = "googlenet_distillation_student"
        resultData["training"]["loss"] = "knowledge_distillation_loss_cross_entropy_plus_kl_divergence"
        resultData["training"]["temperature"] = self.temperature
        resultData["training"]["alpha"] = self.alpha

        resultData["models"] = {
            "teacher": {
                "name": "VGG13",
                "parameters": self.countParameters(teacherModel)
            },
            "student": {
                "name": "GoogLeNet",
                "parameters": self.countParameters(studentModel)
            }
        }

        jsonOutputPath = "/mnt/c/Users/hagen/Desktop/Semester6/DeepLearning/knowledgeDistillation/results/json/googlenet_distillation_student.json"
        diagramOutputPath = "/mnt/c/Users/hagen/Desktop/Semester6/DeepLearning/knowledgeDistillation/results/diagrams/googlenet_distillation_student.png"
        modelOutputPath = "/mnt/c/Users/hagen/Desktop/Semester6/DeepLearning/knowledgeDistillation/results/models/googlenet_distillation_student.pth"

        self.trainer.saveJsonAndDiagram(
            resultData,
            jsonOutputPath,
            diagramOutputPath
        )

        self.trainer.saveModel(
            studentModel,
            modelOutputPath
        )


    def main(self):
        # Modelle holen
        studentModel = self.getGoogLeNet()
        teacherModel = self.getVGG13()

        # Anzahl der Parameter einmal ausgeben
        print("TeacherParams:", self.countParameters(teacherModel))
        print("StudentParams:", self.countParameters(studentModel))

        # Trainingsdaten holen und Loader erstellen
        trainFiles, valFiles, labelFile = self.trainer.loadFiles()
        trainLoader, valLoader = self.createDataLoaders(trainFiles, valFiles, labelFile)

        # Optimizer fuer das Studentenmodell erstellen
        optimizer = torch.optim.Adam(
            studentModel.parameters(),
            lr=self.learningRate
        )

        # Student mit Teacher trainieren
        history, fitInfo = self.fit(studentModel, teacherModel, trainLoader, valLoader, optimizer)

        # JSON Files, Diagramme und Studentenmodell speichern
        self.saveResults(studentModel, teacherModel, history, fitInfo, trainFiles, valFiles)


if "__main__" == __name__:
    KnowledgeDistillation().main()