import json
import os
from collections import defaultdict
import matplotlib
from matplotlib import pyplot as plt

class Plot:
    def __init__(self):
        # Interaktiven Modus aktivieren
        plt.ion()

        # TkAgg als Backend setzen, damit das Plot-Fenster separat angezeigt werden kann (hat bei uns sonst nichts angezeigt)
        matplotlib.use('TkAgg')

        # Listen fuer die Epochen und Metriken initialisieren
        self.epochs = []
        self.loss = []
        self.val_loss = []
        self.accuracy = []
        self.val_accuracy = []

        # Figure mit zwei Diagrammen erstellen (loss und accuracy)
        self.fig, (self.ax_loss, self.ax_acc) = plt.subplots(1, 2, figsize=(10, 4))

        # Linien fuer loss und accuracy erstellen
        self.loss_line, = self.ax_loss.plot([], [], label="loss")
        self.val_loss_line, = self.ax_loss.plot([], [], label="val_loss")
        self.acc_line, = self.ax_acc.plot([], [], label="accuracy")
        self.val_acc_line, = self.ax_acc.plot([], [], label="val_accuracy")

        # Schriften setzen
        self.ax_loss.set_title("Loss")
        self.ax_loss.set_xlabel("Epoch")
        self.ax_loss.set_ylabel("Loss")
        self.ax_loss.legend()
        self.ax_acc.set_title("Accuracy")
        self.ax_acc.set_xlabel("Epoch")
        self.ax_acc.set_ylabel("Accuracy")
        self.ax_acc.legend()

        # Anzeigen ohne ueberlappungen
        self.fig.tight_layout()
        plt.show(block=False)


    def update(self, train_loss, val_loss, train_accuracy, val_accuracy):
        # Aktuelle Epoche bestimmen
        epoch = len(self.epochs) + 1

        # Neue Werte speichern
        self.epochs.append(epoch)
        self.loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.accuracy.append(train_accuracy)
        self.val_accuracy.append(val_accuracy)

        # Linien mit den aktuellen Daten aktualisieren
        self.loss_line.set_data(self.epochs, self.loss)
        self.val_loss_line.set_data(self.epochs, self.val_loss)
        self.acc_line.set_data(self.epochs, self.accuracy)
        self.val_acc_line.set_data(self.epochs, self.val_accuracy)

        # Achsen neu berechnen
        self.ax_loss.relim()
        self.ax_loss.autoscale_view()
        self.ax_acc.relim()
        self.ax_acc.autoscale_view()

        # Neu zeichnen
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        plt.pause(0.001)


    def getJsonFiles(self, folderPath):
        # Alle Json-Dateien rekursiv sammeln
        jsonFiles = []

        for root, dirs, files in os.walk(folderPath):
            for file in files:
                if file.lower().endswith(".json"):
                    jsonFiles.append(os.path.join(root, file))

        return sorted(jsonFiles)


    def loadJsonFile(self, jsonPath):
        # Leere JSON Datei ueberspringen
        if os.path.getsize(jsonPath) == 0:
            return None

        try:
            with open(jsonPath, "r", encoding="utf-8") as file:
                return json.load(file)
        except json.JSONDecodeError:
            # Ungueltige JSON-Datei ueberspringen
            return None


    def createTrainingHistoryDiagram(self, jsonPath, outputPath):
        # TkAgg als Backend setzen, damit das Plot-Fenster angezeigt werden kann
        matplotlib.use("TkAgg")

        # Json-Datei laden
        resultData = self.loadJsonFile(jsonPath)

        if resultData is None:
            return

        # History lesen
        history = resultData["history"]

        loss = history["loss"]
        valLoss = history["val_loss"]
        accuracy = history["accuracy"]
        valAccuracy = history["val_accuracy"]

        # Epochen erzeugen
        epochs = list(range(1, len(loss) + 1))

        # Figure mit zwei Diagrammen erstellen
        fig, (axLoss, axAccuracy) = plt.subplots(1, 2, figsize=(12, 5))

        # Loss-Verlauf zeichnen
        axLoss.plot(epochs, loss, label="loss")
        axLoss.plot(epochs, valLoss, label="val_loss")
        axLoss.set_title("Loss")
        axLoss.set_xlabel("Epoch")
        axLoss.set_ylabel("Loss")
        axLoss.legend()

        # Accuracy-Verlauf zeichnen
        axAccuracy.plot(epochs, accuracy, label="accuracy")
        axAccuracy.plot(epochs, valAccuracy, label="val_accuracy")
        axAccuracy.set_title("Accuracy")
        axAccuracy.set_xlabel("Epoch")
        axAccuracy.set_ylabel("Accuracy")
        axAccuracy.legend()

        # Layout anpassen und speichern
        fig.tight_layout()
        plt.savefig(outputPath, dpi=300)
        plt.show()


    def createTrainingHistoryDiagramsForAll(self, resultsFolder, outputFolder):
        # Alle JSON-Ergebnisdateien aus dem Ergebnisordner sammeln
        jsonFiles = self.getJsonFiles(resultsFolder)

        # Ausgabeordner erstellen, falls er noch nicht existiert
        os.makedirs(outputFolder, exist_ok=True)

        # Fuer jede gueltige JSON-Datei ein eigenes Trainingsverlaufsdiagramm erstellen
        for jsonPath in jsonFiles:
            baseName = os.path.splitext(os.path.basename(jsonPath))[0]

            # Leere JSON-Dateien ueberspringen
            if os.path.getsize(jsonPath) == 0:
                continue

            outputPath = os.path.join(outputFolder, baseName + "_history.png")

            # Diagramm erstellen und ungueltige JSON-Dateien werden uebersprungen
            try:
                self.createTrainingHistoryDiagram(jsonPath, outputPath)
            except json.JSONDecodeError:
                print("Ueberspringe ungueltige Json-Datei:", jsonPath)


    def createBestEpochSummaryDiagram(self, resultsFolder, outputPath, bestBy="val_loss"):
        # TkAgg als Backend setzen, damit das Plot-Fenster angezeigt werden kann
        matplotlib.use("TkAgg")

        # Alle Json-Dateien holen
        jsonFiles = self.getJsonFiles(resultsFolder)

        # Listen fuer das Balkendiagramm
        names = []
        bestLossValues = []
        bestAccuracyValues = []

        # Alle Versuche durchgehen
        for jsonPath in jsonFiles:
            with open(jsonPath, "r", encoding="utf-8") as file:
                resultData = json.load(file)

            history = resultData["history"]

            loss = history["loss"]
            valLoss = history["val_loss"]
            accuracy = history["accuracy"]
            valAccuracy = history["val_accuracy"]

            # Beste Epoche bestimmen
            if bestBy == "val_accuracy":
                bestIndex = max(range(len(valAccuracy)), key=lambda i: valAccuracy[i])
            else:
                bestIndex = min(range(len(valLoss)), key=lambda i: valLoss[i])

            # Dateiname ohne Endung als Label verwenden
            baseName = os.path.splitext(os.path.basename(jsonPath))[0]

            names.append(baseName)
            bestLossValues.append(valLoss[bestIndex])
            bestAccuracyValues.append(valAccuracy[bestIndex])

        # Figure mit zwei Balkendiagrammen erstellen
        fig, (axLoss, axAccuracy) = plt.subplots(1, 2, figsize=(16, 6))

        xPositions = list(range(len(names)))

        # Balkendiagramm fuer val_loss der besten Epoche
        axLoss.bar(xPositions, bestLossValues)
        axLoss.set_title("Val Loss der besten Epoche")
        axLoss.set_xlabel("Versuch")
        axLoss.set_ylabel("Val Loss")
        axLoss.set_xticks(xPositions)
        axLoss.set_xticklabels(names, rotation=90, fontsize=7)

        # Balkendiagramm fuer val_accuracy der besten Epoche
        axAccuracy.bar(xPositions, bestAccuracyValues)
        axAccuracy.set_title("Val Accuracy der besten Epoche")
        axAccuracy.set_xlabel("Versuch")
        axAccuracy.set_ylabel("Val Accuracy")
        axAccuracy.set_xticks(xPositions)
        axAccuracy.set_xticklabels(names, rotation=90, fontsize=7)

        # Layout anpassen und speichern
        fig.tight_layout()
        plt.savefig(outputPath, dpi=300)
        plt.show()


    def createPhaseSummaryDiagram(self, phaseName, phaseEntries, outputPath):
        if len(phaseEntries) == 0:
            return

        names = []
        accuracies = []
        losses = []
        scores = []

        # Relevante Werte aus den Ergebnisdaten sammeln
        for entry in phaseEntries:
            modelName = entry.get("name", "unknown")
            evaluation = entry.get("evaluation", {})

            names.append(modelName)
            accuracies.append(evaluation.get("val_accuracy", 0.0))
            losses.append(evaluation.get("val_loss", 0.0))
            scores.append(evaluation.get("score", 0.0))

        # Modell mit dem besten Score bestimmen
        winnerIndex = max(range(len(scores)), key=lambda i: scores[i])
        winnerName = names[winnerIndex]
        winnerScore = scores[winnerIndex]
        winnerAccuracy = accuracies[winnerIndex]
        winnerLoss = losses[winnerIndex]

        fig, (axAccuracy, axLoss) = plt.subplots(1, 2, figsize=(16, 7))

        xPositions = list(range(len(names)))

        # Farben setzen und Gewinner hervorheben
        accuracyColors = ["tab:blue"] * len(names)
        lossColors = ["tab:orange"] * len(names)

        accuracyColors[winnerIndex] = "tab:green"
        lossColors[winnerIndex] = "tab:green"

        barsAccuracy = axAccuracy.bar(xPositions, accuracies, color=accuracyColors)
        barsLoss = axLoss.bar(xPositions, losses, color=lossColors)

        # Accuracy-Diagramm beschriften
        axAccuracy.set_title("Hoechste Genauigkeit pro Modell")
        axAccuracy.set_xlabel("Modell")
        axAccuracy.set_ylabel("Genauigkeit")
        axAccuracy.set_xticks(xPositions)
        axAccuracy.set_xticklabels(names, rotation=45, ha="right")

        # Loss-Diagramm beschriften
        axLoss.set_title("Niedrigste Verlustwerte pro Modell")
        axLoss.set_xlabel("Modell")
        axLoss.set_ylabel("Verlust")
        axLoss.set_xticks(xPositions)
        axLoss.set_xticklabels(names, rotation=45, ha="right")

        # Werte direkt ueber die Accuracy-Balken schreiben
        for bar, value in zip(barsAccuracy, accuracies):
            axAccuracy.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=8
            )

        # Werte direkt ueber die Loss-Balken schreiben
        for bar, value in zip(barsLoss, losses):
            axLoss.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=8
            )

        # Gewinner und wichtigste Kennzahlen als Ueberschrift anzeigen
        fig.suptitle(
            f"Beste Ergebnisse der Phase: {phaseName}\n"
            f"Gewinner: {winnerName} | Score: {winnerScore:.4f} | "
            f"Genauigkeit: {winnerAccuracy:.4f} | Loss: {winnerLoss:.4f}",
            fontsize=14
        )

        fig.tight_layout()
        plt.savefig(outputPath, dpi=300)
        plt.close(fig)


    def createPhaseSummaryDiagrams(self, resultsFolder, outputFolder):
        # Alle JSON-Ergebnisdateien sammeln
        jsonFiles = self.getJsonFiles(resultsFolder)

        # Ausgabeordner erstellen, falls er noch nicht existiert
        os.makedirs(outputFolder, exist_ok=True)

        phaseGroups = {}

        # JSON-Dateien nach Experimentphase und Modellname gruppieren
        for jsonPath in jsonFiles:
            resultData = self.loadJsonFile(jsonPath)

            # Ungueltige oder unvollstaendige Ergebnisdateien ueberspringen
            if resultData is None:
                continue

            if "history" not in resultData:
                continue

            if "evaluation" not in resultData:
                continue

            phaseName = resultData.get("phase", "unknown")
            modelName = resultData.get("name", "unknown")

            if phaseName not in phaseGroups:
                phaseGroups[phaseName] = {}

            currentScore = resultData.get("evaluation", {}).get("score", 0.0)

            # Falls ein Modell doppelt vorkommt, wird nur der bessere Score behalten
            if modelName in phaseGroups[phaseName]:
                oldData = phaseGroups[phaseName][modelName]
                oldScore = oldData.get("evaluation", {}).get("score", 0.0)

                if currentScore <= oldScore:
                    continue

            phaseGroups[phaseName][modelName] = resultData

        # Fuer jede Phase ein eigenes Vergleichsdiagramm erstellen
        for phaseName, modelDict in phaseGroups.items():
            phaseEntries = list(modelDict.values())

            safePhaseName = str(phaseName).replace("/", "_").replace("\\", "_").replace(" ", "_")
            outputPath = os.path.join(outputFolder, safePhaseName + ".png")

            self.createPhaseSummaryDiagram(phaseName, phaseEntries, outputPath)


if __name__ == "__main__":

    resultsFolder = "/mnt/c/Users/hagen/Desktop/DeepLearning/Experiment/ExperimentResults/json"
    historyOutputFolder = "/mnt/c/Users/hagen/Desktop/DeepLearning/images/history_plots"
    summaryOutputPath = "/mnt/c/Users/hagen/Desktop/DeepLearning/images/best_epoch_summary.png"

    p = Plot()

    # Zusammenfassung der besten Epochen ueber alle Modelle erstellen
    p.createTrainingHistoryDiagramsForAll(resultsFolder, historyOutputFolder)
    p.createBestEpochSummaryDiagram(
        resultsFolder=resultsFolder,
        outputPath=summaryOutputPath,
        bestBy="val_loss"
    )

    # Fuer jede Experimentphase ein eigenes Vergleichsdiagramm erstellen
    phaseSummaryOutputFolder = "/mnt/c/Users/hagen/Desktop/DeepLearning/images/phase_summaries"
    p.createPhaseSummaryDiagrams(
        resultsFolder=resultsFolder,
        outputFolder=phaseSummaryOutputFolder
    )