import os
import re
from dataclasses import dataclass


@dataclass
class ImageInfo:
    label: int
    batch: int
    index: int


class Label:
    def __init__(self, file):
        self.indices = []
        self.labels = []

        # Label-Datei einlesen
        file = open(file, "r")
        content = file.readlines()[1:] #Wir ueberspringen die erste Zeile, weil diese die Spaltennamen enthaelt
        file.close()

        self.setIndicesAndLabels(content)


    def setIndicesAndLabels(self, labelFileContent):
        # Durch die Label-Datei iterieren
        for line in labelFileContent:
            line = line.rstrip()

            # Leere Zeilen ueberspringen
            if not line :
                continue

            # Index entnehmen
            index = int(line.split(",")[0])

            # Bei dem Label "End of speed limit (80km/h)" und "No Speed Limits" gibt es eine Schnittmenge der Bilder (In der Untersuchung der Daten aufgefallen).
            # Aus diesem grund haben wir uns dafuer entschieden dass wir das Label "End of speed limit (80km/h)"(Index 6) nicht verwenden. Die folgende Konsequenz ist das wir
            # die darauf folgenden Labels mittels des Index anpassen muessen.

            # Index 6 ueberspringen
            if index == 6:
                continue

            # Folgende Indizes runterziehen
            if index > 6:
                index -= 1

            self.indices.append(index)

            #
            label = line.split(",")[1]
            self.labels.append(label)

        return self.indices


    def getIndexByLabel(self, label):
        pos = self.labels.index(label)
        return self.indices[pos]


    def getLabelByIndex(self, index):
        labelIndex = self.indices.index(index)
        return self.labels[labelIndex]


    # Diese Methode liest die Label-, Batch- und Index-Nummer aus der Namensformatierung der Bilder aus
    def getInfoFromImagePath(self, imagePath):
        basename = os.path.basename(imagePath)

        if "." in basename:
            basename = basename.split(".")[0]

        pattern = r"^(\d{5})_(\d{5})_(\d{5})$"
        match = re.match(pattern, basename)

        # Entspricht der Name nicht dem Pattern wollen wir reagieren koennen somit geben wir False zurueck
        if not match:
            return False

        label = int(match.group(1))

        # Alles Indizes runterziehen die ueber 6 liegen, weil wir ein Label entfernt haben
        if label > 6:
            label -= 1

        # Informationen zurueck geben
        return ImageInfo(
            label=label,
            batch=int(match.group(2)),
            index=int(match.group(3))
        )


if __name__ == "__main__":
    l = Label("../CleanDataset/labels.txt")
    print(l.getLabelByIndex(5))
    print(l.getLabelByIndex(6))
    print(l.getLabelByIndex(7))
    print(l.getLabelByIndex(61))
    print(l.getIndexByLabel("Speed limit (20km/h)"))