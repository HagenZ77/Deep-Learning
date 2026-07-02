import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


# Eigenes PyTorch-Dataset, das Bildpfade einliest und passende Labels aus dem Dateinamen bestimmt.
class ImageDataset128(Dataset):
    def __init__(self, files, labelFile):
        self.files = files
        self.labelFile = labelFile


    def __len__(self):
        return len(self.files)


    # Laedt ein Bild, wandelt es in RGB um, skaliert es auf 128x128,
    # normalisiert die Pixelwerte auf [0, 1] und bringt die Achsen in PyTorch-Format CxHxW.
    def loadImage(self, path):
        img = Image.open(path).convert("RGB")
        img = img.resize((128, 128))
        img = np.array(img).astype("float32")
        img = img / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img


    # Gibt ein einzelnes Bild mit zugehoerigem Klassenindex fuer den DataLoader zurueck
    def __getitem__(self, index):
        path = self.files[index]

        x = self.loadImage(path)[0]
        y = self.labelFile.getInfoFromImagePath(path).label

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        return x, y
