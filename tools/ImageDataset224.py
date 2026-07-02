import numpy as np
import torch
from PIL import Image
from tools.ImageDataset128 import ImageDataset128


# VGG13 braucht 224x224 Bilderformat, weil diese damit angelernt wurde und wir bei dem transferLearning damit weiterarbeiten
class ImageDataset224(ImageDataset128):
    def loadImage(self, path):
        img = Image.open(path).convert("RGB")
        img = img.resize((224, 224))
        img = np.array(img).astype("float32") / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype="float32")
        std = np.array([0.229, 0.224, 0.225], dtype="float32")

        img = (img - mean) / std
        img = np.transpose(img, (2, 0, 1))

        return img


    # Gibt ein einzelnes Bild mit zugehoerigem Klassenindex fuer den DataLoader zurueck
    def __getitem__(self, index):
        path = self.files[index]

        x = self.loadImage(path)
        y = self.labelFile.getInfoFromImagePath(path).label

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        return x, y
