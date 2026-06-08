import torch.nn as nn

from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet34, ResNet34_Weights
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import vgg11, VGG11_Weights
from torchvision.models import vgg13, VGG13_Weights
from torchvision.models import vgg16, VGG16_Weights
from torchvision.models import alexnet, AlexNet_Weights
from torchvision.models import googlenet, GoogLeNet_Weights


# Diese Klasse erstellt 8 unterschiedliche vortrainierte Netzwerke.
# Der Feature Extractor wird eingefroren.
# Trainiert wird nur der jeweilige Classifier bzw. die FC-Ausgabeschicht.
# Damit wird schnell verglichen, welche Architektur fuer unseren Datensatz geeignet ist.
class TransferModelFactory:
    def __init__(self, class_count):
        self.class_count = class_count


    def freezeParameters(self, parameters):
        for param in parameters:
            param.requires_grad = False


    def createResnet18FcPretrainedTrain(self):
        model = resnet18(weights=ResNet18_Weights.DEFAULT)

        self.freezeParameters(model.parameters())

        model.fc = nn.Linear(model.fc.in_features, self.class_count)

        return model, model.fc.parameters(), "ResNet18_fc_pretrained_train"


    def createResnet34FcPretrainedTrain(self):
        model = resnet34(weights=ResNet34_Weights.DEFAULT)

        self.freezeParameters(model.parameters())

        model.fc = nn.Linear(model.fc.in_features, self.class_count)

        return model, model.fc.parameters(), "ResNet34_fc_pretrained_train"


    def createResnet50FcPretrainedTrain(self):
        model = resnet50(weights=ResNet50_Weights.DEFAULT)

        self.freezeParameters(model.parameters())

        model.fc = nn.Linear(model.fc.in_features, self.class_count)

        return model, model.fc.parameters(), "ResNet50_fc_pretrained_train"


    def createVgg11ClassifierPretrainedTrain(self):
        model = vgg11(weights=VGG11_Weights.DEFAULT)

        self.freezeParameters(model.features.parameters())

        model.classifier[6] = nn.Linear(
            model.classifier[6].in_features,
            self.class_count
        )

        return model, model.classifier.parameters(), "VGG11_classifier_pretrained_train"


    def createVgg13ClassifierPretrainedTrain(self):
        model = vgg13(weights=VGG13_Weights.DEFAULT)

        self.freezeParameters(model.features.parameters())

        model.classifier[6] = nn.Linear(
            model.classifier[6].in_features,
            self.class_count
        )

        return model, model.classifier.parameters(), "VGG13_classifier_pretrained_train"


    def createVgg16ClassifierPretrainedTrain(self):
        model = vgg16(weights=VGG16_Weights.DEFAULT)

        self.freezeParameters(model.features.parameters())

        model.classifier[6] = nn.Linear(
            model.classifier[6].in_features,
            self.class_count
        )

        return model, model.classifier.parameters(), "VGG16_classifier_pretrained_train"


    def createAlexnetClassifierPretrainedTrain(self):
        model = alexnet(weights=AlexNet_Weights.DEFAULT)

        self.freezeParameters(model.features.parameters())

        model.classifier[6] = nn.Linear(
            model.classifier[6].in_features,
            self.class_count
        )

        return model, model.classifier.parameters(), "AlexNet_classifier_pretrained_train"


    def createGooglenetFcPretrainedTrain(self):
        model = googlenet(weights=GoogLeNet_Weights.DEFAULT)

        model.aux_logits = False
        model.aux1 = None
        model.aux2 = None

        self.freezeParameters(model.parameters())

        model.fc = nn.Linear(model.fc.in_features, self.class_count)

        return model, model.fc.parameters(), "GoogLeNet_fc_pretrained_train"


    def getModels(self):
        return [
            self.createResnet18FcPretrainedTrain,
            self.createResnet34FcPretrainedTrain,
            self.createResnet50FcPretrainedTrain,
            self.createVgg11ClassifierPretrainedTrain,
            self.createVgg13ClassifierPretrainedTrain,
            self.createVgg16ClassifierPretrainedTrain,
            self.createAlexnetClassifierPretrainedTrain,
            self.createGooglenetFcPretrainedTrain
        ]