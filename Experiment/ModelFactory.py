import copy
import torch.nn as nn
import torch


class ModelFactory:
    def addDefaultFullyConnectedLayer(self, model: nn.Sequential):
        # Dummy-bild
        x = torch.zeros(1, 3, 128, 128)

        # Dummy einmal durch den conv/pooling-teil schicken
        # damit pytorch selbst berechnet, wie gross der flatten-input ist
        with torch.no_grad():
            x = model(x)

        # Alles zu einem Vektor machen und groesse holen
        fcInputSize = x.view(1, -1).shape[1]

        # Standard-fc-teil anhaengen
        model.append(nn.Flatten())
        model.append(nn.Linear(fcInputSize, 64))
        model.append(nn.ReLU())
        model.append(nn.Linear(64, 62))
        model.append(nn.Softmax(dim=1))

        return model


    def getConvolutionLayerModels(self):
        modelList = []

        # Hier erzeugen wir ein paar Modelle um zu erproben welches davon am besten geeignet ist
        model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        )
        model = self.addDefaultFullyConnectedLayer(model)
        modelList.append((model, "32*5"))


        model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        )
        model = self.addDefaultFullyConnectedLayer(model)
        modelList.append((model, "64*5"))


        model = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        )
        model = self.addDefaultFullyConnectedLayer(model)
        modelList.append((model, "128*5"))


        model = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        )
        model = self.addDefaultFullyConnectedLayer(model)
        modelList.append((model, "256*5"))


        model = nn.Sequential(
            nn.Conv2d(3, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        )
        model = self.addDefaultFullyConnectedLayer(model)
        modelList.append((model, "512*5"))


        model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        )
        model = self.addDefaultFullyConnectedLayer(model)
        modelList.append((model, "32*2/64*2/128*2"))


        model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        )
        model = self.addDefaultFullyConnectedLayer(model)
        modelList.append((model, "64*2/128*2/256*2"))


        model = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        )
        model = self.addDefaultFullyConnectedLayer(model)
        modelList.append((model, "128*2/256*2/512*2"))


        model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        )
        model = self.addDefaultFullyConnectedLayer(model)
        modelList.append((model, "32/64/128/256/512/1024"))


        model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        )
        model = self.addDefaultFullyConnectedLayer(model)
        modelList.append((model, "32/64/128/256/512"))


        model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        )
        model = self.addDefaultFullyConnectedLayer(model)
        modelList.append((model, "64/128/128/128/256"))


        model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            # Kein weiteres Pooling, weil die Feature-Map danach schon sehr klein ist.
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU()
        )
        model = self.addDefaultFullyConnectedLayer(model)
        modelList.append((model, "32/64/128/256/512/512/1024"))

        return modelList


    def getConvolutionKernelSizeModels(self, model: nn.Sequential):
        modelList = []
        # Zu testende kernel Groessen
        kernelSizes = [(3, 3), (5, 5), (7, 7)]

        for ks in kernelSizes:
            newModel = copy.deepcopy(model)

            for index, layer in enumerate(newModel):
                # Nur Conv2d anpassen
                if not isinstance(layer, nn.Conv2d):
                    continue

                # kernel_size und padding anpassen
                newModel[index] = nn.Conv2d(
                    in_channels=layer.in_channels,
                    out_channels=layer.out_channels,
                    kernel_size=ks,
                    stride=layer.stride,
                    padding=ks[0] // 2
                )

            modelList.append((newModel, str(ks)))

        return modelList


    def getPoolingMethodModels(self, model: nn.Sequential):
        newModels = [(model, "MaxPool")]
        newModel = copy.deepcopy(model)

        # Durch Model-Layer iterieren
        for index, layer in enumerate(newModel):
            # Nur MaxPooling2d anpassen
            if not isinstance(layer, nn.MaxPool2d):
                continue

            # MaxPooling setzen
            newModel[index] = nn.AvgPool2d(
                kernel_size=layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding
            )

        newModels.append((newModel, "AvgPool"))

        return newModels


    def getFullyConnectedLayerModels(self, model: nn.Sequential):
        modelList = []

        # Konfigurationen die wir testen wollen
        fcConfigs = [
            [64],
            [128],
            [256],
            [512],
            [256, 64],
            [512, 128],
            [512, 256, 64]
        ]

        for fcLayers in fcConfigs:
            newModel = copy.deepcopy(model)

            linearIndices = []

            for index, layer in enumerate(newModel):
                if isinstance(layer, nn.Linear):
                    linearIndices.append(index)

            firstLinear = newModel[linearIndices[0]]
            lastLinear = newModel[linearIndices[-1]]

            fcInputSize = firstLinear.in_features
            classCount = lastLinear.out_features

            # Alten FC-Teil ab erster Linear-Schicht entfernen
            firstLinearIndex = linearIndices[0]

            while len(newModel) > firstLinearIndex:
                del newModel[firstLinearIndex]

            # Neuen FC-Teil einbauen
            lastSize = fcInputSize

            for hiddenSize in fcLayers:
                newModel.append(nn.Linear(lastSize, hiddenSize))
                newModel.append(nn.ReLU())
                lastSize = hiddenSize

            newModel.append(nn.Linear(lastSize, classCount))
            newModel.append(nn.Softmax(dim=1))

            modelList.append((newModel, "fc_" + "_".join(str(x) for x in fcLayers)))

        return modelList


    def getDropoutPositionsModels(self, model: nn.Sequential):
        # Zum Testen der Positionen nehmen wir erstmal 0.3
        dropoutValue = 0.3
        modelList = []


        def getFlattenIndex(m):
            for index, layer in enumerate(m):
                if isinstance(layer, nn.Flatten):
                    return index


        def insertLayers(m, insertions):
            # Von hinten nach vorne einfuegen, damit indices stabil bleiben
            for index, layer in sorted(insertions, key=lambda x: x[0], reverse=True):
                m.insert(index, layer)

            return m


        # 1 Dropout direkt vor flatten
        newModel = copy.deepcopy(model)
        flattenIndex = getFlattenIndex(newModel)
        newModel.insert(flattenIndex, nn.Dropout2d(dropoutValue))
        modelList.append((newModel, "dropout_before_flatten"))

        # 2 Dropout direkt nach flatten
        newModel = copy.deepcopy(model)
        flattenIndex = getFlattenIndex(newModel)
        newModel.insert(flattenIndex + 1, nn.Dropout(dropoutValue))
        modelList.append((newModel, "dropout_after_flatten"))

        # 3 Dropout nach der ReLU im FC-Teil
        newModel = copy.deepcopy(model)
        flattenIndex = getFlattenIndex(newModel)

        for index, layer in enumerate(newModel):
            if index > flattenIndex and isinstance(layer, nn.ReLU):
                newModel.insert(index + 1, nn.Dropout(dropoutValue))
                break

        modelList.append((newModel, "dropout_after_fc_relu"))

        # 4 Dropout nach letzter Conv-ReLU
        newModel = copy.deepcopy(model)
        flattenIndex = getFlattenIndex(newModel)

        convReluIndices = []

        for index, layer in enumerate(newModel):
            if index >= flattenIndex:
                break

            if isinstance(layer, nn.ReLU):
                convReluIndices.append(index)

        newModel.insert(convReluIndices[-1] + 1, nn.Dropout2d(dropoutValue))
        modelList.append((newModel, "dropout_after_last_conv_relu"))

        # 5 Dropout nach den letzten zwei Conv-ReLU-Schichten
        newModel = copy.deepcopy(model)
        flattenIndex = getFlattenIndex(newModel)

        convReluIndices = []

        for index, layer in enumerate(newModel):
            if index >= flattenIndex:
                break

            if isinstance(layer, nn.ReLU):
                convReluIndices.append(index)

        insertions = [
            (convReluIndices[-2] + 1, nn.Dropout2d(dropoutValue)),
            (convReluIndices[-1] + 1, nn.Dropout2d(dropoutValue))
        ]

        newModel = insertLayers(newModel, insertions)
        modelList.append((newModel, "dropout_after_last_two_conv_relu"))

        return modelList


    def getDropoutValueModels(self, model: nn.Sequential):
        modelList = []

        # Ohne 0.3, denn dieser Wert wurde bereits als default verwendet
        dropoutValues = [0.0, 0.1, 0.2, 0.4, 0.5, 0.6]

        # Mit jedem Dropout-Wert ein Modell erstellen
        for dropoutValue in dropoutValues:
            # Modell kopieren
            newModel = copy.deepcopy(model)

            # Durch Schichten gehen
            for layer in newModel:
                # Ist es eine Dropout-Schicht
                if isinstance(layer, nn.Dropout) or isinstance(layer, nn.Dropout2d):
                    # Dropout-Wert anpassen
                    layer.p = dropoutValue

            # Neues Modell adden
            modelList.append((newModel, str(dropoutValue)))

        return modelList


# Test ConvolutionLayer
"""
models = ModelFactory().getConvolutionLayerModels()
for model in models:
    for i in model.children():
        print(i)
    print("\n\n\n")
"""


# Test ConvolutionKernelSize
"""
model = train.Train().createModel()
models = ModelFactory().getConvolutionKernelSizeModels(model)
"""

# Test PoolingMethod
"""
model = train.Train().createModel()
models = ModelFactory().getPoolingMethodModels(model)
"""

# Test FullyConnectedLayer
"""
model = train.Train().createModel()
models = ModelFactory().getFullyConnectedLayerModels(model)
"""

# Tes DropoutPositions
"""
model = train.Train().createModel()
models = ModelFactory().getDropoutPositionsModels(model)
"""

# Test DropoutValue
"""
model = train.Train().createModel()
models = ModelFactory().getDropoutValueModels(model)
"""

# Test Ausgabe
"""
for model in models:
    for i in model.children():
        print(i)
    print("\n\n\n")
"""