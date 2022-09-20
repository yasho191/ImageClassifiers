from Tensorflow import AlexNet_Tensorflow, DenseNet_Tensorflow, InceptionV3_Tensorflow
from Tensorflow import ResNet_Tensorflow, VGG19_Tensorflow, MobileNetV2_Tesorflow
from tensorflow.keras.models import Model


class Models:
    def __init__(self, model_name: str, shape: tuple, classes: int) -> None:
        
        models = {
            "AlexNet": AlexNet_Tensorflow.AlexNet(shape, classes),
            "DenseNet": DenseNet_Tensorflow.DenseNet(shape, classes),
            "InceptionV3": InceptionV3_Tensorflow.InceptionV3(shape, classes),
            "ResNet": ResNet_Tensorflow.ResNet34(shape, classes),
            "VGG": VGG19_Tensorflow.VGG19(shape, classes),
            "MobileNetV2": MobileNetV2_Tesorflow.MobileNetV2(shape, classes)
        }

        self.model = models[model_name]

    def ret_model(self) -> Model:
        return self.model
