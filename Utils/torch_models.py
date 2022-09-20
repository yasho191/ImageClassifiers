from PyTorch import AlexNet_PyTorch, DenseNet_Pytorch, InceptionV3_Pytorch
from PyTorch import ResNet_Pytorch, VGG19_PyTorch, MobileNetV2_Pytorch
import torch

class Models:
    def __init__(self, model_name: str, classes: int) -> None:
        
        models = {
            "AlexNet": AlexNet_PyTorch.AlexNet(classes),
            "DenseNet": DenseNet_Pytorch.DenseNet(classes),
            "InceptionV3": InceptionV3_Pytorch.InceptionV3(classes),
            "ResNet": ResNet_Pytorch.ResNet34(classes),
            "VGG": VGG19_PyTorch.VGG19(classes),
            "MobileNetV2": MobileNetV2_Pytorch.MobileNetV2(classes)
        }

        self.model = models[model_name]

    def ret_model(self) -> torch.nn.Module:
        return self.model