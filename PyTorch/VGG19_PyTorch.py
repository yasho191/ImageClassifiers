import torch.nn as nn

class VGG19(nn.Module):
    def __init__(self, classes):
        super(VGG19, self).__init__()
        self.classes = classes

        # Conv Model
        self.VGG = nn.Sequential(
            # 2 Convolutional Layers - 64 filters followed by relu Activation
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            # MaxPooling Layer 
            nn.MaxPool2d(kernel_size=(2,2)),
            # 2 Convolutional Layers - 128 filters followed by relu Activation
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            # MaxPooling Layer 
            nn.MaxPool2d(kernel_size=(2,2)),
            # 4 Convolutional Layers - 256 filters followed by relu Activation
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            # MaxPooling Layer
            nn.MaxPool2d(kernel_size=(2,2)),
            # 4 Convolutional Layers - 512 filters followed by relu Activation
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            # MaxPooling Layer
            nn.MaxPool2d(kernel_size=(2,2)),
            # 4 Convolutional Layers - 512 filters followed by relu Activation
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            # Flatten for dense network
            nn.Flatten(),
        )

        # Fully connected Network
        self.densenet = nn.Sequential(
            # Dense Layer
            nn.Linear(in_features=(16*16*512), out_features=4096),
            nn.ReLU(),
            # Dropout 0.2
            nn.Dropout(0.2),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=4096, out_features=self.classes),
            # Softmax Activation
            nn.Softmax(dim=1)
        )

    # Forward Network
    def forward(self, x):
        x = self.VGG(x)
        output = self.densenet(x)
        return output
