import torch.nn as nn

# Declare AlexNet Class
class AlexNet(nn.Module):
    def __init__(self, classes):
        super(AlexNet, self).__init__()
        # Sequential Conv Network
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=3, stride=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
        )

        # Sequential Dense network
        self.densenet = nn.Sequential(
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=4096, out_features=classes),
            nn.Softmax(dim=1)
        )

    def linear(self, x, features):
        x = nn.Linear(in_features=features, out_features=4096)(x)
        x = nn.ReLU()(x)
        x = nn.Dropout(0.2)(x)
        return x

    # Forward Feed
    def forward(self, x):
        x = self.conv_net(x)
        features = x.shape[1]
        x = self.linear(x, features)
        output = self.densenet(x)
        return output
