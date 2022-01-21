import torch.nn as nn
import torch

conv_net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=3, stride=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
        )

data = torch.rand(1, 3, 256, 256)
x = conv_net(data)
print(x.shape)