# DenseNet-121
import torch
import torch.nn as nn

# Basic conv block of densenet consits of -> BatchNorm -> ReLU -> Conv2D
class ConvBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size, strides, padding):
        super(ConvBlock, self).__init__()
        self.batch_norm = nn.BatchNorm2d(in_filters)
        self.relu = nn.ReLU()
        self.conv2d = nn.Conv2d(in_filters, out_filters, kernel_size, stride=strides, padding=padding)

    def forward(self, x):
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.conv2d(x)
        return x

# The dense block of densenet consits of Recurring conv blocks
# The output of every preceding conv block is concatenated and 
# fed to the next conv block as the input
class DenseBlock(nn.Module):
    def __init__(self, layers, in_filters):
        super(DenseBlock, self).__init__()
        self.layers = layers
        self.conv_block1 = ConvBlock(in_filters, 32*4, (1,1), 1, 0)
        self.conv_block2 = ConvBlock(32*4, 32, (3,3), 1, 1)

    def forward(self, x):
        residual = []
        for i in range(self.layers):
            if i == 0:
                x = self.conv_block1(x)
                x = self.conv_block2(x)
                residual.append(x)
            else:
                for j in range(i):
                    x = torch.cat([x, residual[j]], dim=1)
                filters = x.shape[1]
                x = ConvBlock(filters, 32*4, (1, 1), 1, 0)(x)
                x = self.conv_block2(x)
                residual.append(x)

        return x

# DownSample block also called transition block
# Structure: BatchNormalization -> Conv2D -> AveragePooling2D
# The filters given to the Conv2D here are based on the theta = 0.5
# which means the number of filters will be half of the input filters
class DownSample(nn.Module):
    def __init__(self, in_filters):
        super(DownSample, self).__init__()
        self.batch_norm = nn.BatchNorm2d(in_filters)
        self.conv2d = nn.Conv2d(in_filters, in_filters//2, (1, 1))
        self.average_pool = nn.AvgPool2d((2, 2), 2)

    def forward(self, x):
        x = self.batch_norm(x)
        x = self.conv2d(x)
        x = self.average_pool(x)

        return x

# Main Model
class DenseNet(nn.Module):
    def __init__(self, classes):
        super(DenseNet, self).__init__()
        self.classes = classes
        self.conv2d_1 = ConvBlock(3, 32, (7,7), 2, 3)
        self.max_pool = nn.MaxPool2d((3,3), 2, 1)

    def last_block(self, x): 
        x = nn.AvgPool2d((7,7))(x)
        x = nn.Flatten()(x)
        features = x.shape[1]
        x = nn.Linear(features, self.classes)(x)
        x = nn.Softmax(dim=1)(x)
        return x
    
    # Dense Blocks followed by Downsample Blocks
    # Block sizes = 6, 12, 24, 16
    def dense(self, x):
        x = DenseBlock(6, x.shape[1])(x)
        x = DownSample(x.shape[1])(x)
        x = DenseBlock(12, x.shape[1])(x)
        x = DownSample(x.shape[1])(x)
        x = DenseBlock(24, x.shape[1])(x)
        x = DownSample(x.shape[1])(x)
        x = DenseBlock(16, x.shape[1])(x)
        return x
    
    def forward(self, x):
        x = self.conv2d_1(x)
        x = self.max_pool(x)
        x = self.dense(x)
        x = self.last_block(x)

        return x
