import torch.nn as nn
import torch

# relu6 is an modified version of relu where the max value == 6
#         |   ______
#         |  /
#         | /
#  _______|/
# -----------------

class Bottleneck(nn.Module):
    def __init__(self, expansion_factor, in_filters, out_filters, strides):
        super(Bottleneck, self).__init__()
        self.in_filters = in_filters
        self.out_filters = out_filters
        self.hidden_dim = expansion_factor*in_filters
        self.addition = (in_filters == out_filters) and (strides == 1)

        if expansion_factor == 1:
            self.network = nn.Sequential(
                nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, strides, 1, groups=self.hidden_dim, bias=False),
                nn.BatchNorm2d(self.hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(self.hidden_dim, out_filters, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_filters),
            )
        else:
            self.network = nn.Sequential(
                nn.Conv2d(in_filters, self.hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(self.hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, strides, 1, groups=self.hidden_dim, bias=False),
                nn.BatchNorm2d(self.hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(self.hidden_dim, out_filters, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_filters),
            )

    def forward(self, x):
        if self.addition:
            return x + self.network(x)
        else:
            return self.network(x)


# mobilenetv2
class MobileNetV2(nn.Module):
    def __init__(self, classes) -> None:
        super(MobileNetV2, self).__init__()
        self.bottleneck_sequence = [
                                    # t, c, n, s
                                    [1, 16, 1, 1],
                                    [6, 24, 2, 2],
                                    [6, 32, 3, 2],
                                    [6, 64, 4, 2],
                                    [6, 96, 3, 1],
                                    [6, 160, 3, 2],
                                    [6, 320, 1, 1]
                                ]
        self.classes = classes
        # Add layers to a list
        # Considering standard RGB image
        model_layers = [self.conv_block(input=3, output=32, kernel=3, stride=2)]

        # Iterating over all the blocks
        # initial filters after conv_block_1 = 32
        in_filters = 32
        for t, c, n, s in self.bottleneck_sequence:
            for i in range(n):
                if i == 0:
                    # stride will be 2 only for the first iteration
                    model_layers.append(Bottleneck(t, in_filters, c, s))
                else:
                    model_layers.append(Bottleneck(t, in_filters, c, 1))
                in_filters = c

        self.layers = nn.Sequential(*model_layers)
        # after the bottleneck blocks end the output tensor will have 320 filters
        self.last_conv_block = self.conv_block(320, 1280, 1, 1)
        self.average_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(1280, self.classes)

    def conv_block(self, input: int, output: int, kernel: int, stride: int) -> nn.Module:
        pad = 1 if kernel == 3 else 0
        return nn.Sequential(
                            nn.Conv2d(input, output, kernel, stride, padding=pad, bias=False),
                            nn.BatchNorm2d(output),
                            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        x = self.layers(x)
        x = self.last_conv_block(x)
        x = self.average_pooling(x)
        x = torch.reshape(x, (x.shape[0], x.shape[1]))
        x = self.classifier(x)
        return x