import torch
import torch.nn as nn

class identity_block(nn.Module):
    def __init__(self, in_filters, out_filters):
        super(identity_block, self).__init__()
        self.in_filters = in_filters
        self.out_filters = out_filters

        self.conv2d_1 = nn.Conv2d(self.in_filters, self.out_filters, (3, 3), padding=1)
        self.conv2d_2 = nn.Conv2d(self.out_filters, self.out_filters, (3, 3), padding=1)
        self.conv2d_skip = nn.Conv2d(self.in_filters, self.out_filters, (1, 1))
        
        self.batch_norm = nn.BatchNorm2d(self.out_filters)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_skip = x

        x = self.conv2d_1(x)
        x = self.batch_norm(x)
        x = self.relu(x)

        x = self.conv2d_2(x)
        x = self.batch_norm(x)

        x_skip = self.conv2d_skip(x_skip)

        x = torch.add(x, x_skip)
        x = self.relu(x)
        return x


class convolutional_block(nn.Module):
    def __init__(self, in_filters, out_filters):
        super(convolutional_block, self).__init__()
        self.in_filters = in_filters
        self.out_filters = out_filters

        self.conv2d_s2 = nn.Conv2d(self.in_filters, self.out_filters, (3, 3), stride=2, padding=1)
        self.conv2d_s1 = nn.Conv2d(self.out_filters, self.out_filters, (3, 3), stride=1, padding=1)
        self.conv2d_skip = nn.Conv2d(self.in_filters, self.out_filters, (1, 1), stride=2)
        
        self.batch_norm = nn.BatchNorm2d(self.out_filters)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x_skip =x

        x = self.conv2d_s2(x)
        x = self.batch_norm(x)
        x = self.relu(x)

        x = self.conv2d_s1(x)
        x = self.batch_norm(x)

        x_skip = self.conv2d_skip(x_skip)

        x = torch.add(x, x_skip)
        x = self.relu(x)

        return x

def residual_network(x, block_layers, filter_size):
        for i in range(len(block_layers)):
            if i == 0:
                for j in range(block_layers[i]):
                    x = identity_block(64, filter_size)(x)
            else:
                input = filter_size*(2**(i-1))
                output = filter_size*(2**i)
                x = convolutional_block(input, output)(x)
                for j in range(block_layers[i] - 1):
                    x = identity_block(output, output)(x)
        return x

avg_pool = nn.AvgPool2d((2, 2))
flat = nn.Flatten()

t = torch.randn((1, 64, 256, 256))
# Define size of sub-blocks and initial filter size
block_layers = [3, 4, 6, 3]
filter_size = 64
t = residual_network(t, block_layers, filter_size)
t = avg_pool(t)
t = flat(t)
print(t.shape)