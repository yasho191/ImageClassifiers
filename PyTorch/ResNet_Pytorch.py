import torch
import torch.nn as nn

# Identity block
# Size of the Image is maintained
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
        # copy tensor to variable called x_skip
        x_skip = x

        # Layer 1
        x = self.conv2d_1(x)
        x = self.batch_norm(x)
        x = self.relu(x)

        # Layer 2
        x = self.conv2d_2(x)
        x = self.batch_norm(x)

        # Skip Layer
        x_skip = self.conv2d_skip(x_skip)

        # Add Residual
        x = torch.add(x, x_skip)
        x = self.relu(x)

        return x

# Convolutional Block
# Size of the Image is downsampled
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
        # copy tensor to variable called x_skip
        x_skip =x

        # Layer 1
        x = self.conv2d_s2(x)
        x = self.batch_norm(x)
        x = self.relu(x)

        # Layer 2
        x = self.conv2d_s1(x)
        x = self.batch_norm(x)

        # Processing Residue with conv(1,1) stride=2
        x_skip = self.conv2d_skip(x_skip)

        # Add Residual
        x = torch.add(x, x_skip)
        x = self.relu(x)

        return x

class ResNet34(nn.Module):
    def __init__(self, classes):
        super(ResNet34, self).__init__()
        self.classes = classes

        # First Sequential Block with ZeroPad ->
        # Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d
        self.first_block = nn.Sequential(
            nn.ZeroPad2d(3),
            nn.Conv2d(3, 64, (7, 7), 2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), 2, padding=1)
        )

        

        self.avg_pool = nn.AvgPool2d((2, 2), padding=1)

    def residual_network(self, x, block_layers, filter_size):
        for i in range(len(block_layers)):
            if i == 0:
                # For sub-block 1 Residual/Convolutional block not needed
                for j in range(block_layers[i]):
                    x = identity_block(64, filter_size)(x)
            else:
                # One Residual/Convolutional Block followed by Identity blocks
                # The filter size will go on increasing by a factor of 2
                input = filter_size*(2**(i-1))
                output = filter_size*(2**i)
                x = convolutional_block(input, output)(x)
                for j in range(block_layers[i] - 1):
                    x = identity_block(output, output)(x)
        return x
            
    # Last Sequential Block (Dense Block)
    def last_block(self, x):
        x = nn.Flatten()(x)
        x = nn.Linear(in_features=(512*5*5), out_features=512)(x)
        x = nn.ReLU()(x)
        x = nn.Linear(in_features=512, out_features=256)(x)
        x = nn.ReLU()(x)
        x = nn.Linear(in_features=256, out_features=self.classes)(x)
        x = nn.Softmax(dim=1)(x)
        return x
        
    def forward(self, x):
        x = self.first_block(x)
        # Define size of sub-blocks and initial filter size
        block_layers = [3, 4, 6, 3]
        filter_size = 64

        # Residual Network 
        # Convolutional Block -> Identity Blocks
        x = self.residual_network(x, block_layers, filter_size)

        # 2,2 Average pooling
        x = self.avg_pool(x)
        output = self.last_block(x) 

        return output

