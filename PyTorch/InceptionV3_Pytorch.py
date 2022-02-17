import torch
import torch.nn as nn

# Basic Conv Block with structure
# conv2d -> batch normalization -> relu
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,  kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride = stride, padding = padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv2d(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x

# Module A -> 4 paths (3, 3) conv kernel
# size preserved
class ModuleA(nn.Module):
    def __init__(self, in_channels):
        super(ModuleA, self).__init__()

        self.path1 = nn.Sequential(
            ConvBlock(in_channels, 64, kernel_size=(1, 1), stride=1, padding=0),
            ConvBlock(64, 96, kernel_size=(3, 3), stride=1, padding=1),
            ConvBlock(96, 96, kernel_size=(3, 3), stride=1, padding=0)
        )

        self.path2 = nn.Sequential(
            ConvBlock(in_channels, 48, kernel_size=(1, 1), stride=1, padding=0),
            ConvBlock(48, 64, kernel_size=(3, 3), stride=1, padding=1),
        )

        self.path3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3,3), stride=1, padding=1),
            ConvBlock(in_channels, 64, kernel_size=(1, 1), stride=1, padding=0)
        )

        self.path4 = ConvBlock(in_channels, 64, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        x_1 = self.path1(x)
        x_2 = self.path2(x)
        x_3 = self.path3(x)
        x_4 = self.path4(x)
        return torch.cat([x_1, x_2, x_3, x_4], 1)

# Module B -> 4 branches (7, 7) factorized conv kernel
# size preserved
class ModuleB(nn.Module):
    def __init__(self, in_channels, f_7x7):
        super(ModuleB, self).__init__()

        self.path1 = nn.Sequential(
            ConvBlock(in_channels, f_7x7, kernel_size=(1, 1), stride=1, padding=0),
            ConvBlock(f_7x7, f_7x7, kernel_size=(1,7), stride=1, padding=(0, 3)),
            ConvBlock(f_7x7, f_7x7, kernel_size=(7,1), stride=1, padding=(3,0)),
            ConvBlock(f_7x7, f_7x7, kernel_size=(1,7), stride=1, padding=(0,3)),
            ConvBlock(f_7x7, 192, kernel_size=(7,1), stride=1, padding=(3,0))
        )

        self.path2 = nn.Sequential(
            ConvBlock(in_channels, f_7x7, kernel_size=(1, 1), stride=1, padding=0),
            ConvBlock(f_7x7, f_7x7, kernel_size=(1,7), stride=1, padding=(0,3)),
            ConvBlock(f_7x7, 192, kernel_size=(7,1), stride=1, padding=(3,0))
        )

        self.path3 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            ConvBlock(in_channels, 192, kernel_size=1, stride=1, padding=0)
        )

        self.path4 = ConvBlock(in_channels, 192, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        x_1 = self.path1(x)
        x_2 = self.path2(x)
        x_3 = self.path3(x)
        x_4 = self.path4(x)
        return torch.cat([x_1, x_2, x_3, x_4], 1)

# Module C -> 4 branches (3, 3,) conv kernel
# multiple branch splits
# size preserved
class ModuleC(nn.Module):
    def __init__(self, in_channels):
        super(ModuleC, self).__init__()

        self.path1 = nn.Sequential(
            ConvBlock(in_channels, 448, kernel_size=(1, 1), stride=1, padding=0),
            ConvBlock(448, 384, kernel_size=(3, 3), stride=1, padding=1),

        )

        self.path2 = ConvBlock(in_channels, 384, kernel_size=(1, 1), stride=1, padding=0)

        self.path3 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            ConvBlock(in_channels, 192, kernel_size=1, stride=1, padding=0)
        )

        self.path4 = ConvBlock(in_channels, 320, kernel_size=(1,1), stride=1, padding=0)
    
    def forward(self, x):
        x_1 = self.path1(x)
        x_1_top = ConvBlock(384,  384, kernel_size=(1, 3), stride=1, padding=(0, 1))(x_1)
        x_1_bottom = ConvBlock(384, 384, kernel_size=(3,1), stride=1, padding=(1, 0))(x_1)
        x_1 = torch.cat([x_1_top, x_1_bottom], 1)
        x_2 = self.path2(x)
        x_2_top = ConvBlock(384,  384, kernel_size=(1, 3), stride=1, padding=(0, 1))(x_2)
        x_2_bottom = ConvBlock(384, 384, kernel_size=(3,1), stride=1, padding=(1, 0))(x_2)
        x_2 = torch.cat([x_2_top, x_2_bottom], 1)
        x_3 = self.path3(x)
        x_4 = self.path4(x)
        return torch.cat([x_1, x_2, x_3, x_4], 1)

# Size reduction block 
# 3 Branches custom add on filters
class SizeReductionBlock(nn.Module):
    def __init__(self, in_channels, f_3x3, add_ch):
        super(SizeReductionBlock, self).__init__()

        self.path1 = nn.Sequential(
            ConvBlock(in_channels, f_3x3, kernel_size=(1, 1), stride=1, padding=0),
            ConvBlock(f_3x3, 178+add_ch, kernel_size=(3, 3), stride=1, padding=1),
            ConvBlock(178+add_ch, 178+add_ch, kernel_size=(3, 3), stride=2, padding=0)
        )

        self.path2 = nn.Sequential(
            ConvBlock(in_channels, f_3x3, kernel_size=(1, 1), stride=1, padding=0),
            ConvBlock(f_3x3, 302+add_ch, kernel_size=(3, 3), stride=2, padding=0),
        )

        self.path3 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=0)

    def forward(self, x):
        x_1 = self.path1(x)
        x_2 = self.path2(x)
        x_3 = self.path3(x)

        return torch.cat([x_1, x_2, x_3], 1)

# Fine tune auxilary block for early convergence
# AVG POOL -> con2d -> dense(FC) -> relu -> Dropout -> dense -> softmax
class FineTuneBlock(nn.Module):
    def __init__(self, in_channels, classes):
        super(FineTuneBlock, self).__init__()

        self.avg_pool = nn.AvgPool2d(kernel_size=(5, 5), stride=3, padding=0)
        self.conv2d = ConvBlock(in_channels, 128, kernel_size=(1, 1), stride=1, padding=0)
        
        self.dense1 = nn.Linear(2048, 1024)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        self.dense2 = nn.Linear(1024, classes)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv2d(x)
        x = torch.flatten(x, 1)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.softmax(x)

        return x

# Main Inception model
class InceptionV3(nn.Module):
    def __init__(self, classes):
        super(InceptionV3, self).__init__()
        
        # Initial Conv Blocks 3 conv blocks -> MaxPool -> 3 conv blocks
        self.conv2d_1 = ConvBlock(3, 32, kernel_size=3, stride=2, padding=0)
        self.conv2d_2 = ConvBlock(32, 32, kernel_size=3, stride=1, padding=0)
        self.conv2d_3 = ConvBlock(32, 64, kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=0)
        self.conv2d_4 = ConvBlock(64, 80, kernel_size=3, stride=1, padding=0)
        self.conv2d_5 = ConvBlock(80, 192, kernel_size=3, stride=2, padding=0)
        self.conv2d_6 = ConvBlock(192, 288, kernel_size=3, stride=1, padding=1)
        
        # 3 x Module A
        self.moduleA_1 = ModuleA(228)
        self.moduleA_2 = ModuleA(228)
        self.moduleA_3 = ModuleA(228)

        # Size reduction
        self.size_red_1 = SizeReductionBlock(288,f_3x3=64, add_ch=0)

        # 5 x Module B
        self.moduleB_1 = ModuleB(768, f_7x7=128)
        self.moduleB_2 = ModuleB(768, f_7x7=160)
        self.moduleB_3 = ModuleB(768, f_7x7=160)
        self.moduleB_4 = ModuleB(768, f_7x7=160)
        self.moduleB_5 = ModuleB(768, f_7x7=192)

        # Size Reduction
        self.size_red_2 = SizeReductionBlock(768,f_3x3=192, add_ch=16)

        # Early Convergence( Fine Tune Block )
        self.fine_tune_1 = FineTuneBlock(728, classes)

        # 2 x Module C
        self.moduleC_1 = ModuleC(1280)
        self.moduleC_2 = ModuleC(2048)

        # Final Fully connected network
        self.avg_pool = nn.AvgPool2d(kernel_size=(7, 7), stride=1, padding=0)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.batch_norm = nn.BatchNorm2d(4096)
        self.dense_1 = nn.Linear(2048, 4096)
        self.dense_2 = nn.Linear(4096, 4096)
        self.dense_3 = nn.Linear(4096, classes)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.conv2d_1(x)
        x = self.conv2d_2(x)
        x = self.conv2d_3(x)
        x = self.max_pool(x)
        x = self.conv2d_4(x)
        x = self.conv2d_5(x)
        x = self.conv2d_6(x)

        x = self.moduleA_1(x)
        x = self.moduleA_2(x)
        x = self.moduleA_3(x)

        x = self.size_red_1(x)

        x = self.moduleB_1(x)
        x = self.moduleB_2(x)
        x = self.moduleB_3(x)
        x = self.moduleB_4(x)
        x = self.moduleB_5(x)

        output1 = self.size_red_2(x)

        x = self.size_red_2(x)
        x = self.moduleC_1(x)
        x = self.moduleC_2(x)

        x = self.avg_pool(x)
        x = self.dense_1(x)
        x = self.relu(x)
        x = self.batch_norm(x)
        x = self.dropout(x)
        x = self.dense_2(x)
        x = self.relu(x)
        x = self.batch_norm(x)
        x = self.dropout(x)
        x = self.dense_3(x)
        output2 = self.softmax(x)

        return output1, output2
