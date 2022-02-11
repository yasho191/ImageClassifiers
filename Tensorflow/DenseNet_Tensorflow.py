# DenseNet - 121
import os
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.layers import Activation, MaxPool2D
from tensorflow.keras.layers import AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Dense

# Basic conv block of densenet consits of -> BatchNorm -> ReLU -> Conv2D
def ConvBlock(x, filters, kernel_size):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=filters, kernel_size=kernel_size, padding='same')(x)
    return x

# The dense block of densenet consits of Recurring conv blocks
# The output of every preceding conv block is concatenated and 
# fed to the next conv block as the input
def DenseBlock(x, layers):
    residuals = []
    for _ in range(layers):
        x = ConvBlock(x, filters=32*4, kernel_size=(1,1))
        x = ConvBlock(x, filters=32, kernel_size=(3,3))
        residuals.append(x)
        for i in range(_):
            x = Concatenate()([x, residuals[i]])
    return x

# DownSample block also called transition block
# Structure: BatchNormalization -> Conv2D -> AveragePooling2D
# The filters given to the Conv2D here are based on the theta = 0.5
# which means the number of filters will be half of the input filters
def DownSample(x):
    x = BatchNormalization()(x)
    x = Conv2D(filters = x.shape[-1]//2, kernel_size = (1, 1))(x)
    x = AveragePooling2D(pool_size = (2,2), strides = 2, padding = 'same')(x)
    return x

# Main Model
def DenseNet(shape, classes):
    x_input = Input(shape)
    x = Conv2D(32, kernel_size=(7, 7), strides = 2, padding = 'same')(x_input)
    x = MaxPool2D((3, 3), strides = 2, padding = 'same')(x)

    # Dense Blocks followed by Downsample Blocks
    # Block sizes = 6, 12, 24, 16
    x = DenseBlock(x, 6)
    x = DownSample(x)
    x = DenseBlock(x, 12)
    x = DownSample(x)
    x = DenseBlock(x, 24)
    x = DownSample(x)
    x = DenseBlock(x, 16)

    # GlobalAveragePooling2D followed by Dense and Softmax
    x = GlobalAveragePooling2D()(x)
    output = Dense(classes, activation='softmax')(x)

    model = tf.keras.models.Model(inputs = x_input, outputs = output, name = "DenseNet")
    return model


