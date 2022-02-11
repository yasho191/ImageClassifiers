import os
import tensorflow as tf
from tensorflow.keras.layers import Activation, Concatenate
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D

# Basic Conv-Block consists of Conv -> BatchNormalization -> Relu
def ConvBlock(x_input, filters, kernel_size, strides, padding):
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding= padding)(x_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

# Module A 4 branches
def ModuleA(x_base):
    x_1 = ConvBlock(x_base, 64, (1,1), 1, "valid")
    x_1 = ConvBlock(x_1, 96, (3, 3), 1, "same")
    x_1 = ConvBlock(x_1, 96, (3, 3), 1, "same")

    x_2 = ConvBlock(x_base, 48, (1,1), 1, "valid")
    x_2 = ConvBlock(x_2, 64, (3, 3), 1, "same")

    x_3 = MaxPooling2D(pool_size=(3, 3), strides=1, padding="same")(x_base)
    x_3 = ConvBlock(x_3, 64, (1, 1), 1, "valid")

    x_4 = ConvBlock(x_base, 64, (1, 1), 1, "valid")

    x = Concatenate()([x_1, x_2, x_3, x_4])
    return x

# Module B 4 branches
def ModuleB(x_base, f_7x7):
    x_1 = ConvBlock(x_base, f_7x7, (1, 1), 1, "valid")
    x_1 = tf.pad(x_1, [[0, 0], [0, 0], [3, 3], [0, 0]], mode='CONSTANT')
    x_1 = ConvBlock(x_1, f_7x7, (1, 7), 1, "valid")
    x_1 = tf.pad(x_1, [[0, 0], [3, 3], [0, 0], [0, 0]], mode='CONSTANT')
    x_1 = ConvBlock(x_1, f_7x7, (7, 1), 1, "valid")
    x_1 = tf.pad(x_1, [[0, 0], [0, 0], [3, 3], [0, 0]], mode='CONSTANT')
    x_1 = ConvBlock(x_1, f_7x7, (1, 7), 1, "valid")
    x_1 = tf.pad(x_1, [[0, 0], [3, 3], [0, 0], [0, 0]], mode='CONSTANT')
    x_1 = ConvBlock(x_1, 192, (7, 1), 1, "valid")

    x_2 = ConvBlock(x_base, f_7x7, (1, 1), 1, "valid")
    x_2 = tf.pad(x_2, [[0, 0], [0, 0], [3, 3], [0, 0]], mode='CONSTANT')
    x_2 = ConvBlock(x_2, f_7x7, (1, 7), 1, "valid")
    x_2 = tf.pad(x_2, [[0, 0], [3, 3], [0, 0], [0, 0]], mode='CONSTANT')
    x_2 = ConvBlock(x_2, 192, (7, 1), 1, "valid")

    x_3 = MaxPooling2D(pool_size=(3, 3), strides=1, padding="same")(x_base)
    x_3 = ConvBlock(x_3, 192, (1, 1), 1, "valid")

    x_4 = ConvBlock(x_base, 192, (1, 1), 1, "valid")

    x = Concatenate()([x_1, x_2, x_3, x_4])
    return x

# Module C 4 branches
def ModuleC(x_base):
    x_1 = ConvBlock(x_base, 448, (1, 1), 1, "valid")
    x_1 = ConvBlock(x_1, 384, (3, 3), 1, "same")
    x_1_top = tf.pad(x_1, [[0, 0], [0, 0], [1, 1], [0, 0]])
    x_1_top = ConvBlock(x_1_top, 384, (1, 3), 1, "valid")
    x_1_bottom = tf.pad(x_1, [[0, 0], [1, 1], [0, 0], [0, 0]])
    x_1_bottom = ConvBlock(x_1_bottom, 384, (3, 1), 1, "valid")
    x_1 = Concatenate()([x_1_top, x_1_bottom])

    x_2 = ConvBlock(x_base, 384, (1, 1), 1, "valid")
    x_2_top = tf.pad(x_2, [[0, 0], [0, 0], [1, 1], [0, 0]])
    x_2_top = ConvBlock(x_2_top, 384, (1, 3), 1, "valid")
    x_2_bottom = tf.pad(x_2, [[0, 0], [1, 1], [0, 0], [0, 0]])
    x_2_bottom = ConvBlock(x_2_bottom, 384, (3, 1), 1, "valid")
    x_2 = Concatenate()([x_2_top, x_2_bottom])

    x_3 = MaxPooling2D(pool_size=(3, 3), strides=1, padding="same")(x_base)
    x_3 = ConvBlock(x_3, 192, (1, 1), 1, "valid")

    x_4 = ConvBlock(x_base, 320, (1, 1), 1, "valid")

    x = Concatenate()([x_1, x_2, x_3, x_4])
    return x

# Down Sampling Block 3 Branches
def SizeReductionBlock(x_base, f_3x3_r, add_ch):
    x_1 = ConvBlock(x_base, f_3x3_r, (1, 1), 1, "valid")
    x_1 = ConvBlock(x_1, 178+add_ch, (3, 3), 1, "same")
    x_1 = ConvBlock(x_1, 178+add_ch, (3, 3), 2, "valid")

    x_2 = ConvBlock(x_base, f_3x3_r, (1, 1), 1, "valid")
    x_2 = ConvBlock(x_2, 302+add_ch, (3, 3), 2, "valid")

    x_3 = MaxPooling2D(pool_size=(3, 3), strides=2, padding="valid")(x_base)

    x = Concatenate()([x_1, x_2, x_3])
    return x

# Aux Block or Fine tune block Sequential block 
def FineTuneBlock(x_base, classes):
    x = AveragePooling2D(pool_size=(5, 5), strides=3, padding="valid")(x_base)
    x = ConvBlock(x, 128, (1, 1), 1, "valid")
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(classes, activation='softmax')(x)
    return x

def InceptionV3(shape, classes):
    # Input layer
    x_input = Input(shape)
    # 3 Convolutional blocks
    x = ConvBlock(x_input, 32, (3, 3), 2, "valid")
    x = ConvBlock(x, 32, (3, 3), 1, "valid")
    x = ConvBlock(x, 64, (3, 3), 1, "same")
    # Downsampling
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding="valid")(x)
    # 3 Convolutional blocks
    x = ConvBlock(x, 80, (3, 3), 1, "valid")
    x = ConvBlock(x, 192, (3, 3), 2, "valid")
    x = ConvBlock(x, 288, (3, 3), 1, "same")

    # 3 Module A
    x = ModuleA(x)
    x = ModuleA(x)
    x = ModuleA(x)

    # Downsampling
    x = SizeReductionBlock(x, 64, 0)

    # 5 ModuleB -> 7x7 Factorization Blocks
    x = ModuleB(x, 128)
    x = ModuleB(x, 160)
    x = ModuleB(x, 160)
    x = ModuleB(x, 160)
    x = ModuleB(x, 192)

    # Downsampling
    x = SizeReductionBlock(x, 192, 16)

    # First Auxilary Block (Fine tuning)
    output_1 = FineTuneBlock(x, classes)

    # 2 ModuleC
    x = ModuleC(x)
    x = ModuleC(x)

    # Final Dense Block (Fine tuning)
    # Global Average Pooling
    x = GlobalAveragePooling2D()(x)
    # Flatten
    x = Flatten()(x)

    # Fully connected layers with BatchNormalization and Dropout
    x = Dense(4096, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(4096, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    # Final Output
    output_2 = Dense(classes, activation='softmax')(x)

    model_1 = tf.keras.models.Model(inputs = x_input, outputs = output_1, name="InceptionV3_Short")
    model_2 = tf.keras.models.Model(inputs = x_input, outputs = output_2, name="InceptionV3")
    return model_1, model_2

