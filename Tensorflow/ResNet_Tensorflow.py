import tensorflow as tf
import os
from tensorflow.keras.layers import Conv2D, BatchNormalization
from tensorflow.keras.layers import Activation, Dense, Flatten
from tensorflow.keras.layers import Add, Input
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D

def identity_block(x, filter):
    # copy tensor to variable called x_skip
    x_skip = x
    
    # Layer 1
    x = Conv2D(filter, (3,3), padding = 'same')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    
    # Layer 2
    x = Conv2D(filter, (3,3), padding = 'same')(x)
    x = BatchNormalization(axis=3)(x)

    # Skip
    x_skip = Conv2D(filter, (1, 1), padding ="same")(x_skip)
    
    # Add Residue
    x = Add()([x, x_skip])     
    x = Activation('relu')(x)
    
    return x

def convolutional_block(x, filter):
    # copy tensor to variable called x_skip
    x_skip = x
    
    # Layer 1
    x = Conv2D(filter, (3,3), padding = 'same', strides = (2,2))(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    
    # Layer 2
    x = Conv2D(filter, (3,3), padding = 'same')(x)
    x = BatchNormalization(axis=3)(x)
    
    # Processing Residue with conv(1,1) stride=2
    x_skip = Conv2D(filter, (1,1), strides = (2,2))(x_skip)
    
    # Add Residue
    x = Add()([x, x_skip])     
    x = Activation('relu')(x)
    
    return x

def ResNet34(shape, classes):
    # Step 1 (Setup Input Layer)
    x_input = Input(shape)
    x = ZeroPadding2D((3, 3))(x_input)
    
    # Step 2 (Initial Conv layer along with maxPool)
    x = Conv2D(64, kernel_size=7, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)
    
    # Define size of sub-blocks and initial filter size
    block_layers = [3, 4, 6, 3]
    filter_size = 64
    
    # Step 3 Add the Resnet Blocks
    for i in range(4):
        if i == 0:
            # For sub-block 1 Residual/Convolutional block not needed
            for j in range(block_layers[i]):
                x = identity_block(x, filter_size)
        else:
            # One Residual/Convolutional Block followed by Identity blocks
            # The filter size will go on increasing by a factor of 2
            filter_size = filter_size*2
            x = convolutional_block(x, filter_size)
            for j in range(block_layers[i] - 1):
                x = identity_block(x, filter_size)
          
    # Step 4 End Dense Network
    x = AveragePooling2D((2,2), padding = 'same')(x)
    x = Flatten()(x)
    x = Dense(512, activation = 'relu')(x)
    x = Dense(256, activation = 'relu')(x)
    x = Dense(classes, activation = 'softmax')(x)
    
    model = tf.keras.models.Model(inputs = x_input, outputs = x, name = "ResNet34")
    return model