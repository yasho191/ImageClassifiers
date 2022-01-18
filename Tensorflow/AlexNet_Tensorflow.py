import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, Input

# Conv block consisting of Conv -> Activation -> maxpooling
def conv_block(x, kernel_size, filters, conv_strides, pool_size, pool_strides):
    x = Conv2D(kernel_size=kernel_size, filters=filters, padding='same', strides=conv_strides)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=pool_size, strides=pool_strides)(x)
    return x

def AlexNet():
    # Input layer
    x_input = Input(shape=(227, 227, 3))
    # 2 Conv BLocks with MaxPooling
    x = conv_block(x_input, (11, 11), 96, 4, (3, 3), 2)
    x = conv_block(x, (5, 5), 256, 2, (3, 3), 2)
    # 2 Conv Layers without Maxpooling
    x = Conv2D(kernel_size=(3,3), filters=384, padding='same', activation='relu')(x)
    x = Conv2D(kernel_size=(3,3), filters=384, padding='same', activation='relu')(x)
    # 1 Conv block with MaxPooling
    x = conv_block(x, (3, 3), 256, 2, (3, 3), 2)
    x = Flatten()(x)
    # Fully connected layers with Dropout
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.2)(x)
    output = Dense(1000, activation='softmax')(x)

    model = tf.keras.models.Model(inputs = x_input, outputs = output, name = "AlexNet")
    return model

alex = AlexNet()
print(alex.summary())

# For compiling the model use Adam Optimizer or SGD Optimizer
# loss = categorical_crossentropy
# For Experimentation 
# Batch Normalization can be applied, Dropout parameters can be changed
