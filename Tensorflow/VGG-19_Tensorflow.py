import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Activation, Dense, MaxPooling2D, Flatten, Dropout, Input

# Conv Block
def conv_block(x, kernel_size, filters, conv_layers):
    for i in range(conv_layers):
        x = Conv2D(filters=filters, kernel_size=kernel_size, padding='same')(x)
        x = Activation('relu')(x)
    return x

# Main Model
def VGG19(shape, classes):
    x_input = Input(shape=shape)
    x = conv_block(x_input, (3, 3), 64, 2)
    x = MaxPooling2D(pool_size=(2,2), padding='same')(x)
    x = conv_block(x, (3, 3), 128, 2)
    x = MaxPooling2D(pool_size=(2,2), padding='same')(x)
    x = conv_block(x, (3, 3), 256, 4)
    x = MaxPooling2D(pool_size=(2,2), padding='same')(x)
    x = conv_block(x, (3, 3), 512, 4)
    x = MaxPooling2D(pool_size=(2,2), padding='same')(x)
    x = conv_block(x, (3, 3), 512, 4)
    x = Flatten()(x)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(2048, activation='relu')(x)
    output = Dense(classes, activation='softmax')(x)

    model = tf.keras.models.Model(inputs = x_input, outputs = output, name='VGG19')
    return model

vgg19 = VGG19((256, 256, 3), 100)
print(vgg19.summary())
# The implementation can also be done using a Sequential Model