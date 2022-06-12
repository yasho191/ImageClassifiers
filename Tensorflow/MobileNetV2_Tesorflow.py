import tensorflow as tf
from tensorflow.keras.layers import ReLU, Add, GlobalAveragePooling2D
from tensorflow.keras.layers import DepthwiseConv2D, Conv2D, BatchNormalization
from tensorflow.keras.layers import Input, Dropout, Dense

# relu6 is an modified version of relu where the max value == 6
#         |   ______
#         |  /
#         | /
#  _______|/
# -----------------

# BottleNeck1 for Strides == 1
def Bottleneck1(x, expansion_factor: int, out_filters: int):
    x_res = x
    n_filters = x.shape[-1]

    x = Conv2D(expansion_factor * n_filters, (1, 1), 1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU(6.0)(x)
    x = DepthwiseConv2D((3, 3), (1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU(6.0)(x)
    x = Conv2D(out_filters, (1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)

    # Add Residue if and only if in_channels = out_channels
    if n_filters == out_filters:
        x = Add()([x_res, x])

    return x

# BottleNeck2 for Strides == 2
def Bottleneck2(x, expansion_factor: int, out_filters: int):
    n_filters = x.shape[-1]

    x = Conv2D(expansion_factor * n_filters, (1, 1), 1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU(6.0)(x)
    x = DepthwiseConv2D((3, 3), (2, 2), padding='valid', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU(6.0)(x)
    x = Conv2D(out_filters, (1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)

    return x

# mobilenetv2
def MobileNetV2(shape, num_classes):
    x_input = Input(shape)

    # First Block
    x = Conv2D(32, (3, 3), (2, 2), padding='same', use_bias=False)(x_input)
    x = BatchNormalization()(x)
    x = ReLU(6.0)(x)

    # sequence of bottleneck blocks
    bottleneck_sequence = [
        # t, c, n, s
        [1, 16, 1, 1],
        [6, 24, 2, 2],
        [6, 32, 3, 2],
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1]
    ]

    # Iterating over all the blocks
    for t, c, n, s in bottleneck_sequence:
        for i in range(n):
            if i == 0 and s == 2:
                x = Bottleneck2(x, t, c)
            else:
                x = Bottleneck1(x, t, c)

    # End Layers
    x = Conv2D(1280, (1, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU(6.0)(x)
    x = GlobalAveragePooling2D()(x)

    # Classifier
    x = Dropout(0.2)(x)
    x_output = Dense(num_classes, activation='softmax')(x)

    # Build Model
    model = tf.keras.models.Model(inputs = x_input, outputs = x_output, name='MobileNetV3')
    return model
