# DenseNet - 121
import os
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.layers import Activation, MaxPool2D
from tensorflow.keras.layers import AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Dense, Dropout

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


model = DenseNet((224, 224, 3), 1000)
print(model.summary())

# Initialize Variables
EPOCHS = 1000
BATCH_SIZE = 32
TRAIN_IMG_DIR = os.path.join("path_to_dir", "train")
VALID_IMG_DIR = os.path.join("path_to_dir", "valid")
TEST_IMG_DIR = os.path.join("path_to_dir", "test")

# Initialize Data Generators
# Different Augmentations can also be applied on data
train_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255.)
validation_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255.)
test_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255.)

# Get Training, Testing and Validation Data
train_data = train_data_generator.flow_from_directory(
                        TRAIN_IMG_DIR,
                        target_size=(224, 224),
                        batch_size=BATCH_SIZE,
                        class_mode='categorical')

valid_data = validation_data_generator.flow_from_directory(
                        VALID_IMG_DIR,
                        target_size=(224, 224),
                        batch_size=BATCH_SIZE,
                        class_mode='categorical')

test_data = test_data_generator.flow_from_directory(
                        TEST_IMG_DIR,
                        target_size=(224, 224),
                        batch_size=BATCH_SIZE,
                        class_mode='categorical')

# learning rate scheduler can also be applied based on requirements
lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.1,
        patience=5,
)

# Losses can be SparseCategoricalCrossentropy or CategoricalCrossentropy 
# BinaryCrossentropy in case of binary classification
loss = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=False, reduction="auto", name="sparse_categorical_crossentropy"
)

# Compile models with Adam Optimizer or SGD
model.compile(optimizer=tf.keras.optimizers.Adam(), loss=loss, metrics='accuracy')

history = model.fit(
    x = train_data,
    validation_data = valid_data,
    epochs = EPOCHS,
    callbacks=[lr_schedule]
)

# Save model
model.save('path_to_sav_dir/model')

