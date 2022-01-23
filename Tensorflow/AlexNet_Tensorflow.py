import os
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Activation, Flatten
from tensorflow.keras.layers import Input

# Conv block consisting of Conv -> Activation -> maxpooling
def conv_block(x, kernel_size, filters, conv_strides, pool_size, pool_strides):
    x = Conv2D(kernel_size=kernel_size, filters=filters, padding='same', strides=conv_strides)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=pool_size, strides=pool_strides)(x)
    return x

# AlexNet Model
def AlexNet(shape, classes):
    # Input layer
    x_input = Input(shape=shape)
    # 2 Conv BLocks with MaxPooling
    x = conv_block(x_input, (11, 11), 96, 4, (3, 3), 2)
    x = conv_block(x, (5, 5), 256, 1, (3, 3), 2)
    # 2 Conv Layers without Maxpooling
    x = Conv2D(kernel_size=(3,3), filters=384, padding='same', activation='relu')(x)
    x = Conv2D(kernel_size=(3,3), filters=384, padding='same', activation='relu')(x)
    # 1 Conv block with MaxPooling
    x = conv_block(x, (3, 3), 256, 1, (3, 3), 2)
    x = Flatten()(x)
    # Fully connected layers with Dropout
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.2)(x)
    output = Dense(classes, activation='softmax')(x)

    model = tf.keras.models.Model(inputs = x_input, outputs = output, name = "AlexNet")
    return model

# Initialize Model
model = AlexNet((227, 227, 3), 1000)
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
                        target_size=(299, 299),
                        batch_size=BATCH_SIZE,
                        class_mode='categorical')

valid_data = validation_data_generator.flow_from_directory(
                        VALID_IMG_DIR,
                        target_size=(299, 299),
                        batch_size=BATCH_SIZE,
                        class_mode='categorical')

test_data = test_data_generator.flow_from_directory(
                        TEST_IMG_DIR,
                        target_size=(299, 299),
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
