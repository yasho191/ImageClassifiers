import os
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Activation
from tensorflow.keras.layers import Dense, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dropout
from tensorflow.keras.layers import Input

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

model = VGG19((256, 256, 3), 100)
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
