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
    
    # Processing Residue with conv(1,1)
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

# Initialize Model
model = ResNet34((256, 256, 3), 100)
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
                        target_size=(256, 256),
                        batch_size=BATCH_SIZE,
                        class_mode='categorical')

valid_data = validation_data_generator.flow_from_directory(
                        VALID_IMG_DIR,
                        target_size=(256, 256),
                        batch_size=BATCH_SIZE,
                        class_mode='categorical')

test_data = test_data_generator.flow_from_directory(
                        TEST_IMG_DIR,
                        target_size=(256, 256),
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
