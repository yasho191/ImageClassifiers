import os
import tensorflow as tf
import argparse
from tensorflow_models import Models

parser = argparse.ArgumentParser()
parser.add_argument("model", help="Name of the model. Must be one of: 1. AlexNet 2. DenseNet 3. InceptionV3\n4. ResNet\n5. VGG", type=str)
parser.add_argument("shape", help="Input Shape", type=int, default=(256, 256, 3))
parser.add_argument("classes", help="Number of classes", type=int)
parser.add_argument("-e", "--epochs", type=int, default=100)
parser.add_argument("-b", "--batch_size", type=int, default=16)

args = parser.parse_args()

MODEL_NAME = args.model
SHAPE = (args.shape, args.shape, 3)
CLASSES = args.classes
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size

if MODEL_NAME not in ["AlexNet", "DenseNet", "InceptionV3", "ResNet", "VGG"]:
    print(f"Invalid argument for model: {MODEL_NAME}")
    exit(-1)


model = Models(MODEL_NAME, SHAPE, CLASSES)

model = model.ret_model()
print(model.summary())

# Initialize Variables
TRAIN_IMG_DIR = os.path.join('data/train')
VALID_IMG_DIR = os.path.join('data/validate')

# Initialize Data Generators
# Different Augmentations can also be applied on data
train_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255., horizontal_flip=True, rotation_range=270)
validation_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255.)

CLASS_MODE = 'sparse'
# Get Training, Testing and Validation Data
train_data = train_data_generator.flow_from_directory(
                        TRAIN_IMG_DIR,
                        target_size=SHAPE[:2],
                        batch_size=BATCH_SIZE,
                        class_mode=CLASS_MODE)

valid_data = validation_data_generator.flow_from_directory(
                        VALID_IMG_DIR,
                        target_size=SHAPE[:2],
                        batch_size=BATCH_SIZE//2,
                        class_mode=CLASS_MODE)

print("Training Data Indices: ", train_data.class_indices)
print("Validation Data Indices: ", valid_data.class_indices)
# learning rate scheduler can also be applied based on requirements
lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.1,
        patience=5,
)

# Losses can be SparseCategoricalCrossentropy or CategoricalCrossentropy 
# BinaryCrossentropy in case of binary classification

# loss = tf.keras.losses.BinaryCrossentropy(
#     from_logits=False, label_smoothing=0.0, axis=-1, name='binary_crossentropy'
# )

loss = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=False, name='sparse_categorical_crossentropy'
)

# Compile models with Adam Optimizer or SGD
model.compile(optimizer=tf.keras.optimizers.Adam(), loss=loss, metrics='accuracy')

history = model.fit(
    train_data,
    validation_data = valid_data,
    epochs = EPOCHS,
    callbacks=[lr_schedule]
)

# Save model
model.save(f'Models/{MODEL_NAME}_model')