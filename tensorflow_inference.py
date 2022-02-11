import tensorflow as tf
import argparse
import numpy as np
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("model_path", help="Path to pre-trained model")
parser.add_argument("image", help="Path to image")

args = parser.parse_args()

model = tf.keras.models.load_model(args.model_path)
layer0 = model.get_layer(index=0)
shape = layer0.input_shape[0][1:]

image = cv2.imread(args.image)
image = cv2.resize(image, shape[:2], cv2.INTER_CUBIC)

image = np.expand_dims(image, axis=0)
prediction = model.predict(image)
print("Prediction:", prediction)