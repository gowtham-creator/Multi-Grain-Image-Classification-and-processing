import os
import cv2
import pickle
import random
import numpy as np
import time
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow import keras

# Suppress TensorFlow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Function to load the VGG16 model with retries
def load_vgg16_model(retries=3, delay=5):
    for i in range(retries):
        try:
            return VGG16(input_shape=(225, 225, 3), include_top=False, weights='imagenet')
        except Exception as e:
            if i < retries - 1:
                print(f"Error loading VGG16 model: {e}. Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                raise e

# Dataset Preparation
dataset_path = "./dataset/"

# Load or prepare dataset
if "X.obj" not in os.listdir() or "y.obj" not in os.listdir():
    class_map = {}
    index = 0
    X = []
    y = []
    for class_name in os.listdir(dataset_path):
        if class_name not in class_map:
            class_map[class_name] = index
        ohe = [0.0 for _ in range(len(os.listdir(dataset_path)))]
        ohe[index] = 1.0
        index += 1
        for image_file in os.listdir(os.path.join(dataset_path, class_name)):
            image_path = os.path.join(dataset_path, class_name, image_file)
            image = cv2.imread(image_path)
            if image is not None:
                image_resized = cv2.resize(image, (225, 225))
                X.append(image_resized)
                y.append(ohe)
    X = np.array(X, dtype=np.int8)
    y = np.array(y, dtype=float)
    pickle.dump(X, open("X.obj", "wb"))
    pickle.dump(y, open("y.obj", "wb"))
else:
    X = pickle.load(open("X.obj", "rb"))
    y = pickle.load(open("y.obj", "rb"))

# Determine number of classes
num_classes = y.shape[1]

# Check if there's only one image in the dataset
if len(X) == 1:
    print("Warning: Only one sample in the dataset. Using the same sample for training and testing.")
    train_x, test_x, train_y, test_y = X, X, y, y
else:
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=100)

# Model building
if "model.h5" not in os.listdir():
    base_model = load_vgg16_model()
    model = Sequential()
    for layer in base_model.layers:
        model.add(layer)
    model.add(Flatten())
    model.add(Dense(100, activation="relu"))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(25, activation="relu"))
    model.add(Dense(num_classes, activation="softmax"))
else:
    model = keras.models.load_model("model.h5")

while True:
    if input("Train again? (y/n): ") != "y":
        break
    
    if len(X) > 1:
        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=random.randint(0, 1000))
        validation_split = 0.2
    else:
        print("Warning: Only one sample in the dataset. Using the same sample for training and validation.")
        train_x, test_x, train_y, test_y = X, X, y, y
        validation_split = 0.0

    model.summary()
    model.compile(loss="categorical_crossentropy", 
                  optimizer=SGD(learning_rate=0.000001),  # Changed from lr to learning_rate
                  metrics=["acc"])
    
    model.fit(train_x, train_y, epochs=5, batch_size=16, validation_split=validation_split)
    model.save("model2.h5")
    model.evaluate(test_x, test_y)
