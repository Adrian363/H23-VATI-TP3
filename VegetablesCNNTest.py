import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import image
import os
import io
import cv2


from PIL import Image

import VegetablesCNN as vcnn

# Define classes
classes = ['Bean', 'Bitter_Gourd', 'Bottle_Gourd', 'Brinjal', 'Broccoli', 'Cabbage', 'Capsicum', 'Carrot',
           'Cauliflower', 'Cucumber', 'Papaya', 'Patato', 'Pumpkin', 'Radish', 'Tomato']


# Load the data and group them to create the different datasets
def load_data(dataset_phase):
    vegetables_folder = os.listdir("Vegetable Images/{}".format(dataset_phase))

    x_img = []
    y_img = []

    for vegetable in vegetables_folder:

        vegetable_folder_content = os.listdir("Vegetable Images/{}/{}".format(dataset_phase, vegetable))

        for img in vegetable_folder_content:

            image_path = "Vegetable Images/{}/{}/{}".format(dataset_phase, vegetable, img)
            x_img.append(cv2.imread(image_path))
            y_img.append(vegetable)

    return x_img, y_img


# Load all the dataset
x_train, y_train = load_data("train")
x_validation, y_validation = load_data("validation")
x_test, y_test = load_data("test")


x_train = np.array([np.array(val) for val in x_train])

y_train = np.array(y_train)

#y_train_int = np.array([np.array(test) for test in y_train])("informatics")
x_validation = np.array(x_validation)
y_validation = np.array(y_validation)
x_test = np.array(x_test)
y_test = np.array(y_test)

# Create tuple for validation dataset
validation_data = (x_validation, y_validation)

input_shape = (224, 224, 3)
num_classes = len(classes)
batch_size = 32
epochs = 10

vegetables_cnn = vcnn.VegetablesCNN(input_shape, num_classes)
vegetables_cnn.summary()
vegetables_cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

vegetables_cnn.fit(x_train, y_train, batch_size, epochs)





