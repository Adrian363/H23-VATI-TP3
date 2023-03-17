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


x_train, y_train = load_data("train")
x_validation, y_validation = load_data("validation")
x_test, y_test = load_data("test")

