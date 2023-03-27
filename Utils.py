import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
from sklearn.metrics import classification_report
from PIL import Image


# Load the data and group them to create the different datasets
def load_data(dataset_phase, classes):

    #  Load the vegetables folders for the selected dataset phase
    vegetables_folder = os.listdir("Vegetable Images/{}".format(dataset_phase))

    # Images and labels
    x_img = []
    y_img = []

    # For each vegetable folder, load the images and add them to the list
    for vegetable in vegetables_folder:

        # Get the image linked to the selected vegetable
        vegetable_folder_content = os.listdir("Vegetable Images/{}/{}".format(dataset_phase, vegetable))

        # For each image, load it in the x array and add the class name to the y array
        for img in vegetable_folder_content:
            image_path = "Vegetable Images/{}/{}/{}".format(dataset_phase, vegetable, img)
            img_opened = Image.open(image_path).convert('RGB')
            img_opened = img_opened.resize((112, 112))
            x_img.append(np.array(img_opened))
            y_img.append(classes.index(vegetable))

    return np.array(x_img), y_img


# Display the training and validation graphs for loss and accuracy
def display_graph(loss_train, acc_train, loss_val, acc_val):
    plt.plot(loss_train, 'r', label='Training loss')
    plt.plot(loss_val, 'b', label='validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.plot(acc_train, 'r', label='Training accuracy')
    plt.plot(acc_val, 'b', label='validation accuracy')
    plt.title('Training and Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


# Generate the classification report for each class
def generate_report(y_true, y_prediction, labels):
    print(classification_report(y_true, y_prediction, labels=labels))


# Generate and print the kappa coefficient of the trained model
def get_kappa_coefficient(y_true, y_prediction, classes):
    kappa = tfa.metrics.CohenKappa(num_classes=len(classes), sparse_labels=True)
    kappa.update_state(y_true, y_prediction)
    result = kappa.result()
    print("Kappa coefficient: {}".format(result))


# Translate the prediction array to the corresponding class number
def translate_y_pred(y_pred):
    y_translated = np.array(len(y_pred) * [0])

    for i in range(len(y_pred)):
        y_translated[i] = np.argmax(y_pred[i])

    return y_translated
