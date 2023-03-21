# Common
import os
import Utils as utils

import numpy as np
from keras.applications import ResNet50V2

# Data
from keras.preprocessing.image import ImageDataGenerator

# Model
from keras.models import Sequential
from keras.layers import GlobalAvgPool2D as GAP, Dense

# Callbacks
from keras.callbacks import EarlyStopping, ModelCheckpoint
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from tensorflow.python.keras import optimizers


def get_files():
    # Class Names
    root_path = './Vegetable Images/train/'
    class_names = sorted(os.listdir(root_path))
    n_classes = len(class_names)

    # Class Distribution
    class_dis = [len(os.listdir(root_path + name)) for name in class_names]

    # Show
    print(f"Total Number of Classes : {n_classes} \nClass Names : {class_names}")

    # Initialize Generator
    train_gen = ImageDataGenerator(rescale=1 / 255.0, rotation_range=10)
    valid_gen = ImageDataGenerator(rescale=1 / 255.0)
    test_gen = ImageDataGenerator(rescale=1 / 255.0)

    # Load Data / flow_from_directory(CHEMIN des photos, "binary" will be 1D binary labels,
    # color_mode = "rgb" de base, shuffle = true de base, target_size = (256,256) de base
    train_ds = train_gen.flow_from_directory(root_path, class_mode='binary', target_size=(256, 256), shuffle=True)
    valid_ds = valid_gen.flow_from_directory(root_path.replace('train', 'validation'), class_mode='binary',
                                             target_size=(256, 256), shuffle=True)
    test_ds = test_gen.flow_from_directory(root_path.replace('train', 'test'), class_mode='binary',
                                           target_size=(256, 256), shuffle=True)
    return train_ds, valid_ds, test_ds, n_classes, class_names


training, validation, testing, nb_classes, class_names = get_files()

# Define the shape of the images for the input layer
input_shape = (112, 112, 3)

# Define the number of batch and epochs
batch_size = 64
epochs = 2

# Pre-Trained Model
base_model = ResNet50V2(weights="imagenet", input_shape=input_shape, include_top=False)
base_model.trainable = False

# Model Architecture
name = "ResNet50V2"
model = Sequential([
    base_model,
    GAP(), # si je retire erreur : logits and labels must have the same first dimension, got logits shape [2048,15] and labels shape [32]
    Dense(112, activation='relu', kernel_initializer='he_normal'),
    Dense(nb_classes, activation='softmax')
], name=name)

# Callbacks
#model_checkpoint_callback = EarlyStopping(patience=3, restore_best_weights=True)

# Print the model information
model.summary()

# Compile the model with accuracy for the metrics
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model and all the values (loss, accuracy) for training phase and the validation phase
training_values = model.fit(training, batch_size=32, epochs=epochs, validation_data=validation)

# Evaluate the model with the test dataset
score = model.evaluate(testing)

# Print the results of the evaluation
print('Loss Score:', score[0])
print('Accuracy Score:', score[1])

# Display graphs with the evolution of the loss and the accuracy during training and validation phases
utils.display_graph(training_values.history['loss'], training_values.history['accuracy'],
                    training_values.history['val_loss'],
                    training_values.history['val_accuracy'])




# Visualize Predictions
#show_images(model, testing, class_names)

# https://www.kaggle.com/code/utkarshsaxenadn/vegetable-classification-resnet50v2-acc-99/notebook
# https://keras.io/api/layers/core_layers/dense/
# https://keras.io/api/applications/#usage-examples-for-image-classification-models
# https://keras.io/guides/transfer_learning/
# https://keras.io/api/models/model_training_apis/