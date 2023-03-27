# Common
import os
import Utils as utils
import numpy as np

# Model
from keras.models import Sequential
from keras.applications import ResNet50V2
from keras.layers import Dense, Flatten

# Class Names
root_path = './Vegetable Images/train/'
class_names = sorted(os.listdir(root_path))
n_classes = len(class_names)

# Define class number for each vegetable
classes_number = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

# Load the training dataset
x_train, y_train = utils.load_data("train", class_names)
y_train = np.asarray(y_train).astype('int32')

# Load the validation dataset
x_validation, y_validation = utils.load_data("validation", class_names)

# Load the test dataset
x_test, y_test = utils.load_data("test", class_names)
y_test = np.asarray(y_test).astype('int32')

# Create a tuple for the validation dataset
y_validation = np.asarray(y_validation).astype('int32')
validation_data = (x_validation, y_validation)

# Define the shape of the images for the input layer
input_shape = (112, 112, 3)

# Define the number of batch and epochs
batch_size = 32
epochs = 10

# Pre-Trained Model
base_model = ResNet50V2(weights="imagenet", input_shape=input_shape, include_top=False)
for layer in base_model.layers:
    layer.trainable = False

# Model Architecture
name = "ResNet50V2"
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu', kernel_initializer='he_normal'),
    Dense(128, activation='relu', kernel_initializer='he_normal'),
    Dense(64, activation='relu', kernel_initializer='he_normal'),
    Dense(n_classes, activation='softmax')
], name=name)

# Print the model information
model.summary()

# Compile the model with accuracy for the metrics
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model and all the values (loss, accuracy) for training phase and the validation phase
training_values = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=validation_data)

# Evaluate the model with the test dataset
score = model.evaluate(x_test, y_test)

# Print the results of the evaluation
print('Loss Score:', score[0])
print('Accuracy Score:', score[1])

# Display graphs with the evolution of the loss and the accuracy during training and validation phases
utils.display_graph(training_values.history['loss'], training_values.history['accuracy'],
                    training_values.history['val_loss'],
                    training_values.history['val_accuracy'])

# Get the prediction for the test dataset
y_prediction = model.predict(x_test)

# Translate the prediction to obtain the class number for each
y_translated = utils.translate_y_pred(y_prediction)

# Generate rapport of the classification with sklearn
utils.generate_report(y_test, y_translated, classes_number)

# Compile and print the kappa coefficient
utils.get_kappa_coefficient(y_test, y_translated, classes_number)


# https://www.kaggle.com/code/utkarshsaxenadn/vegetable-classification-resnet50v2-acc-99/notebook
# https://keras.io/api/layers/core_layers/dense/
# https://keras.io/api/applications/#usage-examples-for-image-classification-models
# https://keras.io/guides/transfer_learning/
# https://keras.io/api/models/model_training_apis/