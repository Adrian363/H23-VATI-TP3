import VegetablesCNN as vcnn
import Utils as utils
import numpy as np

# Define classes
classes = ['Bean', 'Bitter_Gourd', 'Bottle_Gourd', 'Brinjal', 'Broccoli', 'Cabbage', 'Capsicum', 'Carrot',
           'Cauliflower', 'Cucumber', 'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato']

# Define class number for each vegetable
classes_number = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

# Load the training dataset
# x_train, y_train = load_data("train", classes)
# y_train = np.asarray(y_train).astype('int32')

# Load the validation dataset
x_validation, y_validation = utils.load_data("validation", classes)

# Load the test dataset
x_test, y_test = utils.load_data("test", classes)
y_test = np.asarray(y_test).astype('int32')

# Create a tuple for the validation dataset
y_validation = np.asarray(y_validation).astype('int32')
validation_data = (x_validation, y_validation)

# Define the shape of the images for the input layer
input_shape = (112, 112, 3)

# Define the number of classes for the output layer
num_classes = len(classes)

# Define the number of batch and epochs
batch_size = 64
epochs = 10

# Create the model from the class
vegetables_cnn = vcnn.VegetablesCNN(input_shape, num_classes)

# Print the model information
vegetables_cnn.summary()

# Compile the model with accuracy for the metrics
vegetables_cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model and all the values (loss, accuracy) for training phase and the validation phase
training_values = vegetables_cnn.fit(x_test, y_test, batch_size, epochs, validation_data)

# Evaluate the model with the test dataset
score = vegetables_cnn.evaluate(x_test, y_test)

# Print the results of the evaluation
print('Loss Score:', score[0])
print('Accuracy Score:', score[1])

# Display graphs with the evolution of the loss and the accuracy during training and validation phases
utils.display_graph(training_values.history['loss'], training_values.history['accuracy'],
                    training_values.history['val_loss'],
                    training_values.history['val_accuracy'])

# Get the prediction for the test dataset
y_prediction = vegetables_cnn.predict(x_test)

# Translate the prediction to obtain the class number for each
y_translated = utils.translate_y_pred(y_prediction)

# Generate rapport of the classification with sklearn
utils.generate_report(y_test, y_translated, classes_number)

# Compile and print the kappa coefficient
utils.get_kappa_coefficient(y_test, y_translated, classes_number)
