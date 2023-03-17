import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy


class VegetablesCNN:

    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.models.Sequential()

        # Add convolutional layers to the model
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape))

        return model

    def summary(self):
        self.model.summary()

    def compile(self, optimizer, loss, metrics):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def fit(self, x_train, y_train, batch_size, epochs, validation_data=None):
        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)

    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)

    def predict(self, x_test):
        return self.model.predict(x_test)
