import tensorflow as tf
from keras import layers


class VegetablesCNN:

    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.models.Sequential()

        # Add convolutional layers to the model
        model.add(layers.Conv2D(32, (3, 3), strides=1, activation='relu', input_shape=self.input_shape))
        model.add(layers.Conv2D(64, (3, 3), strides=2, activation='relu'))
        model.add(layers.Conv2D(128, (3, 3), strides=2, activation='relu'))
        model.add(layers.Conv2D(64, (3, 3), strides=1, activation='relu'))

        model.add(layers.Flatten())

        model.add(layers.Dense(self.num_classes, activation="softmax"))

        return model
    def summary(self):
        self.model.summary()

    def compile(self, optimizer, loss, metrics):
        return self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def fit(self, x_train, y_train, batch_size, epochs, validation_data):
        return self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=validation_data)

    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)

    def predict(self, x_test):
        return self.model.predict(x_test)
