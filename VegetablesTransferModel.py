import tensorflow as tf
from tensorflow import keras
import VegetablesCNNTest as VegeCNN

# CREATE ResNet50V2 MODEL
trained_model = tf.keras.applications.ResNet50V2(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
)

# FREEZE TRAINED MODEL
trained_model.trainable = False

# CREATE MODEL ON TOP OF ResNet50V2
inputs = keras.Input(shape=(224, 224, 3))
# We make sure that the base_model is running in inference mode here,
# by passing `training=False`. This is important for fine-tuning, as you will
# learn in a few paragraphs.
x = trained_model(inputs, training=False)
# Convert features of shape `base_model.output_shape[1:]` to vectors
# x = keras.layers.GlobalAveragePooling2D()(x)

# A Dense classifier with a single unit (binary classification)
outputs = keras.layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

# TRAIN MODEL
x_train, y_train = VegeCNN.load_data("train")
x_validation, y_validation = VegeCNN.load_data("validation")
x_test, y_test = VegeCNN.load_data("test")
model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[keras.metrics.BinaryAccuracy()])
model.fit((x_train, y_train), epochs=20, callbacks=..., validation_data=(x_validation, y_validation))
