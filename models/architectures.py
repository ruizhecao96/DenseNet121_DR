import gin
import tensorflow as tf
from tensorflow import keras


@gin.configurable
def DenseNet121(IMG_SIZE, dropout_rate, dense_units, idx_layer):
    # Create base model
    base_model = keras.applications.DenseNet121(
        weights='imagenet',
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False)

    base_model = keras.Model(inputs=base_model.input, outputs=base_model.get_layer(index=idx_layer).output)
    # Freeze base model
    base_model.trainable = False

    x = base_model.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    # x = keras.layers.Flatten()(x)
    x = keras.layers.Dropout(dropout_rate)(x)
    x = keras.layers.Dense(dense_units, activation=None)(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(2, activation=None)(x)
    outputs = keras.layers.Activation('softmax')(x)

    model = keras.Model(base_model.input, outputs, name='DenseNet121')

    return model