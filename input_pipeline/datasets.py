import gin
import logging
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
from input_pipeline.preprocessing import preprocess, aug_and_prepare
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt

AUTOTUNE = tf.data.experimental.AUTOTUNE

def read_tfrecord(example):
    tfrecord_format = {
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'depth': tf.io.FixedLenFeature([], tf.int64),
            'label': tf.io.FixedLenFeature([], tf.int64),
            'image_raw': tf.io.FixedLenFeature([], tf.string),
        }

    example = tf.io.parse_single_example(example, tfrecord_format)
    image = tf.image.decode_jpeg(example["image_raw"], channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [example["height"], example["width"], 3])
    label = tf.cast(example["label"], tf.int32)
    return image, label

def load_dataset(filenames):

    dataset = tf.data.TFRecordDataset(
        filenames
    )  # automatically interleaves reads from multiple files

    dataset = dataset.map(
        read_tfrecord, num_parallel_calls=AUTOTUNE
    )

    return dataset

# run numpy function in tensorflow
def tf_preprocess(input):
      y = tf.numpy_function(preprocess, [input], tf.float32)
      y = tf.reshape(y, [256,256,3])
      return y

@gin.configurable
def load(name, dataset_dir, BATCH_SIZE, tfrecord):
    if name == "idrid":
        logging.info(f"Preparing dataset {name}...")
        # use tfrecord file
        if tfrecord:
            train_dataset = load_dataset(os.path.join(dataset_dir, 'train_image.tfrecords'))
            valid_dataset = load_dataset(os.path.join(dataset_dir, 'valid_image.tfrecords'))
            test_dataset = load_dataset(os.path.join(dataset_dir, 'test_image.tfrecords'))

            train_dataset = train_dataset.map(lambda x, y: (tf_preprocess(x), y), num_parallel_calls=AUTOTUNE)

            '''# Visualized image from training data set
            image, label = next(iter(train_dataset))
            plt.imshow(tf.cast(image, tf.int64))
            plt.axis('off')
            plt.show()'''

            train_dataset = aug_and_prepare(train_dataset, BATCH_SIZE, shuffle=True, augment=True)

            valid_dataset = valid_dataset.map(lambda x, y: (tf_preprocess(x), y), num_parallel_calls=AUTOTUNE)
            valid_dataset = aug_and_prepare(valid_dataset, BATCH_SIZE)

            test_dataset = test_dataset.map(lambda x, y: (tf_preprocess(x), y), num_parallel_calls=AUTOTUNE)
            test_dataset = aug_and_prepare(test_dataset, BATCH_SIZE)

            return train_dataset, valid_dataset, test_dataset

        else:
            # use csv dataset
            train_df = pd.read_csv(os.path.join(dataset_dir, 'train_data.csv'))
            train_df['Image name'] = train_df['Image name'] + ".jpg"
            test_df = pd.read_csv(os.path.join(dataset_dir, 'test_data.csv'))
            test_df['Image name'] = test_df['Image name'] + ".jpg"

            train_datagen = ImageDataGenerator(rotation_range=360,
                                               horizontal_flip=True,
                                               vertical_flip=True,
                                               validation_split=0.15,
                                               preprocessing_function=preprocess,
                                               rescale = 1/255.
                                               )

            train_generator = train_datagen.flow_from_dataframe(train_df,
                                                    x_col='Image name',
                                                    y_col='Retinopathy grade',
                                                    directory = os.path.join(dataset_dir, 'train'),
                                                    target_size=(256, 256),
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='raw',
                                                    subset='training')

            val_generator = train_datagen.flow_from_dataframe(train_df,
                                                    x_col='Image name',
                                                    y_col='Retinopathy grade',
                                                    directory = os.path.join(dataset_dir, 'train'),
                                                    target_size=(256, 256),
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='raw',
                                                    shuffle=False,
                                                    subset='validation')

            test_generator = ImageDataGenerator(preprocessing_function=preprocess,
                                                rescale=1/255.).flow_from_dataframe(test_df,
                                                    x_col='Image name',
                                                    y_col='Retinopathy grade',
                                                    directory = os.path.join(dataset_dir, 'test'),
                                                    target_size=(256, 256),
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='raw',
                                                    shuffle=False)

            return train_generator, val_generator, test_generator