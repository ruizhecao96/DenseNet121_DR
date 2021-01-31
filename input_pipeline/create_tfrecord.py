import os
import tensorflow as tf
import pandas as pd


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def image_example(image_string, label):
    image_shape = tf.image.decode_jpeg(image_string).shape
    feature = {
      'height': _int64_feature(image_shape[0]),
      'width': _int64_feature(image_shape[1]),
      'depth': _int64_feature(image_shape[2]),
      'label': _int64_feature(label),
      'image_raw': _bytes_feature(image_string),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))

def tfrecord(train_df_total, test_df, dataset_dir):
    train_df = train_df_total.iloc[:351, :]        # train dataset 351 images, factor by 0.15
    valid_df = train_df_total.iloc[351:, :]        # valid dataset 62 images



    with tf.io.TFRecordWriter(os.path.join(dataset_dir, "train_image.tfrecords")) as writer:
        for index, row in train_df.iterrows():
            img_string = open(os.path.join(dataset_dir, 'train', row['Image name']+'.jpg'), 'rb').read()
            example = image_example(img_string, row['Retinopathy grade'])
            writer.write(example.SerializeToString())

    with tf.io.TFRecordWriter(os.path.join(dataset_dir, "valid_image.tfrecords")) as writer:
        for index, row in valid_df.iterrows():
            img_string = open(os.path.join(dataset_dir, 'train', row['Image name']+'.jpg'), 'rb').read()
            example = image_example(img_string, row['Retinopathy grade'])
            writer.write(example.SerializeToString())


    with tf.io.TFRecordWriter(os.path.join(dataset_dir, "test_image.tfrecords")) as writer:
        for index, row in test_df.iterrows():
            img_string = open(os.path.join(dataset_dir, 'test', row['Image name']+'.jpg'), 'rb').read()
            example = image_example(img_string, row['Retinopathy grade'])
            writer.write(example.SerializeToString())
