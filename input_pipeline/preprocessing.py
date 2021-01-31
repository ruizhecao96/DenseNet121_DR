import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def crop_image_from_gray(img, tol=7):
    '''crop image for smaller size'''
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol               # crop black part

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if (check_shape == 0):  # image is too dark so that we crop out everything,
            return img  # return original image
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)

        return img


def preprocess(image, image_size=256, sigmaX=10):
    # improve lighting condition, to see details better. Source: https://www.kaggle.com/ratthachat/aptos-eye-preprocessing-in-diabetic-retinopathy
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (image_size, image_size))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)

    return image


def aug_and_prepare(dataset, BATCH_SIZE=32, shuffle=False, augment=False):
    # use random rotation and flip to augment datasets
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    resize_and_rescale = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.Resizing(256, 256),
        tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)
    ])
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.5),
    ])

    dataset = dataset.map(lambda x, y: (resize_and_rescale(x), y), num_parallel_calls=AUTOTUNE).cache()
    if shuffle:
        dataset = dataset.shuffle(300)
        dataset = dataset.repeat()

    dataset = dataset.batch(BATCH_SIZE)
    if augment:
        dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)

    return dataset.prefetch(buffer_size=AUTOTUNE)


# show sample image of Ben's method
'''image = cv2.imread(r'E:\idrid\IDRID_dataset\train\IDRiD_005.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = preprocess(image)
plt.imshow(image)
plt.title("After crop and preprocessing")
plt.axis('off')
plt.show()'''
