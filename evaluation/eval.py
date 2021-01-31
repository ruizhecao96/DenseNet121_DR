import tensorflow as tf
import logging
import seaborn as sns
import matplotlib.pyplot as plt
from evaluation.metrics import ConfusionMatrix


def evaluate(model, ds_test):

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    test_AUC = tf.keras.metrics.AUC(name='test_AUC')
    cm = ConfusionMatrix(2)

    for image, label in ds_test:
        predictions = model(image, training=False)
        t_loss = loss_object(label, predictions)
        test_loss(t_loss)
        test_accuracy(label, predictions)
        cm.update_state(label, predictions)
        predictions = tf.math.argmax(predictions, axis=1)
        test_AUC(label, predictions)

    template = 'Test Loss: {}, Test Accuracy: {}, Confusion Matrix: {}, Test AUC: {}'
    logging.info(template.format(
                                test_loss.result(),
                                test_accuracy.result() * 100,
                                cm.result(),
                                test_AUC.result()
                                ))
    visualize_cm(cm.result())

    return test_accuracy.result().numpy()


def visualize_cm(cm):
    # Visualize Confusion Matrix
    sns.heatmap(cm, annot=True)
    plt.show()
