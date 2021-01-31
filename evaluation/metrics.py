import tensorflow as tf


class ConfusionMatrix(tf.keras.metrics.Metric):

    def __init__(self, num_classes, **kwargs):
        super(ConfusionMatrix, self).__init__(name="confusion_matrix", **kwargs)
        self.num_classes = num_classes
        self.total_cm = self.add_weight("total", shape=(num_classes, num_classes), initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None, **kwargs):
        # convert predictions from probability to boolean
        y_pred = tf.math.argmax(y_pred, axis=1)
        # y_true = tf.cast(y_true, tf.bool)
        # apply confusion matrix
        cm = tf.math.confusion_matrix(y_true, y_pred, dtype=tf.float32, num_classes=self.num_classes)
        self.total_cm.assign_add(cm)

    def reset_states(self):
        for s in self.variables:
            s.assign(tf.zeros(shape=s.shape))

    def result(self):
        return self.total_cm
