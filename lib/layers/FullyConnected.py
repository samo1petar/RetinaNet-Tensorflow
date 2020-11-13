import tensorflow as tf


class FullyConnected(tf.keras.layers.Layer):
    def __init__(self, units : int, **kwargs):
        super(FullyConnected, self).__init__(**kwargs)
        self.units = units

    def build(self, shape):
        self.dense = tf.keras.layers.Dense(units=self.units)

    def call(self, inputs, training: bool = True):
        return self.dense(inputs, training=training)
