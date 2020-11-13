import tensorflow as tf
from lib.layers.Activation import Activation


class ConvBnAct(tf.keras.layers.Layer):
    def __init__(self, filters: int, kernel: int, stride: int = 1, activation: str = 'relu', **kwargs):
        super(ConvBnAct, self).__init__(**kwargs)
        self.filters = filters
        self.kernel = kernel
        self.stride = stride
        self.activation_name = activation

    def build(self, shape):
        self.conv = tf.keras.layers.Conv2D(
            filters     = self.filters,
            kernel_size = self.kernel,
            strides     = self.stride,
            padding     = 'same',
        )
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.activation = Activation(self.activation_name)

    def call(self, inputs, training: bool = False):
        x = self.conv(inputs, training=training)
        x = self.batch_norm(x, training=training)
        x = self.activation(x)
        return x
