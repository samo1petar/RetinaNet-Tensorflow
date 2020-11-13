import tensorflow as tf
from lib.layers.Activation import Activation


class DeConvBnAct(tf.keras.layers.Layer):
    def __init__(self, filters: int, kernel: int, stride: int = 2, activation: str = 'relu', **kwargs):
        super(DeConvBnAct, self).__init__(**kwargs)
        self.filters = filters
        self.kernel = kernel
        self.stride = stride
        self.activation_name = activation

    def build(self, shape):
        self.deconv = tf.keras.layers.Conv2DTranspose(
            filters     = self.filters,
            kernel_size = self.kernel,
            strides     = self.stride,
            padding     = 'same',
        )
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.activation = Activation(self.activation_name)

    def call(self, inputs, training: bool = False):
        x = self.deconv(inputs)
        x = self.batch_norm(x, training=training)
        x = self.activation(x)
        return x
