import tensorflow as tf


class DeConv(tf.keras.layers.Layer):
    def __init__(self, filters: int, kernel: int, stride : int = 1, padding : str ='same', **kwargs):
        super(DeConv, self).__init__(**kwargs)
        self.filters = filters
        self.kernel  = kernel
        self.stride  = stride
        self.padding = padding

    def build(self, shape):
        self.deconv = tf.keras.layers.Conv2DTranspose(
            filters     = self.filters,
            kernel_size = self.kernel,
            strides     = self.stride,
            padding     = self.padding,
        )

    def call(self, inputs, training: bool = True):
        return self.deconv(inputs, training=training)
