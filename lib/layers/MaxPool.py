import tensorflow as tf


class MaxPool(tf.keras.layers.Layer):
    def __init__(
            self,
            kernel  : int = 2,
            stride  : int = 2,
            padding : str = 'same',
            **kwargs
    ):
        super(MaxPool, self).__init__(**kwargs)

        self.kernel  = kernel
        self.stride  = stride
        self.padding = padding

    def build(self, shape):
        self.max_pool = tf.keras.layers.MaxPooling2D(
            pool_size = self.kernel,
            strides   = self.stride,
            padding   = self.padding,
        )

    def call(self, inputs):
        return self.max_pool(inputs)
