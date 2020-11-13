import tensorflow as tf


class GlobalMaxPool(tf.keras.layers.Layer):
    def __init__(
            self,
            **kwargs
    ):
        super(GlobalMaxPool, self).__init__(**kwargs)

    def build(self, shape):
        self.global_max_pool = tf.keras.layers.GlobalMaxPool2D()

    def call(self, inputs):
        return self.global_max_pool(inputs)
