import tensorflow as tf


class GlobalAvgPool(tf.keras.layers.Layer):
    def __init__(
            self,
            **kwargs
    ):
        super(GlobalAvgPool, self).__init__(**kwargs)

    def build(self, shape):
        self.global_avg_pool = tf.keras.layers.GlobalAvgPool2D()

    def call(self, inputs):
        return self.global_avg_pool(inputs)
