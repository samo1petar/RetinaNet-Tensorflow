import tensorflow as tf


class Activation(tf.keras.layers.Layer):
    def __init__(self, activation: str, **kwargs):
        super(Activation, self).__init__(**kwargs)
        self.activation_dict = {
            'relu'    : tf.keras.activations.relu,
            'elu'     : tf.keras.activations.elu,
            'linear'  : tf.keras.activations.linear,
            'softmax' : tf.keras.activations.softmax,
            'sigmoid' : tf.keras.activations.sigmoid,
        }
        assert activation in self.activation_dict
        self.activation = self.activation_dict[activation]

    def call(self, inputs):
        return self.activation(inputs)
