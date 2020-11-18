import tensorflow as tf


def build_head(output_filters, bias_init, channels=8, depth=2):
    """Builds the class/box predictions head.

    Arguments:
      output_filters: Number of convolution filters in the final layer.
      bias_init: Bias Initializer for the final convolution layer.
      channels: Number of channels in conv layers
      depth: Number of conv layers attached to the head

    Returns:
      A keras sequential model representing either the classification
        or the box regression head depending on `output_filters`.
    """
    head = tf.keras.Sequential([tf.keras.Input(shape=[None, None, channels])])
    kernel_init = tf.initializers.RandomNormal(0.0, 0.01)
    for _ in range(depth):
        head.add(
            tf.keras.layers.Conv2D(channels, 3, padding="same", kernel_initializer=kernel_init)
        )
        head.add(tf.keras.layers.ReLU())
    head.add(
        tf.keras.layers.Conv2D(
            output_filters,
            3,
            1,
            padding="same",
            kernel_initializer=kernel_init,
            bias_initializer=bias_init,
        )
    )
    return head
