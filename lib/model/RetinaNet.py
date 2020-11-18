import numpy as np
import tensorflow as tf

from lib.model.FeaturePyramid import FeaturePyramid
from lib.model.heads import build_head


class RetinaNet(tf.keras.Model):
    """A subclassed Keras model implementing the RetinaNet architecture.

    Attributes:
      num_classes: Number of classes in the dataset.
      backbone: The backbone to build the feature pyramid from.
    """

    def __init__(
            self,
            num_classes,
            backbone,
            feature_pyramid_channels,
            head_channels,
            head_depth,
            **kwargs
    ):
        super(RetinaNet, self).__init__(name="RetinaNet", **kwargs)
        self.fpn = FeaturePyramid(backbone, channels=feature_pyramid_channels)
        self.num_classes = num_classes

        prior_probability = tf.constant_initializer(-np.log((1 - 0.01) / 0.01))
        self.cls_head = build_head(9 * num_classes, prior_probability, channels=head_channels, depth=head_depth)
        self.box_head = build_head(9 * 4, "zeros", channels=head_channels, depth=head_depth)

    def call(self, image, training=False):
        features = self.fpn(image, training=training)
        N = tf.shape(image)[0]
        cls_outputs = []
        box_outputs = []

        for feature in features:
            box_outputs.append(tf.reshape(self.box_head(feature), [N, -1, 4]))
            cls_outputs.append(
                tf.reshape(self.cls_head(feature), [N, -1, self.num_classes])
            )
        cls_outputs = tf.concat(cls_outputs, axis=1)
        box_outputs = tf.concat(box_outputs, axis=1)
        return tf.concat([box_outputs, cls_outputs], axis=-1)