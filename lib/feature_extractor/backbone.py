import tensorflow as tf
from lib.feature_extractor.basic_feature_extractor import BasicFE


def get_backbone():
    """Builds ResNet50 with pre-trained imagenet weights"""
    backbone = tf.keras.applications.ResNet50(
        include_top=False, input_shape=[None, None, 3]
    )
    c3_output, c4_output, c5_output = [
        backbone.get_layer(layer_name).output
        for layer_name in ["conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]
    ]
    return tf.keras.Model(
        inputs=[backbone.inputs], outputs=[c3_output, c4_output, c5_output]
    )

def get_backbone_MobileNet_v2():
    """Builds MobileNetv2 with pre-trained imagenet weights"""
    backbone = tf.keras.applications.MobileNetV2(
        include_top=False, input_shape=[None, None, 3]
    )

    block_5_add, block_11_project_BN, out_relu = [
        backbone.get_layer(layer_name).output
        for layer_name in ["block_5_add", "block_11_project_BN", "out_relu"]
    ]

    return tf.keras.Model(
        inputs=[backbone.inputs], outputs=[block_5_add, block_11_project_BN, out_relu]
    )

def get_backbone_conv_small():
    basic_fe = BasicFE(name='basic_feature_extractor')

    inputs = tf.keras.Input(shape=[None, None, 3], name="inputs")

    x1, x2, x3 = basic_fe(inputs)

    return tf.keras.Model(
        inputs=[basic_fe.input], outputs=[x1, x2, x3]
    )