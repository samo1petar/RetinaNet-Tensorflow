import tensorflow as tf
from lib.layers import Conv, ConvBnAct, FullyConnected, GlobalAvgPool, MaxPool


class BasicFE(tf.keras.layers.Layer):
    def __init__(
            self,
            name : str,
            M : int = 1,
    ):
        super(BasicFE, self).__init__(name=name)

        self.c1 = ConvBnAct(8 * M, 3, stride = 2, name='conv_1') # 192 x 256
        self.c2 = ConvBnAct(8 * M, 3, name='conv_2')
        self.p1 = MaxPool(name='pool_1') # 96 x 128
        self.c3 = ConvBnAct(16 * M, 3, name='conv_3')
        self.c4 = ConvBnAct(16 * M, 3, name='conv_4')
        self.p2 = MaxPool(name='pool_2') # 48 x 64
        self.c5 = ConvBnAct(32 * M, 3, name='conv_5')
        self.c6 = ConvBnAct(32 * M, 3, name='conv_6')
        self.p3 = MaxPool(name='pool_3') # 24 x 32
        self.c7 = ConvBnAct(64, 3, name='conv_7')
        self.c8 = ConvBnAct(64, 3, name='conv_8')
        self.p4 = MaxPool(name='pool_4') # 12 x 16
        self.c9 = ConvBnAct(128, 3, name='conv_9')
        self.c10= ConvBnAct(128, 3, name='conv_10')

    def call(self, inputs: tf.Tensor, training: bool = False):

        x = inputs

        x = self.c1(x, training=training)
        x = self.c2(x, training=training)
        x = self.p1(x)
        x = self.c3(x, training=training)
        x = self.c4(x, training=training)
        x = self.p2(x)
        x = self.c5(x, training=training)
        x1 = self.c6(x, training=training)
        x = self.p3(x1)
        x = self.c7(x, training=training)
        x2 = self.c8(x, training=training)
        x = self.p4(x2)
        x = self.c9(x, training=training)
        x3 = self.c10(x, training=training)

        return x1, x2, x3
