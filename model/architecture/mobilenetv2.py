"""Mobilenet V2 backbone"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .nn_ops import *

class MobilenetV2(tf.keras.Model):
    def __init__(self, input_dims):
        super(MobilenetV2, self).__init__()
        self._name=''
        self.input_dims=input_dims
        output_h = input_dims[1] // 2 + input_dims[1] % 2
        output_w = input_dims[2] // 2 + input_dims[2] % 2
        self.conv1_output_dims = [input_dims[0],output_h,output_w, 32]
        self.conv_ = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), use_bias=False,
                                            padding='same', dilation_rate=(1, 1), name='Conv')
        self.conv_batchnorm = tf.keras.layers.BatchNormalization(momentum=0.9997, epsilon=1e-3, center=True,
                                                                  scale=True, name='BatchNorm')#128
        self.relu6 = tf.nn.relu6
        self.inverted_residual0 = inverted_residual(self.conv1_output_dims, 1, 1, 16, 0, 0) #layer1
        self.inverted_residual1 = inverted_residual(self.inverted_residual0.output_dims, 6, 1, 24, 1 ,1)#64
        self.inverted_residual2 = inverted_residual(self.inverted_residual1.output_dims, 6, 1, 24, 0, 2) #layer2
        self.inverted_residual3 = inverted_residual(self.inverted_residual2.output_dims, 6, 1, 32, 1, 3)#32
        self.inverted_residual4 = inverted_residual(self.inverted_residual3.output_dims, 6, 1, 32, 0, 4)
        self.inverted_residual5 = inverted_residual(self.inverted_residual4.output_dims, 6, 1, 32, 0, 5) #layer3
        self.inverted_residual6 = inverted_residual(self.inverted_residual5.output_dims, 6, 1, 64, 1, 6)#16
        self.inverted_residual7 = inverted_residual(self.inverted_residual6.output_dims, 6, 1, 64, 0, 7)
        self.inverted_residual8 = inverted_residual(self.inverted_residual7.output_dims, 6, 1, 64, 0, 8)
        self.inverted_residual9 = inverted_residual(self.inverted_residual8.output_dims, 6, 1, 64, 0, 9) #layer4
        self.inverted_residual10 = inverted_residual(self.inverted_residual9.output_dims, 6, 1, 96, 0, 10)
        self.inverted_residual11 = inverted_residual(self.inverted_residual10.output_dims, 6, 1, 96, 0, 11)
        self.inverted_residual12 = inverted_residual(self.inverted_residual11.output_dims, 6, 1, 96, 0, 12)
        self.inverted_residual13 = inverted_residual(self.inverted_residual12.output_dims, 6, 1, 160, 1, 13)#8
        self.inverted_residual14 = inverted_residual(self.inverted_residual13.output_dims, 6, 1, 160, 0, 14)
        self.inverted_residual15 = inverted_residual(self.inverted_residual14.output_dims, 6, 1, 160, 0, 15) #layer5
        self.inverted_residual16 = inverted_residual(self.inverted_residual15.output_dims, 6, 1, 320, 0, 16) #final
        self.final_dims = self.inverted_residual16.output_dims

    def call(self, inputs, training=None, mask=None):
        with tf.name_scope('MobilenetV2'):
            output = self.conv_(inputs, training=training)
            with tf.name_scope('Conv'):
                output = self.conv_batchnorm(output, training=training)
                output = self.relu6(output)
            _, _, output = self.inverted_residual0(output, training=training)
            layer0, _, output = self.inverted_residual1(output, training=training)
            _, _, output = self.inverted_residual2(output, training=training)
            layer1, _, output = self.inverted_residual3(output, training=training)
            _, _, output = self.inverted_residual4(output, training=training)
            _, _, output = self.inverted_residual5(output, training=training)
            layer2, _, output = self.inverted_residual6(output, training=training)
            _, _, output = self.inverted_residual7(output, training=training)
            _, _, output = self.inverted_residual8(output, training=training)
            _, _, output = self.inverted_residual9(output, training=training)
            _, _, output = self.inverted_residual10(output, training=training)
            _, _, output = self.inverted_residual11(output, training=training)
            _, _, output = self.inverted_residual12(output, training=training)
            layer3, _, output = self.inverted_residual13(output, training=training)
            _, _, output = self.inverted_residual14(output, training=training)
            _, _, output = self.inverted_residual15(output, training=training)
            _, _, final = self.inverted_residual16(output, training=training)
        return layer0, layer1, layer2, layer3, final