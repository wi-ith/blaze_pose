""" Operations for Neural Netwrok """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class inverted_residual(tf.keras.layers.Layer):
    def __init__(self, input_shape, up_sample_rate, atrous_rate, channels, subsample, index_, prefix=''):
        super(inverted_residual, self).__init__()
        # self.i += 1
        stride = 2 if subsample else 1
        self.up_sample_rate = up_sample_rate
        self.atrous_rate = atrous_rate
        self.channels = channels
        self.subsample = subsample
        self.stride = stride
        output_h = input_shape[1] // stride + input_shape[1] % stride
        output_w = input_shape[2] // stride + input_shape[2] % stride
        self.output_dims = [input_shape[0],output_h,output_w, channels]
        self.relu6 = tf.nn.relu6
        self.index_=index_
        if index_ == 0:
            self._name = prefix + 'expanded_conv'
        else:
            self._name = prefix + 'expanded_conv_{}'.format(index_)
        # self._name=''

        if up_sample_rate > 1:
            self.expand = tf.keras.layers.Conv2D(filters=up_sample_rate * input_shape[-1],
                                               kernel_size=1, use_bias=False,
                                               padding='same', dilation_rate=(1, 1), name='expand')
            self.expand_batchnorm = tf.keras.layers.BatchNormalization(momentum=0.9997, epsilon=1e-3, center=True,
                                                                     scale=True, name='BatchNorm')

        self.depthwise = tf.keras.layers.DepthwiseConv2D(kernel_size=3,use_bias=False, strides=(stride, stride),
                                                     padding='same', dilation_rate=(atrous_rate, atrous_rate), name='depthwise')
        self.depthwise_batchnorm = tf.keras.layers.BatchNormalization(momentum=0.9997, epsilon=1e-3, center=True,
                                                                  scale=True, name='BatchNorm')
        self.pointwise = tf.keras.layers.Conv2D(filters=channels, kernel_size=1, use_bias=False,
                                            padding='same', dilation_rate=(1, 1), name='project')
        self.pointwise_batchnorm = tf.keras.layers.BatchNormalization(momentum=0.9997, epsilon=1e-3, center=True,
                                                                  scale=True, name='BatchNorm')
    # def call(self, inputs, training=None, *args, **kwargs):
    def __call__(self, input,training=None):

        with tf.name_scope(self._name):
            if self.up_sample_rate > 1:
                expand_ = self.expand(input,training=training)
                with tf.name_scope('expand'):
                    expand_ = self.expand_batchnorm(expand_,training=training)
                    expand_ = self.relu6(expand_)
            else :
                expand_ = input
            depthwise_ = self.depthwise(expand_,training=training)
            with tf.name_scope('depthwise'):
                depthwise_ = self.depthwise_batchnorm(depthwise_,training=training)
                depthwise_ = self.relu6(depthwise_)
            project_ = self.pointwise(depthwise_,training=training)
            with tf.name_scope('project'):
                project_ =  self.pointwise_batchnorm(project_,training=training)
            if input.shape[-1] == self.channels:
                project_ =tf.keras.layers.add([input,project_])
        return expand_, depthwise_, project_


class CBR(tf.keras.layers.Layer):
    def __init__(self,input_dims, filters, kernel_size, use_bias, padding, dilation_rate, layer_name):
        super(CBR, self).__init__()
        self.input_dims = input_dims
        self.output_dims = [input_dims[0], input_dims[1], input_dims[2], filters]
        self._name = layer_name
        self.relu = tf.nn.relu
        self.conv2d = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, use_bias=use_bias,
                                            padding=padding, dilation_rate=dilation_rate, name=layer_name)
        self.batch_norm = tf.keras.layers.BatchNormalization(momentum=0.9997, epsilon=1e-3, center=True,
                                                                  scale=True, name='BatchNorm')
    def __call__(self, input, relu=True ,training=None, *args, **kwargs):
        output = self.conv2d(input)
        with tf.name_scope(self._name):
            output = self.batch_norm(output)
            if relu:
                output = self.relu(output)
        return output


class DecodingBlock(tf.keras.layers.Layer):
    def __init__(self,input_shape, atrous_rate, channels, upsample, index_, use_batchnorm=True):
        super(DecodingBlock,self).__init__()
        # self.i += 1
        if index_==0:
            self._name = 'decoder_layer'
        else:
            self._name = 'decoder_layer_{}'.format(str(index_))
        self.atrous_rate = atrous_rate
        self.channels = channels
        self.mid_channels = input_shape[3]*3
        self.upsample = upsample[1:3] if upsample!=None else upsample
        # if upsample:
        #     self.resize_shape = (input_shape[1]*2, input_shape[2]*2)
        # else:
        #     self.resize_shape = input_shape[1:3]
        if upsample == None:
            self.output_dims = input_shape
        else:
            self.output_dims = upsample
        self.relu6 = tf.nn.relu6
        # self.index_ = index_
        self.use_batchnorm = use_batchnorm


        self.expand = tf.keras.layers.Conv2D(filters=self.mid_channels,
                                             kernel_size=1, use_bias=False,
                                             padding='same', dilation_rate=(1, 1), name='expand')
        self.expand_batchnorm = tf.keras.layers.BatchNormalization(momentum=0.9997, epsilon=1e-3, center=True,
                                                                   scale=True, name='BatchNorm')

        self.depthwise = tf.keras.layers.DepthwiseConv2D(kernel_size=5, use_bias=False, strides=(1, 1),
                                                         padding='same', dilation_rate=(atrous_rate, atrous_rate),
                                                         name='depthwise')

        self.depthwise_batchnorm = tf.keras.layers.BatchNormalization(momentum=0.9997, epsilon=1e-3, center=True,
                                                                      scale=True, name='BatchNorm')

        self.pointwise = tf.keras.layers.Conv2D(filters=channels, kernel_size=1, use_bias=False,
                                                padding='same', dilation_rate=(1, 1), name='project')

        self.pointwise_batchnorm = tf.keras.layers.BatchNormalization(momentum=0.9997, epsilon=1e-3, center=True,
                                                                      scale=True, name='BatchNorm')

    def __call__(self,input, relu=True ,training=None, *args, **kwargs):


        with tf.name_scope(self._name):
            expand_ = self.expand(input,training=training)
            with tf.name_scope('expand'):
                expand_ = self.expand_batchnorm(expand_,training=training)
                expand_ = self.relu6(expand_)
                if self.upsample != None:
                    expand_ = tf.image.resize(expand_,self.upsample,method='nearest')
            depthwise_ = self.depthwise(expand_,training=training)
            with tf.name_scope('depthwise'):
                depthwise_ = self.depthwise_batchnorm(depthwise_,training=training)
                depthwise_ = self.relu6(depthwise_)
            project_ = self.pointwise(depthwise_,training=training)
            with tf.name_scope('project'):
                if self.use_batchnorm:
                    project_ =  self.pointwise_batchnorm(project_,training=training)
            if input.shape[-1] == self.channels and self.upsample == None:
                project_ =tf.keras.layers.add([input,project_])
        return expand_, depthwise_, project_

class ChannelPadding(tf.keras.layers.Layer):
    def __init__(self, channels):
        super(ChannelPadding, self).__init__()
        self.channels = channels

    def build(self, input_shapes):
        self.pad_shape = tf.constant([[0, 0], [0, 0], [0, 0], [0, self.channels - input_shapes[-1]]])

    def call(self, input):
        return tf.pad(input, self.pad_shape)


class BlazeBlock(tf.keras.layers.Layer):
    def __init__(self, block_num = 3, channel = 48, channel_padding = 1,index=0):
        super(BlazeBlock, self).__init__()
        self._name = 'BlazeBlock_' + index
        self.downsample_a = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=(2, 2), padding='same', activation=None),
            tf.keras.layers.Conv2D(filters=channel, kernel_size=1, activation=None)
        ])
        if channel_padding:
            self.downsample_b = tf.keras.models.Sequential([
                tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
                ChannelPadding(channels=channel)
            ])
        else:
            self.downsample_b = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        self.conv = list()
        for i in range(block_num):
            self.conv.append(tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding='same', activation=None),
            tf.keras.layers.Conv2D(filters=channel, kernel_size=1, activation=None)
        ]))

    def __call__(self, x, training=None):
        with tf.name_scope(self._name):
            x = tf.keras.activations.relu(self.downsample_a(x) + self.downsample_b(x))
            for i in range(len(self.conv)):
                x = tf.keras.activations.relu(x + self.conv[i](x))
            return x