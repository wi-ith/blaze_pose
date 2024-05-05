""" head generators """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .nn_ops import *

class blazepose(tf.keras.Model):
    def __init__(self, backbone, params):

        super(blazepose, self).__init__()
        self.backbone = backbone
        self.params = params
        self.num_key = self.params['architecture']['input_size']['num_keypoints']
        self.type_ = self.params['train']['type']

        # self.final_conv2d = CBR(input_dims=self.backbone.final_dims, filters=1024, kernel_size=1, use_bias=False,
        #                         padding='same', dilation_rate=(1, 1), layer_name='final_conv')

        self.final_conv2d = inverted_residual(self.backbone.final_dims,
                                              up_sample_rate = 6,
                                              atrous_rate = 1,
                                              channels = 192,
                                              subsample = 0,
                                              index_ = 0,
                                              prefix = 'final_')


        self.decoder_layer_1 = DecodingBlock(self.final_conv2d.output_dims, 1, 48,
                                             self.backbone.inverted_residual6.output_dims, 1)

        self.inverted_layer_1 = inverted_residual(self.final_conv2d.output_dims,
                                              up_sample_rate = 6,
                                              atrous_rate = 1,
                                              channels = 48,
                                              subsample = 0,
                                              index_ = 1,
                                              prefix = 'head')

        self.decoder_layer_2 = DecodingBlock(self.decoder_layer_1.output_dims, 1, 48,
                                             self.backbone.inverted_residual3.output_dims, 2)

        self.inverted_layer_2 = inverted_residual(self.decoder_layer_1.output_dims,
                                              up_sample_rate = 6,
                                              atrous_rate = 1,
                                              channels = 48,
                                              subsample = 0,
                                              index_ = 2,
                                              prefix = 'head')

        self.decoder_layer_3 = DecodingBlock(self.decoder_layer_2.output_dims, 1, 48,
                                             self.backbone.inverted_residual1.output_dims, 3)

        self.inverted_layer_3 = inverted_residual(self.decoder_layer_2.output_dims,
                                              up_sample_rate = 6,
                                              atrous_rate = 1,
                                              channels = 48,
                                              subsample = 0,
                                              index_ = 3,
                                              prefix = 'head')

        # self.decoder_layer_4 = DecodingBlock(self.decoder_layer_3.output_dims, 1, 32,
        #                                      self.backbone.conv1_output_dims, 3)
        #
        # self.inverted_layer_4 = inverted_residual(self.decoder_layer_3.output_dims,
        #                                       up_sample_rate = 6,
        #                                       atrous_rate = 1,
        #                                       channels = 32,
        #                                       subsample = 0,
        #                                       index_ = 4,
        #                                       prefix = 'head')

        self.heatmap_pred = tf.keras.layers.Conv2D(filters=self.num_key, kernel_size=1, use_bias=False,
                                                 padding='same', dilation_rate=(1, 1), name='heatmap_pred')

        self.offset_y_pred = tf.keras.layers.Conv2D(filters=self.num_key, kernel_size=1, use_bias=False,
                                                    padding='same', dilation_rate=(1, 1), name='offset_y_pred')

        self.offset_x_pred = tf.keras.layers.Conv2D(filters=self.num_key, kernel_size=1, use_bias=False,
                                                    padding='same', dilation_rate=(1, 1), name='offset_x_pred')

        self.conv12a = BlazeBlock(block_num = 4, channel = 96,index='12a')    # input res: 64
        self.conv12b = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="same", activation=None),
            tf.keras.layers.Conv2D(filters=96, kernel_size=1, activation="relu")
        ])

        self.conv13a = BlazeBlock(block_num = 5, channel = 192,index='13a')   # input res: 32
        self.conv13b = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="same", activation=None),
            tf.keras.layers.Conv2D(filters=192, kernel_size=1, activation="relu")
        ])

        self.conv14a = BlazeBlock(block_num = 6, channel = 288,index='14a')   # input res: 16
        self.conv14b = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="same", activation=None),
            tf.keras.layers.Conv2D(filters=288, kernel_size=1, activation="relu")
        ])

        self.conv15 = tf.keras.models.Sequential([
            BlazeBlock(block_num = 7, channel = 288, channel_padding = 0,index = '15_1'),
            BlazeBlock(block_num = 7, channel = 288, channel_padding = 0,index = '15_2')
        ])

        self.conv16 = tf.keras.models.Sequential([
            tf.keras.layers.GlobalAveragePooling2D(),
            # shape = (1, 1, 1, 288)
            tf.keras.layers.Dense(units=3*self.num_key, activation=None),
            tf.keras.layers.Reshape((self.num_key, 3))
        ])

    def call(self, end_points, training=None, mask=None):
        # outputs_size = tf.shape(inputs)[1:3]
        layer_0, layer_1, layer_2, layer_3, final = end_points
        end_point_features_final = final

        _, _, output = self.final_conv2d(end_point_features_final) #8

        _, _, output = self.decoder_layer_1(output)
        _, _, layer_3_h = self.inverted_layer_1(layer_3)
        output = tf.keras.layers.add([layer_3_h, output]) #16

        _, _, output = self.decoder_layer_2(output)
        _, _, layer_2_h = self.inverted_layer_2(layer_2)
        output = tf.keras.layers.add([layer_2_h, output])#32

        _, _, output = self.decoder_layer_3(output)
        _, _, layer_1_h = self.inverted_layer_3(layer_1)
        output_last = tf.keras.layers.add([layer_1_h, output])#64

        regression_input = tf.identity(output_last)

        # _, _, output_last = self.decoder_layer_4(output)
        # _, _, layer_0_h = self.inverted_layer_4(layer_0)
        # output_last = tf.keras.layers.add([layer_0_h, output_last])#128

        # _, _, output = self.decoder_layer_4(output)
        # output = tf.keras.layers.add([layer_2, output])

        # _, _, output = self.decoder_layer_5(output)
        # output = tf.keras.layers.add([layer_1, output])

        heatmap_logit = tf.keras.activations.sigmoid(self.heatmap_pred(output_last))

        offset_y_logit = tf.keras.activations.sigmoid(self.offset_y_pred(output_last)) - 0.5

        offset_x_logit = tf.keras.activations.sigmoid(self.offset_x_pred(output_last)) - 0.5


        ## regression branch
        if self.type_ == "regression":
            regression_input = tf.keras.backend.stop_gradient(regression_input)
            layer_2 = tf.keras.backend.stop_gradient(layer_2)
            layer_3 = tf.keras.backend.stop_gradient(layer_3)
            end_point_features_final = tf.keras.backend.stop_gradient(end_point_features_final)



        reg_output = self.conv12a(regression_input) + self.conv12b(layer_2)

        reg_output = self.conv13a(reg_output) + self.conv13b(layer_3)

        reg_output = self.conv14a(reg_output) + self.conv14b(end_point_features_final)

        reg_output = self.conv15(reg_output)

        joints = self.conv16(reg_output)

        return heatmap_logit, offset_y_logit, offset_x_logit, joints