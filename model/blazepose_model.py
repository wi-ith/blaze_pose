"""blazepose model"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model import model_interface
from model.architecture import factory
from model import losses
import tensorflow as tf

class BlazePoseModel(model_interface.Model):

    def __init__(self, params):
        super(BlazePoseModel, self).__init__(params)

        # Generating architecture
        self._keras_model = None
        self._params = params
        self._backbone_fn = factory.backbone_generator(params)
        self._head_fn = factory.blazepose_head_generator(self._backbone_fn,params)

        # Loss function
        self._loss = losses.Loss(threshold=params['train']['threshold'],
                                 n_threshold=params['train']['n_threshold'],
                                 num_samples=params['train']['train_samples'],
                                 mode='train')
        self._eval = losses.Loss(threshold=params['train']['threshold'],
                                 n_threshold=params['train']['n_threshold'],
                                 num_samples=params['train']['train_samples'],
                                 mode='eval')

        self._use_bfloat16 = False
        self._optimizer = self.build_optimizer()


    def build_outputs(self, inputs, mode):
        is_training = mode == 'train'

        image = inputs['image']

        features = self._backbone_fn(image, training = is_training)
        if self._params['train']['load_backbone_weights'] !=None and self._params['train']['load_weight_path']==None:
            self._backbone_fn.load_weights(filepath=self._params['train']['load_backbone_weights'])
            print('load pre-trained backbone weights')

        for w in self._backbone_fn.weights:
            print(w.name)

        outputs = self._head_fn(features, training = is_training)
        # resized_logits = tf.image.resize(outputs[2], image.shape[1:3], name='upsample')
        output_dict={
            'heatmap_logit': outputs[0],
            'offset_y': outputs[1],
            'offset_x': outputs[2],
            'reg_logit': outputs[3]
        }
        return output_dict

    def build_loss_fn(self):
        if self._keras_model is None:
            raise ValueError('build_model() must be called first')
        trainable_variables = self._keras_model.trainable_variables
        def _total_loss(outputs, heatmaps, offsetmaps, regression):
            model_loss = self._loss.calc_loss(outputs, heatmaps, offsetmaps, regression)
            l2_regularization_loss = self.weight_decay_loss(trainable_variables)
            total_loss = model_loss['edl_loss'] +\
                         model_loss['x_loss'] +\
                         model_loss['y_loss'] +\
                         l2_regularization_loss
            return{
                'edl_loss' : model_loss['edl_loss'],
                'x_loss': model_loss['x_loss'],
                'y_loss': model_loss['y_loss'],
                'reg_loss' : model_loss['reg_loss'],
                'total_loss' : total_loss,
                'l2_regularization_loss' : l2_regularization_loss
            }
        return _total_loss

    def build_eval_fn(self):
        def eval_fn(batch_output, label):
            return self._eval.calc_loss(batch_output, label)
        return eval_fn

    def build_model(self, params, mode=None):
        if self._keras_model is None:
            input_layer = self.build_input_layers(params, mode = mode)
            outputs = self.model_outputs(input_layer, mode)
            model = tf.keras.Model(inputs = input_layer,outputs = outputs)
            model.optimizer = self.build_optimizer()
            self._keras_model = model

            # print_model_summary(model)

        return self._keras_model


    def build_input_layers(self, params, mode):
        is_training = mode == 'train'
        input_shape = [
            params['architecture']['input_size']['width'],
            params['architecture']['input_size']['height'],
            3
        ]

        if is_training:
            batch_size = params['train']['batch_size']
            input_layer = {
                'image':
                    tf.keras.layers.Input(
                        shape=input_shape,
                        batch_size=batch_size,
                        name='train_image',
                        dtype=tf.bfloat16 if self._use_bfloat16 else tf.float32)
            }
        else:
            batch_size = params['eval']['batch_size']
            input_layer = {
                'image':
                    tf.keras.layers.Input(
                        shape=input_shape,
                        batch_size=batch_size,
                        name='eval_image',
                        dtype=tf.bfloat16 if self._use_bfloat16 else tf.float32),
            }
        return input_layer