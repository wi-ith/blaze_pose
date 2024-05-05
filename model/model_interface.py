from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
from model import optimizers
from model import learning_rates
import tensorflow as tf

class Model(tf.keras.Model):

    __metaclass__ = abc.ABCMeta

    def __init__(self, params):
        super(Model, self).__init__()
        self._optimizer_fn = optimizers.OptimizerFactory(params)
        self._learning_rate = learning_rates.learning_rate_generator(params)
        self._l2_weight_decay = params['train']['l2_weight_decay']

    @abc.abstractmethod
    def build_outputs(self, inputs, mode):
        """Build the graph of the forward path."""
        pass

    @abc.abstractmethod
    def build_model(self, params, mode):
        """Build the model object."""
        pass

    @abc.abstractmethod
    def build_loss_fn(self):
        """Build the model object."""
        pass

    def post_processing(self, labels, outputs):
        """Post-processing function."""
        return labels, outputs

    def model_outputs(self, inputs, mode):
        """Build the model outputs."""
        return self.build_outputs(inputs, mode)

    def build_optimizer(self):
        """Sets up the optimizer."""
        return self._optimizer_fn

    def weight_decay_loss(self, trainable_variables):
        reg_variables = [
            v for v in trainable_variables
        ]

        return self._l2_weight_decay * tf.add_n(
            [tf.nn.l2_loss(v) for v in reg_variables])