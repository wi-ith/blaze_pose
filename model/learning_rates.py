"""Learning rate schedule."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class ExponentialDecay:
    def __init__(self, params):
        self.lr = params['train']['learning_rate']['init_learning_rate']
        self.num_train = params['train']['train_samples']
        self.batch_size = params['train']['batch_size']
        self.lr_decay_epochs = params['train']['learning_rate']['learning_rate_decay_epochs']

    def __call__(self, *args, **kwargs):
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=self.lr,
                                                             decay_steps=int((self.num_train/self.batch_size )*self.lr_decay_epochs),
                                                             decay_rate=self.lr_decay_epochs)
        return lr_schedule

def learning_rate_generator(params):
    if params['train']['learning_rate']['type'] == 'exponential_decay':
        return ExponentialDecay(params)
    else:
        raise ValueError('Invalid learning rate type : {}'.format(params['train']['learning_rate']['type']))