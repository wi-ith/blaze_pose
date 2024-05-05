"""Optimizers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import tensorflow as tf


class OptimizerFactory(object):
  """Class to generate optimizer function."""

  def __init__(self, params):
    """Creates optimized based on the specified flags."""

    if params['train']['optimizer']['type'] == 'momentum':
      self._optimizer = functools.partial(
          tf.keras.optimizers.SGD,
          momentum=params['train']['optimizer']['momentum'],
          nesterov=params['train']['optimizer']['nesterov'])
    elif params['train']['optimizer']['type'] == 'adam':
      self._optimizer = tf.optimizers.Adam
    elif params['train']['optimizer']['type'] == 'adadelta':
      self._optimizer = tf.optimizers.Adadelta
    elif params['train']['optimizer']['type'] == 'adagrad':
      self._optimizer = tf.optimizers.Adagrad
    elif params['train']['optimizer']['type'] == 'rmsprop':
      self._optimizer = functools.partial(
          tf.optimizers.RMSprop, momentum=params['train']['optimizer']['momentum'])
    else:
      raise ValueError('Unsupported optimizer type `{}`.'.format(params.type))

  def __call__(self, learning_rate):
    return self._optimizer(learning_rate=learning_rate)
