from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from typing import Text, Optional
from utils import dataloader_utils as dlu
import glob

class InputFn(object):
    def __init__(self,
                 file_pattern: Text,
                 mode: Text,
                 params):
        assert file_pattern is not None
        assert mode is not None
        self._file_pattern = file_pattern
        self._mode = mode
        self._params = params
        self._is_training = (mode == 'train')

    def __call__(self, *args, **kwargs):
        dataset = tf.data.Dataset.from_tensor_slices([self._file_pattern])
        dataset.cache()

        # if self._is_training:
        #     dataset = dataset.repeat()

        def _parse(x):
            x = tf.data.TFRecordDataset(x)
            return x


        batch_size = self._params['train']['batch_size'] if self._is_training else self._params['eval']['batch_size']
        SHUFFLE_BUFFER = 64

        dataset = dataset.interleave(_parse, cycle_length=32,
                                     block_length=1,
                                     num_parallel_calls=32,
                                     deterministic=True)
        dataset = dataset.shuffle(SHUFFLE_BUFFER)
        dataset = dataset.map(dlu._parse_function, num_parallel_calls=32)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset
