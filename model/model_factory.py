""" detection models factory """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model import blazepose_model

def model_generator(params):
    if params['type'] == 'blazepose' :
        model_fn = blazepose_model.BlazePoseModel(params)

    else:
        raise ValueError('Model %s is not supported.'% params.type)
    return model_fn