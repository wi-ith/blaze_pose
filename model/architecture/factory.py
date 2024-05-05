""" backbone architecture factory """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.architecture import mobilenetv2
from model.architecture import heads

def backbone_generator(params):
    if params['architecture']['backbone'] == 'mobilenetv2':
        backbone_model = mobilenetv2.MobilenetV2(
            input_dims=[params['train']['batch_size'],
                        params['architecture']['input_size']['height'],
                        params['architecture']['input_size']['width']]
        )
    else:
        raise ValueError('{} is not in model factory'.format(params))
    return backbone_model


def blazepose_head_generator(backbone, params):
    heads_model = heads.blazepose(backbone, params)
    return heads_model