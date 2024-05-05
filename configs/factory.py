"""configurations for models"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def config_generator(model):
    if model == 'blazepose':
        from configs import blazepose_cfg
        default_config = blazepose_cfg._CFG
    elif model == 'blazepose_reg':
        from configs import reg_cfg
        default_config = reg_cfg._CFG
    return default_config