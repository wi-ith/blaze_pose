from absl import flags
from absl import app
import glob
import tensorflow as tf
from configs import factory as cfg_factory
from executor import distributed_executor as DetectionDistributedExecutor
from executor.blazepose_executor import BlazePoseDistributedExecutor
from model import model_factory
from dataloader import input_reader
from silence_tensorflow import silence_tensorflow
import argparse


parser = argparse.ArgumentParser(description='',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--model', type=str, default=None,
                    help='blazepose or blazepose_reg')

parser.add_argument('--port', type=str, default=None,
                    help='')
parser.add_argument('--mode', type=str, default=None,
                    help='')

args = parser.parse_args()


tf.config.set_soft_device_placement(True)

params = cfg_factory.config_generator(args.model)

print('optimizer : ', params['train']['optimizer']['type'], '\n'
      'train model : ', params['train']['type'], '\n',
      'train batch size : ', params['train']['batch_size'], '\n',
      'train lr : ', params['train']['learning_rate']['init_learning_rate'])

if params['train']['type'] == 'regression':
    silence_tensorflow()

model_builder = model_factory.model_generator(params)


def _model_fn(params):
    return model_builder.build_model(params, mode='train')

def _eval_model_fn(params):
    return model_builder.build_model(params, mode='eval')

dist_executor = BlazePoseDistributedExecutor(params = params,
                                           model_fn = _model_fn,
                                           eval_model_fn=_eval_model_fn,
                                           loss_fn = model_builder.build_loss_fn,
                                           eval_fn = model_builder.build_eval_fn)

tf_record_pattern = glob.glob(params['train']['tfrecord_dir'] + '*train*')
train_input_fn = input_reader.InputFn(tf_record_pattern, mode = 'train', params=params)

tf_record_pattern = glob.glob(params['train']['tfrecord_dir'] + '*val*')
val_input_fn = input_reader.InputFn(tf_record_pattern, mode = 'eval', params=params)

dist_executor.train(
    train_input_fn = train_input_fn(),
    val_input_fn = val_input_fn()
)
