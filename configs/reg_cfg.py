"""base configuration"""

from configs.base_cfg import BASE_CFG




BASE_CFG['type'] = 'blazepose'
BASE_CFG['train']['train_samples'] = 47563
BASE_CFG['train']['total_epochs'] = 400
BASE_CFG['train']['optimizer']['type'] = 'adam'
BASE_CFG['train']['optimizer']['momentum'] = 0.9
BASE_CFG['train']['optimizer']['nesterov'] = False
BASE_CFG['train']['n_threshold'] = 1
BASE_CFG['train']['threshold'] = 1.0
BASE_CFG['train']['batch_size'] = 16
BASE_CFG['train']['model_dir'] = '/home/kdg/dev/wrapup240126/poseEstimation/blazePose/logs/test/'
BASE_CFG['train']['tfrecord_dir'] = '/home/kdg/dev/tfrecords/foot_pose_superbai_all_addcustom/'
BASE_CFG['train']['type'] = 'regression'

BASE_CFG['train']['sigma_k'] = 0.05
BASE_CFG['train']['learning_rate']['init_learning_rate'] = 0.0001
BASE_CFG['train']['learning_rate']['learning_rate_decay_rate'] = 1
BASE_CFG['train']['learning_rate']['learning_rate_decay_epochs'] = 100
BASE_CFG['train']['load_weight_path'] = BASE_CFG['train']['model_dir']+'151-epoch'
BASE_CFG['train']['load_backbone_weights'] = '/home/kdg/dev/wrapup240126/poseEstimation/blazePose/pretrained/newModel/model.ckpt'
# BASE_CFG['train']['load_backbone_weights'] ='/home/kdg/dev/architec/blaze_pose/pretrained/newModel/mopdel.ckpt'

BASE_CFG['eval']['eval_samples'] = 979
BASE_CFG['eval']['batch_size'] = 4
BASE_CFG['eval']['pck_rate'] = 0.1
BASE_CFG['eval']['bento_save'] = ''#BASE_CFG['train']['model_dir']+'bento/'
# BASE_CFG['eval']['bento_save'] = BASE_CFG['train']['model_dir']+'bento/'
THRESHOLD = 1.0     # range of same depth
N_THRESHOLD = 15

BASE_CFG['architecture']['backbone'] = 'mobilenetv2'
BASE_CFG['architecture']['input_size']['height'] = 256
BASE_CFG['architecture']['input_size']['width'] = 256
BASE_CFG['architecture']['input_size']['output_stride'] = 4
BASE_CFG['architecture']['input_size']['num_keypoints'] = 7

_CFG = BASE_CFG