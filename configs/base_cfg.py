"""base configuration"""

BASE_CFG = {
    'type':'blazepose',
    'pretrained_dir':'',
    'model_dir': '',
    'use_tpu': True,
    'strategy_type': 'tpu',
    'isolate_session_state': False,
    'train': {
        'train_samples':12000,
        'batch_size': 32,
        'total_epochs': 250,
        'num_cores_per_replica': None,
        'input_partition_dims': None,
        'optimizer': {
            'type': 'momentum',
            'momentum': 0.9,
            'nesterov': False,
        },
        'learning_rate': {
            'type':'exponential_decay',
            'init_learning_rate': 0.001,
            'learning_rate_decay_rate': 0.5,
            'learning_rate_decay_epochs': 50,
        },
        'l2_weight_decay': 4e-5,
        'summary_save_step': 500,
        'type' : 'regression'
    },
    'eval': {
        'input_sharding': True,
        'batch_size': 8,
        'eval_samples': 5000,
        'num_images_to_visualize': 0,
    },
    'predict': {
        'batch_size': 8,
    },
    'architecture': {
        'input_size' : {'width' : 224,'height' : 224},
        'backbone': '',
        'num_classes': 1
    }
}