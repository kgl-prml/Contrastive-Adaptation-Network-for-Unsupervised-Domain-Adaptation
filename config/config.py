import os
import numpy as np
from easydict import EasyDict as edict

__C = edict()
cfg = __C

# Dataset options
#
__C.DATASET = edict()
__C.DATASET.NUM_CLASSES = 0
__C.DATASET.DATAROOT = ''
__C.DATASET.SOURCE_NAME = ''
__C.DATASET.TARGET_NAME = ''

# Model options
__C.MODEL = edict()
__C.MODEL.FEATURE_EXTRACTOR = 'resnet101'
__C.MODEL.FC_HIDDEN_DIMS = ()
__C.MODEL.PRETRAINED = True

# data pre-processing options
#
__C.DATA_TRANSFORM = edict()
__C.DATA_TRANSFORM.RESIZE_OR_CROP = 'resize_and_crop'
__C.DATA_TRANSFORM.LOADSIZE = 256
__C.DATA_TRANSFORM.FINESIZE = 224
__C.DATA_TRANSFORM.FLIP = True
__C.DATA_TRANSFORM.WITH_FIVE_CROP = False
__C.DATA_TRANSFORM.NORMALIZE_MEAN = (0.485, 0.456, 0.406)
__C.DATA_TRANSFORM.NORMALIZE_STD = (0.229, 0.224, 0.225)

# Training options
#
__C.TRAIN = edict()
# batch size setting
__C.TRAIN.SOURCE_BATCH_SIZE = 30
__C.TRAIN.TARGET_BATCH_SIZE = 30 
__C.TRAIN.TARGET_CLASS_BATCH_SIZE = 3
__C.TRAIN.SOURCE_CLASS_BATCH_SIZE = 3
__C.TRAIN.NUM_SELECTED_CLASSES = 10
# model setting
__C.TRAIN.STOP_GRAD = 'layer1'
__C.TRAIN.DROPOUT_RATIO = (0.0,)
# learning rate schedule
__C.TRAIN.BASE_LR = 0.001
__C.TRAIN.MOMENTUM = 0.9
__C.TRAIN.LR_MULT = 10
__C.TRAIN.OPTIMIZER = 'SGD'
__C.TRAIN.WEIGHT_DECAY = 0.0005
__C.TRAIN.LR_SCHEDULE = 'inv'
__C.TRAIN.MAX_LOOP = 50
__C.TRAIN.STOP_THRESHOLDS = (0.001, 0.001, 0.001)
__C.TRAIN.MIN_SN_PER_CLASS = 3
__C.TRAIN.LOGGING = True
__C.TRAIN.TEST_INTERVAL = 1.0 # percentage of total iterations each loop
__C.TRAIN.SAVE_CKPT_INTERVAL = 1.0 # percentage of total iterations in each loop
__C.TRAIN.NUM_LOGGING_PER_LOOP = 6.0
__C.TRAIN.UPDATE_EPOCH_PERCENTAGE = 1.0

# optimizer options
__C.ADAM = edict()
__C.ADAM.BETA1 = 0.9
__C.ADAM.BETA2 = 0.999

__C.INV = edict()
__C.INV.ALPHA = 0.001
__C.INV.BETA = 0.75

__C.EXP = edict()
__C.EXP.LR_DECAY_RATE = 0.1
__C.EXP.LR_DECAY_STEP = 30


# Clustering options
__C.CLUSTERING = edict()
__C.CLUSTERING.TARGET_BATCH_SIZE = 100
__C.CLUSTERING.SOURCE_BATCH_SIZE = 100
__C.CLUSTERING.TARGET_DATASET_TYPE = 'SingleDatasetWithoutLabel'
__C.CLUSTERING.BUDGET = 1000
__C.CLUSTERING.EPS = 0.005
__C.CLUSTERING.FILTERING_THRESHOLD = 1.0
__C.CLUSTERING.FEAT_KEY = 'feat'
__C.CLUSTERING.HISTORY_LEN = 2

# CDD options
__C.CDD = edict()
__C.CDD.KERNEL_NUM = (5, 5)
__C.CDD.KERNEL_MUL = (2, 2)
__C.CDD.LOSS_WEIGHT = 0.3
__C.CDD.ALIGNMENT_FEAT_KEYS = ['feat', 'probs']
__C.CDD.INTRA_ONLY = False

# MMD/JMMD options
__C.MMD = edict()
__C.MMD.KERNEL_NUM = (5, 5)
__C.MMD.KERNEL_MUL = (2, 2)
__C.MMD.LOSS_WEIGHT = 0.3
__C.MMD.ALIGNMENT_FEAT_KEYS = ['feat', 'probs']
__C.MMD.JOINT = False

# Testing options
#
__C.TEST = edict()
__C.TEST.BATCH_SIZE = 30
__C.TEST.DATASET_TYPE = 'SingleDataset'
__C.TEST.DOMAIN = ''

# MISC
__C.WEIGHTS = ''
__C.RESUME = ''
__C.EVAL_METRIC = "accuracy" # "mean_accu" as alternative
__C.EXP_NAME = 'exp'
__C.SAVE_DIR = ''
__C.NUM_WORKERS = 3

def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k in a:
        # a must specify keys that are in b
        v = a[k]
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                'for config key: {}').format(type(b[k]),
                                                            type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v

def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f, Loader=yaml.FullLoader))

    _merge_a_into_b(yaml_cfg, __C)

def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
            type(value), type(d[subkey]))
        d[subkey] = value
