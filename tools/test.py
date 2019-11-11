import torch
import argparse
import os
import numpy as np
from torch.backends import cudnn
from model import model
import data.utils as data_utils
from utils.utils import to_cuda, mean_accuracy, accuracy
from data.custom_dataset_dataloader import CustomDatasetDataLoader
import sys
import pprint
from config.config import cfg, cfg_from_file, cfg_from_list
from math import ceil as ceil

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train script.')
    parser.add_argument('--weights', dest='weights',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--adapted', dest='adapted_model',
                        action='store_true',
                        help='if the model is adapted on target')
    parser.add_argument('--exp_name', dest='exp_name',
                        help='the experiment name', 
                        default='exp', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def save_preds(paths, preds, save_path, filename='preds.txt'):
    assert(len(paths) == preds.size(0))
    with open(os.path.join(save_path, filename), 'w') as f:
        for i in range(len(paths)):
            line = paths[i] + ' ' + str(preds[i].item()) + '\n'
            f.write(line)

def prepare_data():
    test_transform = data_utils.get_transform(False)

    target = cfg.TEST.DOMAIN
    dataroot_T = os.path.join(cfg.DATASET.DATAROOT, target)

    with open(os.path.join(cfg.DATASET.DATAROOT, 'category.txt'), 'r') as f:
        classes = f.readlines()
        classes = [c.strip() for c in classes]
    assert(len(classes) == cfg.DATASET.NUM_CLASSES)

    dataloader = None

    dataset_type = cfg.TEST.DATASET_TYPE
    batch_size = cfg.TEST.BATCH_SIZE
    dataloader = CustomDatasetDataLoader(dataset_root=dataroot_T, 
                dataset_type=dataset_type, batch_size=batch_size, 
                transform=test_transform, train=False, 
                num_workers=cfg.NUM_WORKERS, classnames=classes)

    return dataloader

def test(args):
    # prepare data
    dataloader = prepare_data()

    # initialize model
    model_state_dict = None
    fx_pretrained = True

    bn_domain_map = {}
    if cfg.WEIGHTS != '':
        weights_dict = torch.load(cfg.WEIGHTS)
        model_state_dict = weights_dict['weights']
        bn_domain_map = weights_dict['bn_domain_map']
        fx_pretrained = False

    if args.adapted_model:
        num_domains_bn = 2 
    else:
        num_domains_bn = 1

    net = model.danet(num_classes=cfg.DATASET.NUM_CLASSES, 
                 state_dict=model_state_dict,
                 feature_extractor=cfg.MODEL.FEATURE_EXTRACTOR, 
                 fx_pretrained=fx_pretrained, 
                 dropout_ratio=cfg.TRAIN.DROPOUT_RATIO,
                 fc_hidden_dims=cfg.MODEL.FC_HIDDEN_DIMS,
                 num_domains_bn=num_domains_bn) 

    net = torch.nn.DataParallel(net)

    if torch.cuda.is_available():
        net.cuda()

    # test 
    res = {}
    res['path'], res['preds'], res['gt'], res['probs'] = [], [], [], []
    net.eval()

    if cfg.TEST.DOMAIN in bn_domain_map:
        domain_id = bn_domain_map[cfg.TEST.DOMAIN]
    else:
        domain_id = 0

    with torch.no_grad():
        net.module.set_bn_domain(domain_id)
        for sample in iter(dataloader): 
            res['path'] += sample['Path']

            if cfg.DATA_TRANSFORM.WITH_FIVE_CROP:
                n, ncrop, c, h, w = sample['Img'].size()
                sample['Img'] = sample['Img'].view(-1, c, h, w)
                img = to_cuda(sample['Img'])
                probs = net(img)['probs']
                probs = probs.view(n, ncrop, -1).mean(dim=1)
            else:
                img = to_cuda(sample['Img'])
                probs = net(img)['probs']

            preds = torch.max(probs, dim=1)[1]
            res['preds'] += [preds]
            res['probs'] += [probs]

            if 'Label' in sample:
                label = to_cuda(sample['Label'])
                res['gt'] += [label] 
            print('Processed %d samples.' % len(res['path']))

        preds = torch.cat(res['preds'], dim=0)
        save_preds(res['path'], preds, cfg.SAVE_DIR)

        if 'gt' in res and len(res['gt']) > 0:
            gts = torch.cat(res['gt'], dim=0)
            probs = torch.cat(res['probs'], dim=0)
        
            assert(cfg.EVAL_METRIC == 'mean_accu' or cfg.EVAL_METRIC == 'accuracy')
            if cfg.EVAL_METRIC == "mean_accu": 
                eval_res = mean_accuracy(probs, gts)
                print('Test mean_accu: %.4f' % (eval_res))

            elif cfg.EVAL_METRIC == "accuracy":
                eval_res = accuracy(probs, gts)
                print('Test accuracy: %.4f' % (eval_res))

    print('Finished!')

if __name__ == '__main__':
    cudnn.benchmark = True 
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    if args.weights is not None:
        cfg.WEIGHTS = args.weights
    if args.exp_name is not None:
        cfg.EXP_NAME = args.exp_name 

    print('Using config:')
    pprint.pprint(cfg)

    cfg.SAVE_DIR = os.path.join(cfg.SAVE_DIR, cfg.EXP_NAME)
    print('Output will be saved to %s.' % cfg.SAVE_DIR)

    test(args)
