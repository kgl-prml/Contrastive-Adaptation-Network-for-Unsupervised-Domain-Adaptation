import torch
import torch.nn as nn
import os
from . import utils as solver_utils 
from utils.utils import to_cuda, mean_accuracy, accuracy
from torch import optim
from math import ceil as ceil
from config.config import cfg

class BaseSolver:
    def __init__(self, net, dataloader, bn_domain_map={}, resume=None, **kwargs):
        self.opt = cfg
        self.source_name = self.opt.DATASET.SOURCE_NAME
        self.target_name = self.opt.DATASET.TARGET_NAME

        self.net = net
        self.init_data(dataloader)

        self.CELoss = nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            self.CELoss.cuda() 

        self.loop = 0
        self.iters = 0
        self.iters_per_loop = None
        self.history = {}

        self.base_lr = self.opt.TRAIN.BASE_LR
        self.momentum = self.opt.TRAIN.MOMENTUM

        self.bn_domain_map = bn_domain_map

        self.optim_state_dict = None
        self.resume = False
        if resume is not None:
            self.resume = True
            self.loop = resume['loop']
            self.iters = resume['iters']
            self.history = resume['history']
            self.optim_state_dict = resume['optimizer_state_dict']
            self.bn_domain_map = resume['bn_domain_map']
            print('Resume Training from loop %d, iters %d.' % \
			(self.loop, self.iters))

        self.build_optimizer()

    def init_data(self, dataloader):
        self.train_data = {key: dict() for key in dataloader if key != 'test'}
        for key in self.train_data.keys():
            if key not in dataloader:
                continue
            cur_dataloader = dataloader[key]
            self.train_data[key]['loader'] = cur_dataloader 
            self.train_data[key]['iterator'] = None

        if 'test' in dataloader:
            self.test_data = dict()
            self.test_data['loader'] = dataloader['test']
        
    def build_optimizer(self):
        opt = self.opt
        param_groups = solver_utils.set_param_groups(self.net, 
		dict({'FC': opt.TRAIN.LR_MULT}))

        assert opt.TRAIN.OPTIMIZER in ["Adam", "SGD"], \
            "Currently do not support your specified optimizer."

        if opt.TRAIN.OPTIMIZER == "Adam":
            self.optimizer = optim.Adam(param_groups, 
			lr=self.base_lr, betas=[opt.ADAM.BETA1, opt.ADAM.BETA2], 
			weight_decay=opt.TRAIN.WEIGHT_DECAY)

        elif opt.TRAIN.OPTIMIZER == "SGD":
            self.optimizer = optim.SGD(param_groups, 
			lr=self.base_lr, momentum=self.momentum, 
			weight_decay=opt.TRAIN.WEIGHT_DECAY)

        if self.optim_state_dict is not None:
            self.optimizer.load_state_dict(self.optim_state_dict)

    def update_lr(self):
        iters = self.iters
        if self.opt.TRAIN.LR_SCHEDULE == 'exp':
            solver_utils.adjust_learning_rate_exp(self.base_lr, 
			self.optimizer, iters, 
                        decay_rate=self.opt.EXP.LR_DECAY_RATE,
			decay_step=self.opt.EXP.LR_DECAY_STEP)

        elif self.opt.TRAIN.LR_SCHEDULE == 'inv':
            solver_utils.adjust_learning_rate_inv(self.base_lr, self.optimizer, 
		    iters, self.opt.INV.ALPHA, self.opt.INV.BETA)

        else:
            raise NotImplementedError("Currently don't support the specified \
                    learning rate schedule: %s." % self.opt.TRAIN.LR_SCHEDULE)

    def logging(self, loss, accu):
        print('[loop: %d, iters: %d]: ' % (self.loop, self.iters))
        loss_names = ""
        loss_values = ""
        for key in loss:
            loss_names += key + ","
            loss_values += '%.4f,' % (loss[key])
        loss_names = loss_names[:-1] + ': '
        loss_values = loss_values[:-1] + ';'
        loss_str = loss_names + loss_values + (' source %s: %.4f.' % 
                    (self.opt.EVAL_METRIC, accu))
        print(loss_str)

    def model_eval(self, preds, gts):
        assert(self.opt.EVAL_METRIC in ['mean_accu', 'accuracy']), \
             "Currently don't support the evaluation metric you specified."

        if self.opt.EVAL_METRIC == "mean_accu": 
            res = mean_accuracy(preds, gts)
        elif self.opt.EVAL_METRIC == "accuracy":
            res = accuracy(preds, gts)
        return res

    def save_ckpt(self):
        save_path = self.opt.SAVE_DIR
        ckpt_resume = os.path.join(save_path, 'ckpt_%d_%d.resume' % (self.loop, self.iters))
        ckpt_weights = os.path.join(save_path, 'ckpt_%d_%d.weights' % (self.loop, self.iters))
        torch.save({'loop': self.loop,
                    'iters': self.iters,
                    'model_state_dict': self.net.module.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'history': self.history,
                    'bn_domain_map': self.bn_domain_map
                    }, ckpt_resume)

        torch.save({'weights': self.net.module.state_dict(),
                    'bn_domain_map': self.bn_domain_map
                    }, ckpt_weights)

    def complete_training(self):
        if self.loop > self.opt.TRAIN.MAX_LOOP:
            return True

    def register_history(self, key, value, history_len):
        if key not in self.history:
            self.history[key] = [value]
        else:
            self.history[key] += [value]
        
        if len(self.history[key]) > history_len:
            self.history[key] = \
                 self.history[key][len(self.history[key]) - history_len:]
       
    def solve(self):
        print('Training Done!')

    def get_samples(self, data_name):
        assert(data_name in self.train_data)
        assert('loader' in self.train_data[data_name] and \
               'iterator' in self.train_data[data_name])

        data_loader = self.train_data[data_name]['loader']
        data_iterator = self.train_data[data_name]['iterator']
        assert data_loader is not None and data_iterator is not None, \
            'Check your dataloader of %s.' % data_name 

        try:
            sample = next(data_iterator)
        except StopIteration:
            data_iterator = iter(data_loader)
            sample = next(data_iterator)
            self.train_data[data_name]['iterator'] = data_iterator
        return sample

    def get_samples_categorical(self, data_name, category):
        assert(data_name in self.train_data)
        assert('loader' in self.train_data[data_name] and \
               'iterator' in self.train_data[data_name])

        data_loader = self.train_data[data_name]['loader'][category]
        data_iterator = self.train_data[data_name]['iterator'][category]
        assert data_loader is not None and data_iterator is not None, \
            'Check your dataloader of %s.' % data_name

        try:
            sample = next(data_iterator)
        except StopIteration:
            data_iterator = iter(data_loader)
            sample = next(data_iterator)
            self.train_data[data_name]['iterator'][category] = data_iterator

        return sample

    def test(self):
        self.net.eval()
        preds = []
        gts = []
        for sample in iter(self.test_data['loader']):
            data, gt = to_cuda(sample['Img']), to_cuda(sample['Label'])
            logits = self.net(data)['logits']
            preds += [logits]
            gts += [gt]

        preds = torch.cat(preds, dim=0)
        gts = torch.cat(gts, dim=0)

        res = self.model_eval(preds, gts)
        return res

    def clear_history(self, key):
        if key in self.history:
            self.history[key].clear()

    def solve(self):
        pass

    def update_network(self, **kwargs):
        pass

