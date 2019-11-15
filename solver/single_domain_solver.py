import torch
import torch.nn as nn
import os
from utils.utils import to_cuda
from torch import optim
from data.custom_dataset_dataloader import CustomDatasetDataLoader
from math import ceil as ceil
from .base_solver import BaseSolver

class SingleDomainSolver(BaseSolver):
    def __init__(self, net, dataloader, resume=None, **kwargs):
        super(SingleDomainSolver, self).__init__(net, dataloader, \
                        resume=resume, **kwargs)
        assert(len(self.train_data) > 0), "Please specify the training domain."
        self.train_domain = list(self.train_data.keys())[0]

    def solve(self):
        if self.resume:
            self.iters += 1
            self.loop += 1

        self.compute_iters_per_loop()
        while True:
            if self.loop > self.opt.TRAIN.MAX_LOOP: break
            self.update_network()
            self.loop += 1

        print('Training Done!')

    def compute_iters_per_loop(self):
        self.iters_per_loop = len(self.train_data[self.train_domain]['loader']) 
        print('Iterations in one loop: %d' % (self.iters_per_loop))

    def update_network(self):
        # initial configuration
        stop = False
        update_iters = 0
        self.train_data[self.source_name]['iterator'] = iter(self.train_data[self.source_name]['loader'])
        while not stop:
            loss = 0
            # update learning rate
            self.update_lr()

            # set the status of network
            self.net.train()
            self.net.zero_grad()

            # coventional sampling for training on labeled source data
            source_sample = self.get_samples(self.train_domain) 
            source_data, source_gt = source_sample['Img'],\
                          source_sample['Label']

            source_data = to_cuda(source_data)
            source_gt = to_cuda(source_gt)
            self.net.module.set_bn_domain()
            source_preds = self.net(source_data)['logits']

            # compute the cross-entropy loss
            ce_loss = self.CELoss(source_preds, source_gt)
            ce_loss.backward()
            loss += ce_loss
         
            # update the network
            self.optimizer.step()

            if self.opt.TRAIN.LOGGING and (update_iters+1) % \
                      (max(1, self.iters_per_loop // 10)) == 0:
                accu = self.model_eval(source_preds, source_gt)
                cur_loss = {'ce_loss': ce_loss}
                self.logging(cur_loss, accu)

            if self.opt.TRAIN.TEST_INTERVAL > 0 and \
		(self.iters+1) % int(self.opt.TRAIN.TEST_INTERVAL * self.iters_per_loop) == 0:
                with torch.no_grad():
                    self.net.module.set_bn_domain()
                    accu = self.test()
                print('Test at (loop %d, iters %d) with %s: %.4f.' % (
                              self.loop, self.iters, 
                              self.opt.EVAL_METRIC, accu))

            if self.opt.TRAIN.SAVE_CKPT_INTERVAL > 0 and \
		(self.iters+1) % int(self.opt.TRAIN.SAVE_CKPT_INTERVAL * self.iters_per_loop) == 0:
                self.save_ckpt()

            update_iters += 1
            self.iters += 1

            # update stop condition
            if update_iters >= self.iters_per_loop:
                stop = True
            else:
                stop = False

