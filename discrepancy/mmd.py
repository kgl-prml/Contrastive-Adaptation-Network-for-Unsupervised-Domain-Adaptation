import torch
from torch import nn
from utils.utils import to_cuda

class MMD(object):
    def __init__(self, num_layers, kernel_num, kernel_mul, 
		gammas=[], joint=False, **kwargs):

        self.num_layers = num_layers
        assert(len(kernel_num) == num_layers and len(kernel_mul) == num_layers), \
            "Each layer's kernel choice should be specified."

        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.gammas = gammas

        self.joint = joint

    def compute_paired_dist(self, A, B):

        bs_A = A.size(0)
        bs_T = B.size(0)
        feat_len = A.size(1)

        A_expand = A.unsqueeze(1).expand(bs_A, bs_T, feat_len)
        B_expand = B.unsqueeze(0).expand(bs_A, bs_T, feat_len)
        dist = (((A_expand - B_expand))**2).sum(2)
        return dist

    def gamma_estimation(self, dist):
        dist_sum = torch.sum(dist['ss']) + torch.sum(dist['tt']) + \
	    	2 * torch.sum(dist['st'])

        bs_S = dist['ss'].size(0)
        bs_T = dist['tt'].size(0)
        N = bs_S * bs_S + bs_T * bs_T + 2 * bs_S * bs_T - bs_S - bs_T
        gamma = dist_sum.item() / N 
        return gamma

    def compute_kernel_dist(self, dist, gamma, kernel_num, kernel_mul):
        base_gamma = gamma / (kernel_mul ** (kernel_num // 2))
        gamma_list = [base_gamma * (kernel_mul**i) for i in range(kernel_num)]
        gamma_tensor = to_cuda(torch.tensor(gamma_list))

        eps = 1e-5
        gamma_mask = (gamma_tensor < eps).type(torch.cuda.FloatTensor)
        gamma_tensor = (1.0 - gamma_mask) * gamma_tensor + gamma_mask * eps 
        gamma_tensor = gamma_tensor.detach()

        dist = dist.unsqueeze(0) / gamma_tensor.view(-1, 1, 1)
        upper_mask = (dist > 1e5).type(torch.cuda.FloatTensor).detach()
        lower_mask = (dist < 1e-5).type(torch.cuda.FloatTensor).detach()
        normal_mask = 1.0 - upper_mask - lower_mask
        dist = normal_mask * dist + upper_mask * 1e5 + lower_mask * 1e-5
        kernel_val = torch.sum(torch.exp(-1.0 * dist), dim=0)
        return kernel_val

    def kernel_layer_aggregation(self, dist_layers, gamma_layers):
        kernel_dist = {}
        num_layers = self.num_layers 
        for i in range(num_layers):

            dist = dist_layers[i]
            gamma = gamma_layers[i]

            cur_kernel_num = self.kernel_num[i]
            cur_kernel_mul = self.kernel_mul[i]

            if len(kernel_dist.keys()) == 0:
                kernel_dist = {key: self.compute_kernel_dist(dist[key], 
			gamma, cur_kernel_num, cur_kernel_mul) 
                        for key in ['ss', 'tt', 'st']}
                continue

            if not self.joint:
                kernel_dist = {key: kernel_dist[key] + 
		    	self.compute_kernel_dist(dist[key], gamma, cur_kernel_num, cur_kernel_mul) 
                        for key in ['ss', 'tt', 'st']}

            else:
                kernel_dist = {key: kernel_dist[key] * 
		    	self.compute_kernel_dist(dist[key], gamma, cur_kernel_num, cur_kernel_mul) 
                        for key in ['ss', 'tt', 'st']}

        return kernel_dist

    def forward(self, source, target, **kwargs):

        assert(len(source) == self.num_layers and len(target) == self.num_layers), \
            "The length of source (%d) or target (%d) doesn't match with %d, please check." \
            % (len(source), len(target), self.num_layers)

        assert len(source) != 0, "No data provided."

        dist_layers = []
        gamma_layers = []
        for i in range(self.num_layers):

            cur_source = source[i]
            cur_target = target[i]

            dist = {}
            dist['ss'] = self.compute_paired_dist(cur_source, cur_source)  
            dist['tt'] = self.compute_paired_dist(cur_target, cur_target)  
            dist['st'] = self.compute_paired_dist(cur_source, cur_target) 

            dist_layers += [dist]
            if len(self.gammas) > i and self.gammas[i] is not None:
                gamma_layers += [self.gammas[i]]
            else:
                gamma_layers += [self.gamma_estimation(dist)]

        kernel_dist = self.kernel_layer_aggregation(dist_layers, gamma_layers)
        mmd = torch.mean(kernel_dist['ss']) + torch.mean(kernel_dist['tt']) \
                        - 2.0 * torch.mean(kernel_dist['st'])
        return {'mmd': mmd}
