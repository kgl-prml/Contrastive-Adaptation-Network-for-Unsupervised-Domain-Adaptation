import torch.nn as nn

class DomainModule(nn.Module):
    def __init__(self, num_domains, **kwargs):
        super(DomainModule, self).__init__()
        self.num_domains = num_domains
        self.domain = 0

    def set_domain(self, domain=0):
        assert(domain < self.num_domains), \
              "The domain id exceeds the range (%d vs. %d)" \
              % (domain, self.num_domains)
        self.domain = domain

class BatchNormDomain(DomainModule):
    def __init__(self, in_size, num_domains, norm_layer, **kwargs):
        super(BatchNormDomain, self).__init__(num_domains)
        self.bn_domain = nn.ModuleDict() 
        for n in range(self.num_domains):
            self.bn_domain[str(n)] = norm_layer(in_size, **kwargs)

    def forward(self, x):
        out = self.bn_domain[str(self.domain)](x)
        return out

