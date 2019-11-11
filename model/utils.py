import torch
from torch.nn import init

def weights_init_he(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        if 'weight' in m.state_dict().keys():
            m.weight.data.normal_(1.0, 0.02)
        if 'bias' in m.state_dict().keys():
            m.bias.data.fill_(0)
    else:
        if 'weight' in m.state_dict().keys():
            init.kaiming_normal_(m.weight)
        if 'bias' in m.state_dict().keys():
            m.bias.data.fill_(0)


def init_weights(model, state_dict, num_domains=1, BN2BNDomain=False):
    model.apply(weights_init_he)

    if state_dict is not None:

        model_state_dict = model.state_dict()

        keys = set(model_state_dict.keys())
        trained_keys = set(state_dict.keys())

        shared_keys = keys.intersection(trained_keys)
        new_state_dict = {key: state_dict[key] for key in shared_keys}
        if BN2BNDomain:
            for k in (trained_keys - shared_keys):
                if k.find('fc') != -1:
                    continue
                suffix = k.split('.')[-1]
                for d in range(num_domains):
                    bn_key = k.replace(suffix, 'bn_domain.' + str(d) + '.' + suffix) 
                    new_state_dict[bn_key] = state_dict[k]

        model.load_state_dict(new_state_dict) 

    return model
