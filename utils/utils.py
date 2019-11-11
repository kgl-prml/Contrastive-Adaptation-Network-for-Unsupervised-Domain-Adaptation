import torch
import numpy as np

def find_class_by_name(name, modules):
  """Searches the provided modules for the named class and returns it."""
  modules = [getattr(module, name, None) for module in modules]
  return next(a for a in modules if a)

def to_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def to_data(x):
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data.numpy()

def to_onehot(label, num_classes):
    identity = to_cuda(torch.eye(num_classes))
    onehot = torch.index_select(identity, 0, label)
    return onehot

def mean_accuracy(preds, target):
    num_classes = preds.size(1)
    preds = torch.max(preds, dim=1).indices
    accu_class = []
    for c in range(num_classes):
        mask = (target == c)
        c_count = torch.sum(mask).item()
        if c_count == 0: continue
        preds_c = torch.masked_select(preds, mask)
        accu_class += [1.0 * torch.sum(preds_c == c).item() / c_count]
    return 100.0 * np.mean(accu_class)

def accuracy(preds, target):
    preds = torch.max(preds, dim=1).indices
    return 100.0 * torch.sum(preds == target).item() / preds.size(0)


