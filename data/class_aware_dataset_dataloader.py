import torch.utils.data
from .categorical_dataset import CategoricalSTDataset
from math import ceil as ceil

def collate_fn(data):
    # data is a list: index indicates classes
    data_collate = {}
    num_classes = len(data)
    keys = data[0].keys()
    for key in keys:
        if key.find('Label') != -1:
            data_collate[key] = [torch.tensor(data[i][key]) for i in range(num_classes)]
        if key.find('Img') != -1:
            data_collate[key] = [data[i][key] for i in range(num_classes)]
        if key.find('Path') != -1:
            data_collate[key] = [data[i][key] for i in range(num_classes)]

    return data_collate

class ClassAwareDataLoader(object):
    def name(self):
        return 'ClassAwareDataLoader'

    def __init__(self, source_batch_size, target_batch_size,
                source_dataset_root="", target_paths=[], 
                transform=None, classnames=[], 
                class_set=[], num_selected_classes=0, 
                seed=None, num_workers=0, drop_last=True, 
                sampler='RandomSampler', **kwargs):
        
        # dataset type
        self.dataset = CategoricalSTDataset()

        # dataset parameters
        self.source_dataset_root = source_dataset_root
        self.target_paths = target_paths
        self.classnames = classnames
        self.class_set = class_set
        self.source_batch_size = source_batch_size
        self.target_batch_size = target_batch_size
        self.seed = seed
        self.transform = transform

        # loader parameters
        self.num_selected_classes = min(num_selected_classes, len(class_set))
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.sampler = sampler
        self.kwargs = kwargs

    def construct(self):
        self.dataset.initialize(source_root=self.source_dataset_root, 
                  target_paths=self.target_paths,
                  classnames=self.classnames, class_set=self.class_set, 
                  source_batch_size=self.source_batch_size, 
                  target_batch_size=self.target_batch_size, 
                  seed=self.seed, transform=self.transform, 
                  **self.kwargs)

        drop_last = self.drop_last
        sampler = getattr(torch.utils.data, self.sampler)(self.dataset)
        batch_sampler = torch.utils.data.BatchSampler(sampler, 
                                 self.num_selected_classes, drop_last)

        self.dataloader = torch.utils.data.DataLoader(self.dataset, 
                         batch_sampler=batch_sampler,
                         collate_fn=collate_fn,
                         num_workers=int(self.num_workers))

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        dataset_len = 0.0
        cid = 0
        for c in self.class_set:
            c_len = max([len(self.dataset.data_paths[d][cid]) // \
                  self.dataset.batch_sizes[d][cid] for d in ['source', 'target']])
            dataset_len += c_len
            cid += 1

        dataset_len = ceil(1.0 * dataset_len / self.num_selected_classes)
        return dataset_len

