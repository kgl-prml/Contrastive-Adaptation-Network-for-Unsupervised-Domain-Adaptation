import torchvision.transforms as transforms
from PIL import Image
import torch
from config.config import cfg

def get_transform(train=True):
    transform_list = []
    if cfg.DATA_TRANSFORM.RESIZE_OR_CROP == 'resize_and_crop':
        osize = [cfg.DATA_TRANSFORM.LOADSIZE, cfg.DATA_TRANSFORM.LOADSIZE]
        transform_list.append(transforms.Resize(osize, Image.BICUBIC))
        if train:
            transform_list.append(transforms.RandomCrop(cfg.DATA_TRANSFORM.FINESIZE))
        else:
            if cfg.DATA_TRANSFORM.WITH_FIVE_CROP:
                transform_list.append(transforms.FiveCrop(cfg.DATA_TRANSFORM.FINESIZE)) 
            else:
                transform_list.append(transforms.CenterCrop(cfg.DATA_TRANSFORM.FINESIZE))

    elif cfg.DATA_TRANSFORM.RESIZE_OR_CROP == 'crop':
        if train:
            transform_list.append(transforms.RandomCrop(cfg.DATA_TRANSFORM.FINESIZE))
        else:
            if cfg.DATA_TRANSFORM.WITH_FIVE_CROP:
                transform_list.append(transforms.FiveCrop(cfg.DATA_TRANSFORM.FINESIZE)) 
            else:
                transform_list.append(transforms.CenterCrop(cfg.DATA_TRANSFORM.FINESIZE))

    if train and cfg.DATA_TRANSFORM.FLIP:
        transform_list.append(transforms.RandomHorizontalFlip())

    to_normalized_tensor = [transforms.ToTensor(),
                            transforms.Normalize(mean=cfg.DATA_TRANSFORM.NORMALIZE_MEAN,
                                       std=cfg.DATA_TRANSFORM.NORMALIZE_STD)]

    if not train and cfg.DATA_TRANSFORM.WITH_FIVE_CROP:
        transform_list += [transforms.Lambda(lambda crops: torch.stack([
                transforms.Compose(to_normalized_tensor)(crop) for crop in crops]))]
    else:
        transform_list += to_normalized_tensor

    return transforms.Compose(transform_list)
