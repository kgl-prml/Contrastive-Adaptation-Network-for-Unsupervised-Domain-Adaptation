# Contrastive Adaptation Network 
This is the Pytorch implementation for our CVPR 2019 paper [Contrastive Adaptation Network for Unsupervised Domain Adaptation](http://openaccess.thecvf.com/content_CVPR_2019/papers/Kang_Contrastive_Adaptation_Network_for_Unsupervised_Domain_Adaptation_CVPR_2019_paper.pdf). As we reorganized our code based on a new pytorch version, some hyper-parameters are slightly different from the paper.

## Requirements
- Python 3.7
- Pytorch 1.1
- PyYAML 5.1.1

## Dataset
The structure of the dataset should be like

```
Office-31
|_ category.txt
|_ amazon
|  |_ back_pack
|     |_ <im-1-name>.jpg
|     |_ ...
|     |_ <im-N-name>.jpg
|  |_ bike
|     |_ <im-1-name>.jpg
|     |_ ...
|     |_ <im-N-name>.jpg
|  |_ ...
|_ dslr
|  |_ back_pack
|     |_ <im-1-name>.jpg
|     |_ ...
|     |_ <im-N-name>.jpg
|  |_ bike
|     |_ <im-1-name>.jpg
|     |_ ...
|     |_ <im-N-name>.jpg
|  |_ ...
|_ ...
```
The "category.txt" contains the names of all the categories, which is like
```
back_pack
bike
bike_helmet
...
```

## Training
```
./experiments/scripts/train.sh ${config_yaml} ${gpu_ids} ${adaptation_method} ${experiment_name}
```
For example, for the Office-31 dataset,
```
./experiments/scripts/train.sh ./experiments/config/Office-31/CAN/office31_train_amazon2dslr_cfg.yaml 0 CAN office31_a2d
```
for the VisDA-2017 dataset,
```
./experiments/scripts/train.sh ./experiments/config/VisDA-2017/CAN/visda17_train_train2val_cfg.yaml 0 CAN visda17_train2val
```

The experiment log file and the saved checkpoints will be stored at ./experiments/ckpt/${experiment_name}

## Test

```
./experiments/scripts/test.sh ${config_yaml} 0 ${if_adapted} ${experiment_name}
```
Example: 
```
./experiments/scripts/test.sh ./experiments/config/Office-31/office31_test_amazon_cfg.yaml 0 True visda17_test
```

## Citing 
Please cite our paper if you use our code in your research:
```
@inproceedings{kang2019contrastive,
  title={Contrastive Adaptation Network for Unsupervised Domain Adaptation},
  author={Kang, Guoliang and Jiang, Lu and Yang, Yi and Hauptmann, Alexander G},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={4893--4902},
  year={2019}
}
```
## Contact
If you have any questions, please contact me via kgl.prml@gmail.com.

## Thanks to third party
The way of setting configurations is inspired by <https://github.com/rbgirshick/py-faster-rcnn>.

