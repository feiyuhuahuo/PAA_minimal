## PAA_minimal
Minimal PyTorch implementation of [Probabilistic Anchor Assignment with IoU Prediction for Object Detection](https://arxiv.org/abs/2007.08103).  
The original project is [here](https://github.com/kkhoot/PAA).  

## Environments  
1.1 <= PyTorch <= 1.4 (Version > 1.4 will cause a compilation error).  
Python >= 3.6.   
Other common packages.  

## Prepare
- Download COCO 2017 datasets, modify the paths of training and evalution datasets in `config.py`. 
- ```
  # Build DCN, NMS, CUDA FocalLoss.
  cd build_stuff
  python setup.py build develop
  ```

- Download weights and put the weight files in `weights` folder.  

PAA trained weights.  
I trained on two RTX-2080Ti GPUs. Following are results on COCO val2017. SV=score voting.

|cfg        |total iterations| mAP                         | Google Drive                                                                                       |Baidu Cloud                                                       |
|:---------:|:--------------:|:---------------------------:|:--------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------:|
|res50_1x   |120000 (bs=12)  |40.2 (40.5 with SV)| [res50_1x_116000.pth](https://drive.google.com/file/d/1lePvujaE42xHXXN-pxIveHiE8bEt7Azg/view?usp=sharing)    |[password: mksf](https://pan.baidu.com/s/1XDeDwg1Xw9GJCucJNqdNZw) |
|res101_2x  |288000 (bs=10)  |TBD  (40.5 with SV)| [res101_coco_800000.pth](https://drive.google.com/file/d/1KyjhkLEw0D8zP8IiJTTOR0j6PGecKbqS/view?usp=sharing) |[password: oubr](https://pan.baidu.com/s/1uX_v1RPISxgwQ2LdsbJrJQ) |

Backbone pre-trained weights.  

| Backbone  | Google Drive                                                                                    |Baidu Cloud                                                        |
|:---------:|:-----------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------:|
| Resnet50  | [R-50.pkl](https://drive.google.com/file/d/1hIhYjTRzA7qnslwkiBmPFttQURto6VeC/view?usp=sharing)  | [password: i8i3](https://pan.baidu.com/s/1MeTs6Ml4h4dc4Xue3RZdZQ) |
| Resnet101 | [R-101.pkl](https://drive.google.com/file/d/1ZBPXe5n5dLfHjCUn1G6Z91TQFM4kBO_y/view?usp=sharing) | [password: 04ia](https://pan.baidu.com/s/1BACQ3XT2k4Qaa0yC80USpA) |


## Train

```
# Train by res50_1x configuration with a certain batch_size on some specific GPUs.
export CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.launch --nproc_per_node=2 train.py --train_bs=12

# Train with other configuration. (There are 4 configurations in total: res50_1x, res50_15x, res101_2x, res101_dcn_2x.)
python -m torch.distributed.launch --nproc_per_node=2 train.py --train_bs=12 --cfg=res101_2x

# Resume training.
python -m torch.distributed.launch --nproc_per_node=2 train.py --train_bs=12 --cfg=res101_2x --resume=weight/[weight_file]

# Other utilization 
--test_bs=2, set validation batch size.
--val_interval=6000, set validation interval during training.
--val_num=500, set validation number during training.
--score_voting, activate score voting during validation.
--improved_coco, use an improved COCO API to do validation.
```


## Evalution
```
# Evaluate COCO val2017 on a specific GPU.
python val.py --gpu_id=0 --weight=weights/res50_1x_116000.pth

# Evaluate with a specific batch size.
python val.py --gpu_id=0 --weight=weights/res50_1x_116000.pth --test_bs=2

# Specify validation number.
python val.py --gpu_id=0 --weight=weights/res50_1x_116000.pth --val_num=500

# Evaluate with score voting.
python val.py --gpu_id=0 --weight=weights/res50_1x_116000.pth --score_voting

# Use an improved COCO API to do validation.
python val.py --gpu_id=0 --weight=weights/res50_1x_116000.pth --improved_coco
```

## Reference:
- https://github.com/kkhoot/PAA
```
@inproceedings{paa-eccv2020,
  title={Probabilistic Anchor Assignment with IoU Prediction for Object Detection},
  author={Kim, Kang and Lee, Hee Seok},
  booktitle = {ECCV},
  year={2020}
}
```
