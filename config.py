import os

os.makedirs('results/', exist_ok=True)
os.makedirs('weights/', exist_ok=True)


class dict2class:
    def __init__(self, config):
        for k, v in config.items():
            self.__setattr__(k, v)

        print('\n' + '-' * 40 + 'Config' + '-' * 40)
        for k, v in vars(self).items():
            print(f'{k}: {v}')
        print()


data_root = '/home/feiyu/Data/coco2017/'
res50_1x_cfg = {'train_imgs': data_root + 'val2017/',
                'train_ann': data_root + 'annotations/instances_val2017.json',
                'val_imgs': data_root + 'val2017/',
                'val_ann': data_root + 'annotations/instances_val2017.json',
                'num_classes': 81,
                'backbone': 'res50',
                'stem_norm': 'frozen_BN',  # 'frozen_BN' or 'GN'
                'bottleneck_norm': 'frozen_BN',
                'stage_with_dcn': (False, False, False, False),
                'freeze_backbone_at': 2,
                'anchor_strides': (8, 16, 32, 64, 128),
                'anchor_sizes': ((64,), (128,), (256,), (512,), (1024,)),
                'aspect_ratios': (1.,),
                'dcn_tower': False,
                'loss_gamma': 2.,
                'loss_alpha': 0.25,
                'match_iou_thre': 0.1,
                'iou_loss_weight': 0.5,
                'topk': 9,
                'reg_loss_weight': 1.3,
                'test_score_thre': 0.05,
                'test_score_voting': True,
                'pre_nms_topk': 1000,
                'nms_thre': 0.6,
                'resume': None,
                'val_iter': 2000,
                'base_lr': 0.01 / 8,
                'weight_decay': 0.0001,
                'momentum': 0.9,
                'bias_lr_factor': 2,
                'bias_weight_decay': 0,
                'max_iter': 90000,
                'steps': (60000, 80000),
                'warmup_factor': 1 / 3,
                'warmup_iters': 500,
                'max_detections': 100,
                'min_size_train': (800,),
                'min_size_range_train': (-1, -1),
                'max_size_train': 1333,
                'min_size_test': 800,
                'max_size_test': 1333,
                }


# res50_15x_cfg = res50_1x_cfg.copy()
# res50_15x_cfg.update(None)
#
# res101_2x_cfg = res50_1x_cfg.copy()
# res101_2x_cfg.update(None)
#
# res101_dcn_2x_cfg = res50_1x_cfg.copy()
# res101_dcn_2x_cfg.update(None)


def update_config(args=None):
    res50_1x_cfg.update(vars(args))
    return dict2class(res50_1x_cfg)
