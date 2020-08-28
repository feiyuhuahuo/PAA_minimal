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
                'train_bs': 2,
                'test_bs': 1,
                'num_classes': 81,
                'backbone': 'res50',
                'stage_with_dcn': (False, False, False, False),
                'dcn_tower': False,

                'anchor_strides': (8, 16, 32, 64, 128),
                'anchor_sizes': ((64,), (128,), (256,), (512,), (1024,)),
                'aspect_ratios': (1.,),

                'min_size_train': (800,),
                'min_size_range_train': (-1, -1),
                'max_size_train': 1333,

                'iou_loss_w': 0.5,
                'box_loss_w': 1.3,

                'base_lr': 0.01 / 8,  # this is related to train_bs
                'max_iter': 90000,
                'decay_steps': (60000, 80000),

                'val_iter': 2000,
                'resume': None,

                'min_size_test': 800,
                'max_size_test': 1333,
                'nms_topk': 1000,
                'nms_score_thre': 0.05,
                'nms_iou_thre': 0.6,
                'test_score_voting': True,

                # rarely used parameters -----------------------
                'stem_norm': 'frozen_BN',  # 'frozen_BN' or 'GN'
                'bottleneck_norm': 'frozen_BN',
                'freeze_backbone_at': 2,
                'fl_gamma': 2.,  # focal loss gamma, alpha
                'fl_alpha': 0.25,
                'weight_decay': 0.0001,
                'momentum': 0.9,
                'warmup_factor': 1 / 3,
                'warmup_iters': 500,  # TODO: this parameter seems useless
                'fpn_topk': 9,
                'match_iou_thre': 0.1,
                'max_detections': 100}

res50_15x_cfg = res50_1x_cfg.copy()
res50_15x_cfg.update({'max_iter': 135000,
                      'decay_steps': (90000, 120000)})

res101_2x_cfg = res50_1x_cfg.copy()
res101_2x_cfg.update({'backbone': 'res101',
                      'min_size_range_train': (640, 800),
                      'max_iter': 180000,
                      'decay_steps': (120000, 160000)})

res101_dcn_2x_cfg = res50_1x_cfg.copy()
res101_dcn_2x_cfg.update({'backbone': 'res101',
                          'stage_with_dcn': (False, True, True, True),
                          'dcn_tower': True,
                          'min_size_range_train': (640, 800),
                          'max_iter': 180000,
                          'decay_steps': (120000, 160000)})


def update_config(args=None):
    config = res50_1x_cfg  # change the desired config here
    config.update(vars(args))
    return dict2class(config)
