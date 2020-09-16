import os
import torch
import torch.distributed as dist
import pdb

os.makedirs('results/', exist_ok=True)
os.makedirs('weights/', exist_ok=True)


class res50_1x_cfg:
    def __init__(self, args_attr, val_mode=False):
        data_root = '/home/feiyu/Data/coco2017/'
        self.gpu_id = 0
        if not val_mode:
            self.train_bs = 12
            self.bs_per_gpu = None
        self.test_bs = 1
        if not val_mode:
            self.train_imgs = data_root + 'train2017/'
            self.train_ann = data_root + 'annotations/instances_train2017.json'
        self.val_imgs = data_root + 'val2017/'
        self.val_ann = data_root + 'annotations/instances_val2017.json'
        self.num_classes = 81
        self.backbone = 'res50'
        if not val_mode:
            self.weight = 'weights/R-50.pkl'
            self.resume = None
        self.stage_with_dcn = (False, False, False, False)
        self.dcn_tower = False
        self.anchor_strides = (8, 16, 32, 64, 128)
        self.anchor_sizes = ((64,), (128,), (256,), (512,), (1024,))
        self.aspect_ratios = (1.,)

        if not val_mode:
            self.min_size_train = 800
            self.max_size_train = 1333
            self.val_interval = 4000

            self.box_loss_w = 1.3
            self.iou_loss_w = 0.5

            self.bs_factor = self.train_bs / 16
            self.base_lr = 0.01 * self.bs_factor
            self.max_iter = int(90000 / self.bs_factor)
            self.decay_steps = (int(60000 / self.bs_factor), int(80000 / self.bs_factor))

        self.min_size_test = 800
        self.max_size_test = 1333
        self.nms_topk = 1000
        self.nms_score_thre = 0.05
        self.nms_iou_thre = 0.6
        self.test_score_voting = False

        # rarely used parameters ----------------------------------
        if not val_mode:
            self.fl_gamma = 2.  # focal loss gamma, alpha
            self.fl_alpha = 0.25
            self.weight_decay = 0.0001
            self.momentum = 0.9
            self.warmup_factor = 1 / 3
            self.warmup_iters = 500
        self.freeze_backbone_at = 2
        self.fpn_topk = 9
        self.match_iou_thre = 0.1
        self.max_detections = 100

        self.para_list = list(vars(self).keys())  # for ordered printing

        for k, v in args_attr.items():
            self.__setattr__(k, v)

    def print_cfg(self):
        print()
        print('-' * 30 + self.__class__.__name__ + '-' * 30)
        for k in self.para_list:
            if k not in ('local_rank', 'bs_factor', 'para_list'):
                print(f'{k}: {getattr(self, k)}')
        print()


class res50_15x_cfg(res50_1x_cfg):
    def __init__(self, args_attr, val_mode=False):
        super().__init__(args_attr, val_mode)
        if not val_mode:
            self.max_iter = int(135000 / self.bs_factor)
            self.decay_steps = (int(90000 / self.bs_factor), int(120000 / self.bs_factor))


class res101_2x_cfg(res50_1x_cfg):
    def __init__(self, args_attr, val_mode=False):
        super().__init__(args_attr, val_mode)
        self.backbone = 'res101'
        if not val_mode:
            self.weight = 'weights/R-101.pkl'
            self.min_size_train = (640, 800)
            self.max_iter = int(180000 / self.bs_factor)
            self.decay_steps = (int(120000 / self.bs_factor), int(160000 / self.bs_factor))


class res101_dcn_2x_cfg(res50_1x_cfg):
    def __init__(self, args_attr, val_mode=False):
        super().__init__(args_attr, val_mode)
        self.backbone = 'res101'
        self.stage_with_dcn = (False, True, True, True)
        self.dcn_tower = True
        if not val_mode:
            self.weight = 'weights/R-101.pkl'
            self.min_size_train = (640, 800)
            self.max_iter = int(180000 / self.bs_factor)
            self.decay_steps = (int(120000 / self.bs_factor), int(160000 / self.bs_factor))


def get_config(args, val_mode=False):
    if val_mode:
        assert args.gpu_id.isdigit(), f'Only one GPU can be used in val mode, got {args.gpu_id}.'
    else:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")

        # Only launched by torch.distributed.launch, 'WORLD_SIZE' can be add to environment variables.
        num_gpus = int(os.environ["WORLD_SIZE"])
        assert args.train_bs % num_gpus == 0, 'Training batch size must be divisible by GPU number.'
        args.bs_per_gpu = int(args.train_bs / num_gpus)

        if os.environ.get('CUDA_VISIBLE_DEVICES'):
            args.gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES')

    cfg = res50_1x_cfg(vars(args), val_mode)  # change the desired config here

    if val_mode:
        cfg.print_cfg()
    elif dist.get_rank() == 0:
        cfg.print_cfg()

    return cfg
