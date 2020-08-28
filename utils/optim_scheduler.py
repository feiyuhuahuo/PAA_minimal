import pickle
import torch
import pdb
from bisect import bisect_right


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, cfg):
        self.steps = cfg.decay_steps
        self.warmup_factor = cfg.warmup_factor
        self.warmup_iters = cfg.warmup_iters
        self.gamma = 0.1
        super().__init__(optimizer, last_epoch=-1)

    def get_lr(self):
        lrs = []
        for lr in self.base_lrs:
            new_lr = lr * self.warmup_factor * self.gamma ** bisect_right(self.steps, self.last_epoch)
            lrs.append(new_lr)

        return lrs


def make_optimizer(cfg, model):
    params = []
    bias_lr_factor = 2  # implicit hyper-parameters
    bias_weight_decay = 0

    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue

        lr, weight_decay = cfg.base_lr, cfg.weight_decay

        if "bias" in key:
            lr *= bias_lr_factor
            weight_decay = bias_weight_decay

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    return torch.optim.SGD(params, lr, momentum=cfg.momentum)
