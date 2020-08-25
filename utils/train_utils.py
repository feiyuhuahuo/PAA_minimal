import pickle
import torch
import logging
from bisect import bisect_right


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, steps, warmup_factor, warmup_iters):
        self.steps = steps
        self.gamma = 0.1
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        super().__init__(optimizer, last_epoch=-1)

    def get_lr(self):
        return [base_lr * self.warmup_factor * self.gamma ** bisect_right(self.steps, self.last_epoch)
                for base_lr in self.base_lrs]


def make_optimizer(cfg, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue

        lr, weight_decay = cfg.base_lr, cfg.weight_decay
        if "bias" in key:
            lr *= cfg.bias_lr_factor
            weight_decay = cfg.bias_weight_decay

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    return torch.optim.SGD(params, lr, momentum=cfg.momentum)
