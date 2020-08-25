import argparse
import os
import torch
import datetime
import time
import numpy as np
from config import update_config
from data.data_loader import make_data_loader
from modeling.paa import PAA
from val import inference
from utils.checkpoint import Checkpointer
from utils.train_utils import WarmupMultiStepLR, make_optimizer
from utils import timer
import pdb

parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
parser.add_argument("--model", default='paa_res5000.pth')
args = parser.parse_args()
cfg = update_config(args)

model = PAA(cfg).cuda()
model.train()

# if cfg.MODEL.USE_SYNCBN:  # TODO: figure this out
#     model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

np.random.seed(10)
torch.manual_seed(10)
torch.cuda.manual_seed_all(10)

optimizer = make_optimizer(cfg, model)
scheduler = WarmupMultiStepLR(optimizer, cfg.steps, warmup_factor=cfg.warmup_factor, warmup_iters=cfg.warmup_iters)
checkpointer = Checkpointer(cfg, model, optimizer, scheduler)
ckpt_iter = checkpointer.load()

data_loader = make_data_loader(cfg, training=True, start_iter=ckpt_iter)

max_iter = len(data_loader)
timer.init()
for i, (images, targets, _) in enumerate(data_loader, ckpt_iter):
    if i > 0:
        timer.start()

    images = images.to(torch.device("cuda"))
    targets = [target.to(torch.device("cuda")) for target in targets]

    with timer.counter('for+loss'):
        loss_dict = model(images, targets)

    with timer.counter('backward'):
        losses = sum(loss for loss in loss_dict.values())
        losses_reduced = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()

    with timer.counter('update'):
        optimizer.step()
        scheduler.step()  # in pytorch >= 1.1.0, scheduler.step() should be run after optimizer.step()

    time_this = time.perf_counter()
    if i > ckpt_iter :
        batch_time = time_this - time_last
        timer.add_batch_time(batch_time)
    time_last = time_this

    if i > ckpt_iter and i % 4 == 0:
        cur_lr = optimizer.param_groups[0]['lr']
        time_name = ['batch', 'data', 'for+loss', 'backward', 'update']
        t_t, t_d, t_fl, t_b, t_u = timer.get_times(time_name)
        seconds = (max_iter - i) * t_t
        eta = str(datetime.timedelta(seconds=seconds)).split('.')[0]
        # seems when printing, need to call .item(), not sure
        l_c, l_b, l_iou = loss_dict['loss_cls'].item(), loss_dict['loss_reg'].item(), loss_dict['loss_iou_pred'].item()
        print(f'step: {i} | lr: {cur_lr:.2e} | l_class: {l_c:.3f} | l_box: {l_b:.3f} | l_iou: {l_iou:.3f} | '
              f't_t: {t_t:.3f} | t_d: {t_d:.3f} | t_fl: {t_fl:.3f} | t_b: {t_b:.3f} | t_u: {t_u:.3f} | ETA: {eta}')

    if i > ckpt_iter and i % cfg.val_iter == 0 or i == max_iter:
        checkpointer.save(cur_iter=i)
        inference(model, cfg, training=True)
