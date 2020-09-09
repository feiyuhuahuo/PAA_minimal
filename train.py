import argparse
import datetime
import time
import torch
import tensorboardX
import numpy as np
import torch.distributed as dist
from config import get_config
from data.data_loader import make_data_loader
from modeling.paa import PAA
from val import inference
from utils.checkpoint import Checkpointer
from utils.optim_scheduler import WarmupMultiStepLR, make_optimizer
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import timer
import pdb

parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
parser.add_argument('--local_rank', type=int)
parser.add_argument("--train_bs", type=int, default=4, help='total training batch size')
parser.add_argument("--test_bs", type=int, default=1, help='-1 to disable val')
args = parser.parse_args()
cfg = get_config(args)

model = PAA(cfg)
model.train().cuda()  # True if BN is used
model = DDP(model, device_ids=[cfg.local_rank], output_device=cfg.local_rank, broadcast_buffers=False)

# if cfg.MODEL.USE_SYNCBN:  # TODO: figure this out
#     model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

# np.random.seed(10)
# torch.manual_seed(10)
# torch.cuda.manual_seed_all(10)

optimizer = make_optimizer(cfg, model)
scheduler = WarmupMultiStepLR(optimizer, cfg)  # TODO: figure out the lr principle， maybe need to modify
checkpointer = Checkpointer(cfg, model, optimizer, scheduler)
ckpt_iter = checkpointer.ckpt_iter
data_loader = make_data_loader(cfg, start_iter=ckpt_iter)
max_iter = len(data_loader)
timer.init()

for i, (img_list_batch, box_list_batch) in enumerate(data_loader, ckpt_iter):
    if i > 0:
        timer.start()

    img_tensor_batch = torch.stack([aa.img for aa in img_list_batch], dim=0)

    for box_list in box_list_batch:
        box_list.to_cuda()

    with timer.counter('for+loss'):
        category_loss, box_loss, iou_loss = model(img_tensor_batch, box_list_batch)

    with timer.counter('backward'):
        losses = category_loss + box_loss + iou_loss
        optimizer.zero_grad()
        losses.backward()

    with timer.counter('update'):
        optimizer.step()
        scheduler.step()  # in pytorch >= 1.1.0, scheduler.step() should be run after optimizer.step()

    time_this = time.perf_counter()
    if i > ckpt_iter:
        batch_time = time_this - time_last
        timer.add_batch_time(batch_time)
    time_last = time_this

    if i > ckpt_iter and i % 20 == 0:
        cur_lr = optimizer.param_groups[0]['lr']
        time_name = ['batch', 'data', 'for+loss', 'backward', 'update']
        t_t, t_d, t_fl, t_b, t_u = timer.get_times(time_name)
        seconds = (max_iter - i) * t_t
        eta = str(datetime.timedelta(seconds=seconds)).split('.')[0]
        # seems when printing, need to call .item(), not sure
        l_c, l_b, l_iou = category_loss.item(), box_loss.item(), iou_loss.item()

        print(f'step: {i} | lr: {cur_lr:.2e} | l_class: {l_c:.3f} | l_box: {l_b:.3f} | l_iou: {l_iou:.3f} | '
              f't_t: {t_t:.3f} | t_d: {t_d:.3f} | t_fl: {t_fl:.3f} | t_b: {t_b:.3f} | t_u: {t_u:.3f} | ETA: {eta}')

    if i > ckpt_iter and i % cfg.val_interval == 0 or i == max_iter:
        if dist.get_rank() == 0:
            checkpointer.save(cur_iter=i)
            inference(model.module, cfg, during_train=True)
            model.train()
