import argparse
import datetime
import time
import torch
import torch.distributed as dist
from config import get_config
from data.data_loader import make_data_loader
from model.paa import PAA
from val import inference
from utils.checkpoint import Checkpointer
from utils.utils import Optimizer
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import timer
import pdb

parser = argparse.ArgumentParser(description='PAA_Minimal Training')
parser.add_argument('--local_rank', type=int)
parser.add_argument('--cfg', type=str, default='res50_1x', help='(res50_1x, res50_15x, res101_2x, res101_dcn_2x)')
parser.add_argument('--train_bs', type=int, default=4, help='total training batch size')
parser.add_argument('--test_bs', type=int, default=1, help='-1 to disable val')
parser.add_argument('--resume', type=str, default=None, help='the weight for resume training')
parser.add_argument('--val_interval', type=int, default=4000, help='validation interval during training')
parser.add_argument('--val_num', default=-1, type=int, help='Number of validation images, -1 for all.')
parser.add_argument('--score_voting', action='store_true', default=False, help='Using score voting.')
parser.add_argument('--improved_coco', action='store_true', default=False, help='Improved COCO-EVAL written by myself.')

args = parser.parse_args()
cfg = get_config(args)

model = PAA(cfg)
model.train().cuda()  # broadcast_buffers is True if BN is used
model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=False)

# if cfg.MODEL.USE_SYNCBN:  # TODO: figure this out
#     model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

optim = Optimizer(model, cfg)
checkpointer = Checkpointer(cfg, model.module, optim.optimizer)
start_iter = int(cfg.resume.split('_')[-1].split('.')[0]) if cfg.resume else 0
data_loader = make_data_loader(cfg, start_iter=start_iter)
max_iter = len(data_loader) - 1
timer.reset()
main_gpu = dist.get_rank() == 0
num_gpu = dist.get_world_size()

for i, (img_list_batch, box_list_batch) in enumerate(data_loader, start_iter):
    if main_gpu and i == start_iter + 1:
        timer.start()

    optim.update_lr(step=i)

    img_tensor_batch = torch.stack([aa.img for aa in img_list_batch], dim=0).cuda()
    for box_list in box_list_batch:
        box_list.to_cuda()

    with timer.counter('for+loss'):
        category_loss, box_loss, iou_loss = model(img_tensor_batch, box_list_batch)
        all_loss = torch.stack([category_loss, box_loss, iou_loss], dim=0)
        dist.reduce(all_loss, dst=0)

        if main_gpu:  # get the mean loss across all GPUS
            l_c = all_loss[0].item() / num_gpu  # seems when printing, need to call .item(), not sure
            l_b = all_loss[1].item() / num_gpu
            l_iou = all_loss[2].item() / num_gpu

    with timer.counter('backward'):
        losses = category_loss + box_loss + iou_loss
        optim.optimizer.zero_grad()
        losses.backward()

    with timer.counter('update'):
        optim.optimizer.step()

    time_this = time.time()
    if i > start_iter:
        batch_time = time_this - time_last
        timer.add_batch_time(batch_time)
    time_last = time_this

    if i > start_iter and i % 20 == 0 and main_gpu:
        cur_lr = optim.optimizer.param_groups[0]['lr']
        time_name = ['batch', 'data', 'for+loss', 'backward', 'update']
        t_t, t_d, t_fl, t_b, t_u = timer.get_times(time_name)
        seconds = (max_iter - i) * t_t
        eta = str(datetime.timedelta(seconds=seconds)).split('.')[0]

        print(f'step: {i} | lr: {cur_lr:.2e} | l_class: {l_c:.3f} | l_box: {l_b:.3f} | l_iou: {l_iou:.3f} | '
              f't_t: {t_t:.3f} | t_d: {t_d:.3f} | t_fl: {t_fl:.3f} | t_b: {t_b:.3f} | t_u: {t_u:.3f} | ETA: {eta}')

    if main_gpu and (i > start_iter and i % cfg.val_interval == 0 or i == max_iter):  # pay attention to the logic here
        checkpointer.save(cur_iter=i)
        inference(model.module, cfg, during_training=True)
        model.train()
        timer.reset()  # training time and val time share the same Obj, so reset it to avoid confusion

    if main_gpu and i != 1 and i % cfg.val_interval == 1:
        timer.start()  # the first iter after validation should not be included
