#!/usr/bin/env python 
# -*- coding:utf-8 -*-
from collections import OrderedDict
import torch
import pdb

# aa = torch.load('weights/paa_res50.pth')
# bb = OrderedDict()
# cc = OrderedDict()
#
# for k, v in aa.items():
#     if k != 'model':
#         bb[k] = v
#     else:
#         for kk, vv in v.items():
#             kk_new = kk.split('rpn.')[-1]
#             cc[kk_new] = vv
#
#         bb[k] = cc
#
# torch.save(bb, 'paa_res10111.pth')

import os
import argparse
import pdb
import torch.distributed

parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
parser.add_argument('--local_rank', type=int)
args = parser.parse_args()

torch.cuda.set_device(args.local_rank)
torch.distributed.init_process_group(backend="nccl", init_method="env://")
print(torch.distributed.get_rank())
