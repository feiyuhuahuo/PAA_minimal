#!/usr/bin/env python 
# -*- coding:utf-8 -*-
from collections import OrderedDict
import torch
import pdb

# aa = torch.load('weights/paa_res101.pth')
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

aa = ['a', 2, 3, 'e', 10]
for i, a in enumerate(aa, 3):
    print(i, a)
