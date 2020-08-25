#!/usr/bin/env python 
# -*- coding:utf-8 -*-
from collections import OrderedDict
import torch
import pdb

aa = torch.load('paa_res50.pth')
bb = OrderedDict()
cc = OrderedDict()

for k, v in aa.items():
    if k != 'model':
        bb[k] = v
    else:
        for kk, vv in v.items():
            kk_new = kk.split('module.')[-1]
            cc[kk_new] = vv

        bb[k] = cc

torch.save(bb, 'paa_res5000.pth')
