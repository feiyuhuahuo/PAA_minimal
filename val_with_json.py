#!/usr/bin/env python 
# -*- coding:utf-8 -*-
from pycocotools.coco import COCO
from my_cocoeval.cocoeval import SelfEval
coco = COCO('/home/feiyu/Data/coco2017/annotations/instances_val2017.json')

coco_dt = coco.loadRes('results/thre_03/faster_rcnn_r101.json')
bbox_eval = SelfEval(coco, coco_dt, all_points=True)
bbox_eval.evaluate()
bbox_eval.accumulate()
bbox_eval.summarize()
