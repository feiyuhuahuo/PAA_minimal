import torch.nn as nn
import math
import torch
from collections import OrderedDict
from modeling import fpn as fpn_module
from modeling import resnet
from modeling.loss import PAALoss
from utils.anchor_generator import AnchorGenerator
from modeling.layers import DFConv2d
import pdb


class Scale(nn.Module):  # TODO: figure out the useage
    def __init__(self, init_value=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class PAAHead(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        num_classes = cfg.num_classes - 1
        num_anchors = len(cfg.aspect_ratios)

        cls_tower, bbox_tower = [], []
        for i in range(4):
            if cfg.dcn_tower and i == 3:
                conv_func = DFConv2d
            else:
                conv_func = nn.Conv2d

            cls_tower.append(conv_func(256, 256, kernel_size=3, stride=1, padding=1, bias=True))
            cls_tower.append(nn.GroupNorm(32, 256))
            cls_tower.append(nn.ReLU())

            bbox_tower.append(conv_func(256, 256, kernel_size=3, stride=1, padding=1, bias=True))
            bbox_tower.append(nn.GroupNorm(32, 256))
            bbox_tower.append(nn.ReLU())

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))

        self.cls_logits = nn.Conv2d(256, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
        self.bbox_pred = nn.Conv2d(256, num_anchors * 4, kernel_size=3, stride=1, padding=1)
        self.iou_pred = nn.Conv2d(256, num_anchors * 1, kernel_size=3, stride=1, padding=1)
        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

        all_modules = [self.cls_tower, self.bbox_tower, self.cls_logits, self.bbox_pred, self.iou_pred]

        # initialization
        for modules in all_modules:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)

        # initialize the bias for focal loss
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(self, x):
        logits, bbox_reg, iou_pred = [], [], []

        for i, feature in enumerate(x):
            cls_tower = self.cls_tower(feature)
            box_tower = self.bbox_tower(feature)

            logits.append(self.cls_logits(cls_tower))
            bbox_reg.append(self.scales[i](self.bbox_pred(box_tower)))
            iou_pred.append(self.iou_pred(box_tower))

        return logits, bbox_reg, iou_pred


class PAA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        body = resnet.ResNet(cfg)
        fpn = fpn_module.FPN(in_channels_list=[0, 512, 1024, 2048], out_channels=256)
        self.backbone = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
        self.head = PAAHead(cfg)
        self.paa_loss = PAALoss(cfg)
        self.anchor_generator = AnchorGenerator(cfg)

    def forward(self, img_tensor_batch, box_list_batch=None):
        features = self.backbone(img_tensor_batch)
        c_pred, box_pred, iou_pred = self.head(features)
        anchors = self.anchor_generator(features)
        self.paa_loss.anchors = anchors

        if self.training:
            return self.paa_loss(c_pred, box_pred, iou_pred, box_list_batch)
        else:
            return c_pred, box_pred, iou_pred, anchors
