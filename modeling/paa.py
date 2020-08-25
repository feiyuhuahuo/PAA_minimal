import torch.nn as nn
from utils.boxlist_ops import to_image_list
from collections import OrderedDict
from modeling import fpn as fpn_module
from modeling import resnet
from modeling.utils import PAAPostProcessor
import pdb
import math
import torch
from modeling.loss import PAALossComputation
from modeling.anchor_generator import AnchorGenerator


class Scale(nn.Module):
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


class PAAModule(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.head = PAAHead(cfg)
        self.loss_evaluator = PAALossComputation(cfg)
        self.post_process = PAAPostProcessor(pre_nms_thresh=cfg.test_score_thre,
                                             pre_nms_top_n=cfg.pre_nms_topk,
                                             nms_thresh=cfg.nms_thre,
                                             fpn_post_nms_top_n=cfg.max_detections,
                                             min_size=0,
                                             num_classes=cfg.num_classes,
                                             score_voting=cfg.test_score_voting)

        self.anchor_generator = AnchorGenerator(cfg.anchor_sizes, cfg.aspect_ratios, cfg.anchor_strides)
        self.fpn_strides = cfg.anchor_strides

    def forward(self, images, features, targets=None):
        c_pred, box_pred, iou_pred = self.head(features)

        anchors = self.anchor_generator(images, features)

        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(h, w, self.fpn_strides[level], feature.device)
            locations.append(locations_per_level)

        if self.training:
            losses = self.loss_evaluator(c_pred, box_pred, iou_pred, targets, anchors, locations)
            losses_dict = {"loss_cls": losses[0], "loss_reg": losses[1], 'loss_iou_pred': losses[2]}
            return None, losses_dict
        else:
            boxes = self.post_process(c_pred, box_pred, iou_pred, anchors)
            return boxes, None

    def compute_locations_per_level(self, h, w, stride, device):
        shifts_x = torch.arange(0, w * stride, step=stride, dtype=torch.float32, device=device)
        shifts_y = torch.arange(0, h * stride, step=stride, dtype=torch.float32, device=device)
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations


class PAA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        body = resnet.ResNet(cfg)
        fpn = fpn_module.FPN(in_channels_list=[0, 512, 1024, 2048], out_channels=256)
        self.backbone = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
        self.rpn = PAAModule(cfg)

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        images = to_image_list(images)
        features = self.backbone(images.tensors)

        proposals, proposal_losses = self.rpn(images, features, targets)

        if self.training:
            return proposal_losses

        return proposals
