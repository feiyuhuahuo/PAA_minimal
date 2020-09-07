#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch
from utils.utils import permute_and_flatten, decode
from utils.box_list import BoxList, cat_boxlist, boxlist_ml_nms, boxlist_iou
import pdb


def select_over_all_levels(cfg, box_list):
    results = []
    for i in range(len(box_list)):
        result = boxlist_ml_nms(box_list[i], cfg.nms_iou_thre)  # multi-class nms
        num_detections = result.box.shape[0]

        # Limit to max_per_image detections    **over all classes**
        if num_detections > cfg.max_detections > 0:
            score = result.score
            image_thre, _ = torch.kthvalue(score.cpu(), num_detections - cfg.max_detections + 1)
            keep = score >= image_thre.item()
            keep = torch.nonzero(keep).squeeze(1)
            result = result[keep]

        if cfg.test_score_voting:
            boxes_al = box_list[i].box
            boxlist = box_list[i]
            labels = box_list[i].label
            scores = box_list[i].score
            sigma = 0.025
            result_labels = result.label

            for j in range(1, cfg.num_classes):
                inds = (labels == j).nonzero().reshape(-1)
                scores_j = scores[inds]
                boxes_j = boxes_al[inds, :].reshape(-1, 4)
                boxlist_for_class = BoxList(boxes_j, boxlist.img_size, mode="x1y1x2y2")
                result_inds = (result_labels == j).nonzero().reshape(-1)
                boxlist_nmsed = result[result_inds]
                ious = boxlist_iou(boxlist_nmsed, boxlist_for_class)

                voted_boxes = []
                for bi in range(boxlist_nmsed.box.shape[0]):
                    cur_ious = ious[bi]
                    pos_inds = (cur_ious > 0.01).nonzero().squeeze(1)
                    pos_ious = cur_ious[pos_inds]
                    pos_boxes = boxlist_for_class.box[pos_inds]
                    pos_scores = scores_j[pos_inds]
                    pis = (torch.exp(-(1 - pos_ious) ** 2 / sigma) * pos_scores).unsqueeze(1)
                    voted_box = torch.sum(pos_boxes * pis, dim=0) / torch.sum(pis, dim=0)
                    voted_boxes.append(voted_box.unsqueeze(0))

                if voted_boxes:
                    result.box[result_inds] = torch.cat(voted_boxes, dim=0)

        results.append(result)

    return results


def post_process(cfg, c_batch, box_batch, iou_batch, anchors, resized_size):
    total_boxes = []

    for c_fpn, box_fpn, iou_fpn, anchor_fpn in zip(c_batch, box_batch, iou_batch, anchors):
        N, _, H, W = c_fpn.shape
        A = box_fpn.size(1) // 4  # 'A' means num_anchors per location
        C = c_fpn.size(1) // A

        c_fpn = permute_and_flatten(c_fpn, N, A, C, H, W)  # shape: (n, num_anchor, 80)
        c_fpn = c_fpn.sigmoid()

        box_fpn = permute_and_flatten(box_fpn, N, A, 4, H, W)  # shape: (n, num_anchor, 4)
        box_fpn = box_fpn.reshape(N, -1, 4)

        iou_fpn = permute_and_flatten(iou_fpn, N, A, 1, H, W)
        iou_fpn = iou_fpn.reshape(N, -1).sigmoid()

        # multiply classification and IoU to get the score
        score_fpn = (c_fpn * iou_fpn[:, :, None]).sqrt()

        # use class score to do the pre-threshold
        candi_i_fpn = c_fpn > cfg.nms_score_thre  # TODO: if use score_fpn to do score threshold?
        nms_topk_fpn = candi_i_fpn.reshape(N, -1).sum(dim=1)
        nms_topk_fpn = nms_topk_fpn.clamp(max=cfg.nms_topk)

        results = []
        for score, box, nms_topk, candi_i, size in zip(score_fpn, box_fpn, nms_topk_fpn, candi_i_fpn, resized_size):
            score = score[candi_i]  # TODO: too much thre is not elegant, too handcrafted
            score, topk_i = score.topk(nms_topk, sorted=False)  # use score to get the topk

            candi_i = candi_i.nonzero()[topk_i, :]

            box_selected = box[candi_i[:, 0], :].reshape(-1, 4)
            anchor_selected = anchor_fpn.box[candi_i[:, 0], :].reshape(-1, 4)

            anchor_selected = anchor_selected.cuda()

            box_decoded = decode(box_selected, anchor_selected)
            boxlist = BoxList(box_decoded, size, mode='x1y1x2y2', label=candi_i[:, 1] + 1, score=score)

            boxlist.clip_to_image(remove_empty=False)
            boxlist.remove_small_box(min_size=0)
            results.append(boxlist)

        total_boxes.append(results)

    box_list_fpn_batch = list(zip(*total_boxes))  # bind together the fpn box_lists which belong to the same batch
    box_list_batch = [cat_boxlist(aa) for aa in box_list_fpn_batch]

    return select_over_all_levels(cfg, box_list_batch)
