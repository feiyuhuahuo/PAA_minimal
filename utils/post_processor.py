#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch
from utils.utils import permute_and_flatten, decode
from utils.box_list import BoxList, cat_boxlist, boxlist_ml_nms, boxlist_iou


def remove_small_boxes(boxlist, min_size):
    # Only keep boxes with both sides >= min_size
    # TODO maybe add an API for querying the ws / hs
    xywh_boxes = boxlist.convert("xywh").bbox
    _, _, ws, hs = xywh_boxes.unbind(dim=1)
    keep = ((ws >= min_size) & (hs >= min_size)).nonzero().squeeze(1)
    return boxlist[keep]


def select_over_all_levels(cfg, box_list):
    results = []
    for i in range(len(box_list)):
        result = boxlist_ml_nms(box_list[i], cfg.nms_iou_thre)  # multi-class nms
        num_detections = len(result)

        # Limit to max_per_image detections    **over all classes**
        if num_detections > cfg.max_detections > 0:
            cls_scores = result.get_field("scores")
            image_thre, _ = torch.kthvalue(cls_scores.cpu(), num_detections - cfg.max_detections + 1)
            keep = cls_scores >= image_thre.item()
            keep = torch.nonzero(keep).squeeze(1)
            result = result[keep]

        if cfg.test_score_voting:
            boxes_al = box_list[i].bbox
            boxlist = box_list[i]
            labels = box_list[i].get_field("labels")
            scores = box_list[i].get_field("scores")
            sigma = 0.025
            result_labels = result.get_field("labels")
            for j in range(1, cfg.num_classes):
                inds = (labels == j).nonzero().view(-1)
                scores_j = scores[inds]
                boxes_j = boxes_al[inds, :].view(-1, 4)
                boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
                result_inds = (result_labels == j).nonzero().view(-1)
                boxlist_for_class_nmsed = result[result_inds]
                ious = boxlist_iou(boxlist_for_class_nmsed, boxlist_for_class)

                voted_boxes = []
                for bi in range(len(boxlist_for_class_nmsed)):
                    cur_ious = ious[bi]
                    pos_inds = (cur_ious > 0.01).nonzero().squeeze(1)
                    pos_ious = cur_ious[pos_inds]
                    pos_boxes = boxlist_for_class.bbox[pos_inds]
                    pos_scores = scores_j[pos_inds]
                    pis = (torch.exp(-(1 - pos_ious) ** 2 / sigma) * pos_scores).unsqueeze(1)
                    voted_box = torch.sum(pos_boxes * pis, dim=0) / torch.sum(pis, dim=0)
                    voted_boxes.append(voted_box.unsqueeze(0))

                if voted_boxes:
                    voted_boxes = torch.cat(voted_boxes, dim=0)
                    boxlist_for_class_nmsed_ = BoxList(voted_boxes,
                                                       boxlist_for_class_nmsed.size,
                                                       mode="xyxy")
                    boxlist_for_class_nmsed_.add_field("scores",
                                                       boxlist_for_class_nmsed.get_field('scores'))
                    result.bbox[result_inds] = boxlist_for_class_nmsed_.bbox

        results.append(result)

    return results


def post_process(cfg, c_batch, box_batch, iou_batch, anchor_batch):
    total_boxes = []
    anchor_batch = list(zip(*anchor_batch))

    for c_fpn, box_fpn, iou_fpn, anchor_fpn in zip(c_batch, box_batch, iou_batch, anchor_batch):
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
        for score, box, nms_topk, candi_i, anchor in zip(score_fpn, box_fpn, nms_topk_fpn, candi_i_fpn, anchor_fpn):
            score = score[candi_i]  # TODO: too much thre is not elegant, too handcrafted
            score, topk_i = score.topk(nms_topk, sorted=False)  # use score to get the topk

            candi_i = candi_i.nonzero()[topk_i, :]

            box_selected = box[candi_i[:, 0], :].reshape(-1, 4)
            anchor_selected = anchor.bbox[candi_i[:, 0], :].reshape(-1, 4)

            box_decoded = decode(box_selected, anchor_selected)
            boxlist = BoxList(box_decoded, anchor.size, mode="xyxy")
            boxlist.add_field("labels", candi_i[:, 1] + 1)
            boxlist.add_field("scores", score)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = remove_small_boxes(boxlist, min_size=0)
            results.append(boxlist)

        total_boxes.append(results)

    box_list = list(zip(*total_boxes))  # bind together the fpn box_lists which belong to the same batch
    box_list = [cat_boxlist(boxlist) for boxlist in box_list]

    return select_over_all_levels(cfg, box_list)
