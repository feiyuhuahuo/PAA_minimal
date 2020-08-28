import torch
import pdb
import math
import torch.nn as nn
from utils.bounding_box import BoxList
from utils.boxlist_ops import cat_boxlist, boxlist_ml_nms, boxlist_iou


def get_group_gn(dim, dim_per_gp, num_groups):
    """get number of groups used by GroupNorm, based on number of channels."""
    assert dim_per_gp == -1 or num_groups == -1, \
        "GroupNorm: can only specify G or C/G."

    if dim_per_gp > 0:
        assert dim % dim_per_gp == 0, \
            "dim: {}, dim_per_gp: {}".format(dim, dim_per_gp)
        group_gn = dim // dim_per_gp
    else:
        assert dim % num_groups == 0, \
            "dim: {}, num_groups: {}".format(dim, num_groups)
        group_gn = num_groups

    return group_gn


def group_norm(out_channels, affine=True, divisor=1):
    out_channels = out_channels // divisor
    dim_per_gp = cfg.MODEL.GROUP_NORM.DIM_PER_GP // divisor
    num_groups = cfg.MODEL.GROUP_NORM.NUM_GROUPS // divisor
    eps = cfg.MODEL.GROUP_NORM.EPSILON  # default: 1e-5
    return torch.nn.GroupNorm(get_group_gn(out_channels, dim_per_gp, num_groups),
                              out_channels,
                              eps,
                              affine)


def cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def permute_and_flatten(layer, N, A, C, H, W):
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)
    layer = layer.reshape(N, -1, C)
    return layer


def concat_fpn_pred(c_pred, box_pred, iou_pred, anchors):
    bs = c_pred[0].shape[0]
    c_all_level, box_all_level = [], []

    for c_per_level, box_per_level in zip(c_pred, box_pred):
        N, AxC, H, W = c_per_level.shape
        Ax4 = box_per_level.shape[1]
        A = Ax4 // 4
        C = AxC // A
        c_per_level = permute_and_flatten(c_per_level, N, A, C, H, W)
        box_per_level = permute_and_flatten(box_per_level, N, A, 4, H, W)

        c_all_level.append(c_per_level)
        box_all_level.append(box_per_level)

    c_flatten = cat(c_all_level, dim=1).reshape(-1, C)
    box_flatten = cat(box_all_level, dim=1).reshape(-1, 4)

    iou_pred_flatten = [aa.permute(0, 2, 3, 1).reshape(bs, -1, 1) for aa in iou_pred]
    iou_pred_flatten = torch.cat(iou_pred_flatten, dim=1).reshape(-1)

    anchor_flatten = torch.cat([cat_boxlist(anchor_per_img).bbox for anchor_per_img in anchors], dim=0)

    return c_flatten, box_flatten, iou_pred_flatten, anchor_flatten


def encode(gt_boxes, anchors):
    TO_REMOVE = 1  # TODO remove
    ex_widths = anchors[:, 2] - anchors[:, 0] + TO_REMOVE
    ex_heights = anchors[:, 3] - anchors[:, 1] + TO_REMOVE
    ex_ctr_x = (anchors[:, 2] + anchors[:, 0]) / 2
    ex_ctr_y = (anchors[:, 3] + anchors[:, 1]) / 2

    gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0] + TO_REMOVE
    gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1] + TO_REMOVE
    gt_ctr_x = (gt_boxes[:, 2] + gt_boxes[:, 0]) / 2
    gt_ctr_y = (gt_boxes[:, 3] + gt_boxes[:, 1]) / 2

    wx, wy, ww, wh = (10., 10., 5., 5.)
    targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = ww * torch.log(gt_widths / ex_widths)
    targets_dh = wh * torch.log(gt_heights / ex_heights)

    return torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)


def decode(preds, anchors):
    anchors = anchors.to(preds.dtype)

    TO_REMOVE = 1  # TODO remove
    widths = anchors[:, 2] - anchors[:, 0] + TO_REMOVE
    heights = anchors[:, 3] - anchors[:, 1] + TO_REMOVE
    ctr_x = (anchors[:, 2] + anchors[:, 0]) / 2
    ctr_y = (anchors[:, 3] + anchors[:, 1]) / 2

    wx, wy, ww, wh = (10., 10., 5., 5.)
    dx = preds[:, 0::4] / wx
    dy = preds[:, 1::4] / wy
    dw = preds[:, 2::4] / ww
    dh = preds[:, 3::4] / wh

    # Prevent sending too large values into torch.exp()
    dw = torch.clamp(dw, max=math.log(1000. / 16))
    dh = torch.clamp(dh, max=math.log(1000. / 16))

    pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
    pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
    pred_w = torch.exp(dw) * widths[:, None]
    pred_h = torch.exp(dh) * heights[:, None]

    pred_boxes = torch.zeros_like(preds)
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * (pred_w - 1)
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * (pred_h - 1)
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * (pred_w - 1)
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * (pred_h - 1)
    return pred_boxes


class PAAPostProcessor:
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    @staticmethod
    def remove_small_boxes(boxlist, min_size):
        # Only keep boxes with both sides >= min_size
        # TODO maybe add an API for querying the ws / hs
        xywh_boxes = boxlist.convert("xywh").bbox
        _, _, ws, hs = xywh_boxes.unbind(dim=1)
        keep = ((ws >= min_size) & (hs >= min_size)).nonzero().squeeze(1)
        return boxlist[keep]

    def __call__(self, c_batch, box_batch, iou_batch, anchor_batch):
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
            candi_i_fpn = c_fpn > self.cfg.nms_score_thre  # TODO: if use score_fpn to do score threshold?
            nms_topk_fpn = candi_i_fpn.reshape(N, -1).sum(dim=1)
            nms_topk_fpn = nms_topk_fpn.clamp(max=self.cfg.nms_topk)

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
                boxlist = self.remove_small_boxes(boxlist, min_size=0)
                results.append(boxlist)

            total_boxes.append(results)

        box_list = list(zip(*total_boxes))  # bind together the fpn box_lists which belong to the same batch
        box_list = [cat_boxlist(boxlist) for boxlist in box_list]

        return self.select_over_all_levels(box_list)

    def select_over_all_levels(self, box_list):
        results = []
        for i in range(len(box_list)):
            result = boxlist_ml_nms(box_list[i], self.cfg.nms_iou_thre)  # multi-class nms
            num_detections = len(result)

            # Limit to max_per_image detections    **over all classes**
            if num_detections > self.cfg.max_detections > 0:
                cls_scores = result.get_field("scores")
                image_thre, _ = torch.kthvalue(cls_scores.cpu(), num_detections - self.cfg.max_detections + 1)
                keep = cls_scores >= image_thre.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]

            if self.cfg.test_score_voting:
                boxes_al = box_list[i].bbox
                boxlist = box_list[i]
                labels = box_list[i].get_field("labels")
                scores = box_list[i].get_field("scores")
                sigma = 0.025
                result_labels = result.get_field("labels")
                for j in range(1, self.cfg.num_classes):
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


def match(iou_matrix, high_thre, low_thre):
    if iou_matrix.numel() == 0:
        # empty targets or proposals not supported during training
        if iou_matrix.shape[0] == 0:
            raise ValueError('No ground-truth boxes available for one of the images')
        else:
            raise ValueError('No proposal boxes available for one of the images')

    # find max IoU gt for each anchor
    matched_vals, match_i = iou_matrix.max(dim=0)
    match_i_clone = match_i.clone()

    # Assign candidate match_i with low quality to negative (unassigned) values
    below_low_thre = matched_vals < low_thre
    between_thre = (matched_vals >= low_thre) & (matched_vals < high_thre)
    match_i[below_low_thre] = -1
    match_i[between_thre] = -2

    # For each gt, find the prediction with which it has the highest IoU
    max_dt_per_gt, cc = iou_matrix.max(dim=1)
    # Find highest quality match available, even if it is low
    dt_index_per_gt = torch.nonzero(iou_matrix == max_dt_per_gt[:, None])

    index_to_update = dt_index_per_gt[:, 1]
    match_i[index_to_update] = match_i_clone[index_to_update]

    return match_i
