import torch
import pdb
import math
import torch.nn as nn
from utils.bounding_box import BoxList
from utils.boxlist_ops import cat_boxlist, boxlist_ml_nms, remove_small_boxes, boxlist_iou


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


class PAAPostProcessor(torch.nn.Module):
    def __init__(self, pre_nms_thresh, pre_nms_top_n, nms_thresh, fpn_post_nms_top_n,
                 min_size, num_classes, score_voting=True):
        super().__init__()
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.min_size = min_size
        self.num_classes = num_classes
        self.score_voting = score_voting

    def forward_for_single_feature_map(self, box_cls, box_regression, iou_pred, anchors):
        N, _, H, W = box_cls.shape
        A = box_regression.size(1) // 4
        C = box_cls.size(1) // A

        # put in the same format as anchors
        box_cls = permute_and_flatten(box_cls, N, A, C, H, W)
        box_cls = box_cls.sigmoid()

        box_regression = permute_and_flatten(box_regression, N, A, 4, H, W)
        box_regression = box_regression.reshape(N, -1, 4)

        candidate_inds = box_cls > self.pre_nms_thresh
        pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)

        # multiply the classification scores with IoU scores
        if iou_pred is not None:
            iou_pred = permute_and_flatten(iou_pred, N, A, 1, H, W)
            iou_pred = iou_pred.reshape(N, -1).sigmoid()
            box_cls = (box_cls * iou_pred[:, :, None]).sqrt()

        results = []
        for per_box_cls_, per_box_regression, per_pre_nms_top_n, per_candidate_inds, per_anchors \
                in zip(box_cls, box_regression, pre_nms_top_n, candidate_inds, anchors):
            per_box_cls = per_box_cls_[per_candidate_inds]

            per_box_cls, top_k_indices = per_box_cls.topk(per_pre_nms_top_n, sorted=False)

            per_candidate_nonzeros = per_candidate_inds.nonzero()[top_k_indices, :]

            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1] + 1

            detections = decode(per_box_regression[per_box_loc, :].view(-1, 4),
                                per_anchors.bbox[per_box_loc, :].view(-1, 4))
            boxlist = BoxList(detections, per_anchors.size, mode="xyxy")
            boxlist.add_field("labels", per_class)
            boxlist.add_field("scores", per_box_cls)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = remove_small_boxes(boxlist, self.min_size)
            results.append(boxlist)

        return results

    def forward(self, box_cls, box_regression, iou_pred, anchors):
        sampled_boxes = []
        anchors = list(zip(*anchors))
        if iou_pred is None:
            iou_pred = [None] * len(box_cls)
        for _, (o, b, i, a) in enumerate(zip(box_cls, box_regression, iou_pred, anchors)):
            sampled_boxes.append(self.forward_for_single_feature_map(o, b, i, a))

        boxlists = list(zip(*sampled_boxes))
        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]
        boxlists = self.select_over_all_levels(boxlists)

        return boxlists

    # TODO very similar to filter_results from PostProcessor
    # but filter_results is per image
    # TODO Yang: solve this issue in the future. No good solution
    # right now.
    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            # multiclass nms
            result = boxlist_ml_nms(boxlists[i], self.nms_thresh)
            number_of_detections = len(result)

            # Limit to max_per_image detections **over all classes**
            if number_of_detections > self.fpn_post_nms_top_n > 0:
                cls_scores = result.get_field("scores")
                image_thresh, _ = torch.kthvalue(cls_scores.cpu(), number_of_detections - self.fpn_post_nms_top_n + 1)
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            if self.score_voting:
                boxes_al = boxlists[i].bbox
                boxlist = boxlists[i]
                labels = boxlists[i].get_field("labels")
                scores = boxlists[i].get_field("scores")
                sigma = 0.025
                result_labels = result.get_field("labels")
                for j in range(1, self.num_classes):
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
