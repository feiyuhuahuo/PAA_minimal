import torch
import math
import pdb


def cat(tensors, dim=0):
    # Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def permute_and_flatten(layer, N, A, C, H, W):
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)
    layer = layer.reshape(N, -1, C)
    return layer


def concat_fpn_pred(c_pred, box_pred, iou_pred, anchor_cat):
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
    anchor_flatten = torch.cat([anchor_cat.box] * bs, dim=0)

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


class ProgressBar:
    def __init__(self, length, max_val):
        self.max_val = max_val
        self.length = length
        self.cur_val = 0

        self.cur_num_bars = -1
        self.update_str()

    def update_str(self):
        num_bars = int(self.length * (self.cur_val / self.max_val))

        if num_bars != self.cur_num_bars:
            self.cur_num_bars = num_bars
            self.string = '█' * num_bars + '░' * (self.length - num_bars)

    def get_bar(self, new_val):
        self.cur_val = new_val

        if self.cur_val > self.max_val:
            self.cur_val = self.max_val
        self.update_str()
        return self.string
