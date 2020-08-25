import torch
import pdb
from torch import nn
from modeling.utils import concat_box_prediction_layers, encode, decode
from modeling.sigmoid_focal_loss import focal_loss_cuda, focal_loss_cpu
from modeling.matcher import Matcher
from utils.boxlist_ops import boxlist_iou, cat_boxlist
import sklearn.mixture as skm


class PAALossComputation:
    def __init__(self, cfg):
        self.cfg = cfg
        # self.c_loss_func = SigmoidFocalLoss(cfg.loss_gamma, cfg.loss_alpha)
        self.iou_bce_loss = nn.BCEWithLogitsLoss(reduction="sum")
        self.matcher = Matcher(cfg.matcher_iou_thre, cfg.matcher_iou_thre, True)

    @staticmethod
    def GIoULoss(pred, target, anchor, weight=None):
        pred_boxes = decode(pred.view(-1, 4), anchor.view(-1, 4))
        pred_x1 = pred_boxes[:, 0]
        pred_y1 = pred_boxes[:, 1]
        pred_x2 = pred_boxes[:, 2]
        pred_y2 = pred_boxes[:, 3]
        pred_x2 = torch.max(pred_x1, pred_x2)
        pred_y2 = torch.max(pred_y1, pred_y2)
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)

        gt_boxes = decode(target.view(-1, 4), anchor.view(-1, 4))
        target_x1 = gt_boxes[:, 0]
        target_y1 = gt_boxes[:, 1]
        target_x2 = gt_boxes[:, 2]
        target_y2 = gt_boxes[:, 3]
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)

        x1_intersect = torch.max(pred_x1, target_x1)
        y1_intersect = torch.max(pred_y1, target_y1)
        x2_intersect = torch.min(pred_x2, target_x2)
        y2_intersect = torch.min(pred_y2, target_y2)
        area_intersect = torch.zeros(pred_x1.size()).to(pred)
        mask = (y2_intersect > y1_intersect) * (x2_intersect > x1_intersect)
        area_intersect[mask] = (x2_intersect[mask] - x1_intersect[mask]) * (y2_intersect[mask] - y1_intersect[mask])

        x1_enclosing = torch.min(pred_x1, target_x1)
        y1_enclosing = torch.min(pred_y1, target_y1)
        x2_enclosing = torch.max(pred_x2, target_x2)
        y2_enclosing = torch.max(pred_y2, target_y2)
        area_enclosing = (x2_enclosing - x1_enclosing) * (y2_enclosing - y1_enclosing) + 1e-7

        area_union = pred_area + target_area - area_intersect + 1e-7
        ious = area_intersect / area_union
        gious = ious - (area_enclosing - area_union) / area_enclosing

        losses = 1 - gious

        if weight is not None and weight.sum() > 0:
            return losses * weight
        else:
            assert losses.numel() != 0
            return losses

    def prepare_iou_based_targets(self, targets, anchors):
        cls_labels, reg_targets, matched_idx_all = [], [], []

        for i in range(len(targets)):
            targets_per_im = targets[i]
            assert targets_per_im.mode == "xyxy"
            anchors_per_im = cat_boxlist(anchors[i])

            iou_matrix = boxlist_iou(targets_per_im, anchors_per_im)

            matched_idxs = self.matcher(iou_matrix)
            targets_per_im = targets_per_im.copy_with_fields(['labels'])
            matched_targets = targets_per_im[matched_idxs.clamp(min=0)]

            category = matched_targets.get_field("labels").to(dtype=torch.float32)

            # Background (negative examples)
            bg_indices = matched_idxs == -1
            category[bg_indices] = 0

            # discard indices that are between thresholds, "useless"
            inds_to_discard = matched_idxs == -2
            category[inds_to_discard] = -1

            matched_idx_all.append(matched_idxs.view(1, -1))

            offset = encode(matched_targets.bbox, anchors_per_im.bbox)
            cls_labels.append(category)
            reg_targets.append(offset)

        return cls_labels, reg_targets, matched_idx_all

    def compute_paa(self, targets, anchors, labels_all, loss_all, matched_idx_all):
        """
        Args:
            targets (batch_size): list of BoxLists for GT bboxes
            anchors (batch_size, feature_lvls): anchor boxes per feature level
            labels_all (batch_size x num_anchors): assigned labels
            loss_all (batch_size x numa_nchors): calculated loss
            matched_idx_all (batch_size x numa_nchors): best-matched GG bbox indexes
        """
        device = loss_all.device
        cls_labels = []
        reg_targets = []
        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            assert targets_per_im.mode == "xyxy"
            bboxes_per_im = targets_per_im.bbox
            labels_per_im = targets_per_im.get_field("labels")
            anchors_per_im = cat_boxlist(anchors[im_i])
            labels_all_per_im = labels_all[im_i]
            loss_all_per_im = loss_all[im_i]
            matched_idx_all_per_im = matched_idx_all[im_i]
            assert labels_all_per_im.shape == matched_idx_all_per_im.shape

            num_anchors_per_level = [len(anchors_per_level.bbox) for anchors_per_level in anchors[im_i]]

            # select candidates based on IoUs between anchors and GTs
            candidate_idxs = []
            num_gt = bboxes_per_im.shape[0]
            for gt in range(num_gt):
                candidate_idxs_per_gt = []
                star_idx = 0
                for level, anchors_per_level in enumerate(anchors[im_i]):
                    end_idx = star_idx + num_anchors_per_level[level]
                    loss_per_level = loss_all_per_im[star_idx:end_idx]
                    labels_per_level = labels_all_per_im[star_idx:end_idx]
                    matched_idx_per_level = matched_idx_all_per_im[star_idx:end_idx]
                    match_idx = ((matched_idx_per_level == gt) & (labels_per_level > 0)).nonzero()[:, 0]

                    if match_idx.numel() > 0:
                        _, topk_idxs = loss_per_level[match_idx].topk(
                            min(match_idx.numel(), self.cfg.topk), largest=False)
                        topk_idxs_per_level_per_gt = match_idx[topk_idxs]
                        candidate_idxs_per_gt.append(topk_idxs_per_level_per_gt + star_idx)

                    star_idx = end_idx

                if candidate_idxs_per_gt:
                    candidate_idxs.append(torch.cat(candidate_idxs_per_gt))
                else:
                    candidate_idxs.append(None)

            # fit 2-mode GMM per GT box
            n_labels = anchors_per_im.bbox.shape[0]
            cls_labels_per_im = torch.zeros(n_labels, dtype=torch.long).to(device)
            matched_gts = torch.zeros_like(anchors_per_im.bbox)
            fg_inds = matched_idx_all_per_im >= 0
            matched_gts[fg_inds] = bboxes_per_im[matched_idx_all_per_im[fg_inds]]

            for gt in range(num_gt):
                if candidate_idxs[gt] is not None:
                    if candidate_idxs[gt].numel() > 1:
                        candidate_loss = loss_all_per_im[candidate_idxs[gt]]
                        candidate_loss, inds = candidate_loss.sort()
                        candidate_loss = candidate_loss.view(-1, 1).cpu().numpy()
                        min_loss, max_loss = candidate_loss.min(), candidate_loss.max()
                        means_init = [[min_loss], [max_loss]]
                        weights_init = [0.5, 0.5]
                        precisions_init = [[[1.0]], [[1.0]]]
                        gmm = skm.GaussianMixture(2,
                                                  weights_init=weights_init,
                                                  means_init=means_init,
                                                  precisions_init=precisions_init)
                        gmm.fit(candidate_loss)
                        components = gmm.predict(candidate_loss)
                        scores = gmm.score_samples(candidate_loss)
                        components = torch.from_numpy(components).to(device)
                        scores = torch.from_numpy(scores).to(device)

                        fgs = components == 0
                        bgs = components == 1
                        if fgs.nonzero().numel() > 0:
                            # Fig 3. (c)
                            fg_max_score = scores[fgs].max().item()
                            fg_max_idx = (fgs & (scores == fg_max_score)).nonzero().min()
                            is_neg = inds[fgs | bgs]
                            is_pos = inds[:fg_max_idx + 1]
                        else:
                            # just treat all samples as positive for high recall.
                            is_pos = inds
                            is_neg = None
                    else:
                        is_pos = 0
                        is_neg = None

                    if is_neg is not None:
                        neg_idx = candidate_idxs[gt][is_neg]
                        cls_labels_per_im[neg_idx] = 0

                    pos_idx = candidate_idxs[gt][is_pos]
                    cls_labels_per_im[pos_idx] = labels_per_im[gt].view(-1, 1)
                    matched_gts[pos_idx] = bboxes_per_im[gt].view(-1, 4)

            reg_targets_per_im = encode(matched_gts, anchors_per_im.bbox)
            cls_labels.append(cls_labels_per_im)
            reg_targets.append(reg_targets_per_im)

        return cls_labels, reg_targets

    def compute_ious(self, boxes1, boxes2):
        area1 = (boxes1[:, 2] - boxes1[:, 0] + 1) * (boxes1[:, 3] - boxes1[:, 1] + 1)
        area2 = (boxes2[:, 2] - boxes2[:, 0] + 1) * (boxes2[:, 3] - boxes2[:, 1] + 1)
        lt = torch.max(boxes1[:, :2], boxes2[:, :2])
        rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])
        wh = (rb - lt + 1).clamp(min=0)
        inter = wh[:, 0] * wh[:, 1]
        return inter / (area1 + area2 - inter)

    def __call__(self, c_pred, box_pred, iou_pred, targets, anchors, locations):
        iou_based_c, iou_based_offset, matched_idx_all = self.prepare_iou_based_targets(targets, anchors)
        matched_idx_all = torch.cat(matched_idx_all, dim=0)
        bs = len(iou_based_c)
        iou_based_c = torch.cat(iou_based_c, dim=0).int()
        iou_based_offset = torch.cat(iou_based_offset, dim=0)

        box_cls_flatten, box_regression_flatten = concat_box_prediction_layers(c_pred, box_pred)
        anchors_flatten = torch.cat([cat_boxlist(anchors_per_image).bbox for anchors_per_image in anchors], dim=0)

        iou_pred_flatten = [ip.permute(0, 2, 3, 1).reshape(bs, -1, 1) for ip in iou_pred]
        iou_pred_flatten = torch.cat(iou_pred_flatten, dim=1).reshape(-1)

        pos_i = torch.nonzero(iou_based_c > 0).squeeze(1)

        if pos_i.numel() > 0:
            n_loss_per_box = 1

            # compute anchor scores (losses) for all anchors
            logits = box_cls_flatten.detach()
            focal_loss = focal_loss_cuda if logits.is_cuda else focal_loss_cpu
            c_loss = focal_loss(logits, iou_based_c, self.cfg.loss_gamma, self.cfg.loss_alpha)

            box_loss = self.GIoULoss(box_regression_flatten.detach(), iou_based_offset, anchors_flatten, weight=None)
            box_loss = box_loss[iou_based_c > 0].view(-1)

            iou_based_reg_loss_full = torch.full((c_loss.shape[0],),
                                                 fill_value=100000000,
                                                 device=c_loss.device)
            iou_based_reg_loss_full[pos_i] = box_loss.view(-1, n_loss_per_box).mean(1)
            combined_loss = c_loss.sum(dim=1) + iou_based_reg_loss_full
            assert not torch.isnan(combined_loss).any()

            # compute labels and targets using PAA
            labels, reg_targets = self.compute_paa(targets, anchors,
                                                   iou_based_c.view(bs, -1), combined_loss.view(bs, -1),
                                                   matched_idx_all)

            num_gpus = 1
            labels_flatten = torch.cat(labels, dim=0).int()
            reg_targets_flatten = torch.cat(reg_targets, dim=0)
            pos_i = torch.nonzero(labels_flatten > 0).squeeze(1)
            total_num_pos = pos_i.new_tensor([pos_i.numel()]).item()
            num_pos_per_gpu = max(total_num_pos / float(num_gpus), 1.0)

            box_regression_flatten = box_regression_flatten[pos_i]
            reg_targets_flatten = reg_targets_flatten[pos_i]
            anchors_flatten = anchors_flatten[pos_i]

            # compute iou prediction targets
            iou_pred_flatten = iou_pred_flatten[pos_i]
            gt_boxes = decode(reg_targets_flatten, anchors_flatten)
            boxes = decode(box_regression_flatten, anchors_flatten).detach()
            ious = self.compute_ious(gt_boxes, boxes)

            # compute iou losses
            iou_pred_loss = self.iou_bce_loss(iou_pred_flatten, ious) / num_pos_per_gpu * self.cfg.iou_loss_weight
            sum_ious_targets_per_gpu = ious.sum().item() / float(num_gpus)

            # set regression loss weights to ious between predicted boxes and GTs
            reg_loss = self.GIoULoss(box_regression_flatten, reg_targets_flatten, anchors_flatten, weight=ious)
            reg_loss = reg_loss[labels_flatten[pos_i] > 0].view(-1)

            logits = box_cls_flatten
            focal_loss = focal_loss_cuda if logits.is_cuda else focal_loss_cpu
            cls_loss = focal_loss(logits, labels_flatten.int(), self.cfg.loss_gamma, self.cfg.loss_alpha)
        else:
            reg_loss = box_regression_flatten.sum()

        category_loss = cls_loss.sum() / num_pos_per_gpu
        box_loss = reg_loss.sum() / sum_ious_targets_per_gpu * self.cfg.reg_loss_weight

        return [category_loss, box_loss, iou_pred_loss]
