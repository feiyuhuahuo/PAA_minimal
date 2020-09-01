import torch
import pdb
from torch import nn
from modeling.utils import concat_fpn_pred, encode, decode, match
from modeling.sigmoid_focal_loss import focal_loss_cuda, focal_loss_cpu
from utils.boxlist_ops import boxlist_iou, cat_boxlist
import sklearn.mixture as skm


class PAALoss:
    def __init__(self, cfg):
        self.cfg = cfg
        self.iou_bce_loss = nn.BCEWithLogitsLoss(reduction="sum")

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

    def prepare_initial_targets(self, targets, anchors):
        category_all, offset_all, index_all = [], [], []

        for targets_per_img, anchor in zip(targets, anchors):
            assert targets_per_img.mode == "xyxy"
            anchors_per_img = cat_boxlist(anchor)

            iou_matrix = boxlist_iou(targets_per_img, anchors_per_img)  # shape: (num_gt, num_anchor)
            # shape: (num_anchor,), record the matched gt index for each dt, -1 for background, -2 for ignored.
            matched_index = match(iou_matrix, self.cfg.match_iou_thre, self.cfg.match_iou_thre)
            targets_per_img = targets_per_img.copy_with_fields(['labels'])
            # '.clamp()' here to make index operation available, this won't affect 'category' and 'matched_index'
            matched_targets = targets_per_img[matched_index.clamp(min=0)]

            offset = encode(matched_targets.bbox, anchors_per_img.bbox)

            category = matched_targets.get_field("labels").to(dtype=torch.float32)
            category[matched_index == -1] = 0  # Background (negative examples)
            category[matched_index == -2] = -1  # ignore indices that are between thresholds

            offset_all.append(offset)
            category_all.append(category)
            index_all.append(matched_index)

        category_all = torch.cat(category_all, dim=0).int()
        offset_all = torch.cat(offset_all, dim=0)
        index_all = torch.cat(index_all, dim=0)

        return category_all, offset_all, index_all

    def compute_paa(self, target_batch, anchor_batch, c_init_batch, score_batch, index_init_batch):
        bs = len(target_batch)
        c_init_batch = c_init_batch.reshape(bs, -1)
        score_batch = score_batch.reshape(bs, -1)
        index_init_batch = index_init_batch.reshape(bs, -1)
        device = score_batch.device

        final_c_batch, final_offset_batch = [], []
        for i in range(len(target_batch)):
            target = target_batch[i]
            assert target.mode == "xyxy", 'target mode incorrect'

            box_gt = target.bbox
            c_gt = target.get_field("labels")
            anchor = cat_boxlist(anchor_batch[i])

            c_init = c_init_batch[i]
            score = score_batch[i]
            index_init = index_init_batch[i]
            assert c_init.shape == index_init.shape

            num_anchor_per_fpn = [len(anchor_per_fpn.bbox) for anchor_per_fpn in anchor_batch[i]]

            final_c = torch.zeros(anchor.bbox.shape[0], dtype=torch.long).to(device)  # '0' represents background
            final_box_gt = torch.zeros_like(anchor.bbox)

            for gt_i in range(box_gt.shape[0]):
                candi_i_per_gt = []
                start_i = 0

                for j in range(len(num_anchor_per_fpn)):
                    end_i = start_i + num_anchor_per_fpn[j]

                    score_per_fpn = score[start_i:end_i]
                    index_init_per_fpn = index_init[start_i:end_i]

                    # get the matched anchor index for a certain gt in a certain fpn
                    matched_i = (index_init_per_fpn == gt_i).nonzero()[:, 0]
                    matched_num = matched_i.numel()

                    if matched_num > 0:
                        _, topk_i = score_per_fpn[matched_i].topk(min(matched_num, self.cfg.fpn_topk), largest=False)
                        topk_i_per_fpn = matched_i[topk_i]
                        candi_i_per_gt.append(topk_i_per_fpn + start_i)

                    start_i = end_i

                if candi_i_per_gt:
                    candi_i_per_gt = torch.cat(candi_i_per_gt)

                    # only if there are more than 1 candidate, gmm would be done
                    if candi_i_per_gt.numel() > 1:
                        candi_score = score[candi_i_per_gt]
                        candi_score, candi_index = candi_score.sort()
                        candi_score = candi_score.reshape(-1, 1).cpu().numpy()

                        gmm = skm.GaussianMixture(n_components=2,
                                                  weights_init=[0.5, 0.5],
                                                  means_init=[[candi_score.min()], [candi_score.max()]],
                                                  precisions_init=[[[1.0]], [[1.0]]])
                        gmm.fit(candi_score)

                        gmm_component = gmm.predict(candi_score)
                        gmm_score = gmm.score_samples(candi_score)

                        gmm_component = torch.from_numpy(gmm_component).to(device)
                        gmm_score = torch.from_numpy(gmm_score).to(device)

                        fg = gmm_component == 0
                        if fg.nonzero().numel() > 0:
                            _, fg_max_i = gmm_score[fg].max(dim=0)  # Fig 3. (c)
                            is_pos = candi_index[:fg_max_i + 1]
                        else:
                            is_pos = candi_index  # just treat all samples as positive for high recall.
                    else:
                        is_pos = 0  # if there is only one candidate, treat it as positive

                    pos_i = candi_i_per_gt[is_pos]
                    final_c[pos_i] = c_gt[gt_i].reshape(-1, 1)
                    final_box_gt[pos_i] = box_gt[gt_i].reshape(-1, 4)

            # 'neg_i' and 'pos_i' derives from 'candi_i_per_gt' derives from 'matched_i' derives from 'index_init'
            final_offset = encode(final_box_gt, anchor.bbox)

            final_c_batch.append(final_c)
            final_offset_batch.append(final_offset)

        final_c_batch = torch.cat(final_c_batch, dim=0).int()
        final_offset_batch = torch.cat(final_offset_batch, dim=0)

        return final_c_batch, final_offset_batch

    @staticmethod
    def compute_ious(boxes1, boxes2):
        area1 = (boxes1[:, 2] - boxes1[:, 0] + 1) * (boxes1[:, 3] - boxes1[:, 1] + 1)
        area2 = (boxes2[:, 2] - boxes2[:, 0] + 1) * (boxes2[:, 3] - boxes2[:, 1] + 1)
        lt = torch.max(boxes1[:, :2], boxes2[:, :2])
        rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])
        wh = (rb - lt + 1).clamp(min=0)
        inter = wh[:, 0] * wh[:, 1]
        return inter / (area1 + area2 - inter)

    def __call__(self, c_pred, box_pred, iou_pred, targets, anchors):
        # TODO: figure out whether anchors for per image are the same
        # c_init_batch: (bs * num_anchor,), 0 for background, -1 for ignored
        # offset_init_batch: (bs * num_anchor, 4)
        # index_init_batch: (bs * num_anchor,), -1 for bakground, -2 for ignored
        c_init_batch, offset_init_batch, index_init_batch = self.prepare_initial_targets(targets, anchors)
        pos_i_init = torch.nonzero(c_init_batch > 0).reshape(-1)

        c_pred_f, box_pred_f, iou_pred_f, anchor_f = concat_fpn_pred(c_pred, box_pred, iou_pred, anchors)

        if pos_i_init.numel() > 0:  # compute anchor scores (losses) for all anchors, gradient is not needed.
            c_loss = focal_loss_cuda(c_pred_f.detach(), c_init_batch, self.cfg.fl_gamma, self.cfg.fl_alpha)
            box_loss = self.GIoULoss(box_pred_f.detach(), offset_init_batch, anchor_f, weight=None)
            box_loss = box_loss[c_init_batch > 0].reshape(-1)

            box_loss_full = torch.full((c_loss.shape[0],), fill_value=10000, device=c_loss.device)
            assert box_loss.max() < 10000, 'box_loss_full initial value error'
            box_loss_full[pos_i_init] = box_loss

            score_batch = c_loss.sum(dim=1) + box_loss_full
            assert not torch.isnan(score_batch).any()  # all the elements should not be nan

            # compute labels and targets using PAA
            final_c_batch, final_offset_batch = self.compute_paa(targets, anchors, c_init_batch,
                                                                 score_batch, index_init_batch)

            pos_i_final = torch.nonzero(final_c_batch > 0).reshape(-1)
            num_pos = pos_i_final.numel()

            box_pred_f = box_pred_f[pos_i_final]
            final_offset_batch = final_offset_batch[pos_i_final]
            anchor_f = anchor_f[pos_i_final]
            iou_pred_f = iou_pred_f[pos_i_final]

            gt_boxes = decode(final_offset_batch, anchor_f)
            box_pred_decoded = decode(box_pred_f, anchor_f).detach()
            iou_gt = self.compute_ious(gt_boxes, box_pred_decoded)

            cls_loss = focal_loss_cuda(c_pred_f, final_c_batch.int(), self.cfg.fl_gamma, self.cfg.fl_alpha)
            box_loss = self.GIoULoss(box_pred_f, final_offset_batch, anchor_f, weight=iou_gt)
            box_loss = box_loss[final_c_batch[pos_i_final] > 0].reshape(-1)
            iou_pred_loss = self.iou_bce_loss(iou_pred_f, iou_gt) / num_pos * self.cfg.iou_loss_w
            iou_gt_sum = iou_gt.sum().item()
        else:
            box_loss = box_f.sum()

        category_loss = cls_loss.sum() / num_pos
        box_loss = box_loss.sum() / iou_gt_sum * self.cfg.box_loss_w

        return category_loss, box_loss, iou_pred_loss
