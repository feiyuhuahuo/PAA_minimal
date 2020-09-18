import numpy as np
from collections import defaultdict
from my_cocoeval import mask as maskUtils
from terminaltables import AsciiTable
import matplotlib.pyplot as plt
import pdb

NAMES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
         'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
         'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
         'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
         'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
         'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
         'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
         'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
         'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')


class SelfEval:
    def __init__(self, cocoGt, cocoDt, all_points=False):
        self.gt = defaultdict(list)
        self.dt = defaultdict(list)
        self.all_points = all_points

        # np.arange and np.linspace can not get the accurate number, e.g. 0.8500000000000003 and 0.8999999999
        self.iou_thre = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        self.recall_points = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)

        self.max_det = 100
        self.area = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.area_name = ['all', 'small', 'medium', 'large']

        self.imgIds = list(np.unique(cocoGt.getImgIds()))
        self.catIds = list(np.unique(cocoGt.getCatIds()))

        gts = cocoGt.loadAnns(cocoGt.getAnnIds(imgIds=self.imgIds, catIds=self.catIds))
        dts = cocoDt.loadAnns(cocoDt.getAnnIds(imgIds=self.imgIds, catIds=self.catIds))

        self.C, self.A, self.T, self.N = len(self.catIds), len(self.area), len(self.iou_thre), len(self.imgIds)

        # key is a tuple (gt['image_id'], gt['category_id']), value is a list.
        for gt in gts:
            # if gt['iscrowd'] == 0:  # TODO: why this makes the result different
            self.gt[gt['image_id'], gt['category_id']].append(gt)
        for dt in dts:
            self.dt[dt['image_id'], dt['category_id']].append(dt)

    def evaluate(self):
        self.match_record = [[[None for _ in range(self.N)] for _ in range(self.A)] for _ in range(self.C)]

        for c, cat_id in enumerate(self.catIds):
            for a, area in enumerate(self.area):
                for n, img_id in enumerate(self.imgIds):
                    print(f'\rMatching ground-truths and detections: C: {c}, A: {a}, N: {n}', end='')

                    gt_list, dt_list = self.gt[img_id, cat_id], self.dt[img_id, cat_id]
                    if len(gt_list) == 0 and len(dt_list) == 0:
                        continue
                    else:
                        # different sorting method generates slightly different results.
                        # 'mergesort' is used to be consistent as the COCO Matlab implementation.
                        index = np.argsort([-aa['score'] for aa in dt_list], kind='mergesort')
                        dt_list = [dt_list[i] for i in index]
                        dt_list = dt_list[0: self.max_det]  # if len(one_dt) < self.max_det, no influence

                        for one_gt in gt_list:
                            if one_gt['iscrowd'] or one_gt['area'] < area[0] or one_gt['area'] > area[1]:
                                one_gt['_ignore'] = 1
                            else:
                                one_gt['_ignore'] = 0

                        # sort ignored gt to last
                        index = np.argsort([aa['_ignore'] for aa in gt_list], kind='mergesort')
                        gt_list = [gt_list[i] for i in index]

                        box_gt = [aa['bbox'] for aa in gt_list]
                        box_dt = [aa['bbox'] for aa in dt_list]

                        iscrowd = [int(aa['iscrowd']) for aa in gt_list]
                        IoUs = maskUtils.iou(box_dt, box_gt, iscrowd)  # shape: (num_dt, num_gt)

                        gt_matched = np.zeros((self.T, len(gt_list)))
                        gt_ignore = np.array([aa['_ignore'] for aa in gt_list])
                        dt_matched = np.zeros((self.T, len(dt_list)))
                        dt_ignore = np.zeros((self.T, len(dt_list)))

                        # Pay attent to the logic, if (len(gt_list) == 0) xor (len(dt_list) == 0) is True,
                        # the below 'if' will not be entered, but the dt scores and num_gt should always be recorded.
                        if len(IoUs) != 0:  # this equal to (len(gt_list) != 0) and (len(dt_list) != 0)
                            for t, one_thre in enumerate(self.iou_thre):
                                for d, one_dt in enumerate(dt_list):
                                    iou = one_thre
                                    g_temp = -1
                                    for g in range(len(gt_list)):
                                        # if this gt already matched, and not a crowd, continue
                                        if gt_matched[t, g] > 0 and not iscrowd[g]:
                                            continue
                                        # if dt matched a ignore gt, break, because all the ignore gts are at last
                                        if g_temp > -1 and gt_ignore[g_temp] == 0 and gt_ignore[g] == 1:
                                            break
                                        # continue to next gt unless better match made
                                        if IoUs[d, g] < iou:
                                            continue
                                        # if match successful and best so far, store appropriately
                                        iou = IoUs[d, g]
                                        g_temp = g

                                    # if match made store id of match for both dt and gt
                                    if g_temp == -1:
                                        continue

                                    dt_ignore[t, d] = gt_ignore[g_temp]
                                    dt_matched[t, d] = gt_list[g_temp]['id']
                                    gt_matched[t, g_temp] = one_dt['id']

                        # set unmatched detections which are outside of area range to ignore
                        dt_out_range = [aa['area'] < area[0] or aa['area'] > area[1] for aa in dt_list]
                        dt_out_range = np.array(dt_out_range)[None, :]
                        dt_out_range = np.logical_and(dt_matched == 0, np.repeat(dt_out_range, repeats=self.T, axis=0))
                        dt_ignore = np.logical_or(dt_ignore, dt_out_range)
                        num_gt = np.count_nonzero(gt_ignore == 0)

                        self.match_record[c][a][n] = {'dt_match': dt_matched,
                                                      'dt_score': [aa['score'] for aa in dt_list],
                                                      'dt_ignore': dt_ignore,
                                                      'num_gt': num_gt}

    def accumulate(self):
        print('\nComputing recalls and precisions...')

        R = len(self.recall_points)

        self.p_record = [[[None for _ in range(self.T)] for _ in range(self.A)] for _ in range(self.C)]
        self.r_record = [[[None for _ in range(self.T)] for _ in range(self.A)] for _ in range(self.C)]
        self.s_record = [[[None for _ in range(self.T)] for _ in range(self.A)] for _ in range(self.C)]

        # TODO: check if the logic is right, especially when there are absent categories when evaling part of images
        for c in range(self.C):
            for a in range(self.A):
                temp_dets = self.match_record[c][a][:]
                temp_dets = [aa for aa in temp_dets if aa is not None]

                num_gt = sum([aa['num_gt'] for aa in temp_dets])
                if num_gt == 0:  # The gt category is absent, so it should be excluded when computing mAP.
                    continue

                if len(temp_dets) == 0:
                    # If continue directly, the realted record would be 'None',
                    # which is excluded when computing mAP in summarize().
                    for t in range(self.T):
                        self.p_record[c][a][t] = np.array([0.])
                        self.r_record[c][a][t] = np.array([0.])
                        self.s_record[c][a][t] = np.array([0.])
                    continue

                scores = np.concatenate([aa['dt_score'] for aa in temp_dets])
                index = np.argsort(-scores, kind='mergesort')
                score_sorted = scores[index]

                dt_matched = np.concatenate([aa['dt_match'] for aa in temp_dets], axis=1)[:, index]
                dt_ignore = np.concatenate([aa['dt_ignore'] for aa in temp_dets], axis=1)[:, index]

                tps = np.logical_and(dt_matched, np.logical_not(dt_ignore))  # shape: (thre_num, dt_num)
                fps = np.logical_and(np.logical_not(dt_matched), np.logical_not(dt_ignore))

                tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)

                for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                    tp = np.array(tp)
                    fp = np.array(fp)
                    recall = (tp / num_gt).tolist()
                    precision = (tp / (fp + tp + np.spacing(1))).tolist()

                    # numpy is slow without cython optimization for accessing elements
                    # use python list can get significant speed improvement
                    p_smooth = precision.copy()
                    for i in range(len(tp) - 1, 0, -1):
                        if p_smooth[i] > p_smooth[i - 1]:
                            p_smooth[i - 1] = p_smooth[i]

                    if self.all_points:
                        p_reduced, s_reduced = [], []
                        r_reduced = list(set(recall))
                        r_reduced.sort()

                        for one_r in r_reduced:
                            index = recall.index(one_r)  # the first precision w.r.t the recall is always the highest
                            p_reduced.append(p_smooth[index])
                            s_reduced.append(score_sorted[index])

                        stair_h, stair_w, stair_s = [], [], []
                        for i in range(len(p_reduced)):  # get the falling edge of the stairs
                            if (i != len(p_reduced) - 1) and (p_reduced[i] > p_reduced[i + 1]):
                                stair_h.append(p_reduced[i])
                                stair_w.append(r_reduced[i])
                                stair_s.append(s_reduced[i])

                        stair_h.append(p_reduced[-1])  # add the final point which is out of range in the above loop
                        stair_w.append(r_reduced[-1])
                        stair_s.append(s_reduced[-1])

                        stair_w.insert(0, 0.)  # insert 0. at index 0 to do np.diff()
                        stair_w = np.diff(stair_w)
                        self.p_record[c][a][t] = np.array(stair_h)
                        self.r_record[c][a][t] = np.array(stair_w)
                        self.s_record[c][a][t] = np.array(stair_s)
                    else:
                        index = np.searchsorted(recall, self.recall_points, side='left')
                        score_101, precision_101 = np.zeros((R,)), np.zeros((R,))
                        # if recall is < 1.0, then there will always be some points out of the recall range,
                        # so use try...except... to deal with it automatically.
                        try:
                            for ri, pi in enumerate(index):
                                precision_101[ri] = p_smooth[pi]
                                score_101[ri] = score_sorted[pi]
                        except:
                            pass

                        self.p_record[c][a][t] = precision_101
                        num_points = len(precision_101)
                        # COCO's ap = mean of the 101 precision points, I use this way to keep the code compatibility,
                        # so the width of the stair is 1 / num_points. This can get the same AP. But recall is
                        # different. COCO's recall is the last value of all recall values, and mine is the last value
                        # of 101 recall values.
                        self.r_record[c][a][t] = np.array([1 / num_points] * num_points)
                        self.s_record[c][a][t] = score_101

    @staticmethod
    def mr4(array):
        return round(float(np.mean(array)), 4)

    def summarize(self):
        print('Summarizing...')
        AP_matrix = np.zeros((self.C, self.A, self.T)) - 1
        AR_matrix = np.zeros((self.C, self.A, self.T)) - 1
        if self.all_points:
            MPP_matrix = np.zeros((self.C, self.A, self.T, 5)) - 1

        for c in range(self.C):
            for a in range(self.A):
                for t in range(self.T):
                    if self.p_record[c][a][t] is not None:  # exclude absent categories, the related AP is -1
                        AP_matrix[c, a, t] = (self.p_record[c][a][t] * self.r_record[c][a][t]).sum()
                        # In all points mode, recall is always the sum of 'stair_w', but in 101 points mode,
                        # we need to find where precision reduce to 0., and thus calculate the recall.
                        if self.all_points:
                            AR_matrix[c, a, t] = self.r_record[c][a][t].sum()
                            r_cumsum = np.cumsum(self.r_record[c][a][t])
                            ap_array = self.p_record[c][a][t] * r_cumsum
                            index = np.argmax(ap_array)
                            p_max = self.p_record[c][a][t][index]
                            r_max = r_cumsum[index]
                            s_s = self.s_record[c][a][t][index]
                            mpp = ap_array[index]
                            # If ap == 0 for a certain threshold, ff should be taken into calculation because
                            # it's not an absent category, so ff should be 0 instead of nan.
                            ff = 0. if AP_matrix[c, a, t] == 0 else mpp / AP_matrix[c, a, t]
                            MPP_matrix[c, a, t] = np.array([p_max, r_max, s_s, mpp, ff])
                        else:
                            r_mask = self.p_record[c][a][t] != 0
                            AR_matrix[c, a, t] = (self.r_record[c][a][t])[r_mask].sum()

        table_c_list = [['Category', 'AP', 'Recall'] * 3]
        c_line = ['all', self.mr4(AP_matrix[:, 0, :]), self.mr4(AR_matrix[:, 0, :])]

        if self.all_points:  # max practical precision
            table_mpp_list = [['Category', 'P_max', 'R_max', 'Score', 'MPP', 'FF'] * 3]
            mpp_line = ['all', self.mr4(MPP_matrix[:, 0, :, 0]), self.mr4(MPP_matrix[:, 0, :, 1]),
                        self.mr4(MPP_matrix[:, 0, :, 2]), self.mr4(MPP_matrix[:, 0, :, 3]),
                        self.mr4(MPP_matrix[:, 0, :, 4])]

        for i in range(self.C):
            if -1 in AP_matrix[i, 0, :]:  # if this category is absent
                assert AP_matrix[i, 0, :].sum() == -len(self.iou_thre), 'Not all ap is -1 in absent category'
                c_line += [NAMES[i], 'absent', 'absent']
                if self.all_points:
                    mpp_line += [NAMES[i], 'absent', 'absent', 'absent', 'absent', 'absent']
            else:
                c_line += [NAMES[i], self.mr4(AP_matrix[i, 0, :]), self.mr4(AR_matrix[i, 0, :])]
                if self.all_points:
                    mpp_line += [NAMES[i], self.mr4(MPP_matrix[i, 0, :, 0]), self.mr4(MPP_matrix[i, 0, :, 1]),
                                 self.mr4(MPP_matrix[i, 0, :, 2]), self.mr4(MPP_matrix[i, 0, :, 3]),
                                 self.mr4(MPP_matrix[i, 0, :, 4])]
            if (i + 2) % 3 == 0:
                table_c_list.append(c_line)
                c_line = []

                if self.all_points:
                    table_mpp_list.append(mpp_line)
                    mpp_line = []

        table_iou_list = [['IoU'] + self.iou_thre, ['AP'], ['Recall']]
        for i in range(self.T):
            ap_m = AP_matrix[:, 0, i]  # absent category is not included
            ar_m = AR_matrix[:, 0, i]
            table_iou_list[1].append(self.mr4(ap_m[ap_m > -1]))
            table_iou_list[2].append(self.mr4(ar_m[ar_m > -1]))

        table_area_list = [['Area'] + self.area_name, ['AP'], ['Recall']]
        for i in range(self.A):
            ap_m = AP_matrix[:, i, :]
            ar_m = AR_matrix[:, i, :]
            table_area_list[1].append(self.mr4(ap_m[ap_m > -1]))
            table_area_list[2].append(self.mr4(ar_m[ar_m > -1]))

        table_c = AsciiTable(table_c_list)
        table_iou = AsciiTable(table_iou_list)
        table_area = AsciiTable(table_area_list)

        if self.all_points:
            print()
            table_mpp = AsciiTable(table_mpp_list)
            print(table_mpp.table)

        print()
        print(table_c.table)  # bug, can not print '\n', or table is not perfect
        print()
        print(table_iou.table)
        print()
        print(table_area.table)

    def draw_curve(self):
        recalls = self.r_record[0][0][0].cumsum()
        precisions = self.p_record[0][0][0]
        pdb.set_trace()  # the len of different thre is different
        plt.plot(recalls, precisions)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.show()
        pdb.set_trace()
        plt.hlines()
