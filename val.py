import argparse
import time
import torch
import numpy as np
from config import get_config
from data.data_loader import make_data_loader
from model.paa import PAA
from utils.utils import ProgressBar
from utils.post_processor import post_process
import json
from utils import timer
import pdb

parser = argparse.ArgumentParser(description='PAA_Minimal Evaluation')
parser.add_argument('--gpu_id', default='0', type=str, help='The GPUs to use.')
parser.add_argument('--weight', type=str, default='weights/res50_1x_116000.pth', help='The validation model.')
parser.add_argument('--test_bs', default='1', type=str, help='Test batch size.')
parser.add_argument('--val_num', default=-1, type=int, help='Number of validation images, -1 for all.')
parser.add_argument('--score_voting', action='store_true', default=False, help='Using score voting.')
parser.add_argument('--improved_coco', action='store_true', default=False, help='Improved COCO API written by myself.')

def compute_thre_per_class(coco_eval):
    # Compute the score threshold per class according to the highest f-measure. Then it can be used in visualization.

    # dimension of precision: [TxRxKxAxM]
    precision = coco_eval.eval['precision']
    # we compute thresholds with IOU being 0.5
    precision = precision[0, :, :, 0, -1]
    scores = coco_eval.eval['scores']
    scores = scores[0, :, :, 0, -1]

    recall = np.linspace(0, 1, num=precision.shape[0])
    recall = recall[:, None]

    f_measure = (2 * precision * recall) / (np.maximum(precision + recall, 1e-6))
    max_f_measure = f_measure.max(axis=0)
    max_f_measure_inds = f_measure.argmax(axis=0)
    scores = scores[max_f_measure_inds, range(len(max_f_measure_inds))]

    print("Maximum f-measures for classes:")
    print(list(max_f_measure))
    print("Score thresholds for classes (used in demos for visualization purposes):")
    print(list(scores))


def inference(model, cfg, during_training=False):
    model.eval()
    predictions, coco_results = {}, []
    val_loader = make_data_loader(cfg, during_training=during_training)
    dataset = val_loader.dataset
    dl = len(val_loader)
    bar = ProgressBar(length=40, max_val=dl)
    timer.reset()

    # with torch.no_grad():
    #     for i, (img_list_batch, _) in enumerate(val_loader):
    #         if i == 1:
    #             timer.start()
    #
    #         with timer.counter('forward'):
    #             img_tensor_batch = torch.stack([aa.img for aa in img_list_batch], dim=0).cuda()
    #             c_pred, box_pred, iou_pred, anchors = model(img_tensor_batch)
    #
    #         with timer.counter('post_process'):
    #             resized_size = [aa.resized_size for aa in img_list_batch]
    #             pred_batch = post_process(cfg, c_pred, box_pred, iou_pred, anchors, resized_size)
    #
    #         with timer.counter('accumulate'):
    #             for pred in pred_batch:
    #                 pred.to_cpu()
    #
    #             for img_list, pred in zip(img_list_batch, pred_batch):
    #                 if pred.box.shape[0] == 0:
    #                     continue
    #
    #                 original_id = dataset.id_img_map[img_list.id]
    #                 pred.resize(img_list.ori_size)
    #                 pred.convert_mode("x1y1wh")
    #
    #                 boxes = pred.box.tolist()
    #                 score = pred.score.tolist()
    #                 label = pred.label.tolist()
    #
    #                 mapped_labels = [dataset.to_category_id[i] for i in label]
    #                 coco_results.extend([{"image_id": original_id,
    #                                       "category_id": mapped_labels[k],
    #                                       "bbox": box,
    #                                       "score": score[k]} for k, box in enumerate(boxes)])
    #
    #         aa = time.perf_counter()
    #         if i > 0:
    #             batch_time = aa - temp
    #             timer.add_batch_time(batch_time)
    #
    #             time_name = ['batch', 'data', 'forward', 'post_process', 'accumulate']
    #             t_t, t_d, t_f, t_pp, t_acc = timer.get_times(time_name)
    #             fps, t_fps = 1 / (t_d + t_f + t_pp), 1 / t_t
    #             bar_str = bar.get_bar(i + 1)
    #             print(f'\rTesting: {bar_str} {i + 1}/{dl}, fps: {fps:.2f} | total fps: {t_fps:.2f} | t_t: {t_t:.3f} | '
    #                   f't_d: {t_d:.3f} | t_f: {t_f:.3f} | t_pp: {t_pp:.3f} | t_acc: {t_acc:.3f}', end='')
    #
    #         temp = aa
    #
    # print('\n\nTest ended, doing evaluation...')
    #
    # json_name = cfg.weight.split('/')[-1].split('.')[0]
    # file_path = f'results/{json_name}.json'
    # with open(file_path, "w") as f:
    #     json.dump(coco_results, f)

    coco_dt = dataset.coco.loadRes('results/res50_1x_116000.json')

    if cfg.val_api == 'Improved COCO':
        from my_cocoeval.cocoeval import SelfEval
        bbox_eval = SelfEval(dataset.coco, coco_dt, all_points=True)
    else:
        from pycocotools.cocoeval import COCOeval
        bbox_eval = COCOeval(dataset.coco, coco_dt, iouType='bbox')

    bbox_eval.evaluate()
    bbox_eval.accumulate()
    bbox_eval.summarize()

    if not during_training:
        if cfg.val_api == 'Improved COCO':
            bbox_eval.draw_curve()
        else:
            compute_thre_per_class(bbox_eval)


if __name__ == '__main__':
    args = parser.parse_args()
    cfg_str = args.weight.split('/')[-1].split('.')[0]
    iter_str = cfg_str.split('_')[-1]
    args.cfg = cfg_str.replace('_' + iter_str, '')

    cfg = get_config(args, val_mode=True)
    model = PAA(cfg).cuda()
    model.load_state_dict(torch.load(cfg.weight), strict=True)
    inference(model, cfg)
