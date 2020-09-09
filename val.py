import argparse
import time
import torch
import numpy as np
from config import get_config
from data.data_loader import make_data_loader
from modeling.paa import PAA
from utils.utils import ProgressBar
from utils.post_processor import post_process
import json
from utils import timer
from pycocotools.cocoeval import COCOeval
import pdb

parser = argparse.ArgumentParser(description="PyTorch Object Detection Evaluation")
parser.add_argument("--weight", type=str, default='weights/paa_res50.pth')
parser.add_argument('--gpu_id', default='0', type=str, help='The GPUs to use.')
parser.add_argument('--test_bs', default='1', type=str, help='The batch size allocated to each GPU.')


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


def inference(model, cfg, during_train=False):
    model.eval()
    predictions, coco_results = {}, []
    val_loader = make_data_loader(cfg, val=True)
    dataset = val_loader.dataset
    dl = len(val_loader)
    bar = ProgressBar(length=40, max_val=dl)
    timer.init()

    with torch.no_grad():
        for i, (img_list_batch, _) in enumerate(val_loader):
            if i > 0:
                timer.start()

            with timer.counter('forward'):
                img_tensor_batch = torch.stack([aa.img for aa in img_list_batch], dim=0).cuda()
                c_pred, box_pred, iou_pred, anchors = model(img_tensor_batch)

            with timer.counter('post_process'):
                resized_size = [aa.resized_size for aa in img_list_batch]
                pred_batch = post_process(cfg, c_pred, box_pred, iou_pred, anchors, resized_size)

            with timer.counter('accumulate'):
                for pred in pred_batch:
                    pred.to_cpu()

                for img_list, pred in zip(img_list_batch, pred_batch):
                    if pred.box.shape[0] == 0:
                        continue

                    original_id = dataset.id_img_map[img_list.id]
                    pred.resize(img_list.ori_size)
                    pred.convert_mode("x1y1wh")

                    boxes = pred.box.tolist()
                    score = pred.score.tolist()
                    label = pred.label.tolist()

                    mapped_labels = [dataset.to_category_id[i] for i in label]
                    coco_results.extend([{"image_id": original_id,
                                          "category_id": mapped_labels[k],
                                          "bbox": box,
                                          "score": score[k]} for k, box in enumerate(boxes)])

            aa = time.perf_counter()
            if i > 0:
                batch_time = aa - temp
                timer.add_batch_time(batch_time)

                time_name = ['batch', 'data', 'forward', 'post_process', 'accumulate']
                t_t, t_d, t_f, t_pp, t_acc = timer.get_times(time_name)
                fps, t_fps = 1 / (t_d + t_f + t_pp), 1 / t_t
                bar_str = bar.get_bar(i + 1)
                print(f'\rTesting: {bar_str} {i + 1}/{dl}, fps: {fps:.2f} | total fps: {t_fps:.2f} | t_t: {t_t:.3f} | '
                      f't_d: {t_d:.3f} | t_f: {t_f:.3f} | t_pp: {t_pp:.3f} | t_acc: {t_acc:.3f}', end='')

            temp = aa

    print('\n\nTesting ended, doing evaluation...')
    file_path = f'results/{cfg.backbone}_bbox.json'
    with open(file_path, "w") as f:
        json.dump(coco_results, f)

    coco_dt = dataset.coco.loadRes(file_path)
    coco_eval = COCOeval(dataset.coco, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    if not during_train:
        compute_thre_per_class(coco_eval)


if __name__ == '__main__':
    args = parser.parse_args()
    cfg = get_config(args, val_mode=True)
    model = PAA(cfg).cuda()
    model.load_state_dict(torch.load(cfg.weight)['model'], strict=True)
    inference(model, cfg)
