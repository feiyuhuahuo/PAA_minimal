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
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pdb

parser = argparse.ArgumentParser(description="PyTorch Object Detection Evaluation")
parser.add_argument("--weight", type=str, default='weights/paa_res50.pth')
parser.add_argument('--gpu_id', default='0', type=str, help='The GPUs to use.')
parser.add_argument('--alloc', default='2', type=str, help='The batch size allocated to each GPU.')


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
    val_loader = make_data_loader(cfg, training=False)
    dataset = val_loader.dataset
    dl = len(val_loader)
    bar = ProgressBar(length=40, max_val=dl)
    timer.init()

    with torch.no_grad():
        for i, (images, targets, image_ids) in enumerate(val_loader):
            if i > 0:
                timer.start()

            with timer.counter('forward'):
                c_pred, box_pred, iou_pred, anchors = model(images.to(torch.device("cuda")))

            with timer.counter('post_process'):
                pred_batch = post_process(cfg, c_pred, box_pred, iou_pred, anchors)
                pred_batch = [aa.to(torch.device("cpu")) for aa in pred_batch]

            with timer.counter('accumulate'):
                for img_id, prediction in zip(image_ids, pred_batch):
                    original_id = dataset.id_img_map[img_id]
                    if len(prediction) == 0:
                        continue

                    img_info = dataset.get_img_info(img_id)
                    image_width = img_info["width"]
                    image_height = img_info["height"]
                    prediction = prediction.resize((image_width, image_height))
                    prediction = prediction.convert("xywh")

                    boxes = prediction.bbox.tolist()
                    scores = prediction.get_field("scores").tolist()
                    labels = prediction.get_field("labels").tolist()

                    mapped_labels = [dataset.contiguous_id_to_class_id[i] for i in labels]
                    coco_results.extend([{"image_id": original_id,
                                          "category_id": mapped_labels[k],
                                          "bbox": box,
                                          "score": scores[k]} for k, box in enumerate(boxes)])

            aa = time.perf_counter()
            if i > 0:
                batch_time = aa - temp
                timer.add_batch_time(batch_time)
            temp = aa

            if i > 0:
                time_name = ['batch', 'data', 'forward', 'post_process', 'accumulate']
                t_t, t_d, t_f, t_pp, t_acc = timer.get_times(time_name)
                fps, t_fps = 1 / (t_d + t_f + t_pp), 1 / t_t
                bar_str = bar.get_bar(i + 1)
                print(f'\rTesting: {bar_str} {i + 1}/{dl}, fps: {fps:.2f} | total fps: {t_fps:.2f} | t_t: {t_t:.3f} | '
                      f't_d: {t_d:.3f} | t_f: {t_f:.3f} | t_pp: {t_pp:.3f} | t_acc: {t_acc:.3f}', end='')

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
