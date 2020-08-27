import logging
import tempfile
import os
import json
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from collections import OrderedDict
from utils.bounding_box import BoxList
from utils.boxlist_ops import boxlist_iou


def do_coco_evaluation(dataset, predictions, cfg):
    coco_results = []
    for image_id, prediction in enumerate(predictions):
        original_id = dataset.id_img_map[image_id]
        if len(prediction) == 0:
            continue

        img_info = dataset.get_img_info(image_id)
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

    file_path = f'results/{cfg.backbone}_bbox.json'
    with open(file_path, "w") as f:
        json.dump(coco_results, f)

    coco_dt = dataset.coco.loadRes(file_path)
    coco_eval = COCOeval(dataset.coco, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    compute_thresholds_for_classes(coco_eval)


def compute_thresholds_for_classes(coco_eval):
    '''
    The function is used to compute the thresholds corresponding to best f-measure.
    The resulting thresholds are used in atss_demo.py.
    :param coco_eval:
    :return:
    '''
    import numpy as np
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
