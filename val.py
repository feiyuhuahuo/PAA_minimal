import argparse
import os
import pdb
import torch
import numpy as np
from config import get_config
from data.data_loader import make_data_loader
from modeling.paa import PAA
from tqdm import tqdm
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

parser = argparse.ArgumentParser(description="PyTorch Object Detection Evaluation")
parser.add_argument("--test_bs", type=int, default=1)
parser.add_argument("--weight", type=str, default='weights/paa_res50.pth')


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

    with torch.no_grad():
        for _, batch in enumerate(tqdm(val_loader)):
            images, targets, image_ids = batch
            output = model(images.to(torch.device("cuda")))
            output = [aa.to(torch.device("cpu")) for aa in output]

            for img_id, prediction in zip(image_ids, output):
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
