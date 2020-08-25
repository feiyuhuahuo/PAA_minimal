import argparse
import os
import pdb
import torch
from torch.utils.tensorboard import SummaryWriter
from config import update_config
from data.data_loader import make_data_loader
from modeling.paa import PAA
from utils.checkpoint import Checkpointer
from tqdm import tqdm
from utils.coco_eval import do_coco_evaluation

parser = argparse.ArgumentParser(description="PyTorch Object Detection Evaluation")
parser.add_argument("--model", default='weights/paa_res50.pth')


def inference(model, cfg, training=False):
    model.eval()
    predictions = {}
    val_loader = make_data_loader(cfg, training=training)

    with torch.no_grad():
        for _, batch in enumerate(tqdm(val_loader)):
            images, targets, image_ids = batch
            output = model(images.to(torch.device("cuda")))
            output = [o.to(torch.device("cpu")) for o in output]
            predictions.update({img_id: result for img_id, result in zip(image_ids, output)})

    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        print("Number of images that were gathered from multiple processes is not "
              "a contiguous set. Some images might be missing from the evaluation")

    predictions = [predictions[i] for i in image_ids]
    if not training:
        torch.save(predictions, 'predictions.pth')

    do_coco_evaluation(dataset=data_loader.dataset, predictions=predictions, cfg=cfg)


if __name__ == '__main__':
    args = parser.parse_args()
    cfg = update_config(args)
    model = PAA(cfg).cuda()
    model.load_state_dict(torch.load(cfg.model)['model'], strict=True)
    inference(model, cfg, training=False)
