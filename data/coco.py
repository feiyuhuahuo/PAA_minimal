import torch
import torchvision
from data import transforms as T
from utils.bounding_box import BoxList
import pdb


class COCODataset(torchvision.datasets.coco.CocoDetection):
    def __init__(self, cfg, training):
        img_path = cfg.train_imgs if training else cfg.val_imgs
        ann_file = cfg.train_ann if training else cfg.val_ann
        super().__init__(img_path, ann_file)

        self.ids = sorted(self.ids)  # sort indices for reproducible results

        if training:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if self.has_valid_annotation(anno):  # filter images without detection annotations
                    ids.append(img_id)

            self.ids = ids

            if cfg.min_size_range_train[0] == -1:
                min_size = cfg.min_size_train
            else:
                assert len(cfg.min_size_range_train) == 2, "min_size_range_train error "
                min_size = list(range(cfg.min_size_range_train[0], cfg.min_size_range_train[1] + 1))

            max_size = cfg.max_size_train
            flip_prob = 0.5
        else:
            min_size = cfg.min_size_test
            max_size = cfg.max_size_test
            flip_prob = 0

        self.class_id_to_contiguous_id = {v: i + 1 for i, v in enumerate(self.coco.getCatIds())}
        self.contiguous_id_to_class_id = {v: k for k, v in self.class_id_to_contiguous_id.items()}
        self.id_img_map = {k: v for k, v in enumerate(self.ids)}
        self.transform = T.Compose([T.Resize(min_size, max_size),
                                    # T.RandomHorizontalFlip(flip_prob),
                                    T.ToTensor(),
                                    T.Normalize(mean=[102.9801, 115.9465, 122.7717], std=[1., 1., 1.],
                                                to_bgr255=True)])

    def get_img_info(self, index):
        img_id = self.id_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data

    @staticmethod
    def has_valid_annotation(anno):
        # if it's empty, there is no annotation
        if len(anno) == 0:
            return False
        # if all boxes have close to zero area, there is no annotation
        if all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno):
            return False
        # keypoints task have a slight different critera for considering
        # if an annotation is valid
        if "keypoints" not in anno[0]:
            return True

        return False

    def __getitem__(self, idx):
        img, anno = super().__getitem__(idx)

        anno = [aa for aa in anno if aa["iscrowd"] == 0]  # filter crowd annotations
        boxes = [aa["bbox"] for aa in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")
        classes = [aa["category_id"] for aa in anno]
        classes = [self.class_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        target = target.clip_to_image(remove_empty=True)

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target, idx
