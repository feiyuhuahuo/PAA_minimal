import torch
import torchvision
import PIL
from data.transforms import train_aug, val_aug
from utils.box_list import BoxList
import pdb


class ImageList:
    def __init__(self, img, ori_size, id):
        self.img = img
        self.ori_size = ori_size
        self.id = id

    def to_device(self, device):
        self.tensors = self.tensors.to(device)

    def __repr__(self):
        if isinstance(self.img, torch.Tensor):
            s = f'\nimg: {self.img.shape}, {self.img.dtype}, {self.img.device}, need_grad: {self.img.requires_grad}'
        elif isinstance(self.img, PIL.Image.Image):
            s = f'\nimg: {type(self.img)}'
        else:
            raise TypeError(f'Unrecognized img type, got {type(self.img)}.')

        for k, v in vars(self).items():
            if k != 'img':
                s += f'\n{k}: {v}'

        return s + '\n'


class COCODataset(torchvision.datasets.coco.CocoDetection):
    def __init__(self, cfg, valing):
        self.cfg = cfg
        self.valing = valing

        img_path = cfg.train_imgs if not valing else cfg.val_imgs
        ann_file = cfg.train_ann if not valing else cfg.val_ann
        super().__init__(img_path, ann_file)
        self.ids = sorted(self.ids)  # sort indices for reproducible results

        if not valing:
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)

                if not self.has_valid_annotation(anno):  # filter images without detection annotations
                    self.ids.remove(img_id)

            self.aug = train_aug
        else:
            self.aug = val_aug

        self.to_contiguous_id = {v: i + 1 for i, v in enumerate(self.coco.getCatIds())}
        self.to_category_id = {v: k for k, v in self.to_contiguous_id.items()}
        self.id_img_map = {k: v for k, v in enumerate(self.ids)}

    def get_img_info(self, index):
        img_id = self.id_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data

    @staticmethod
    def has_valid_annotation(anno):
        if len(anno) == 0:
            return False
        # if all boxes have close to zero area, there is no annotation
        if all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno):
            return False

        return True

    def __getitem__(self, index):
        img, anno = super().__getitem__(index)

        anno = [aa for aa in anno if aa["iscrowd"] == 0]  # filter crowd annotations
        box = [aa["bbox"] for aa in anno]
        box = torch.as_tensor(box).reshape(-1, 4)

        category = [aa["category_id"] for aa in anno]
        category = [self.to_contiguous_id[c] for c in category]

        # img is a PIL object, and it's size = (img_width, img_height)
        img_list = ImageList(img, ori_size=img.size, id=index)

        box_list = BoxList(box, img.size, 'x1y1wh', label=torch.tensor(category))
        box_list.convert_mode('x1y1x2y2')
        box_list.clip_to_image(remove_empty=True)

        img_list, box_list = self.aug(img_list, box_list, self.cfg)

        return img_list, box_list

    def __len__(self):
        if (not self.valing) or (self.cfg.val_num == -1):
            return len(self.ids)
        else:
            return min(self.cfg.val_num, len(self.ids))
