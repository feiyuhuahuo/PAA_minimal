import random
import math
from torchvision.transforms import functional as F
from utils.box_list import BoxList


def resize(img_list, box_list=None, min_size=None, max_size=None):
    if isinstance(min_size, int):
        size = min_size
    elif isinstance(min_size, tuple):
        size = random.randint(min_size[0], min_size[1])
    else:
        raise TypeError(f'The type of min_size_train shoule be int or tuple, got {type(min_size)}')

    assert img_list.img.size == img_list.ori_size, 'img size error when resizing.'
    w, h = img_list.ori_size

    if max_size is not None:
        min_original_size = float(min((w, h)))
        max_original_size = float(max((w, h)))
        if max_original_size / min_original_size * size > max_size:
            size = int(round(max_size * min_original_size / max_original_size))

    if (w <= h and w == size) or (h <= w and h == size):
        resize_h = h, resize_w = w
    else:
        if w < h:
            resize_w = size
            resize_h = int(size * h / w)
        else:
            resize_h = size
            resize_w = int(size * w / h)

    resized_img = F.resize(img_list.img, (resize_h, resize_w))
    img_list.img = resized_img
    img_list.resized_size = (resize_w, resize_h)

    if box_list is None:
        return img_list
    else:
        assert isinstance(box_list, BoxList), f'target error, should be a Boxlist, got a {type(box_list)}.'
        box_list.resize(new_size=(resize_w, resize_h))

    return img_list, box_list


def random_flip(img_list, box_list, h_prob=0.5, v_prob=None):
    if h_prob and random.random() < h_prob:
        new_img = F.hflip(img_list.img)
        img_list.img = new_img
        assert img_list.resized_size == box_list.img_size, 'img size != box size when flipping.'
        box_list.box_flip(method='h_flip')
    if v_prob and random.random() < v_prob:
        raise NotImplementedError('Vertical flip has not been implemented.')

    return img_list, box_list


def to_tensor(img_list):
    new_img = F.to_tensor(img_list.img)
    img_list.img = new_img
    return img_list


def normalize(img_list, mean=(102.9801, 115.9465, 122.7717), std=(1., 1., 1.)):
    new_img = img_list.img[[2, 1, 0]] * 255  # to BGR, 255
    new_img = F.normalize(new_img, mean=mean, std=std)
    img_list.img = new_img
    return img_list


def train_aug(img_list, box_list, cfg):
    img_list, box_list = resize(img_list, box_list, min_size=cfg.min_size_train, max_size=cfg.max_size_train)
    img_list, box_list = random_flip(img_list, box_list, h_prob=0.5)
    img_list = to_tensor(img_list)
    img_list = normalize(img_list)
    return img_list, box_list


def val_aug(img_list, box_list, cfg):
    img_list = resize(img_list, box_list=None, min_size=cfg.min_size_test, max_size=cfg.max_size_test)
    img_list = to_tensor(img_list)
    img_list = normalize(img_list)
    return img_list, None
