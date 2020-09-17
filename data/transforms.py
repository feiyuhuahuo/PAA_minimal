import random
import pdb
from torchvision.transforms import functional as F
from utils.box_list import BoxList


def resize(img_list, box_list=None, min_size=None, max_size=None):
    assert type(min_size) in (int, tuple), f'The type of min_size_train shoule be int or tuple, got {type(min_size)}.'
    if isinstance(min_size, tuple):
        min_size = random.randint(min_size[0], min_size[1])

    assert img_list.img.size == img_list.ori_size, 'img size error when resizing.'
    w, h = img_list.ori_size

    short_side, long_side = min(w, h), max(w, h)
    if min_size / short_side * long_side > max_size:
        scale = max_size / long_side
    else:
        scale = min_size / short_side

    new_h, new_w = int(scale * h), int(scale * w)
    assert (min(new_h, new_w)) <= min_size and (max(new_h, new_w) <= max_size), 'Scale error when resizing.'

    resized_img = F.resize(img_list.img, (new_h, new_w))
    img_list.img = resized_img
    img_list.resized_size = (new_w, new_h)

    if box_list is None:
        return img_list
    else:
        box_list.resize(new_size=(new_w, new_h))

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
