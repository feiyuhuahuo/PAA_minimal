import torch
from build_stuff import _C
import copy
import pdb


class BoxList:
    def __init__(self, box, img_size, mode, **kwargs):
        assert isinstance(box, torch.Tensor), f'box should be a tensor, got {type(box)}'
        assert box.size(1) == 4, f'The last dimension of box should be 4, got {box.size(1)}'
        assert mode in ('x1y1x2y2', 'x1y1wh', 'xcycwh'), "mode should be 'x1y1x2y2' or 'x1y1wh' or 'xcycwh'"

        self.box = box
        self.img_size = img_size
        self.mode = mode

        for k, v in kwargs.items():
            setattr(self, k, v)

    def convert_mode(self, to_mode):
        TO_REMOVE = 1
        if self.mode == to_mode:
            print(f'mode already is {to_mode}, nothing changed.')
        else:
            if to_mode == 'x1y1x2y2':
                if self.mode == 'x1y1wh':
                    x1, y1, w, h = self.box.split(1, dim=1)
                    x2 = x1 + (w - TO_REMOVE).clamp(min=0)
                    y2 = y1 + (h - TO_REMOVE).clamp(min=0)
                    self.box = torch.cat((x1, y1, x2, y2), dim=1)
                    self.mode = 'x1y1x2y2'
                elif self.mode == 'xcycwh':
                    raise NotImplementedError

            elif to_mode == 'x1y1wh':
                if self.mode == 'x1y1x2y2':
                    x1, y1, x2, y2 = self.box.split(1, dim=1)
                    self.box = torch.cat((x1, y1, x2 - x1 + TO_REMOVE, y2 - y1 + TO_REMOVE), dim=1)
                    self.mode = 'x1y1wh'
                elif self.mode == 'xcycwh':
                    raise NotImplementedError

            elif to_mode == 'xcycwh':
                raise NotImplementedError

            else:
                raise ValueError('Unrecognized mode.')

    def resize(self, new_size):
        ratios = [float(s) / float(s_ori) for s, s_ori in zip(new_size, self.img_size)]

        if self.mode == 'x1y1x2y2':
            x1, y1, x2, y2 = self.box.split(1, dim=1)
            x1 *= ratios[0]
            x2 *= ratios[0]
            y1 *= ratios[1]
            y2 *= ratios[1]
            self.box = torch.cat((x1, y1, x2, y2), dim=1)

        elif self.mode == 'x1y1wh' or self.mode == 'xcycwh':
            raise NotImplementedError

        self.img_size = new_size

    def box_flip(self, method='h_flip'):
        img_w, img_h = self.img_size

        if self.mode == 'x1y1x2y2':
            x1, y1, x2, y2 = self.box.split(1, dim=1)
            if method == 'h_flip':
                TO_REMOVE = 1
                new_x1 = img_w - x2 - TO_REMOVE
                new_x2 = img_w - x1 - TO_REMOVE
                new_y1 = y1
                new_y2 = y2
            elif method == 'v_flip':
                new_x1 = x1
                new_x2 = x2
                new_y1 = img_h - y2
                new_y2 = img_h - y1
            else:
                raise ValueError(f'flip method should in (h_flip, v_flip), got {method}')

            self.box = torch.cat((new_x1, new_y1, new_x2, new_y2), dim=1)

        elif self.mode == 'x1y1wh' or self.mode == 'xcycwh':
            raise NotImplementedError

    def to_cpu(self):
        for k, v in vars(self).items():
            if isinstance(v, torch.Tensor):
                setattr(self, k, v.cpu())

    def to_gpu(self, gpu):
        for k, v in vars(self).items():
            if isinstance(v, torch.Tensor):
                assert not v.requires_grad, 'Tensors that require grad can not be allocated.'
                setattr(self, k, v.to(gpu))

    def __getitem__(self, item):
        # When to get a part of the box_list, itself should not be changed, so return a new box_list.
        new_box_list = copy.deepcopy(self)  # use deepcopy just in case
        for k, v in vars(new_box_list).items():
            if k not in ('img_size', 'mode'):
                if isinstance(v, torch.Tensor):
                    setattr(new_box_list, k, v[item])
                else:
                    raise NotImplementedError('Index op for non-tensor attr has not been implemented.')

        # the ids of two objects may be the same because of their non-overlapping lifetime, use 'is' just in case.
        assert new_box_list is not self
        return new_box_list

    def clip_to_image(self, remove_empty=True):
        TO_REMOVE = 1
        self.box[:, 0].clamp_(min=0, max=self.img_size[0] - TO_REMOVE)
        self.box[:, 1].clamp_(min=0, max=self.img_size[1] - TO_REMOVE)
        self.box[:, 2].clamp_(min=0, max=self.img_size[0] - TO_REMOVE)
        self.box[:, 3].clamp_(min=0, max=self.img_size[1] - TO_REMOVE)

        if remove_empty:
            keep = (self.box[:, 3] > self.box[:, 1]) & (self.box[:, 2] > self.box[:, 0])
            self.box = self.box[keep]

    def remove_small_box(self, min_size):
        assert self.mode == 'x1y1x2y2', 'Incorrect mode when removing small boxes.'
        ws, hs = (self.box[:, 2] - self.box[:, 0]), (self.box[:, 3] - self.box[:, 1])
        keep = ((ws >= min_size) & (hs >= min_size)).nonzero().squeeze(1)
        self.box = self.box[keep]

    def area(self):
        box = self.box
        TO_REMOVE = 1
        if self.mode == 'x1y1x2y2':
            area = (box[:, 2] - box[:, 0] + TO_REMOVE) * (box[:, 3] - box[:, 1] + TO_REMOVE)
        elif self.mode in ('x1y1wh', 'xcycwh'):
            area = box[:, 2] * box[:, 3]
        else:
            raise ValueError('Unrecognized mode.')

        return area

    def __repr__(self):
        s = f'\nbox: shape: {self.box.shape}, dtype: {self.box.dtype}, device: {self.box.device}, ' \
            f'grad: {self.box.requires_grad}'

        for k, v in vars(self).items():
            if k != 'box':
                if isinstance(v, torch.Tensor):
                    s += f'\n{k}: shape: {v.shape}, dtype: {v.dtype}, device: {v.device}, grad: {v.requires_grad}'
                else:
                    s += f'\n{k}: {v}'

        return s + '\n'


def boxlist_ml_nms(box_list, nms_thresh, max_proposals=-1):
    if nms_thresh <= 0:
        return box_list

    assert box_list.mode == 'x1y1x2y2', f'mode here should be x1y1x2y2, got {box_list.mode}, need to check the code.'
    box, score, label = box_list.box, box_list.score, box_list.label
    keep = _C.ml_nms(box, score, label.float(), nms_thresh)

    if max_proposals > 0:
        keep = keep[: max_proposals]

    return box_list[keep]


def boxlist_iou(boxlist1, boxlist2):
    area1 = boxlist1.area()
    area2 = boxlist2.area()

    box1, box2 = boxlist1.box, boxlist2.box

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    TO_REMOVE = 1

    wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou


def cat_boxlist(box_list_fpn):
    # Concatenates a list of BoxList (having the same image size) into a single BoxList
    assert isinstance(box_list_fpn, (list, tuple))
    assert all(isinstance(bbox, BoxList) for bbox in box_list_fpn)

    img_size = box_list_fpn[0].img_size
    assert all(bbox.img_size == img_size for bbox in box_list_fpn)

    mode = box_list_fpn[0].mode
    assert all(bbox.mode == mode for bbox in box_list_fpn)

    attr = list(vars(box_list_fpn[0]).keys())
    assert all([list(vars(bbox).keys()) == attr for bbox in box_list_fpn])

    kwargs = {}
    for k in attr:
        if k == 'box':
            cat_box = torch.cat([getattr(box_list, k) for box_list in box_list_fpn], dim=0)
        elif k in ('mode', 'img_size'):
            pass
        else:
            aa = []
            for box_list in box_list_fpn:
                v = getattr(box_list, k)
                if isinstance(v, torch.Tensor):
                    aa.append(v)
                else:
                    raise NotImplementedError('Cat op for Non-tensor attr has not been implemented.')

            kwargs[k] = torch.cat(aa, dim=0)

    return BoxList(cat_box, img_size, mode, **kwargs)
