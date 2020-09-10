import numpy as np
import torch
from torch import nn
from utils.box_list import BoxList
import pdb


class BufferList(nn.Module):
    def __init__(self, buffers=None):
        super().__init__()
        if buffers is not None:
            self.extend(buffers)

    def extend(self, buffers):
        offset = len(self)
        for i, buffer in enumerate(buffers):
            self.register_buffer(str(offset + i), buffer)
        return self

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.values())


class AnchorGenerator:
    def __init__(self, cfg):
        super().__init__()
        self.anchor_sizes = cfg.anchor_sizes
        self.aspect_ratios = cfg.aspect_ratios
        self.anchor_strides = cfg.anchor_strides
        assert len(self.anchor_strides) == len(self.anchor_sizes), "len(anchor_strides) != len(sizes)"

        cell_anchors = [generate_cell(stride, size, self.aspect_ratios).float()
                        for stride, size in zip(self.anchor_strides, self.anchor_sizes)]

        self.cell_anchors = BufferList(cell_anchors)

    def __call__(self, feature_maps):
        grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]

        anchor_list = []
        for size, stride, base_anchors in zip(grid_sizes, self.anchor_strides, self.cell_anchors):
            grid_height, grid_width = size
            shifts_x = torch.arange(0, grid_width * stride, step=stride, dtype=torch.float32)
            shifts_y = torch.arange(0, grid_height * stride, step=stride, dtype=torch.float32)
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

            anchor = (shifts.reshape(-1, 1, 4) + base_anchors.reshape(1, -1, 4)).reshape(-1, 4)
            one_list = BoxList(anchor, img_size=None, mode='x1y1x2y2')
            one_list.to_cuda()

            anchor_list.append(one_list)

        return anchor_list


def generate_cell(stride, sizes, aspect_ratios):
    # Generates a matrix of anchor boxes in (x1, y1, x2, y2) format. Anchors are centered on stride / 2.
    scales = np.array(sizes, dtype=np.float) / stride
    aspect_ratios = np.array(aspect_ratios, dtype=np.float)
    anchor = np.array([1, 1, stride, stride], dtype=np.float) - 0.5
    anchors = _ratio_enum(anchor, aspect_ratios)
    anchors = np.vstack([_scale_enum(anchors[i, :], scales) for i in range(anchors.shape[0])])

    return torch.from_numpy(anchors)


def _whctrs(anchor):
    # Return width, height, x center, and y center for an anchor.
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
    # Given a vector of widths (ws) and heights (hs) around a center (x_ctr, y_ctr), output a set of anchors.
    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors


def _ratio_enum(anchor, ratios):
    # Enumerate a set of anchors for each aspect ratio wrt an anchor.
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _scale_enum(anchor, scales):
    # Enumerate a set of anchors for each scale wrt an anchor.
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors
