import torch
import pickle
from collections import OrderedDict
import pdb


class Checkpointer:
    def __init__(self, cfg, model, optimizer=None, scheduler=None):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.c2_stage_names = {"res50": ["1.2", "2.3", "3.5", "4.2"], "res101": ["1.2", "2.3", "3.22", "4.2"]}
        self.ckpt_iter = self.load()

    @staticmethod
    def rename_basic_resnet_weights(layer_keys):
        layer_keys = [k.replace("_", ".") for k in layer_keys]
        layer_keys = [k.replace(".w", ".weight") for k in layer_keys]
        layer_keys = [k.replace(".bn", "_bn") for k in layer_keys]
        layer_keys = [k.replace(".b", ".bias") for k in layer_keys]
        layer_keys = [k.replace("_bn.s", "_bn.scale") for k in layer_keys]
        layer_keys = [k.replace(".biasranch", ".branch") for k in layer_keys]
        layer_keys = [k.replace("bbox.pred", "bbox_pred") for k in layer_keys]
        layer_keys = [k.replace("cls.score", "cls_score") for k in layer_keys]
        layer_keys = [k.replace("res.conv1_", "conv1_") for k in layer_keys]

        # Affine-Channel -> BatchNorm enaming
        layer_keys = [k.replace("_bn.scale", "_bn.weight") for k in layer_keys]

        # Make torchvision-compatible
        layer_keys = [k.replace("conv1_bn.", "bn1.") for k in layer_keys]
        layer_keys = [k.replace("res2.", "layer1.") for k in layer_keys]
        layer_keys = [k.replace("res3.", "layer2.") for k in layer_keys]
        layer_keys = [k.replace("res4.", "layer3.") for k in layer_keys]
        layer_keys = [k.replace("res5.", "layer4.") for k in layer_keys]
        layer_keys = [k.replace(".branch2a.", ".conv1.") for k in layer_keys]
        layer_keys = [k.replace(".branch2a_bn.", ".bn1.") for k in layer_keys]
        layer_keys = [k.replace(".branch2b.", ".conv2.") for k in layer_keys]
        layer_keys = [k.replace(".branch2b_bn.", ".bn2.") for k in layer_keys]
        layer_keys = [k.replace(".branch2c.", ".conv3.") for k in layer_keys]
        layer_keys = [k.replace(".branch2c_bn.", ".bn3.") for k in layer_keys]
        layer_keys = [k.replace(".branch1.", ".downsample.0.") for k in layer_keys]
        layer_keys = [k.replace(".branch1_bn.", ".downsample.1.") for k in layer_keys]

        # GroupNorm
        layer_keys = [k.replace("conv1.gn.s", "bn1.weight") for k in layer_keys]
        layer_keys = [k.replace("conv1.gn.bias", "bn1.bias") for k in layer_keys]
        layer_keys = [k.replace("conv2.gn.s", "bn2.weight") for k in layer_keys]
        layer_keys = [k.replace("conv2.gn.bias", "bn2.bias") for k in layer_keys]
        layer_keys = [k.replace("conv3.gn.s", "bn3.weight") for k in layer_keys]
        layer_keys = [k.replace("conv3.gn.bias", "bn3.bias") for k in layer_keys]
        layer_keys = [k.replace("downsample.0.gn.s", "downsample.1.weight") for k in layer_keys]
        layer_keys = [k.replace("downsample.0.gn.bias", "downsample.1.bias") for k in layer_keys]

        return layer_keys

    def rename_resnet_weights(self, weights, stage_names):
        original_keys = sorted(weights.keys())
        layer_keys = sorted(weights.keys())

        # for X-101, rename output to fc1000 to avoid conflicts afterwards
        layer_keys = [k if k != "pred_b" else "fc1000_b" for k in layer_keys]
        layer_keys = [k if k != "pred_w" else "fc1000_w" for k in layer_keys]

        layer_keys = self.rename_basic_resnet_weights(layer_keys)
        layer_keys = self.rename_fpn_weights(layer_keys, stage_names)

        key_map = {k: v for k, v in zip(original_keys, layer_keys)}
        new_weights = OrderedDict()
        for k in original_keys:
            v = weights[k]
            if "_momentum" in k:
                continue
            if 'weight_order' in k:
                continue
            w = torch.from_numpy(v)
            new_weights[key_map[k]] = w

        return new_weights

    @staticmethod
    def rename_fpn_weights(layer_keys, stage_names):
        for mapped_idx, stage_name in enumerate(stage_names, 1):
            suffix = ""
            if mapped_idx < 4:
                suffix = ".lateral"
            layer_keys = [k.replace(f"fpn.inner.layer{stage_name}.sum{suffix}", f"fpn_inner{mapped_idx}")
                          for k in layer_keys]

            layer_keys = [k.replace(f"fpn.layer{stage_name}.sum", f"fpn_layer{mapped_idx}") for k in layer_keys]

        return layer_keys

    @staticmethod
    def _load_c2_pickled_weights(file_path):
        with open(file_path, "rb") as f:
            if torch._six.PY3:
                data = pickle.load(f, encoding="latin1")
            else:
                data = pickle.load(f)

        return data["blobs"] if "blobs" in data else data

    def rename_dcn_weights(self, state_dict):
        import re
        layer_keys = sorted(state_dict.keys())
        for ix, stage_with_dcn in enumerate(self.cfg.stage_with_dcn, 1):
            if not stage_with_dcn:
                continue
            for old_key in layer_keys:
                pattern = ".*layer{}.*conv2.*".format(ix)
                r = re.match(pattern, old_key)
                if r is None:
                    continue
                for param in ["weight", "bias"]:
                    if old_key.find(param) is -1:
                        continue
                    new_key = old_key.replace("conv2.{}".format(param), "conv2.conv.{}".format(param))
                    state_dict[new_key] = state_dict[old_key]
                    del state_dict[old_key]
        return state_dict

    def load_resnet_c2_format(self, f):
        state_dict = self._load_c2_pickled_weights(f)
        stages = self.c2_stage_names[self.cfg.backbone]
        state_dict = self.rename_resnet_weights(state_dict, stages)
        state_dict = self.rename_dcn_weights(state_dict)
        return dict(model=state_dict)

    @staticmethod
    def align_load(model, loaded_state_dict):
        """
        align the model state_dict() and the pretrained weight's state_dict(), then load it.
        """
        model_state_dict = model.state_dict()

        current_keys = sorted(list(model_state_dict.keys()))
        loaded_keys = sorted(list(loaded_state_dict.keys()))
        # get a matrix of string matches, where each (i, j) entry correspond to the size of the
        # loaded_key string, if it matches
        match_matrix = [len(j) if i.endswith(j) else 0 for i in current_keys for j in loaded_keys]
        match_matrix = torch.as_tensor(match_matrix).view(len(current_keys), len(loaded_keys))
        max_match_size, idxs = match_matrix.max(1)
        # remove indices that correspond to no-match
        idxs[max_match_size == 0] = -1

        for idx_new, idx_old in enumerate(idxs.tolist()):
            if idx_old == -1:
                continue
            key = current_keys[idx_new]
            key_old = loaded_keys[idx_old]
            model_state_dict[key] = loaded_state_dict[key_old]

        model.load_state_dict(model_state_dict)

    def save(self, cur_iter):
        state_dict = {'model': self.model.state_dict(),
                      'optimizer': self.optimizer.state_dict(),
                      'scheduler': self.scheduler.state_dict(),
                      'iteration': cur_iter}

        save_file = f'weights/{self.cfg.backbone}_{cur_iter}.pth'
        print(f'Saving checkpoint to {save_file}')
        torch.save(state_dict, save_file)

    def load(self):
        if self.cfg.resume is not None:
            print(f'Resume training from {self.cfg.resume}.')
            ckpt = torch.load(self.cfg.resume)
            assert 'model' in ckpt and 'optimizer' in ckpt and 'scheduler' in ckpt, 'ckpt error.'
            self.model.load_state_dict(ckpt['model'], strict=True)
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.scheduler.load_state_dict(ckpt['scheduler'])
            return ckpt['iteration']
        else:
            print(f'Initialize training from {self.cfg.weight}.')
            ckpt = self.load_resnet_c2_format(self.cfg.weight)
            self.align_load(self.model, ckpt['model'])
            return 0

# https://dl.fbaipublicfiles.com/detectron/ImageNetPretrained/MSRA/R-50.pkl
# https://dl.fbaipublicfiles.com/detectron/ImageNetPretrained/47261647/R-50-GN.pkl
# https://dl.fbaipublicfiles.com/detectron/ImageNetPretrained/MSRA/R-101.pkl
# https://dl.fbaipublicfiles.com/detectron/ImageNetPretrained/47592356/R-101-GN.pkl
# https://dl.fbaipublicfiles.com/detectron/ImageNetPretrained/20171220/X-101-32x8d.pkl
