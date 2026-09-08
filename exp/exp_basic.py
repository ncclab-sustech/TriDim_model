"""Base experiment class and the paper model registry."""

import os

import torch

from models import eeg_mixer_v11_1_spatial_multilevel


class Exp_Basic:
    """Shared device and model setup for classification experiments."""

    MODEL_REGISTRY = {"eeg_mixer_v11_1_spatial_multilevel": eeg_mixer_v11_1_spatial_multilevel}

    def __init__(self, args):
        self.args = args
        self.model_dict = dict(self.MODEL_REGISTRY)
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError

    def _acquire_device(self):
        if not self.args.use_gpu:
            print("Use CPU")
            return torch.device("cpu")

        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if not visible:
            visible = self.args.devices if self.args.use_multi_gpu else str(self.args.gpu)
            os.environ["CUDA_VISIBLE_DEVICES"] = visible
        mode = "multi" if self.args.use_multi_gpu else "single"
        print(f"Use GPU ({mode}): visible={visible} -> cuda:0")
        return torch.device("cuda:0")

    def _get_data(self):
        raise NotImplementedError

    def vali(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError
