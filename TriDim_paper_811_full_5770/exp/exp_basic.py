"""Base experiment class and the paper model registry."""

import os

import torch

from models import (
    eeg_mixer_v11_1_spatial_multilevel,
    eeg_mixer_v11_1_spatial_multilevel_noC,
    eeg_mixer_v11_1_spatial_multilevel_noK,
    eeg_mixer_v11_1_spatial_multilevel_noT,
    eeg_mixer_v11_1_spatial_multilevel_onlyC,
    eeg_mixer_v11_1_spatial_multilevel_onlyK,
    eeg_mixer_v11_1_spatial_multilevel_onlyT,
    eeg_mixer_v11_1_spatial_multilevel_noxattn,
    eeg_mixer_v11_1_spatial_multilevel_sharedffn,
    eeg_mixer_v11_1_spatial_multilevel_indepattn,
    eeg_mixer_v11_1_spatial_multilevel_seqckt,
    eeg_mixer_v11_1_spatial_multilevel_flattf,
    eeg_mixer_v11_1_spatial_multilevel_crisscross,
)


class Exp_Basic:
    """Shared device and model setup for classification experiments."""

    MODEL_REGISTRY = {
        "eeg_mixer_v11_1_spatial_multilevel": eeg_mixer_v11_1_spatial_multilevel,
        "eeg_mixer_v11_2_no_stem": eeg_mixer_v11_1_spatial_multilevel,
        "eeg_mixer_v11_1_spatial_multilevel_noC": eeg_mixer_v11_1_spatial_multilevel_noC,
        "eeg_mixer_v11_1_spatial_multilevel_noK": eeg_mixer_v11_1_spatial_multilevel_noK,
        "eeg_mixer_v11_1_spatial_multilevel_noT": eeg_mixer_v11_1_spatial_multilevel_noT,
        "eeg_mixer_v11_1_spatial_multilevel_onlyC": eeg_mixer_v11_1_spatial_multilevel_onlyC,
        "eeg_mixer_v11_1_spatial_multilevel_onlyK": eeg_mixer_v11_1_spatial_multilevel_onlyK,
        "eeg_mixer_v11_1_spatial_multilevel_onlyT": eeg_mixer_v11_1_spatial_multilevel_onlyT,
        "eeg_mixer_v11_1_spatial_multilevel_noxattn": eeg_mixer_v11_1_spatial_multilevel_noxattn,
        "eeg_mixer_v11_1_spatial_multilevel_sharedffn": eeg_mixer_v11_1_spatial_multilevel_sharedffn,
        "eeg_mixer_v11_1_spatial_multilevel_indepattn": eeg_mixer_v11_1_spatial_multilevel_indepattn,
        "eeg_mixer_v11_1_spatial_multilevel_seqckt": eeg_mixer_v11_1_spatial_multilevel_seqckt,
        "eeg_mixer_v11_1_spatial_multilevel_flattf": eeg_mixer_v11_1_spatial_multilevel_flattf,
        "eeg_mixer_v11_1_spatial_multilevel_crisscross": eeg_mixer_v11_1_spatial_multilevel_crisscross,
    }

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
