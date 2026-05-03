import os
import torch
from models import eeg_basis_mixer_v2_nobasis, eeg_basis_mixer_v2_nobasis_onlyT, eeg_basis_mixer_v2_nobasis_noT, eeg_basis_mixer_v2_nobasis_layernorm, eeg_basis_mixer_v2_nobasis_mlla, eeg_mixer_v3_tconv, eeg_mixer_v4_outAttentionPoolingFusion
class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        # Only the nobasis baseline is registered in this release.
        self.model_dict = {
            "eeg_basis_mixer_v2_nobasis": eeg_basis_mixer_v2_nobasis,
            "eeg_basis_mixer_v2_nobasis_onlyT": eeg_basis_mixer_v2_nobasis_onlyT,
            "eeg_basis_mixer_v2_nobasis_noT": eeg_basis_mixer_v2_nobasis_noT,
            "eeg_basis_mixer_v2_nobasis_layernorm": eeg_basis_mixer_v2_nobasis_layernorm,
            "eeg_basis_mixer_v2_nobasis_mlla": eeg_basis_mixer_v2_nobasis_mlla,
            "eeg_mixer_v3_tconv": eeg_mixer_v3_tconv,
            "eeg_mixer_v4_outAttentionPoolingFusion": eeg_mixer_v4_outAttentionPoolingFusion
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            if self.args.use_multi_gpu:
                os.environ["CUDA_VISIBLE_DEVICES"] = self.args.devices
                device = torch.device("cuda:0")
                print("Use GPU (multi): visible={} -> cuda:0".format(self.args.devices))
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
                device = torch.device("cuda:0")
                print("Use GPU (single): visible={} -> cuda:0".format(self.args.gpu))
        else:
            device = torch.device("cpu")
            print("Use CPU")
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
