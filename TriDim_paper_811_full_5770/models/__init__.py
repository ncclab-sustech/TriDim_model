"""TriDim model variants used in the paper experiments.

Each module exposes ``Model`` as the training entry point.  The seven variants
are kept as separate modules because these are the exact implementations used
for the reported C/S/L axis and block-level ablations. Historical module names
use K for the within-patch S axis and T for the across-patch L axis so that
released checkpoints remain load-compatible.
"""

from . import eeg_mixer_v11_1_spatial_multilevel
from . import eeg_mixer_v11_1_spatial_multilevel_noC
from . import eeg_mixer_v11_1_spatial_multilevel_noK
from . import eeg_mixer_v11_1_spatial_multilevel_noT
from . import eeg_mixer_v11_1_spatial_multilevel_onlyC
from . import eeg_mixer_v11_1_spatial_multilevel_onlyK
from . import eeg_mixer_v11_1_spatial_multilevel_onlyT
from . import eeg_mixer_v11_1_spatial_multilevel_noxattn
from . import eeg_mixer_v11_1_spatial_multilevel_sharedffn
from . import eeg_mixer_v11_1_spatial_multilevel_indepattn
from . import eeg_mixer_v11_1_spatial_multilevel_seqckt
from . import eeg_mixer_v11_1_spatial_multilevel_flattf
from . import eeg_mixer_v11_1_spatial_multilevel_crisscross

__all__ = [
    "eeg_mixer_v11_1_spatial_multilevel",
    "eeg_mixer_v11_1_spatial_multilevel_noC",
    "eeg_mixer_v11_1_spatial_multilevel_noK",
    "eeg_mixer_v11_1_spatial_multilevel_noT",
    "eeg_mixer_v11_1_spatial_multilevel_onlyC",
    "eeg_mixer_v11_1_spatial_multilevel_onlyK",
    "eeg_mixer_v11_1_spatial_multilevel_onlyT",
    "eeg_mixer_v11_1_spatial_multilevel_noxattn",
    "eeg_mixer_v11_1_spatial_multilevel_sharedffn",
    "eeg_mixer_v11_1_spatial_multilevel_indepattn",
    "eeg_mixer_v11_1_spatial_multilevel_seqckt",
    "eeg_mixer_v11_1_spatial_multilevel_flattf",
    "eeg_mixer_v11_1_spatial_multilevel_crisscross",
]
