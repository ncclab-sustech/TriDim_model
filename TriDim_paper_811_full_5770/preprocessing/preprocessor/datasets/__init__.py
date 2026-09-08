"""Paper-dataset preprocessing registry."""

from __future__ import annotations

from .ad65 import scan_ad65_plan, write_ad65_zarr
from .bcic2a import scan_bcic2a_plan, write_bcic2a_zarr
from .faced_new import scan_faced_new_plan, write_faced_new_zarr
from .physionet_mi import scan_physionet_mi_plan, write_physionet_mi_zarr
from .seed import scan_seed_plan, write_seed_zarr
from .seed_v_10s import scan_seed_v_10s_plan, write_seed_v_10s_zarr
from .shu_mi import scan_shu_mi_plan, write_shu_mi_zarr
from .sleepedf import scan_sleepedf_plan, write_sleepedf_zarr


DATASET_REGISTRY: dict[str, dict] = {
    "ad65": {
        "scan": scan_ad65_plan,
        "write": write_ad65_zarr,
        "output_dir_name": "AD65_wsn",
        "default_root": "",
        "help": (
            "Path to AD65/OpenNeuro ds004504 containing "
            "sub-*_task-eyesclosed_eeg.set."
        ),
    },
    "bcic2a": {
        "scan": scan_bcic2a_plan,
        "write": write_bcic2a_zarr,
        "output_dir_name": "BCIC2A_wsn",
        "default_root": "",
        "help": "Path to the BCI Competition IV 2a raw/label directories.",
    },
    "faced_new": {
        "scan": scan_faced_new_plan,
        "write": write_faced_new_zarr,
        "output_dir_name": "FACED_new",
        "default_root": "",
        "help": "Path to the FACED release containing Processed_data_full.",
    },
    "physionet_mi": {
        "scan": scan_physionet_mi_plan,
        "write": write_physionet_mi_zarr,
        "output_dir_name": "Physionet_MI_wsn",
        "default_root": "",
        "help": "Path to the PhysioNet EEGMMIDB files.",
    },
    "seed": {
        "scan": scan_seed_plan,
        "write": write_seed_zarr,
        "output_dir_name": "SEED",
        "default_root": "",
        "help": "Path to the SEED preprocessed MAT files and label.mat.",
    },
    "seed_v_10s": {
        "scan": scan_seed_v_10s_plan,
        "write": write_seed_v_10s_zarr,
        "output_dir_name": "SEED_V_10s",
        "default_root": "",
        "help": "Path to the SEED-V CNT files and stimulus metadata.",
    },
    "shu_mi": {
        "scan": scan_shu_mi_plan,
        "write": write_shu_mi_zarr,
        "output_dir_name": "SHU_MI",
        "default_root": "",
        "help": "Path to the SHU-MI MAT release.",
    },
    "sleepedf": {
        "scan": scan_sleepedf_plan,
        "write": write_sleepedf_zarr,
        "output_dir_name": "SleepEDF_full",
        "default_root": "",
        "help": "Path to the Sleep-EDF Expanded cassette/telemetry release.",
    },
}
