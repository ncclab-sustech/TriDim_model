"""SEED-V 10-second window variant."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..plan import DatasetPlan
from .seed_v import _scan_seed_v, _write_seed_v_zarr

SEGMENT_SEC = 10.0


def scan_seed_v_10s_plan(root: Path, smoke: bool = False) -> DatasetPlan:
    return _scan_seed_v(root, segment_seconds=SEGMENT_SEC, dataset_key="SEED_V_10s", smoke=smoke)


def write_seed_v_10s_zarr(root_path: Path, dataset_dir: Path, plan: DatasetPlan) -> dict[str, Any]:
    return _write_seed_v_zarr(root_path, dataset_dir, plan, log_tag="SEED_V_10s")
