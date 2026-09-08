"""Zarr store creation and batch writing utilities."""
from __future__ import annotations

import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numcodecs
import numpy as np
import zarr

from .constants import SUBJECT_ID_STR_DTYPE, TARGET_SFREQ
from .plan import DatasetPlan
from .validation import sample_order_sha256


def _require_zarr_v2() -> None:
    try:
        major = int(str(zarr.__version__).split(".", 1)[0])
    except (AttributeError, ValueError):
        major = 0
    if major >= 3:
        raise RuntimeError(
            "This writer uses the Zarr 2 storage API. Install the pinned "
            "requirements with 'zarr<3' before preprocessing."
        )


def create_zarr_store(dataset_dir: Path, plan: DatasetPlan) -> zarr.Group:
    _require_zarr_v2()
    zarr_path = dataset_dir / f"{plan.dataset_key}.zarr"
    if zarr_path.exists():
        shutil.rmtree(zarr_path)
    compressor = numcodecs.Blosc(cname="zstd", clevel=3, shuffle=numcodecs.Blosc.SHUFFLE)
    n = len(plan.rows)
    bytes_per_sample = max(1, plan.c_max * plan.t * np.dtype("float32").itemsize)
    target_chunk_bytes = 16 * 1024 * 1024
    chunk_n = max(1, min(64, n, target_chunk_bytes // bytes_per_sample))
    meta_chunk_n = max(1, min(4096, n))
    root = zarr.open_group(str(zarr_path), mode="w")
    root.create_dataset(
        "signals",
        shape=(n, plan.c_max, plan.t),
        chunks=(chunk_n, plan.c_max, plan.t),
        dtype="float32",
        compressor=compressor,
        fill_value=0.0,
    )
    root.create_dataset(
        "channel_mask",
        shape=(n, plan.c_max),
        chunks=(meta_chunk_n, plan.c_max),
        dtype="bool",
        compressor=compressor,
        fill_value=False,
    )
    root.create_dataset(
        "bad_channel_mask",
        shape=(n, plan.c_max),
        chunks=(meta_chunk_n, plan.c_max),
        dtype="bool",
        compressor=compressor,
        fill_value=False,
    )
    root.create_dataset(
        "channel_counts",
        shape=(n,),
        chunks=(meta_chunk_n,),
        dtype="uint16",
        compressor=compressor,
        fill_value=0,
    )
    root.create_dataset(
        "labels",
        shape=(n,),
        chunks=(meta_chunk_n,),
        dtype="int32",
        compressor=compressor,
        fill_value=-1,
    )
    root.create_dataset(
        "label_values",
        shape=(n,),
        chunks=(meta_chunk_n,),
        dtype="float32",
        compressor=compressor,
        fill_value=np.nan,
    )
    root.create_dataset(
        "subject_ids",
        shape=(n,),
        chunks=(meta_chunk_n,),
        dtype=SUBJECT_ID_STR_DTYPE,
        compressor=compressor,
        fill_value="",
    )
    root.create_dataset(
        "qc_flags",
        shape=(n,),
        chunks=(meta_chunk_n,),
        dtype="uint32",
        compressor=compressor,
        fill_value=0,
    )
    root.create_dataset(
        "valid_time_samples",
        shape=(n,),
        chunks=(meta_chunk_n,),
        dtype="int32",
        compressor=compressor,
        fill_value=0,
    )
    notch_attr = plan.dataset_config.get("notch_hz", "per_sample")
    root.attrs.update(
        {
            "schema_version": "downstream_zarr_v1",
            "dataset_key": plan.dataset_key,
            "task_type": plan.task_type,
            "sampling_rate": int(TARGET_SFREQ),
            "signal_unit": "uV",
            "bandpass_low": float(plan.dataset_config.get("bandpass_low", 0.3)),
            "bandpass_high": float(plan.dataset_config.get("bandpass_high", 75.0)),
            "notch_hz": notch_attr,
            "channel_order": "source",
            "normalization": "none",
            "segment_seconds": plan.segment_seconds,
            "n_samples": n,
            "c_max": plan.c_max,
            "t": plan.t,
            "sample_order_sha256": sample_order_sha256(plan.rows),
        }
    )
    return root


def row_groups_by_source(rows: list[dict[str, Any]]) -> dict[str, list[tuple[int, dict[str, Any]]]]:
    groups: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for idx, row in enumerate(rows):
        groups[str(row["source_relpath"])].append((idx, row))
    return dict(groups)


def write_batch(
    root: zarr.Group,
    indices: list[int],
    signals: list[np.ndarray],
    channel_masks: list[np.ndarray],
    bad_masks: list[np.ndarray],
    channel_counts: list[int],
    labels: list[int],
    label_values: list[float],
    subject_ids: list[str],
    qc_flags: list[int],
    valid_time_samples: list[int],
) -> None:
    if not indices:
        raise ValueError("Cannot write an empty Zarr batch")
    order = np.argsort(indices)
    sorted_indices = [indices[i] for i in order]
    contiguous = sorted_indices == list(range(sorted_indices[0], sorted_indices[-1] + 1))
    packed_signals = np.stack([signals[i] for i in order]).astype(np.float32, copy=False)
    packed_channel_masks = np.stack([channel_masks[i] for i in order])
    packed_bad_masks = np.stack([bad_masks[i] for i in order])
    packed_counts = np.asarray([channel_counts[i] for i in order], dtype=np.uint16)
    packed_labels = np.asarray([labels[i] for i in order], dtype=np.int32)
    packed_values = np.asarray([label_values[i] for i in order], dtype=np.float32)
    packed_subjects = np.asarray([subject_ids[i] for i in order], dtype=SUBJECT_ID_STR_DTYPE)
    packed_qc = np.asarray([qc_flags[i] for i in order], dtype=np.uint32)
    packed_valid = np.asarray([valid_time_samples[i] for i in order], dtype=np.int32)

    if contiguous:
        sl = slice(sorted_indices[0], sorted_indices[-1] + 1)
        root["signals"][sl] = packed_signals
        root["channel_mask"][sl] = packed_channel_masks
        root["bad_channel_mask"][sl] = packed_bad_masks
        root["channel_counts"][sl] = packed_counts
        root["labels"][sl] = packed_labels
        root["label_values"][sl] = packed_values
        root["subject_ids"][sl] = packed_subjects
        root["qc_flags"][sl] = packed_qc
        root["valid_time_samples"][sl] = packed_valid
    else:
        for local_i, global_i in enumerate(sorted_indices):
            root["signals"][global_i] = packed_signals[local_i]
            root["channel_mask"][global_i] = packed_channel_masks[local_i]
            root["bad_channel_mask"][global_i] = packed_bad_masks[local_i]
            root["channel_counts"][global_i] = packed_counts[local_i]
            root["labels"][global_i] = packed_labels[local_i]
            root["label_values"][global_i] = packed_values[local_i]
            root["subject_ids"][global_i] = packed_subjects[local_i]
            root["qc_flags"][global_i] = packed_qc[local_i]
            root["valid_time_samples"][global_i] = packed_valid[local_i]
