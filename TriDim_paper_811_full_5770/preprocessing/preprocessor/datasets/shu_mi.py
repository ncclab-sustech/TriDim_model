"""SHU-MI motor imagery dataset with content-hash deduplication."""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.io

from ..constants import TARGET_SFREQ
from ..dsp import extract_segment, finalize_segment, preprocess_continuous_uv
from ..io_utils import relpath, write_json, write_parquet_compat
from ..plan import DatasetPlan
from ..splits import apply_subject_disjoint_split
from ..validation import build_summary, qc_flag_mapping
from ..writer import create_zarr_store, row_groups_by_source, write_batch

SHU_LABELS = {0: "left", 1: "right"}
SEGMENT_SEC = 4.0
RAW_SFREQ = 250.0
SHU_CHANNELS = [
    "Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8", "FC1", "FC2", "FC5", "FC6",
    "Cz", "C3", "C4", "T3", "T4", "A1", "A2", "CP1", "CP2", "CP5", "CP6",
    "Pz", "P3", "P4", "T5", "T6", "PO3", "PO4", "Oz", "O1", "O2",
]
_MAT_RE = re.compile(r"^(?P<sub>sub-\d+)_ses-(?P<ses>\d+)_task_motorimagery_eeg\.mat$")


def _resolve_root(root: Path) -> Path:
    candidates = [root, root / "19228725", root / "mat"]
    for candidate in candidates:
        if (candidate / "mat").exists():
            return candidate
        if candidate.name == "mat" and candidate.exists():
            return candidate.parent
    if any(root.glob("sub-*_task_motorimagery_eeg.mat")):
        return root
    raise FileNotFoundError(f"No SHU-MI mat payload under {root}")


def _signal_sha256(data: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest()


def _dedup_trials(
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        by_hash[item["content_hash"]].append(item)

    kept: list[dict[str, Any]] = []
    dropped_duplicate = 0
    dropped_conflict = 0
    for items in by_hash.values():
        labels = {int(item["label_id"]) for item in items}
        if len(labels) > 1:
            dropped_conflict += len(items)
            continue
        items_sorted = sorted(
            items,
            key=lambda item: (item["source_relpath"], item["trial_index"]),
        )
        kept.append(items_sorted[0])
        dropped_duplicate += len(items_sorted) - 1
    return kept, {
        "dedup_dropped_duplicate_trials": int(dropped_duplicate),
        "dedup_dropped_conflict_trials": int(dropped_conflict),
        "dedup_unique_content_hashes": len(by_hash),
    }


def scan_shu_mi_plan(root: Path, smoke: bool = False) -> DatasetPlan:
    data_root = _resolve_root(root)
    mat_dir = data_root / "mat" if (data_root / "mat").exists() else data_root
    mat_files = sorted(mat_dir.glob("sub-*_task_motorimagery_eeg.mat"))
    if smoke:
        mat_files = mat_files[:2]
    if not mat_files:
        raise FileNotFoundError(f"No SHU-MI mat files under {mat_dir}")

    indices = list(range(len(SHU_CHANNELS)))
    channel_names = list(SHU_CHANNELS)

    candidates: list[dict[str, Any]] = []
    for mat_path in mat_files:
        match = _MAT_RE.match(mat_path.name)
        if match is None:
            continue
        subject_id = match.group("sub")
        session_id = f"{subject_id}_ses-{match.group('ses')}"
        mat = scipy.io.loadmat(str(mat_path))
        data = np.asarray(mat["data"], dtype=np.float64)
        labels = np.asarray(mat["labels"]).reshape(-1)
        if data.ndim != 3:
            raise ValueError(f"Unexpected SHU-MI data shape in {mat_path}: {data.shape}")
        n_trials = min(data.shape[0], labels.shape[0])
        for trial_idx in range(n_trials):
            raw_label = int(labels[trial_idx])
            if raw_label not in (1, 2):
                continue
            trial = data[trial_idx][indices]
            candidates.append(
                {
                    "subject_id": subject_id,
                    "session_id": session_id,
                    "source_relpath": relpath(mat_path, data_root),
                    "trial_index": int(trial_idx),
                    "label_id": int(raw_label - 1),
                    "label_raw": "left" if raw_label == 1 else "right",
                    "content_hash": _signal_sha256(trial),
                    "channel_count": len(channel_names),
                }
            )

    kept, dedup_stats = _dedup_trials(candidates)
    rows: list[dict[str, Any]] = []
    for item in kept:
        trial_idx = int(item["trial_index"])
        rows.append(
            {
                "sample_id": (
                    f"SHU_MI:{item['subject_id']}:{Path(item['source_relpath']).stem}"
                    f":trial{trial_idx:04d}"
                ),
                "dataset_key": "SHU_MI",
                "subject_id": item["subject_id"],
                "session_id": item["session_id"],
                "segment_id": f"{Path(item['source_relpath']).stem}_trial{trial_idx:04d}",
                "source_relpath": item["source_relpath"],
                "source_format": "mat",
                "segment_start_sec": 0.0,
                "segment_end_sec": SEGMENT_SEC,
                "label_id": int(item["label_id"]),
                "label_raw": item["label_raw"],
                "channel_names_json": json.dumps(channel_names),
                "raw_sfreq": RAW_SFREQ,
                "notch_hz": 50.0,
                "qc_flags": 0,
                "channel_count": int(item["channel_count"]),
                "trial_index": trial_idx,
                "content_hash": item["content_hash"],
            }
        )

    rows.sort(key=lambda r: (r["subject_id"], r["session_id"], r["trial_index"]))
    apply_subject_disjoint_split(rows)
    dataset_config = {
        "dataset_key": "SHU_MI",
        "source_root": str(data_root),
        "source_format": "mat",
        "segment_seconds": SEGMENT_SEC,
        "target_sampling_rate": TARGET_SFREQ,
        "signal_unit": "uV",
        "notch_hz": 50.0,
        "bandpass_low": 0.3,
        "bandpass_high": 75.0,
        "channel_rule": "retain the fixed 32-channel release order, including A1/A2",
        "sample_order": "subject/session/trial source order; independent of split assignment",
        "qc_flags": qc_flag_mapping(),
        "label_note": "Raw labels left/right preserved from mat labels {1,2}.",
        **dedup_stats,
    }
    return DatasetPlan(
        dataset_key="SHU_MI",
        rows=rows,
        c_max=len(channel_names),
        t=int(SEGMENT_SEC * TARGET_SFREQ),
        segment_seconds=SEGMENT_SEC,
        task_type="binary_classification",
        label_vocab={str(k): {"name": v} for k, v in SHU_LABELS.items()},
        dataset_config=dataset_config,
        channel_count_distribution={str(len(channel_names)): len(rows)},
        expected_n_samples=len(rows),
    )


def write_shu_mi_zarr(root_path: Path, dataset_dir: Path, plan: DatasetPlan) -> dict[str, Any]:
    data_root = _resolve_root(root_path)
    root = create_zarr_store(dataset_dir, plan)
    groups = row_groups_by_source(plan.rows)
    qc_counter: Counter[int] = Counter()
    label_counter: Counter[int] = Counter()
    start_time = time.time()
    indices = list(range(len(SHU_CHANNELS)))
    channel_names = list(SHU_CHANNELS)

    for file_idx, (source_rel, indexed_rows) in enumerate(groups.items(), start=1):
        mat_path = data_root / source_rel
        logging.info("SHU_MI %d/%d processing %s", file_idx, len(groups), source_rel)
        mat = scipy.io.loadmat(str(mat_path))
        data = np.asarray(mat["data"], dtype=np.float64)

        indices_out: list[int] = []
        signals: list[np.ndarray] = []
        channel_masks: list[np.ndarray] = []
        bad_masks: list[np.ndarray] = []
        channel_counts_list: list[int] = []
        labels: list[int] = []
        label_values: list[float] = []
        subject_ids: list[str] = []
        qc_flags_list: list[int] = []
        valid_time_list: list[int] = []

        for idx, row in indexed_rows:
            trial = np.asarray(data[int(row["trial_index"])][indices], dtype=np.float64)
            processed, __eff_high, file_qc = preprocess_continuous_uv(
                trial, RAW_SFREQ, notch_hz=50.0
            )
            segment, valid_samples, seg_qc = extract_segment(processed, 0.0, plan.t, TARGET_SFREQ)
            signal, channel_mask, bad_mask, final_qc = finalize_segment(
                segment,
                plan.c_max,
                plan.t,
                channel_names,
                file_qc | seg_qc,
                valid_samples,
                min_expected_channels=len(channel_names),
            )
            row["channel_names_json"] = json.dumps(channel_names)
            row["qc_flags"] = int(final_qc)
            indices_out.append(idx)
            signals.append(signal)
            channel_masks.append(channel_mask)
            bad_masks.append(bad_mask)
            channel_counts_list.append(len(channel_names))
            labels.append(int(row["label_id"]))
            label_values.append(float(row["label_id"]))
            subject_ids.append(str(row["subject_id"]))
            qc_flags_list.append(int(final_qc))
            valid_time_list.append(int(valid_samples))
            qc_counter[int(final_qc)] += 1
            label_counter[int(row["label_id"])] += 1

        write_batch(
            root,
            indices_out,
            signals,
            channel_masks,
            bad_masks,
            channel_counts_list,
            labels,
            label_values,
            subject_ids,
            qc_flags_list,
            valid_time_list,
        )

    write_parquet_compat(pd.DataFrame(plan.rows), dataset_dir / "sample_index.parquet")
    write_json(dataset_dir / "label_vocab.json", plan.label_vocab)
    write_json(dataset_dir / "dataset_config.json", plan.dataset_config)
    summary = build_summary(plan, label_counter, qc_counter, elapsed_sec=time.time() - start_time)
    write_json(dataset_dir / "conversion_summary.json", summary)
    return summary
