"""SEED emotion-recognition preprocessing for the paper's 10-s inputs."""
from __future__ import annotations

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

SEED_LABELS = {0: "negative", 1: "neutral", 2: "positive"}
RAW_TO_LABEL_ID = {-1: 0, 0: 1, 1: 2}
CANONICAL_TRIAL_LABELS = (1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1)
SEGMENT_SEC = 10.0
RAW_SFREQ = 200.0
SPLIT_SEED = 20260725
TRIAL_KEY_RE = re.compile(r".*_eeg(?P<trial>\d+)$", flags=re.IGNORECASE)

SEED_62_CHANNELS = [
    "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2",
    "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6",
    "FT8", "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8", "TP7", "CP5",
    "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "P7", "P5", "P3", "P1",
    "PZ", "P2", "P4", "P6", "P8", "PO7", "PO5", "PO3", "POZ", "PO4", "PO6",
    "PO8", "CB1", "O1", "OZ", "O2", "CB2",
]
C_MAX = len(SEED_62_CHANNELS)


def _subject_id(path: Path) -> str:
    return path.stem.split("_", 1)[0]


def _natural_file_key(path: Path) -> tuple[int, str]:
    subject = _subject_id(path)
    return (int(subject) if subject.isdigit() else 10**9, path.name.lower())


def _as_channel_time(array: np.ndarray, source: str) -> np.ndarray:
    trial = np.asarray(array, dtype=np.float64).squeeze()
    if trial.ndim != 2:
        raise ValueError(f"{source}: expected a 2-D trial, got {trial.shape}")
    if trial.shape[0] == C_MAX:
        return trial
    if trial.shape[1] == C_MAX:
        return trial.T
    raise ValueError(f"{source}: expected 62 channels, got {trial.shape}")


def _trial_arrays(mat: dict[str, Any]) -> list[tuple[int, str, np.ndarray]]:
    trials: list[tuple[int, str, np.ndarray]] = []
    for key, value in mat.items():
        match = TRIAL_KEY_RE.fullmatch(key)
        if match:
            trial_index = int(match.group("trial")) - 1
            trials.append((trial_index, key, _as_channel_time(value, key)))
    if trials:
        return sorted(trials, key=lambda item: item[0])

    # Some repackaged copies expose one [trial, channel, time] tensor.
    if "EEG" in mat:
        eeg = np.asarray(mat["EEG"])
        if eeg.ndim != 3:
            raise ValueError(f"EEG: expected 3 dimensions, got {eeg.shape}")
        return [
            (idx, "EEG", _as_channel_time(eeg[idx], f"EEG[{idx}]"))
            for idx in range(eeg.shape[0])
        ]
    return []


def _load_trial_labels(root: Path) -> tuple[int, ...]:
    label_files = sorted(root.rglob("label.mat"))
    if not label_files:
        return CANONICAL_TRIAL_LABELS
    mat = scipy.io.loadmat(str(label_files[0]))
    if "label" not in mat:
        raise ValueError(f"{label_files[0]} does not contain 'label'")
    labels = tuple(int(value) for value in np.asarray(mat["label"]).reshape(-1))
    if set(labels) - set(RAW_TO_LABEL_ID):
        raise ValueError(f"Unexpected SEED labels in {label_files[0]}: {sorted(set(labels))}")
    return labels


def scan_seed_plan(root: Path, smoke: bool = False) -> DatasetPlan:
    if not root.exists():
        raise FileNotFoundError(root)
    labels = _load_trial_labels(root)
    mat_files = [
        path for path in sorted(root.rglob("*.mat"), key=_natural_file_key)
        if path.name.lower() != "label.mat"
    ]
    if smoke:
        mat_files = mat_files[:2]

    rows: list[dict[str, Any]] = []
    channel_counts: Counter[str] = Counter()
    window_samples = int(round(SEGMENT_SEC * RAW_SFREQ))

    for mat_path in mat_files:
        trials = _trial_arrays(scipy.io.loadmat(str(mat_path)))
        if not trials:
            continue
        subject_id = _subject_id(mat_path)
        channel_counts[str(C_MAX)] += 1
        for trial_index, variable, trial in trials:
            if trial_index >= len(labels):
                raise ValueError(
                    f"{mat_path.name}: trial {trial_index + 1} has no entry in label.mat"
                )
            raw_label = labels[trial_index]
            label_id = RAW_TO_LABEL_ID[raw_label]
            n_windows = trial.shape[1] // window_samples
            for window_index in range(n_windows):
                start_sec = float(window_index) * SEGMENT_SEC
                rows.append({
                    "sample_id": (
                        f"SEED:{subject_id}:{mat_path.stem}:"
                        f"trial{trial_index + 1:02d}:win{window_index:03d}"
                    ),
                    "dataset_key": "SEED",
                    "subject_id": subject_id,
                    "session_id": mat_path.stem,
                    "trial_id": f"{mat_path.stem}_trial{trial_index + 1:02d}",
                    "segment_id": (
                        f"{mat_path.stem}_trial{trial_index + 1:02d}_"
                        f"win{window_index:03d}"
                    ),
                    "source_relpath": relpath(mat_path, root),
                    "source_format": "mat",
                    "source_variable": variable,
                    "source_trial_index": int(trial_index),
                    "segment_start_sec": start_sec,
                    "segment_end_sec": start_sec + SEGMENT_SEC,
                    "label_id": label_id,
                    "label_raw": raw_label,
                    "channel_names_json": json.dumps(SEED_62_CHANNELS),
                    "raw_sfreq": RAW_SFREQ,
                    "notch_hz": 50.0,
                    "qc_flags": 0,
                    "valid_time_samples": 0,
                    "channel_count": C_MAX,
                    "trial_index": int(trial_index),
                    "window_index": int(window_index),
                })

    if not rows:
        raise RuntimeError(f"No SEED trial arrays were found under {root}")
    rows.sort(
        key=lambda row: (
            int(row["subject_id"]) if str(row["subject_id"]).isdigit() else 10**9,
            row["session_id"],
            row["trial_index"],
            row["window_index"],
        )
    )
    apply_subject_disjoint_split(rows, seed=SPLIT_SEED)

    dataset_config = {
        "dataset_key": "SEED",
        "source_root": str(root),
        "source_format": "mat",
        "segment_seconds": SEGMENT_SEC,
        "target_sampling_rate": TARGET_SFREQ,
        "signal_unit": "uV",
        "notch_hz": 50.0,
        "bandpass_low": 0.3,
        "bandpass_high": 75.0,
        "channel_names": SEED_62_CHANNELS,
        "channel_rule": "retain the documented 62-channel SEED order",
        "label_rule": "label.mat values -1/0/1 map to negative/neutral/positive",
        "segment_rule": "non-overlapping 10-s windows within each source trial",
        "sample_order": "subject/session/trial/window source order; independent of split",
        "split_rule": (
            "deterministic subject-disjoint hash split 70/15/15 "
            f"(seed={SPLIT_SEED})"
        ),
        "qc_flags": qc_flag_mapping(),
    }
    return DatasetPlan(
        dataset_key="SEED",
        rows=rows,
        c_max=C_MAX,
        t=int(SEGMENT_SEC * TARGET_SFREQ),
        segment_seconds=SEGMENT_SEC,
        task_type="multiclass_classification",
        label_vocab={str(key): {"name": value} for key, value in SEED_LABELS.items()},
        dataset_config=dataset_config,
        channel_count_distribution=dict(channel_counts),
        expected_n_samples=len(rows),
    )


def write_seed_zarr(root_path: Path, dataset_dir: Path, plan: DatasetPlan) -> dict[str, Any]:
    root = create_zarr_store(dataset_dir, plan)
    groups = row_groups_by_source(plan.rows)
    qc_counter: Counter[int] = Counter()
    label_counter: Counter[int] = Counter()
    start_time = time.time()

    for file_index, (source_rel, indexed_rows) in enumerate(groups.items(), start=1):
        mat_path = root_path / source_rel
        logging.info("SEED %d/%d processing %s", file_index, len(groups), source_rel)
        trial_lookup = {
            trial_index: trial
            for trial_index, _variable, trial in _trial_arrays(
                scipy.io.loadmat(str(mat_path))
            )
        }
        by_trial: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
        for index, row in indexed_rows:
            by_trial[int(row["source_trial_index"])].append((index, row))

        batch: dict[str, list[Any]] = {
            "indices": [], "signals": [], "channel_masks": [], "bad_masks": [],
            "channel_counts": [], "labels": [], "label_values": [],
            "subject_ids": [], "qc_flags": [], "valid_time_samples": [],
        }
        for trial_index in sorted(by_trial):
            if trial_index not in trial_lookup:
                raise ValueError(f"{mat_path.name}: missing trial {trial_index + 1}")
            processed, _effective_high, trial_qc = preprocess_continuous_uv(
                trial_lookup[trial_index], raw_sfreq=RAW_SFREQ, notch_hz=50.0
            )
            for index, row in by_trial[trial_index]:
                segment, valid_samples, segment_qc = extract_segment(
                    processed, row["segment_start_sec"], plan.t, TARGET_SFREQ
                )
                signal, channel_mask, bad_mask, final_qc = finalize_segment(
                    segment, plan.c_max, plan.t, SEED_62_CHANNELS,
                    trial_qc | segment_qc, valid_samples,
                    min_expected_channels=C_MAX,
                )
                row["qc_flags"] = int(final_qc)
                row["valid_time_samples"] = int(valid_samples)
                batch["indices"].append(index)
                batch["signals"].append(signal)
                batch["channel_masks"].append(channel_mask)
                batch["bad_masks"].append(bad_mask)
                batch["channel_counts"].append(C_MAX)
                batch["labels"].append(int(row["label_id"]))
                batch["label_values"].append(float(row["label_id"]))
                batch["subject_ids"].append(str(row["subject_id"]))
                batch["qc_flags"].append(int(final_qc))
                batch["valid_time_samples"].append(int(valid_samples))
                qc_counter[int(final_qc)] += 1
                label_counter[int(row["label_id"])] += 1

        if batch["indices"]:
            write_batch(root=root, **batch)

    write_parquet_compat(pd.DataFrame(plan.rows), dataset_dir / "sample_index.parquet")
    write_json(dataset_dir / "label_vocab.json", plan.label_vocab)
    write_json(dataset_dir / "dataset_config.json", plan.dataset_config)
    summary = build_summary(
        plan, label_counter, qc_counter, elapsed_sec=time.time() - start_time
    )
    write_json(dataset_dir / "conversion_summary.json", summary)
    return summary
