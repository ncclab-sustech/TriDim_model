"""FACED: Finer-grained Affective Computing EEG Dataset (nine emotion categories).

Source layout
-------------
The release ships one pickled ``numpy`` array per subject under
``Processed_data_full`` (``sub000.pkl`` ... ``sub122.pkl``, 123 subjects). Each
array is shaped ``(28, 32, 7500)``: 28 video clips, 32 electrodes, and the last
30 seconds of every clip sampled at 250 Hz. The upstream pipeline already
adjusted units to uV, band-passed 0.05-47 Hz, interpolated bad channels,
removed ocular ICs and re-referenced to the common average, so this builder only
re-filters, resamples to the canonical 200 Hz grid and windows the trials.

Channels
--------
Electrode order is fixed by the release (cohort 2 naming; cohort 1 is reordered
upstream to match). The final two entries are the right/left mastoids, named
``A2``/``A1`` after the upstream rename. The paper experiments retain all
32 released channels, including these two reference channels.

Labels
------
Clip index (1-based in the stimulus table) maps to the nine emotion categories
in blocks of three, except for the four neutral clips:

    1-3   anger        -> 0        16-19  amusement   -> 5
    4-6   disgust      -> 1        20-22  inspiration -> 6
    7-9   fear         -> 2        23-25  joy         -> 7
    10-12 sadness      -> 3        26-28  tenderness  -> 8
    13-16 neutral      -> 4

This is the ``cls9`` ordering used by the reference validation code shipped with
the release, so a 9-way head trained on the legacy store stays valid.
"""
from __future__ import annotations

import json
import logging
import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..constants import TARGET_SFREQ
from ..dsp import extract_segment, finalize_segment, preprocess_continuous_uv
from ..io_utils import relpath, write_json, write_parquet_compat
from ..plan import DatasetPlan
from ..splits import apply_subject_disjoint_split
from ..validation import build_summary, qc_flag_mapping
from ..writer import create_zarr_store, row_groups_by_source, write_batch

logger = logging.getLogger(__name__)

DATASET_KEY = "FACED_new"
SOURCE_FORMAT = "faced_pickle"
MONTAGE_NAME = "faced_processed_32ch"

# Electrode order of the pre-processed release; A2/A1 are retained because the
# paper training store contains all 32 released channels.
FACED_SOURCE_CHANNELS: list[str] = [
    "Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8", "FC1", "FC2", "FC5",
    "FC6", "Cz", "C3", "C4", "T7", "T8", "CP1", "CP2", "CP5", "CP6",
    "Pz", "P3", "P4", "P7", "P8", "PO3", "PO4", "Oz", "O1", "O2",
    "A2", "A1",
]
FACED_EEG_INDICES = list(range(len(FACED_SOURCE_CHANNELS)))
FACED_EEG_CHANNELS = list(FACED_SOURCE_CHANNELS)
FACED_DROPPED_CHANNELS: list[str] = []
C_MAX = len(FACED_EEG_CHANNELS)  # 32

FACED_LABELS: dict[int, str] = {
    0: "anger",
    1: "disgust",
    2: "fear",
    3: "sadness",
    4: "neutral",
    5: "amusement",
    6: "inspiration",
    7: "joy",
    8: "tenderness",
}
# Number of consecutive video clips per emotion, in clip order.
CLIPS_PER_EMOTION: tuple[int, ...] = (3, 3, 3, 3, 4, 3, 3, 3, 3)
TRIAL_LABEL_IDS: tuple[int, ...] = tuple(
    label_id
    for label_id, count in enumerate(CLIPS_PER_EMOTION)
    for _ in range(count)
)
N_TRIALS = len(TRIAL_LABEL_IDS)  # 28

SEGMENT_SEC = 10.0
TRIAL_SEC = 30.0
T_SAMPLES = int(SEGMENT_SEC * TARGET_SFREQ)  # 2000
WINDOWS_PER_TRIAL = int(TRIAL_SEC // SEGMENT_SEC)  # 3
RAW_SFREQ = 250.0
NOTCH_HZ = 50.0
BANDPASS_LOW = 0.3
# The release is already low-passed at 47 Hz; asking for more only adds ringing.
BANDPASS_HIGH = 47.0

PICKLE_DIR_CANDIDATES = ("Processed_data_full", "Processed_data", "processed")
SMOKE_SUBJECT_COUNT = 2


def _resolve_pickle_dir(root: Path) -> Path:
    for name in PICKLE_DIR_CANDIDATES:
        candidate = root / name
        if candidate.is_dir() and any(candidate.glob("sub*.pkl")):
            return candidate
    if any(root.glob("sub*.pkl")):
        return root
    raise FileNotFoundError(f"No FACED sub*.pkl files under {root}")


def _load_subject_trials(pkl_path: Path) -> np.ndarray:
    with pkl_path.open("rb") as handle:
        data = pickle.load(handle)
    array = np.asarray(data)
    if array.ndim != 3:
        raise ValueError(f"{pkl_path.name}: expected a 3-D trial array, got {array.shape}")
    if array.shape[1] != len(FACED_SOURCE_CHANNELS):
        raise ValueError(
            f"{pkl_path.name}: expected {len(FACED_SOURCE_CHANNELS)} electrodes, "
            f"got {array.shape[1]}"
        )
    return array


def scan_faced_new_plan(root: Path, smoke: bool = False) -> DatasetPlan:
    if not root.exists():
        raise FileNotFoundError(root)

    pickle_dir = _resolve_pickle_dir(root)
    pkl_files = sorted(pickle_dir.glob("sub*.pkl"))
    if smoke:
        pkl_files = pkl_files[:SMOKE_SUBJECT_COUNT]
    if not pkl_files:
        raise FileNotFoundError(f"No FACED sub*.pkl files under {pickle_dir}")

    rows: list[dict[str, Any]] = []
    channel_counts: Counter[int] = Counter()
    short_trial_subjects: list[str] = []

    for file_idx, pkl_path in enumerate(pkl_files, start=1):
        subject_id = pkl_path.stem
        trials = _load_subject_trials(pkl_path)
        n_trials, __n_channels, n_times = trials.shape
        if n_trials != N_TRIALS:
            short_trial_subjects.append(f"{subject_id}:{n_trials}")
            n_trials = min(n_trials, N_TRIALS)
        trial_sec = float(n_times) / RAW_SFREQ
        n_windows = min(WINDOWS_PER_TRIAL, int(trial_sec // SEGMENT_SEC))
        if n_windows <= 0:
            logger.warning(
                "FACED %s trials are shorter than one %.0fs window (%.2fs); skipping",
                subject_id, SEGMENT_SEC, trial_sec,
            )
            continue
        channel_counts[C_MAX] += 1
        source_rel = relpath(pkl_path, root)
        logger.info(
            "FACED scan %d/%d %s: %d trials x %d windows",
            file_idx, len(pkl_files), subject_id, n_trials, n_windows,
        )

        for trial_index in range(n_trials):
            label_id = TRIAL_LABEL_IDS[trial_index]
            for window_idx in range(n_windows):
                start_sec = float(window_idx) * SEGMENT_SEC
                rows.append({
                    "sample_id": f"{DATASET_KEY}:{subject_id}:{trial_index:02d}:{window_idx}",
                    "dataset_key": DATASET_KEY,
                    "subject_id": subject_id,
                    "session_id": subject_id,
                    "trial_id": str(trial_index),
                    "trial_index": int(trial_index),
                    "segment_id": f"{subject_id}_{trial_index:02d}_{window_idx}",
                    "source_relpath": source_rel,
                    "source_format": SOURCE_FORMAT,
                    "raw_inner_path": "",
                    "segment_start_sec": start_sec,
                    "segment_end_sec": start_sec + SEGMENT_SEC,
                    "event_start_sec": np.nan,
                    "event_end_sec": np.nan,
                    "event_id": f"trial_{trial_index:02d}",
                    "label_id": int(label_id),
                    "label_raw": FACED_LABELS[label_id],
                    "label_value": np.nan,
                    "label_source": "fixed_trial_label_table",
                    "channel_names_json": json.dumps(FACED_EEG_CHANNELS),
                    "channel_type_json": json.dumps(["eeg"] * C_MAX),
                    "montage_name": MONTAGE_NAME,
                    "raw_sfreq": RAW_SFREQ,
                    "notch_hz": NOTCH_HZ,
                    "bandpass_high_effective": BANDPASS_HIGH,
                    "split": "unassigned",
                    "segment_kind": "trial_10s_chunk",
                    "qc_flags": 0,
                    "source_file_name": pkl_path.name,
                    "channel_count": C_MAX,
                    "trial_window_idx": int(window_idx),
                    "valid_time_samples": 0,
                })

    if not rows:
        raise RuntimeError(f"No valid FACED segments built from {root}")

    apply_subject_disjoint_split(rows)
    rows.sort(key=lambda r: (r["subject_id"], r["trial_index"], r["trial_window_idx"]))

    label_vocab = {
        str(label_id): {"name": name, "raw_aliases": [name, str(label_id)]}
        for label_id, name in FACED_LABELS.items()
    }
    dataset_config = {
        "dataset_key": DATASET_KEY,
        "source_root": str(root),
        "source_format": SOURCE_FORMAT,
        "source_subdir": relpath(pickle_dir, root),
        "segment_seconds": SEGMENT_SEC,
        "segment_stride_seconds": SEGMENT_SEC,
        "trial_seconds": TRIAL_SEC,
        "target_sampling_rate": TARGET_SFREQ,
        "signal_unit": "uV",
        "notch_hz": NOTCH_HZ,
        "bandpass_low": BANDPASS_LOW,
        "bandpass_high": BANDPASS_HIGH,
        "bandpass_note": (
            "upstream pre-processing already band-passed 0.05-47 Hz, so the "
            "rebuild keeps the 47 Hz upper edge instead of the shared 75 Hz default"
        ),
        "c_max": C_MAX,
        "channel_layout": "fixed",
        "channel_names": FACED_EEG_CHANNELS,
        "channel_name_count": C_MAX,
        "channel_source_order": FACED_SOURCE_CHANNELS,
        "channel_rule": "fixed 32-channel release order; A2/A1 are retained",
        "montage_name": MONTAGE_NAME,
        "min_expected_channels": C_MAX,
        "label_rule": (
            "clip order -> emotion in blocks of three (four for neutral): "
            "anger/disgust/fear/sadness/neutral/amusement/inspiration/joy/tenderness "
            "as label_id 0..8"
        ),
        "segment_rule": (
            f"each {TRIAL_SEC:.0f}s trial is cut into {WINDOWS_PER_TRIAL} "
            f"non-overlapping {SEGMENT_SEC:.0f}s windows"
        ),
        "split_rule": "subject_disjoint",
        "compatibility_note": (
            "The channel count, label ids, and 10-s window grid match the paper "
            "store. The released source is already filtered to 0.05-47 Hz; this "
            "converter applies the documented shared filtering/resampling chain, "
            "so bit identity requires validation against the archived store."
        ),
        "qc_flags": qc_flag_mapping(),
        "short_trial_subjects": short_trial_subjects,
    }
    return DatasetPlan(
        dataset_key=DATASET_KEY,
        rows=rows,
        c_max=C_MAX,
        t=T_SAMPLES,
        segment_seconds=SEGMENT_SEC,
        task_type="multiclass_classification",
        label_vocab=label_vocab,
        dataset_config=dataset_config,
        channel_count_distribution={str(k): int(v) for k, v in sorted(channel_counts.items())},
        expected_n_samples=len(rows),
    )


def write_faced_new_zarr(root_path: Path, dataset_dir: Path, plan: DatasetPlan) -> dict[str, Any]:
    zarr_root = create_zarr_store(dataset_dir, plan)
    groups = row_groups_by_source(plan.rows)
    qc_counter: Counter[int] = Counter()
    label_counter: Counter[int] = Counter()
    start_time = time.time()

    for file_idx, (source_rel, indexed_rows) in enumerate(groups.items(), start=1):
        pkl_path = root_path / source_rel
        logger.info(
            "FACED %d/%d processing %s (%d segments)",
            file_idx, len(groups), source_rel, len(indexed_rows),
        )
        trials = _load_subject_trials(pkl_path)

        rows_by_trial: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
        for idx, row in indexed_rows:
            rows_by_trial[int(row["trial_index"])].append((idx, row))

        indices: list[int] = []
        signals: list[np.ndarray] = []
        channel_masks: list[np.ndarray] = []
        bad_masks: list[np.ndarray] = []
        channel_counts_list: list[int] = []
        labels: list[int] = []
        label_values: list[float] = []
        subject_ids: list[str] = []
        qc_flags_list: list[int] = []
        valid_time_list: list[int] = []

        for trial_index in sorted(rows_by_trial):
            trial_uv = np.asarray(trials[trial_index][FACED_EEG_INDICES, :], dtype=np.float64)
            processed, effective_high, trial_qc = preprocess_continuous_uv(
                trial_uv,
                raw_sfreq=RAW_SFREQ,
                notch_hz=NOTCH_HZ,
                bandpass_low=BANDPASS_LOW,
                bandpass_high=BANDPASS_HIGH,
            )
            for idx, row in rows_by_trial[trial_index]:
                segment, valid_samples, seg_qc = extract_segment(
                    processed,
                    start_sec=float(row["segment_start_sec"]),
                    t_samples=plan.t,
                )
                signal, channel_mask, bad_mask, qc_flags = finalize_segment(
                    segment=segment,
                    c_max=plan.c_max,
                    t_samples=plan.t,
                    channel_names=FACED_EEG_CHANNELS,
                    inherited_qc=int(row["qc_flags"]) | trial_qc | seg_qc,
                    valid_time_samples=valid_samples,
                    min_expected_channels=C_MAX,
                )
                row["qc_flags"] = int(qc_flags)
                row["valid_time_samples"] = int(valid_samples)
                row["bandpass_high_effective"] = float(effective_high)

                indices.append(idx)
                signals.append(signal)
                channel_masks.append(channel_mask)
                bad_masks.append(bad_mask)
                channel_counts_list.append(C_MAX)
                labels.append(int(row["label_id"]))
                label_values.append(np.nan)
                subject_ids.append(str(row["subject_id"]))
                qc_flags_list.append(int(qc_flags))
                valid_time_list.append(int(valid_samples))
                qc_counter[int(qc_flags)] += 1
                label_counter[int(row["label_id"])] += 1

        write_batch(
            zarr_root, indices, signals, channel_masks, bad_masks,
            channel_counts_list, labels, label_values, subject_ids,
            qc_flags_list, valid_time_list,
        )
        del trials

    write_parquet_compat(pd.DataFrame(plan.rows), dataset_dir / "sample_index.parquet")
    write_json(dataset_dir / "label_vocab.json", plan.label_vocab)
    write_json(dataset_dir / "dataset_config.json", plan.dataset_config)
    summary = build_summary(plan, label_counter, qc_counter, elapsed_sec=time.time() - start_time)
    write_json(dataset_dir / "conversion_summary.json", summary)
    return summary
