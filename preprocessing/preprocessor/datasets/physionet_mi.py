"""Physionet EEG Motor Movement/Imagery Dataset: 4-class motor imagery.

Source: PhysioNet EEG Motor Movement/Imagery Database (eegmmidb).
- 64-channel EEG at 160 Hz across subjects S001+.
- Runs R04/R08/R12: left fist (T1) vs right fist (T2) imagery.
- Runs R06/R10/R14: both fists (T1) vs both feet (T2) imagery.
- T0 events (rest) are discarded.
- Each imagery epoch is 4 seconds (0 to +4s from event onset).

Label mapping (consistent with CBraMod):
  R04/R08/R12: T1 -> 0 (left_fist),  T2 -> 1 (right_fist)
  R06/R10/R14: T1 -> 2 (both_fists), T2 -> 3 (both_feet)
"""
from __future__ import annotations

import json
import logging
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd

from ..constants import TARGET_SFREQ
from ..dsp import extract_segment, finalize_segment, preprocess_continuous_uv
from ..io_utils import relpath, write_json, write_parquet_compat
from ..plan import DatasetPlan
from ..validation import build_summary, qc_flag_mapping
from ..writer import create_zarr_store, row_groups_by_source, write_batch

logger = logging.getLogger(__name__)

PHYSIONET_MI_CHANNELS = [
    "Fc5.", "Fc3.", "Fc1.", "Fcz.", "Fc2.", "Fc4.", "Fc6.",
    "C5..", "C3..", "C1..", "Cz..", "C2..", "C4..", "C6..",
    "Cp5.", "Cp3.", "Cp1.", "Cpz.", "Cp2.", "Cp4.", "Cp6.",
    "Fp1.", "Fpz.", "Fp2.",
    "Af7.", "Af3.", "Afz.", "Af4.", "Af8.",
    "F7..", "F5..", "F3..", "F1..", "Fz..", "F2..", "F4..", "F6..", "F8..",
    "Ft7.", "Ft8.",
    "T7..", "T8..", "T9..", "T10.",
    "Tp7.", "Tp8.",
    "P7..", "P5..", "P3..", "P1..", "Pz..", "P2..", "P4..", "P6..", "P8..",
    "Po7.", "Po3.", "Poz.", "Po4.", "Po8.",
    "O1..", "Oz..", "O2..", "Iz..",
]

MI_TASKS = ["04", "06", "08", "10", "12", "14"]
LR_TASKS = {"04", "08", "12"}

PHYSIONET_MI_LABEL_NAMES = {
    0: "left_fist",
    1: "right_fist",
    2: "both_fists",
    3: "both_feet",
}

SEGMENT_SEC = 4.0
RAW_SFREQ = 160.0
T_SAMPLES = int(SEGMENT_SEC * TARGET_SFREQ)  # 800
C_MAX = len(PHYSIONET_MI_CHANNELS)  # 64


def _find_subject_dirs(root: Path, smoke: bool = False) -> list[Path]:
    """Find all S??? subject directories."""
    base = root / "files" / "eegmmidb" / "1.0.0"
    if not base.exists():
        base = root
    dirs = sorted(
        p for p in base.iterdir()
        if p.is_dir() and re.match(r"S\d{3}$", p.name)
    )
    if smoke:
        dirs = dirs[:2]
    return dirs


def _event_to_label(event_id: int, task: str) -> int | None:
    """Map MNE event_id + task code to 0-indexed label. Returns None for T0 (rest)."""
    if event_id == 1:
        return None
    if task in LR_TASKS:
        return event_id - 2  # T1(2)->0, T2(3)->1
    return event_id  # T1(2)->2, T2(3)->3


def scan_physionet_mi_plan(root: Path, smoke: bool = False) -> DatasetPlan:
    if not root.exists():
        raise FileNotFoundError(root)

    subject_dirs = _find_subject_dirs(root, smoke=smoke)
    if not subject_dirs:
        raise RuntimeError(f"No S??? directories found under {root}")

    rows: list[dict[str, Any]] = []
    skipped_files = 0

    for subj_dir in subject_dirs:
        subject_id = subj_dir.name  # e.g. "S001"

        for task in MI_TASKS:
            edf_name = f"{subject_id}R{task}.edf"
            edf_path = subj_dir / edf_name
            if not edf_path.exists():
                skipped_files += 1
                continue

            try:
                raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose=False)
            except Exception as exc:
                logger.warning("Cannot read %s: %s", edf_path, exc)
                skipped_files += 1
                continue

            sfreq = raw.info["sfreq"]
            duration_sec = raw.n_times / sfreq
            ch_names = raw.ch_names

            events_from_annot, event_dict = mne.events_from_annotations(
                raw, verbose=False,
            )

            for epoch_idx, event_row in enumerate(events_from_annot):
                event_sample, _, event_id = int(event_row[0]), int(event_row[1]), int(event_row[2])
                label_id = _event_to_label(event_id, task)
                if label_id is None:
                    continue

                onset_sec = event_sample / sfreq
                end_sec = onset_sec + SEGMENT_SEC
                if end_sec > duration_sec + 0.05:
                    continue

                rows.append({
                    "sample_id": f"Physionet_MI:{subject_id}:{subject_id}R{task}:epoch{epoch_idx:03d}",
                    "dataset_key": "Physionet_MI",
                    "subject_id": subject_id,
                    "session_id": f"{subject_id}R{task}",
                    "segment_id": f"{subject_id}R{task}_epoch{epoch_idx:03d}",
                    "source_relpath": relpath(edf_path, root),
                    "source_format": "edf",
                    "segment_start_sec": onset_sec,
                    "segment_end_sec": end_sec,
                    "label_id": label_id,
                    "label_raw": str(event_id),
                    "task_code": task,
                    "channel_names_json": json.dumps(PHYSIONET_MI_CHANNELS),
                    "montage_name": "physionet_eegmmidb_64_source_order",
                    "raw_sfreq": sfreq,
                    "notch_hz": 60.0,
                    "qc_flags": 0,
                    "channel_count": len(ch_names),
                })

    if not rows:
        raise RuntimeError(f"No valid MI epochs found under {root}")

    rows.sort(key=lambda r: (r["subject_id"], r["session_id"], r["segment_id"]))

    logger.info(
        "Physionet_MI scan: %d epochs from %d subjects (%d files skipped)",
        len(rows), len(subject_dirs), skipped_files,
    )

    label_vocab = {
        str(k): {"name": v} for k, v in PHYSIONET_MI_LABEL_NAMES.items()
    }
    dataset_config = {
        "dataset_key": "Physionet_MI",
        "source_root": "<provided-at-runtime>",
        "source_format": "edf",
        "tasks": MI_TASKS,
        "segment_seconds": SEGMENT_SEC,
        "target_sampling_rate": TARGET_SFREQ,
        "signal_unit": "uV",
        "notch_hz": 60.0,
        "bandpass_low": 0.3,
        "bandpass_high": 75.0,
        "channel_names": PHYSIONET_MI_CHANNELS,
        "label_rule": "R04/08/12: T1->0(left), T2->1(right); R06/10/14: T1->2(fists), T2->3(feet); T0 discarded",
        "segment_rule": "event onset to onset+4s",
        "qc_flags": qc_flag_mapping(),
    }
    ch_dist = Counter(row["channel_count"] for row in rows)
    return DatasetPlan(
        dataset_key="Physionet_MI",
        rows=rows,
        c_max=C_MAX,
        t=T_SAMPLES,
        segment_seconds=SEGMENT_SEC,
        task_type="multiclass_classification",
        label_vocab=label_vocab,
        dataset_config=dataset_config,
        channel_count_distribution={str(k): v for k, v in sorted(ch_dist.items())},
        expected_n_samples=len(rows),
    )


def write_physionet_mi_zarr(
    root_path: Path, dataset_dir: Path, plan: DatasetPlan,
) -> dict[str, Any]:
    zarr_root = create_zarr_store(dataset_dir, plan)
    groups = row_groups_by_source(plan.rows)
    qc_counter: Counter[int] = Counter()
    label_counter: Counter[int] = Counter()
    start_time = time.time()

    for file_idx, (source_rel, indexed_rows) in enumerate(groups.items(), start=1):
        edf_path = root_path / source_rel
        logger.info(
            "Physionet_MI %d/%d processing %s (%d epochs)",
            file_idx, len(groups), source_rel, len(indexed_rows),
        )

        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
        raw.pick(PHYSIONET_MI_CHANNELS)
        raw.reorder_channels(PHYSIONET_MI_CHANNELS)
        data_uv = raw.get_data() * 1e6  # V -> µV

        first_row = indexed_rows[0][1]
        raw_sfreq = float(first_row["raw_sfreq"])
        notch_hz = float(first_row["notch_hz"])

        processed, effective_high, file_qc = preprocess_continuous_uv(
            data_uv, raw_sfreq=raw_sfreq, notch_hz=notch_hz,
        )

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

        for idx, row in indexed_rows:
            segment, valid_samples, seg_qc = extract_segment(
                processed,
                start_sec=float(row["segment_start_sec"]),
                t_samples=plan.t,
            )
            signal, ch_mask, bad_mask, qc_flags = finalize_segment(
                segment=segment,
                c_max=plan.c_max,
                t_samples=plan.t,
                channel_names=PHYSIONET_MI_CHANNELS,
                inherited_qc=int(row["qc_flags"]) | file_qc | seg_qc,
                valid_time_samples=valid_samples,
                min_expected_channels=C_MAX,
            )
            row["qc_flags"] = int(qc_flags)
            row["valid_time_samples"] = int(valid_samples)
            row["bandpass_high_effective"] = float(effective_high)

            indices.append(idx)
            signals.append(signal)
            channel_masks.append(ch_mask)
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
        del processed

    index_df = pd.DataFrame(plan.rows)
    write_parquet_compat(index_df, dataset_dir / "sample_index.parquet")
    write_json(dataset_dir / "label_vocab.json", plan.label_vocab)
    write_json(dataset_dir / "dataset_config.json", plan.dataset_config)
    summary = build_summary(plan, label_counter, qc_counter, elapsed_sec=time.time() - start_time)
    write_json(dataset_dir / "conversion_summary.json", summary)
    return summary
