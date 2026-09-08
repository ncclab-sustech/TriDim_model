"""SEED-V emotion recognition from Neuroscan CNT recordings."""
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

from ..channel_filters import pick_eeg_channels
from ..constants import TARGET_SFREQ
from ..dsp import extract_segment, finalize_segment, preprocess_continuous_uv
from ..io_utils import relpath, write_json, write_parquet_compat
from ..plan import DatasetPlan
from ..splits import apply_subject_disjoint_split
from ..validation import build_summary, qc_flag_mapping
from ..writer import create_zarr_store, row_groups_by_source, write_batch

SEED_V_LABELS = {
    0: "Disgust",
    1: "Fear",
    2: "Sad",
    3: "Neutral",
    4: "Happy",
}
EMOTION_TO_ID = {name.lower(): idx for idx, name in SEED_V_LABELS.items()}
SEGMENT_SEC_DEFAULT = 4.0
_CNT_RE = re.compile(r"^(?P<sub>\d+)_(?P<ses>\d+)_(?P<date>\d+)\.cnt$")


def _resolve_cnt_dir(root: Path) -> Path:
    for candidate in [root / "EEG_raw_cnt", root / "EEG_raw", root]:
        if any(candidate.glob("*.cnt")):
            return candidate
    raise FileNotFoundError(f"No SEED-V CNT files under {root}")


def _parse_timestamps(root: Path) -> dict[int, tuple[list[float], list[float]]]:
    path = root / "trial_start_end_timestamp.txt"
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text()
    sessions: dict[int, tuple[list[float], list[float]]] = {}
    current: int | None = None
    starts: list[float] = []
    ends: list[float] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        sess_match = re.match(r"Session\s+(\d+)", line, flags=re.IGNORECASE)
        if sess_match:
            if current is not None:
                sessions[current] = (starts, ends)
            current = int(sess_match.group(1))
            starts, ends = [], []
            continue
        if line.lower().startswith("start_second"):
            starts = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+", line.split(":", 1)[1])]
        elif line.lower().startswith("end_second"):
            ends = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+", line.split(":", 1)[1])]
    if current is not None:
        sessions[current] = (starts, ends)
    return sessions


def _parse_session_emotions(root: Path) -> dict[int, list[str]]:
    xlsx = root / "emotion_label_and_stimuli_order.xlsx"
    if not xlsx.exists():
        raise FileNotFoundError(xlsx)
    df = pd.read_excel(xlsx, header=None)
    session_emotions: dict[int, list[str]] = {}
    for __idx, row in df.iterrows():
        values = [str(v).strip() for v in row.tolist() if pd.notna(v)]
        if not values:
            continue
        joined = " ".join(values)
        sess_match = re.search(r"Session\s+(\d+)", joined, flags=re.IGNORECASE)
        if not sess_match:
            continue
        sess_id = int(sess_match.group(1))
        emotions: list[str] = []
        for value in values:
            key = re.sub(r"[^a-z]", "", value.lower())
            if key in EMOTION_TO_ID:
                emotions.append(SEED_V_LABELS[EMOTION_TO_ID[key]])
        if emotions:
            session_emotions[sess_id] = emotions
    if not session_emotions:
        raise RuntimeError(f"Could not parse session emotions from {xlsx}")
    return session_emotions


def _scan_seed_v(root: Path, *, segment_seconds: float, dataset_key: str, smoke: bool = False) -> DatasetPlan:
    if not root.exists():
        raise FileNotFoundError(root)
    cnt_dir = _resolve_cnt_dir(root)
    timestamps = _parse_timestamps(root)
    session_emotions = _parse_session_emotions(root)
    cnt_files = sorted(cnt_dir.glob("*.cnt"))
    if smoke:
        cnt_files = cnt_files[:2]

    rows: list[dict[str, Any]] = []
    channel_count_distribution: Counter[str] = Counter()

    for cnt_path in cnt_files:
        match = _CNT_RE.match(cnt_path.name)
        if match is None:
            continue
        subject_id = match.group("sub")
        session_id = match.group("ses")
        sess_num = int(session_id)
        if sess_num not in timestamps or sess_num not in session_emotions:
            logging.warning("SEED-V missing metadata for %s", cnt_path.name)
            continue
        starts, ends = timestamps[sess_num]
        emotions = session_emotions[sess_num]
        n_trials = min(len(starts), len(ends), len(emotions))

        raw = mne.io.read_raw_cnt(str(cnt_path), preload=False, verbose=False)
        indices, channel_names, __excluded = pick_eeg_channels(raw.ch_names)
        if not indices:
            continue
        sfreq = float(raw.info["sfreq"])
        duration_sec = float(raw.n_times) / sfreq
        channel_count_distribution[str(len(channel_names))] += 1

        for trial_idx in range(n_trials):
            trial_start = float(starts[trial_idx])
            trial_end = float(ends[trial_idx])
            if trial_end <= trial_start:
                continue
            label_raw = emotions[trial_idx]
            label_id = EMOTION_TO_ID[label_raw.lower()]
            n_segments = int((trial_end - trial_start) // segment_seconds)
            for seg_idx in range(n_segments):
                start = trial_start + float(seg_idx) * segment_seconds
                end = start + segment_seconds
                if end > duration_sec + 1e-6:
                    break
                rows.append(
                    {
                        "sample_id": (
                            f"{dataset_key}:{subject_id}:ses{session_id}:"
                            f"trial{trial_idx:02d}:seg{seg_idx:04d}"
                        ),
                        "dataset_key": dataset_key,
                        "subject_id": subject_id,
                        "session_id": f"{subject_id}_{session_id}",
                        "segment_id": f"{cnt_path.stem}_t{trial_idx:02d}_seg{seg_idx:04d}",
                        "source_relpath": relpath(cnt_path, root),
                        "source_format": "cnt",
                        "segment_start_sec": start,
                        "segment_end_sec": end,
                        "label_id": label_id,
                        "label_raw": label_raw,
                        "channel_names_json": json.dumps(channel_names),
                        "raw_sfreq": sfreq,
                        "notch_hz": 50.0,
                        "qc_flags": 0,
                        "channel_count": len(channel_names),
                        "trial_index": int(trial_idx),
                    }
                )

    rows.sort(
        key=lambda row: (
            int(row["subject_id"]) if str(row["subject_id"]).isdigit() else 10**9,
            row["session_id"],
            row["trial_index"],
            row["segment_start_sec"],
        )
    )
    apply_subject_disjoint_split(rows)
    c_max = max((int(row["channel_count"]) for row in rows), default=62)
    dataset_config = {
        "dataset_key": dataset_key,
        "source_root": str(root),
        "source_format": "cnt",
        "segment_seconds": segment_seconds,
        "target_sampling_rate": TARGET_SFREQ,
        "signal_unit": "uV",
        "notch_hz": 50.0,
        "bandpass_low": 0.3,
        "bandpass_high": 75.0,
        "qc_flags": qc_flag_mapping(),
        "split_rule": "subject_disjoint",
        "sample_order": "subject/session/trial/segment source order; independent of split",
        "label_note": "Raw emotion names preserved from emotion_label_and_stimuli_order.xlsx.",
    }
    return DatasetPlan(
        dataset_key=dataset_key,
        rows=rows,
        c_max=c_max,
        t=int(segment_seconds * TARGET_SFREQ),
        segment_seconds=segment_seconds,
        task_type="multiclass_classification",
        label_vocab={str(k): {"name": v} for k, v in SEED_V_LABELS.items()},
        dataset_config=dataset_config,
        channel_count_distribution=dict(channel_count_distribution),
        expected_n_samples=len(rows),
    )


def _write_seed_v_zarr(root_path: Path, dataset_dir: Path, plan: DatasetPlan, log_tag: str) -> dict[str, Any]:
    root = create_zarr_store(dataset_dir, plan)
    groups = row_groups_by_source(plan.rows)
    qc_counter: Counter[int] = Counter()
    label_counter: Counter[int] = Counter()
    start_time = time.time()

    for file_idx, (source_rel, indexed_rows) in enumerate(groups.items(), start=1):
        cnt_path = root_path / source_rel
        logging.info("%s %d/%d processing %s", log_tag, file_idx, len(groups), source_rel)
        raw = mne.io.read_raw_cnt(str(cnt_path), preload=True, verbose=False)
        indices, channel_names, __excluded = pick_eeg_channels(raw.ch_names)
        raw.pick([raw.ch_names[i] for i in indices])
        data_uv = raw.get_data() * 1e6
        raw_sfreq = float(raw.info["sfreq"])
        for __idx, row in indexed_rows:
            row["channel_names_json"] = json.dumps(channel_names)
            row["channel_count"] = len(channel_names)

        processed, __eff_high, file_qc = preprocess_continuous_uv(data_uv, raw_sfreq, notch_hz=50.0)
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
            segment, valid_samples, seg_qc = extract_segment(
                processed, row["segment_start_sec"], plan.t, TARGET_SFREQ
            )
            signal, channel_mask, bad_mask, final_qc = finalize_segment(
                segment,
                plan.c_max,
                plan.t,
                channel_names,
                file_qc | seg_qc,
                valid_samples,
                min_expected_channels=60,
            )
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


def scan_seed_v_plan(root: Path, smoke: bool = False) -> DatasetPlan:
    return _scan_seed_v(root, segment_seconds=SEGMENT_SEC_DEFAULT, dataset_key="SEED_V", smoke=smoke)


def write_seed_v_zarr(root_path: Path, dataset_dir: Path, plan: DatasetPlan) -> dict[str, Any]:
    return _write_seed_v_zarr(root_path, dataset_dir, plan, log_tag="SEED_V")
