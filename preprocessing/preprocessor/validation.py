"""Store validation and summary building utilities."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import zarr

from .constants import TARGET_SFREQ, QC_SHORT_PADDED, QC_NAN_INF_REPLACED, QC_HIGH_AMPLITUDE, QC_FLAT_CHANNEL, QC_LOW_CHANNEL_COUNT, QC_MISSING_CHANNEL_NAME, QC_SOURCE_ARTIFACT, QC_EVENT_CLIPPED
from .plan import DatasetPlan


def sample_order_sha256(rows: list[dict[str, Any]]) -> str:
    """Fingerprint row identity and order for index-based split manifests."""
    digest = hashlib.sha256()
    identity_fields = (
        "dataset_key",
        "subject_id",
        "session_id",
        "source_relpath",
        "segment_id",
        "trial_id",
        "segment_start_sec",
        "label_id",
    )
    for row in rows:
        identity = row.get("sample_id")
        if identity is None:
            identity = {
                key: row.get(key)
                for key in identity_fields
                if key in row
            }
        digest.update(
            json.dumps(identity, sort_keys=True, separators=(",", ":"), default=str)
            .encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def qc_flag_mapping() -> dict[str, int]:
    return {
        "SHORT_PADDED": QC_SHORT_PADDED,
        "NAN_INF_REPLACED": QC_NAN_INF_REPLACED,
        "HIGH_AMPLITUDE": QC_HIGH_AMPLITUDE,
        "FLAT_CHANNEL": QC_FLAT_CHANNEL,
        "LOW_CHANNEL_COUNT": QC_LOW_CHANNEL_COUNT,
        "MISSING_CHANNEL_NAME": QC_MISSING_CHANNEL_NAME,
        "SOURCE_ARTIFACT": QC_SOURCE_ARTIFACT,
        "EVENT_CLIPPED": QC_EVENT_CLIPPED,
    }


def build_summary(
    plan: DatasetPlan,
    label_counter: Counter[int],
    qc_counter: Counter[int],
    elapsed_sec: float,
) -> dict[str, Any]:
    segment_kind_counts = Counter(str(row.get("segment_kind", "")) for row in plan.rows)
    subject_count = len({str(row["subject_id"]) for row in plan.rows})
    source_count = len({str(row["source_relpath"]) for row in plan.rows})
    qc_bit_counts = {name: 0 for name in qc_flag_mapping()}
    for flags in qc_counter:
        for name, bit in qc_flag_mapping().items():
            if int(flags) & bit:
                qc_bit_counts[name] += qc_counter[flags]
    return {
        "dataset_key": plan.dataset_key,
        "n_samples": len(plan.rows),
        "expected_n_samples": plan.expected_n_samples,
        "n_subjects": subject_count,
        "n_source_files": source_count,
        "shape": [len(plan.rows), plan.c_max, plan.t],
        "sampling_rate": int(TARGET_SFREQ),
        "signal_unit": "uV",
        "normalization": "none",
        "label_counts": {str(k): int(v) for k, v in sorted(label_counter.items())},
        "segment_kind_counts": {str(k): int(v) for k, v in sorted(segment_kind_counts.items())},
        "qc_flag_value_counts": {str(k): int(v) for k, v in sorted(qc_counter.items())},
        "qc_bit_counts": {str(k): int(v) for k, v in sorted(qc_bit_counts.items())},
        "channel_count_distribution": plan.channel_count_distribution,
        "sample_order_sha256": sample_order_sha256(plan.rows),
        "elapsed_sec": round(float(elapsed_sec), 3),
    }


def validate_store(dataset_dir: Path, plan: DatasetPlan) -> dict[str, Any]:
    zarr_path = dataset_dir / f"{plan.dataset_key}.zarr"
    root = zarr.open_group(str(zarr_path), mode="r")
    n = len(plan.rows)
    sample_index = pd.read_parquet(dataset_dir / "sample_index.parquet")
    sample_index_rows = sample_index.to_dict(orient="records")
    checks: dict[str, Any] = {
        "zarr_exists": zarr_path.exists(),
        "signals_shape": list(root["signals"].shape),
        "signals_dtype": str(root["signals"].dtype),
        "sampling_rate": root.attrs.get("sampling_rate"),
        "signal_unit": root.attrs.get("signal_unit"),
        "normalization": root.attrs.get("normalization"),
        "sample_index_rows": int(sample_index.shape[0]),
        "sample_order_sha256": sample_order_sha256(sample_index_rows),
    }
    expected = {
        "signals_shape": [n, plan.c_max, plan.t],
        "signals_dtype": "float32",
        "sampling_rate": int(TARGET_SFREQ),
        "signal_unit": "uV",
        "normalization": "none",
        "sample_index_rows": n,
        "sample_order_sha256": sample_order_sha256(plan.rows),
    }
    failures = {
        key: {"actual": checks[key], "expected": value}
        for key, value in expected.items()
        if checks[key] != value
    }
    if failures:
        raise ValueError(f"Zarr schema validation failed: {failures}")
    sample_indices = np.linspace(0, max(0, n - 1), num=min(16, n), dtype=int)
    for idx in sample_indices:
        arr = root["signals"][int(idx)]
        if not np.isfinite(arr).all():
            raise ValueError(f"Non-finite signal at index {idx}")
    checks["validated_sample_indices"] = sample_indices.tolist()
    return checks
