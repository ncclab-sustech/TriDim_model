from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DatasetPlan:
    dataset_key: str
    rows: list[dict[str, Any]]
    c_max: int
    t: int
    segment_seconds: float
    task_type: str
    label_vocab: dict[str, Any]
    dataset_config: dict[str, Any]
    channel_count_distribution: dict[str, int]
    expected_n_samples: int | None = None
