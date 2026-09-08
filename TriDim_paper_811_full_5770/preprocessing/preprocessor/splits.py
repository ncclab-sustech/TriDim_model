"""Deterministic leakage-safe split helpers."""
from __future__ import annotations

import hashlib
from collections.abc import Iterable


def _score(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}\x1f{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def subject_disjoint_split(
    subject_ids: Iterable[str],
    *,
    train: float = 0.7,
    val: float = 0.15,
    seed: int = 20260725,
) -> dict[str, str]:
    """Assign every subject to exactly one deterministic split.

    Hash assignment is stable when new subjects are added and therefore avoids
    silently reshuffling an existing benchmark.
    """
    if not (0.0 < train < 1.0 and 0.0 <= val < 1.0 and train + val < 1.0):
        raise ValueError("split fractions must satisfy 0 < train and train+val < 1")
    result: dict[str, str] = {}
    for subject_id in sorted({str(value) for value in subject_ids}):
        score = _score(subject_id, seed)
        result[subject_id] = (
            "train" if score < train else "val" if score < train + val else "test"
        )
    return result


def apply_subject_disjoint_split(
    rows: list[dict],
    *,
    subject_key: str = "subject_id",
    split_key: str = "split",
    seed: int = 20260725,
) -> None:
    assignments = subject_disjoint_split(
        (str(row[subject_key]) for row in rows), seed=seed
    )
    for row in rows:
        row[split_key] = assignments[str(row[subject_key])]


def subject_session_disjoint_split(
    pairs: Iterable[tuple[str, str]],
    *,
    train: float = 0.7,
    val: float = 0.15,
    seed: int = 20260725,
) -> dict[tuple[str, str], str]:
    """Assign every (subject, session) pair to exactly one deterministic split."""
    keys = sorted({(str(subject), str(session)) for subject, session in pairs})
    units = [f"{subject}\x1f{session}" for subject, session in keys]
    by_unit = subject_disjoint_split(units, train=train, val=val, seed=seed)
    return {(subject, session): by_unit[f"{subject}\x1f{session}"] for subject, session in keys}


def apply_subject_session_disjoint_split(
    rows: list[dict],
    *,
    subject_key: str = "subject_id",
    session_key: str = "session_id",
    split_key: str = "split",
    seed: int = 20260725,
) -> None:
    assignments = subject_session_disjoint_split(
        ((str(row[subject_key]), str(row[session_key])) for row in rows),
        seed=seed,
    )
    for row in rows:
        row[split_key] = assignments[(str(row[subject_key]), str(row[session_key]))]
