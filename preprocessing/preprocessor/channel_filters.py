"""Shared EEG channel classification and selection.

Prefix matching alone is not enough: TUH and other clinical corpora often
label auxiliary sensors with an ``EEG`` prefix. Names are normalized and
checked against an auxiliary-sensor deny list. TCP bipolar labels such as
``FP1-F7`` are kept when both endpoints are scalp (or scalp-vs-reference).
Standalone mastoid/reference electrodes (A1/A2/M1/M2) are optional.
"""
from __future__ import annotations

import re
from collections.abc import Iterable


_AUX_TOKENS = {
    "ECG", "EKG", "EOG", "HEO", "VEO", "ROC", "LOC", "EMG", "RESP",
    "PHOTIC", "PHOT", "TRIGGER", "STATUS", "EVENT", "IBI", "BURST", "BURSTS",
    "SUPPR", "DC", "SAO2", "SPO2", "PULSE", "PLETH", "AIRFLOW",
    "THOR", "ABD", "SNORE", "POSITION", "MARKER", "VNS", "STIM",
    "PG1", "PG2", "PG", "SP1", "SP2", "SP", "RLC", "LUC",
    "X1", "X2", "X3", "CHIN", "LEG", "ARM",
}
_REFERENCE_ONLY = {"A1", "A2", "M1", "M2", "CS1", "CS2", "REF", "LE", "AVG", "AR"}
_SCALP_RE = re.compile(
    r"^(?:"
    r"FP|AF|AFF|F|FFT|FFC|FT|FC|FCC|FTT|"
    r"T|TTP|TP|TPP|C|CCP|CP|CPP|"
    r"P|PPO|PO|POO|O|OI|I|"
    r"A|M"
    r")[0-9Z]+H?$|"
    r"^(?:CZ|FZ|PZ|FPZ|AFZ|FCZ|FCCZ|CPZ|CPPZ|POZ|POOZ|OZ|IZ|CB[12])$",
    re.IGNORECASE,
)
_NUMERIC_ELECTRODE_RE = re.compile(r"^E[0-9]+$", re.IGNORECASE)
_DIGIT_ONLY_RE = re.compile(r"^\d+$")
# Misc / unused amplifier channels such as ``23A-23R``.
_MISC_PAIR_RE = re.compile(r"^\d{1,2}[A-Z]-\d{1,2}[A-Z]$", re.IGNORECASE)


def canonical_channel_name(name: str) -> str:
    """Return a stable source-referenced channel label."""
    value = str(name).strip().upper()
    value = re.sub(r"^EEG[\s_-]+", "", value)
    value = re.sub(r"^POL[\s_-]+", "", value)
    value = re.sub(r"[\s_-]+(?:REF|LE|AVG|AR)$", "", value)
    value = re.sub(r"[.\s]+$", "", value)
    return value.strip()


def _is_scalp_electrode(name: str) -> bool:
    canonical = canonical_channel_name(name)
    if canonical in _REFERENCE_ONLY:
        return False
    return bool(_NUMERIC_ELECTRODE_RE.fullmatch(canonical) or _SCALP_RE.fullmatch(canonical))


def _is_reference_electrode(name: str) -> bool:
    return canonical_channel_name(name) in _REFERENCE_ONLY


def _is_eeg_endpoint(name: str) -> bool:
    # Mastoids/refs are valid bipolar endpoints (e.g. F3-A2) even when standalone
    # mastoid channels are dropped from the montage.
    if _is_reference_electrode(name):
        return True
    return _is_scalp_electrode(name)


def _has_aux_token(original: str, canonical: str) -> bool:
    tokens = {token for token in re.split(r"[^A-Z0-9]+", original) if token}
    tokens |= {token for token in re.split(r"[^A-Z0-9]+", canonical) if token}
    if tokens & _AUX_TOKENS:
        return True
    return any(token in original or token in canonical for token in _AUX_TOKENS)


def is_eeg_channel(name: str, *, keep_mastoid_references: bool = False) -> bool:
    """True for scalp EEG channels or EEG bipolar pairs.

    Standalone mastoid/reference electrodes are kept only when
    ``keep_mastoid_references`` is True (common for TUH clinical montages).
    Reference-to-reference pairs such as ``A2-A1`` are always dropped.
    """
    original = str(name).strip().upper()
    if not original or original in {".", "-"}:
        return False
    canonical = canonical_channel_name(original)
    if not canonical:
        return False
    if _DIGIT_ONLY_RE.fullmatch(canonical):
        return False
    if re.fullmatch(r"-+(?:\d+)?", canonical):
        return False
    if canonical.startswith(".-"):
        return False
    if _MISC_PAIR_RE.fullmatch(canonical):
        return False
    if _has_aux_token(original, canonical):
        return False

    if "-" in canonical:
        parts = canonical.split("-")
        if len(parts) == 3 and parts[-1].isdigit():
            parts = parts[:2]
        if len(parts) != 2:
            return False
        left, right = parts
        if _is_reference_electrode(left) and _is_reference_electrode(right):
            return False
        return _is_eeg_endpoint(left) and _is_eeg_endpoint(right)

    if "_" in canonical:
        parts = canonical.split("_")
        if len(parts) != 2:
            return False
        left, right = parts
        if _is_reference_electrode(left) and _is_reference_electrode(right):
            return False
        return _is_eeg_endpoint(left) and _is_eeg_endpoint(right)

    if _is_reference_electrode(canonical):
        return bool(keep_mastoid_references)
    return _is_scalp_electrode(canonical)


def pick_eeg_channels(
    names: Iterable[str],
    *,
    keep_mastoid_references: bool = False,
) -> tuple[list[int], list[str], list[str]]:
    """Return source indices, original kept names, and excluded source names."""
    indices: list[int] = []
    eeg_names: list[str] = []
    excluded: list[str] = []
    for index, name in enumerate(names):
        if is_eeg_channel(name, keep_mastoid_references=keep_mastoid_references):
            indices.append(index)
            eeg_names.append(str(name))
        else:
            excluded.append(str(name))
    return indices, eeg_names, excluded


def resolve_eeg_channel_indices(
    names: list[str],
    *,
    keep_mastoid_references: bool = False,
    preferred_order: list[str] | None = None,
) -> tuple[list[int], list[str]]:
    """Select EEG channels, optionally reordering to a preferred montage."""
    if not names:
        return [], []
    if preferred_order:
        lookup: dict[str, int] = {}
        for idx, name in enumerate(names):
            key = canonical_channel_name(name)
            lookup.setdefault(key, idx)
            lookup.setdefault(str(name).strip().upper(), idx)
        indices: list[int] = []
        kept: list[str] = []
        for wanted in preferred_order:
            key = canonical_channel_name(wanted)
            idx = lookup.get(key, lookup.get(str(wanted).strip().upper()))
            if idx is None:
                continue
            indices.append(idx)
            kept.append(str(names[idx]))
        return indices, kept

    return pick_eeg_channels(names, keep_mastoid_references=keep_mastoid_references)[:2]
