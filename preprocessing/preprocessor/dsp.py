"""Core DSP functions shared by all dataset builders.

preprocess_continuous_uv  -- notch + bandpass + resample
extract_segment           -- cut (C, T) window from continuous signal
finalize_segment          -- pad to (C_max, T), generate masks and QC
"""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, resample_poly, sosfiltfilt

from .constants import (
    FLAT_STD_UV,
    HIGH_AMPLITUDE_UV,
    QC_FLAT_CHANNEL,
    QC_HIGH_AMPLITUDE,
    QC_LOW_CHANNEL_COUNT,
    QC_MISSING_CHANNEL_NAME,
    QC_NAN_INF_REPLACED,
    QC_SHORT_PADDED,
    TARGET_SFREQ,
)


def preprocess_continuous_uv(
    data_uv: np.ndarray,
    raw_sfreq: float,
    notch_hz: float = 50.0,
    bandpass_low: float = 0.3,
    bandpass_high: float = 75.0,
    target_sfreq: float = TARGET_SFREQ,
) -> tuple[np.ndarray, float, int]:
    """Filter and resample continuous signal shaped (C, T)."""
    qc_flags = 0
    if not np.isfinite(data_uv).all():
        data_uv = np.nan_to_num(data_uv, nan=0.0, posinf=0.0, neginf=0.0)
        qc_flags |= QC_NAN_INF_REPLACED

    data = np.asarray(data_uv, dtype=np.float64)
    nyquist_guard = 0.99 * 0.5 * float(raw_sfreq)
    effective_high = min(float(bandpass_high), nyquist_guard)
    if not 0.0 < float(bandpass_low) < effective_high:
        raise ValueError(
            "Invalid band-pass range after applying the source Nyquist limit: "
            f"low={bandpass_low}, high={effective_high}, fs={raw_sfreq}"
        )
    if notch_hz and notch_hz < 0.5 * raw_sfreq:
        b_notch, a_notch = iirnotch(w0=notch_hz, Q=30.0, fs=raw_sfreq)
        data = filtfilt(b_notch, a_notch, data, axis=-1)

    sos = butter(
        5,
        [bandpass_low, effective_high],
        btype="bandpass",
        fs=raw_sfreq,
        output="sos",
    )
    data = sosfiltfilt(sos, data, axis=-1)

    if not math.isclose(raw_sfreq, target_sfreq):
        ratio = Fraction(target_sfreq / raw_sfreq).limit_denominator(1000)
        data = resample_poly(data, up=ratio.numerator, down=ratio.denominator, axis=-1)

    if not np.isfinite(data).all():
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        qc_flags |= QC_NAN_INF_REPLACED
    return data.astype(np.float32, copy=False), effective_high, qc_flags


def extract_segment(
    data: np.ndarray,
    start_sec: float,
    t_samples: int,
    target_sfreq: float = TARGET_SFREQ,
) -> tuple[np.ndarray, int, int]:
    start = int(round(float(start_sec) * target_sfreq))
    end = start + t_samples
    qc_flags = 0
    if start < 0:
        qc_flags |= QC_SHORT_PADDED
    read_start = max(start, 0)
    read_end = min(end, data.shape[-1])
    valid = max(0, read_end - read_start)
    out = np.zeros((data.shape[0], t_samples), dtype=np.float32)
    if valid > 0:
        out_start = read_start - start
        out[:, out_start : out_start + valid] = data[:, read_start:read_end]
    if valid < t_samples:
        qc_flags |= QC_SHORT_PADDED
    return out, valid, qc_flags


def finalize_segment(
    segment: np.ndarray,
    c_max: int,
    t_samples: int,
    channel_names: list[str],
    inherited_qc: int,
    valid_time_samples: int,
    min_expected_channels: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    channel_count = len(channel_names)
    signal = np.zeros((c_max, t_samples), dtype=np.float32)
    signal[:channel_count, :] = segment[:channel_count, :]
    channel_mask = np.zeros((c_max,), dtype=bool)
    channel_mask[:channel_count] = True
    bad_channel_mask = np.zeros((c_max,), dtype=bool)
    qc_flags = int(inherited_qc)

    if any(not str(ch).strip() for ch in channel_names):
        qc_flags |= QC_MISSING_CHANNEL_NAME
    if channel_count < min_expected_channels:
        qc_flags |= QC_LOW_CHANNEL_COUNT

    valid_end = max(0, min(valid_time_samples, t_samples))
    valid_signal = signal[:channel_count, :valid_end] if valid_end else signal[:channel_count, :]
    if not np.isfinite(valid_signal).all():
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
        qc_flags |= QC_NAN_INF_REPLACED
        valid_signal = signal[:channel_count, :valid_end] if valid_end else signal[:channel_count, :]

    if valid_signal.size:
        high_channels = np.any(np.abs(valid_signal) > HIGH_AMPLITUDE_UV, axis=1)
        if bool(np.any(high_channels)):
            qc_flags |= QC_HIGH_AMPLITUDE
            bad_channel_mask[:channel_count] |= high_channels
        flat_channels = np.std(valid_signal, axis=1) < FLAT_STD_UV
        if bool(np.any(flat_channels)):
            qc_flags |= QC_FLAT_CHANNEL
            bad_channel_mask[:channel_count] |= flat_channels
    return signal, channel_mask, bad_channel_mask, qc_flags
