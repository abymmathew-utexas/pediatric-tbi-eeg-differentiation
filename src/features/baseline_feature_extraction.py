from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.signal import welch


EEG_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}


def _bandpower(signal_1d: np.ndarray, sfreq: float, fmin: float, fmax: float) -> float:
    freqs, psd = welch(signal_1d, fs=sfreq, nperseg=min(len(signal_1d), 512))
    mask = (freqs >= fmin) & (freqs < fmax)
    if not np.any(mask):
        return 0.0
    return float(np.trapezoid(psd[mask], freqs[mask]))


def _extract_channel_features(channel_signal: np.ndarray, sfreq: float) -> dict[str, float]:
    feats: dict[str, float] = {
        "mean": float(np.mean(channel_signal)),
        "std": float(np.std(channel_signal)),
        "min": float(np.min(channel_signal)),
        "max": float(np.max(channel_signal)),
        "rms": float(np.sqrt(np.mean(np.square(channel_signal)))),
        "energy": float(np.sum(np.square(channel_signal))),
    }

    for band_name, (fmin, fmax) in EEG_BANDS.items():
        feats[f"bp_{band_name}"] = _bandpower(channel_signal, sfreq, fmin, fmax)

    total_bp = sum(feats[f"bp_{name}"] for name in EEG_BANDS)
    for band_name in EEG_BANDS:
        key = f"bp_{band_name}_rel"
        feats[key] = feats[f"bp_{band_name}"] / total_bp if total_bp > 0 else 0.0

    return feats


def extract_epoch_features(epochs: np.ndarray, sfreq: float) -> pd.DataFrame:
    rows: list[dict[str, float]] = []

    for epoch_idx, epoch in enumerate(epochs):
        row: dict[str, float] = {"epoch_idx": epoch_idx}

        for ch_idx, ch_signal in enumerate(epoch):
            ch_feats = _extract_channel_features(ch_signal, sfreq)
            for k, v in ch_feats.items():
                row[f"ch{ch_idx}_{k}"] = v

        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["epoch_idx"])

    return pd.DataFrame(rows)


def build_feature_table(records: Iterable[tuple[str, np.ndarray]], sfreq: float) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []

    for source_id, epochs in records:
        df = extract_epoch_features(epochs, sfreq)
        if df.empty:
            continue
        df.insert(0, "source_id", source_id)
        parts.append(df)

    if not parts:
        return pd.DataFrame(columns=["source_id", "epoch_idx"])

    return pd.concat(parts, ignore_index=True)
