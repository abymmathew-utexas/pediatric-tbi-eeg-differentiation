from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import mne
import numpy as np


SUPPORTED_EXTENSIONS = {".edf", ".bdf", ".fif", ".vhdr", ".set"}


@dataclass
class PreprocessConfig:
    l_freq: float = 1.0
    h_freq: float = 40.0
    notch_freq: float | None = 60.0
    target_sfreq: float = 256.0
    epoch_length_sec: float = 4.0
    epoch_overlap_sec: float = 2.0


def discover_eeg_files(input_dir: Path) -> list[Path]:
    files = [
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(files)


def load_raw_eeg(file_path: Path) -> mne.io.BaseRaw:
    ext = file_path.suffix.lower()
    if ext in {".edf", ".bdf"}:
        return mne.io.read_raw_edf(file_path, preload=True, verbose="ERROR")
    if ext == ".fif":
        return mne.io.read_raw_fif(file_path, preload=True, verbose="ERROR")
    if ext == ".vhdr":
        return mne.io.read_raw_brainvision(file_path, preload=True, verbose="ERROR")
    if ext == ".set":
        return mne.io.read_raw_eeglab(file_path, preload=True, verbose="ERROR")
    raise ValueError(f"Unsupported EEG file: {file_path}")


def preprocess_raw(raw: mne.io.BaseRaw, config: PreprocessConfig) -> mne.io.BaseRaw:
    raw = raw.copy().pick("eeg")
    raw.filter(l_freq=config.l_freq, h_freq=config.h_freq, verbose="ERROR")

    if config.notch_freq is not None:
        raw.notch_filter(freqs=[config.notch_freq], verbose="ERROR")

    raw.resample(config.target_sfreq)
    return raw


def epoch_raw(raw: mne.io.BaseRaw, config: PreprocessConfig) -> np.ndarray:
    data = raw.get_data()
    sfreq = raw.info["sfreq"]

    window = int(config.epoch_length_sec * sfreq)
    step = int((config.epoch_length_sec - config.epoch_overlap_sec) * sfreq)
    if step <= 0:
        raise ValueError("epoch_overlap_sec must be smaller than epoch_length_sec")

    n_samples = data.shape[1]
    starts = range(0, max(0, n_samples - window + 1), step)
    epochs = [data[:, start:start + window] for start in starts]

    if not epochs:
        return np.empty((0, data.shape[0], window), dtype=np.float32)

    return np.stack(epochs).astype(np.float32)


def preprocess_file(file_path: Path, config: PreprocessConfig) -> np.ndarray:
    raw = load_raw_eeg(file_path)
    raw_pp = preprocess_raw(raw, config)
    return epoch_raw(raw_pp, config)


def iter_preprocessed(file_paths: Iterable[Path], config: PreprocessConfig):
    for file_path in file_paths:
        try:
            epochs = preprocess_file(file_path, config)
            yield file_path, epochs
        except Exception as exc:
            print(f"[WARN] Failed to preprocess {file_path}: {exc}")
