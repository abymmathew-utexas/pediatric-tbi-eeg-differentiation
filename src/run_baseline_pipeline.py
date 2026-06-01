from __future__ import annotations

import argparse
from pathlib import Path

from data_preprocessing.baseline_preprocess import (
    PreprocessConfig,
    discover_eeg_files,
    iter_preprocessed,
)
from features.baseline_feature_extraction import build_feature_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baseline EEG preprocessing + feature extraction pipeline"
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="Folder with EEG files")
    parser.add_argument("--output-csv", type=Path, required=True, help="Path to output features CSV")
    parser.add_argument("--sfreq", type=float, default=256.0, help="Target sampling rate")
    parser.add_argument("--epoch-sec", type=float, default=4.0, help="Epoch length in seconds")
    parser.add_argument("--overlap-sec", type=float, default=2.0, help="Epoch overlap in seconds")
    parser.add_argument("--l-freq", type=float, default=1.0, help="Bandpass low cutoff")
    parser.add_argument("--h-freq", type=float, default=40.0, help="Bandpass high cutoff")
    parser.add_argument(
        "--notch",
        type=float,
        default=60.0,
        help="Notch filter frequency; set <=0 to disable",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    notch_freq = args.notch if args.notch > 0 else None
    config = PreprocessConfig(
        l_freq=args.l_freq,
        h_freq=args.h_freq,
        notch_freq=notch_freq,
        target_sfreq=args.sfreq,
        epoch_length_sec=args.epoch_sec,
        epoch_overlap_sec=args.overlap_sec,
    )

    eeg_files = discover_eeg_files(args.input_dir)
    if not eeg_files:
        raise FileNotFoundError(f"No supported EEG files found in {args.input_dir}")

    print(f"[INFO] Found {len(eeg_files)} EEG files")

    records = []
    for file_path, epochs in iter_preprocessed(eeg_files, config):
        if epochs.size == 0:
            print(f"[WARN] No epochs extracted: {file_path}")
            continue
        records.append((file_path.stem, epochs))
        print(f"[INFO] Processed {file_path.name}: {epochs.shape[0]} epochs")

    features_df = build_feature_table(records, sfreq=config.target_sfreq)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(args.output_csv, index=False)

    print(f"[INFO] Saved feature table: {args.output_csv}")
    print(f"[INFO] Feature shape: {features_df.shape}")


if __name__ == "__main__":
    main()
