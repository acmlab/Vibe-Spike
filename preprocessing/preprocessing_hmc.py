import argparse
import os
import pickle
import random
from collections import Counter
from pathlib import Path

import mne
import numpy as np
import pandas as pd


DROP_CHANNELS = ["EMG chin", "EOG E1-M2", "EOG E2-M2", "ECG"]
REQUIRED_CHANNELS = ["EEG F4-M1", "EEG C4-M1", "EEG O2-M1", "EEG C3-M2"]
OUTPUT_CH_NAMES = ["F4", "C4", "O2", "C3"]

SYMBOLS_HMC = {
    "Sleep stage W": 0,
    "Sleep stage N1": 1,
    "Sleep stage N2": 2,
    "Sleep stage N3": 3,
    "Sleep stage R": 4,
}


def save_pickle(obj, filename):
    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def find_annotation_file(edf_path: Path) -> Path:
    candidate = edf_path.with_name(edf_path.stem + "_sleepscoring.txt")
    if not candidate.exists():
        raise FileNotFoundError(f"Annotation file not found for {edf_path.name}: {candidate}")
    return candidate


def read_and_preprocess_edf(
    edf_path: Path,
    target_sfreq: float = 100.0,
    l_freq: float = 0.3,
    h_freq: float = 45.0,
    notch_freq: float = 50.0,
    n_jobs: int = 1,
):
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")

    drop_now = [ch for ch in DROP_CHANNELS if ch in raw.ch_names]
    if drop_now:
        raw.drop_channels(drop_now)

    missing = [ch for ch in REQUIRED_CHANNELS if ch not in raw.ch_names]
    if missing:
        raw.close()
        raise ValueError(f"Missing required channels: {missing}")

    raw.pick(REQUIRED_CHANNELS)
    raw.reorder_channels(REQUIRED_CHANNELS)

    raw.filter(l_freq=l_freq, h_freq=h_freq, n_jobs=n_jobs, verbose="ERROR")

    orig_sfreq = float(raw.info["sfreq"])
    if notch_freq < orig_sfreq / 2.0:
        raw.notch_filter(freqs=[notch_freq], n_jobs=n_jobs, verbose="ERROR")

    raw.resample(target_sfreq, n_jobs=n_jobs, verbose="ERROR")

    signals = raw.get_data(units="uV").astype(np.float32)
    times = raw.times.astype(np.float64)
    sfreq = float(raw.info["sfreq"])
    return raw, signals, times, sfreq


def build_epochs_from_annotations(signals, times, sfreq, ann_df):
    ann_df = normalize_columns(ann_df)

    required_cols = ["Recording onset", "Duration", "Annotation"]
    for col in required_cols:
        if col not in ann_df.columns:
            raise ValueError(f"Missing column '{col}' in annotation file. Found: {list(ann_df.columns)}")

    epoch_len = int(round(sfreq * 30.0))
    feats = []
    labels = []

    summary = {
        "valid_rows": 0,
        "skipped_duration": 0,
        "skipped_label": 0,
        "skipped_length": 0,
    }

    for _, row in ann_df.iterrows():
        duration = float(row["Duration"])
        annotation = str(row["Annotation"]).strip()

        if abs(duration - 30.0) > 1e-6:
            summary["skipped_duration"] += 1
            continue

        if annotation not in SYMBOLS_HMC:
            summary["skipped_label"] += 1
            continue

        onset = float(row["Recording onset"])
        start = int(np.searchsorted(times, onset, side="left"))
        end = start + epoch_len

        if end > signals.shape[1]:
            summary["skipped_length"] += 1
            continue

        epoch = signals[:, start:end]
        if epoch.shape != (signals.shape[0], epoch_len):
            summary["skipped_length"] += 1
            continue

        feats.append(epoch.astype(np.float32))
        labels.append(int(SYMBOLS_HMC[annotation]))
        summary["valid_rows"] += 1

    if len(feats) == 0:
        return np.zeros((0, signals.shape[0], epoch_len), dtype=np.float32), np.zeros((0,), dtype=np.int64), summary

    feats = np.stack(feats, axis=0)
    labels = np.asarray(labels, dtype=np.int64)
    return feats, labels, summary


def process_recording(edf_path: Path, out_dir: Path, target_sfreq: float, n_jobs: int):
    raw, signals, times, sfreq = read_and_preprocess_edf(
        edf_path=edf_path,
        target_sfreq=target_sfreq,
        l_freq=0.3,
        h_freq=45.0,
        notch_freq=50.0,
        n_jobs=n_jobs,
    )

    ann_path = find_annotation_file(edf_path)
    ann_df = pd.read_csv(ann_path)
    epochs, labels, summary = build_epochs_from_annotations(signals, times, sfreq, ann_df)

    base = edf_path.stem
    label_counter = Counter()

    for idx, (x, y) in enumerate(zip(epochs, labels)):
        sample = {
            "X": x,
            "ch_names": OUTPUT_CH_NAMES,
            "y": int(y),
            "fs": int(round(sfreq)),
            "record_id": base,
            "epoch_index": idx,
        }
        save_pickle(sample, out_dir / f"{base}-{idx}.pkl")
        label_counter[int(y)] += 1

    raw.close()

    return {
        "record_id": base,
        "num_epochs": int(len(labels)),
        "label_counter": dict(label_counter),
        **summary,
    }


def split_files(edf_files, seed, train_ratio, val_ratio, test_ratio):
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-8:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    edf_files = list(edf_files)
    rng = random.Random(seed)
    rng.shuffle(edf_files)

    n = len(edf_files)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_test = n - n_train - n_val

    train_files = edf_files[:n_train]
    val_files = edf_files[n_train:n_train + n_val]
    test_files = edf_files[n_train + n_val:n_train + n_val + n_test]
    return train_files, val_files, test_files


def process_split(file_list, split_name, out_dir, target_sfreq, n_jobs):
    split_dir = out_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    split_stats = []
    total_labels = Counter()
    num_failed = 0

    print(f"\n=== Processing {split_name}: {len(file_list)} recordings ===")
    for edf_path in file_list:
        print(f"  {edf_path.name}")
        try:
            stats = process_recording(edf_path, split_dir, target_sfreq=target_sfreq, n_jobs=n_jobs)
            split_stats.append(stats)
            total_labels.update(stats["label_counter"])
        except Exception as e:
            num_failed += 1
            print(f"  [skip] {edf_path.name}: {e}")

    total_epochs = sum(s["num_epochs"] for s in split_stats)
    print(f"--- {split_name} summary ---")
    print(f"saved epochs : {total_epochs}")
    print(f"failed files : {num_failed}")
    print(f"label counts : {dict(sorted(total_labels.items()))}")

    return {
        "split": split_name,
        "num_recordings": len(file_list),
        "num_processed_recordings": len(split_stats),
        "num_failed_recordings": num_failed,
        "num_epochs": total_epochs,
        "label_counts": dict(sorted(total_labels.items())),
        "records": split_stats,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=str, required=True, help="Raw HMC data directory")
    parser.add_argument("--processed", type=str, required=True, help="Output processed directory")
    parser.add_argument("--target_sfreq", type=float, default=100.0, help="Final sampling rate")
    parser.add_argument("--seed", type=int, default=42, help="Split seed")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--n_jobs", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    args = parser.parse_args()

    raw_root = Path(args.raw)
    out_dir = Path(args.processed)
    out_dir.mkdir(parents=True, exist_ok=True)

    edf_files = sorted(raw_root.rglob("*.edf"))
    if len(edf_files) == 0:
        raise ValueError(f"No .edf files found under {raw_root}")

    train_files, val_files, test_files = split_files(
        edf_files,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    train_summary = process_split(train_files, "train", out_dir, args.target_sfreq, args.n_jobs)
    val_summary = process_split(val_files, "eval", out_dir, args.target_sfreq, args.n_jobs)
    test_summary = process_split(test_files, "test", out_dir, args.target_sfreq, args.n_jobs)

    meta = {
        "target_sfreq": args.target_sfreq,
        "required_channels": REQUIRED_CHANNELS,
        "output_ch_names": OUTPUT_CH_NAMES,
        "label_map": SYMBOLS_HMC,
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "train_summary": train_summary,
        "eval_summary": val_summary,
        "test_summary": test_summary,
    }

    with open(out_dir / "meta_summary.pkl", "wb") as f:
        pickle.dump(meta, f)

    print("\nDone.")
    print(f"Saved metadata to: {out_dir / 'meta_summary.pkl'}")


if __name__ == "__main__":
    main()
