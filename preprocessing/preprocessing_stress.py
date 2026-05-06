import argparse
import pickle
import re
from pathlib import Path
from typing import Tuple

import lmdb
import mne
import numpy as np


SELECTED_CHANNELS = [
    "EEG Fp1",
    "EEG Fp2",
    "EEG F3",
    "EEG F4",
    "EEG F7",
    "EEG F8",
    "EEG T3",
    "EEG T4",
    "EEG C3",
    "EEG C4",
    "EEG T5",
    "EEG T6",
    "EEG P3",
    "EEG P4",
    "EEG O1",
    "EEG O2",
    "EEG Fz",
    "EEG Cz",
    "EEG Pz",
    "EEG A2-A1",
]


def parse_subject_and_label(file_name: str) -> Tuple[int, int]:
    stem = Path(file_name).stem
    m = re.fullmatch(r"Subject(\d+)_(\d+)", stem)
    if m is None:
        raise ValueError(f"Unexpected file name format: {file_name}")
    subject_id = int(m.group(1))
    suffix = int(m.group(2))
    if suffix not in (1, 2):
        raise ValueError(f"Unexpected suffix in file name: {file_name}")
    return subject_id, suffix - 1


def build_split(subject_id: int) -> str:
    if 0 <= subject_id <= 27:
        return "train"
    if 28 <= subject_id <= 31:
        return "val"
    if 32 <= subject_id <= 35:
        return "test"
    raise ValueError(f"Subject id out of expected range 0..35: {subject_id}")


def save_to_lmdb(db, key: str, value: dict) -> None:
    with db.begin(write=True) as txn:
        txn.put(key.encode(), pickle.dumps(value))


def zscore_per_recording(eeg: np.ndarray, eps: float = 1e-6):
    mu = eeg.mean(axis=1, keepdims=True)
    std = eeg.std(axis=1, keepdims=True)
    std = np.maximum(std, eps)
    eeg_z = (eeg - mu) / std
    return eeg_z.astype(np.float32), mu.astype(np.float32), std.astype(np.float32)


def preprocess_file(edf_path: Path, target_fs: int, l_freq: float, h_freq: float, window_seconds: float, out_type: str, apply_recording_zscore: bool = True):
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")

    missing = [ch for ch in SELECTED_CHANNELS if ch not in raw.ch_names]
    if missing:
        raise ValueError(f"{edf_path.name}: missing channels {missing}")

    raw.pick(SELECTED_CHANNELS)
    raw.reorder_channels(SELECTED_CHANNELS)
    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose="ERROR")

    if int(round(raw.info["sfreq"])) != int(target_fs):
        raw.resample(target_fs, verbose="ERROR")

    eeg = raw.get_data(units="uV").astype(np.float32)
    raw.close()

    if apply_recording_zscore:
        eeg, mu, std = zscore_per_recording(eeg)
    else:
        mu = np.zeros((eeg.shape[0], 1), dtype=np.float32)
        std = np.ones((eeg.shape[0], 1), dtype=np.float32)

    win = int(round(window_seconds * target_fs))
    remainder = eeg.shape[1] % win
    if remainder != 0:
        eeg = eeg[:, :-remainder]

    if eeg.shape[1] == 0:
        return [], mu, std

    eeg = eeg.reshape(len(SELECTED_CHANNELS), -1, win).transpose(1, 0, 2)

    samples = []
    for sample in eeg:
        sample = sample.astype(np.float32)
        if out_type == "cbramod":
            sample = sample.reshape(len(SELECTED_CHANNELS), int(window_seconds), target_fs)
        samples.append(sample)

    return samples, mu, std


def main():
    parser = argparse.ArgumentParser(description="Preprocess MentalArithmetic / EEGMAT dataset with recording-level z-score")
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--processed", type=str, required=True)
    parser.add_argument("--type", type=str, choices=["reve", "cbramod"], default="reve")
    parser.add_argument("--target_fs", type=int, default=100)
    parser.add_argument("--l_freq", type=float, default=0.5)
    parser.add_argument("--h_freq", type=float, default=45.0)
    parser.add_argument("--window_seconds", type=float, default=5.0)
    parser.add_argument("--map_size_gb", type=float, default=2.0)
    parser.add_argument("--no_recording_zscore", action="store_true")
    args = parser.parse_args()

    root_dir = Path(args.root)
    processed_dir = Path(args.processed)
    processed_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in root_dir.glob("*.edf")])
    if len(files) == 0:
        raise ValueError(f"No EDF files found in {root_dir}")

    db = lmdb.open(str(processed_dir), map_size=int(args.map_size_gb * (1024 ** 3)), subdir=True)
    dataset = {"train": [], "val": [], "test": []}

    file_counts = {"train": 0, "val": 0, "test": 0}
    window_counts = {"train": 0, "val": 0, "test": 0}
    label_counts = {"train": {0: 0, 1: 0}, "val": {0: 0, 1: 0}, "test": {0: 0, 1: 0}}

    print("Found EDF files:", len(files))
    print(f"Band-pass: {args.l_freq}-{args.h_freq} Hz")
    print(f"Final sampling rate: {args.target_fs} Hz")
    print(f"Window length: {args.window_seconds} s")
    print(f"Output type: {args.type}")
    print(f"Recording-level z-score: {not args.no_recording_zscore}")

    for edf_path in files:
        subject_id, label = parse_subject_and_label(edf_path.name)
        split = build_split(subject_id)

        try:
            samples, mu, std = preprocess_file(
                edf_path=edf_path,
                target_fs=args.target_fs,
                l_freq=args.l_freq,
                h_freq=args.h_freq,
                window_seconds=args.window_seconds,
                out_type=args.type,
                apply_recording_zscore=not args.no_recording_zscore,
            )
        except Exception as e:
            print(f"[skip] {edf_path.name}: {e}")
            continue

        file_counts[split] += 1
        label_counts[split][label] += 1
        print(f"{edf_path.name} -> split={split}, label={label}, windows={len(samples)}")

        for i, sample in enumerate(samples):
            sample_key = f"{edf_path.stem}-{i}"
            data_dict = {
                "sample": sample.astype(np.float32),
                "label": int(label),
                "subject_id": int(subject_id),
                "fs": int(args.target_fs),
                "ch_names": list(SELECTED_CHANNELS),
                "source_file": edf_path.name,
                "recording_mean": mu,
                "recording_std": std,
                "recording_zscore": bool(not args.no_recording_zscore),
            }
            save_to_lmdb(db, sample_key, data_dict)
            dataset[split].append(sample_key)
            window_counts[split] += 1

    meta = {
        "dataset_name": "MentalArithmetic / EEGMAT",
        "selected_channels": list(SELECTED_CHANNELS),
        "target_fs": int(args.target_fs),
        "bandpass": [float(args.l_freq), float(args.h_freq)],
        "window_seconds": float(args.window_seconds),
        "recording_zscore": bool(not args.no_recording_zscore),
        "split_rule": {
            "train_subjects": "Subject00-Subject27",
            "val_subjects": "Subject28-Subject31",
            "test_subjects": "Subject32-Subject35",
        },
        "file_counts": file_counts,
        "window_counts": window_counts,
        "label_counts_per_split": label_counts,
    }

    save_to_lmdb(db, "__keys__", dataset)
    save_to_lmdb(db, "__meta__", meta)
    db.close()

    print("\nDone.")
    for split in ("train", "val", "test"):
        print(
            f"{split}: files={file_counts[split]}, windows={window_counts[split]}, "
            f"label0={label_counts[split][0]}, label1={label_counts[split][1]}"
        )


if __name__ == "__main__":
    main()
