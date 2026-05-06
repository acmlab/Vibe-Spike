
import argparse
import os
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lmdb
import mne
import numpy as np


SELECTED_CHANNELS = [
    "EEG Fp1-LE", "EEG Fp2-LE", "EEG F3-LE", "EEG F4-LE",
    "EEG C3-LE", "EEG C4-LE", "EEG P3-LE", "EEG P4-LE",
    "EEG O1-LE", "EEG O2-LE", "EEG F7-LE", "EEG F8-LE",
    "EEG T3-LE", "EEG T4-LE", "EEG T5-LE", "EEG T6-LE",
    "EEG Fz-LE", "EEG Cz-LE", "EEG Pz-LE",
]


def parse_record_info(file_path: Path) -> Optional[Dict]:
    stem = file_path.stem.strip()
    tokens = re.split(r"[\s_]+", stem)
    tokens = [t for t in tokens if t]

    label_token = None
    condition = None
    session = None

    for t in tokens:
        tu = t.upper()
        if tu in {"H", "MDD"} and label_token is None:
            label_token = tu
        elif re.fullmatch(r"S\d+", tu):
            session = tu
        elif tu in {"EC", "EO", "TASK"}:
            condition = tu

    if label_token is None or condition is None:
        return None

    label = 1 if label_token == "MDD" else 0
    return {
        "label_name": label_token,
        "label": label,
        "condition": condition,
        "session": session,
    }


def iter_files_original_style(root_dir: Path):
    files_H, files_MDD = [], []
    for file_path in root_dir.rglob("*.edf"):
        meta = parse_record_info(file_path)
        if meta is None:
            continue
        if meta["condition"] == "TASK":
            continue

        record = {
            "path": file_path,
            "name": file_path.name,
            "relpath": str(file_path.relative_to(root_dir)),
            **meta,
        }

        if meta["label_name"] == "MDD":
            files_MDD.append(record)
        else:
            files_H.append(record)

    files_H = sorted(files_H, key=lambda x: (x["name"], x["relpath"]))
    files_MDD = sorted(files_MDD, key=lambda x: (x["name"], x["relpath"]))
    return files_H, files_MDD


def build_original_split(files_H: List[Dict], files_MDD: List[Dict]):
    files_dict = {
        "train": [],
        "val": [],
        "test": [],
    }

    files_dict["train"].extend(files_H[:40])
    files_dict["train"].extend(files_MDD[:42])

    files_dict["val"].extend(files_H[40:48])
    files_dict["val"].extend(files_MDD[42:52])

    files_dict["test"].extend(files_H[48:])
    files_dict["test"].extend(files_MDD[52:])

    return files_dict


def open_lmdb(path: Path, map_size: int):
    path.mkdir(parents=True, exist_ok=True)
    return lmdb.open(str(path), map_size=map_size, subdir=True)


def save_to_lmdb(db, key: str, value: dict):
    with db.begin(write=True) as txn:
        txn.put(key.encode(), pickle.dumps(value))


def preprocess_recording(
    edf_path: Path,
    target_fs: int,
    window_seconds: float,
    l_freq: float,
    h_freq: float,
    notch_freq: float,
    scale_uv_divisor: Optional[float],
):
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")

    missing = [ch for ch in SELECTED_CHANNELS if ch not in raw.ch_names]
    if missing:
        raise ValueError(f"Missing channels in {edf_path.name}: {missing}")

    raw.pick(SELECTED_CHANNELS)
    raw.reorder_channels(SELECTED_CHANNELS)

    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose="ERROR")

    orig_fs = float(raw.info["sfreq"])
    if notch_freq < orig_fs / 2.0:
        raw.notch_filter(freqs=[notch_freq], verbose="ERROR")

    if int(round(orig_fs)) != int(target_fs):
        raw.resample(target_fs, verbose="ERROR")

    data = raw.get_data(units="uV").astype(np.float32)  # [C, T]
    raw.close()

    if scale_uv_divisor is not None and scale_uv_divisor > 0:
        data = data / float(scale_uv_divisor)

    win = int(round(window_seconds * target_fs))
    n_total = data.shape[1]
    tail = n_total % win
    if tail != 0:
        data = data[:, :-tail]

    if data.shape[1] == 0:
        return []

    data = data.reshape(len(SELECTED_CHANNELS), -1, win)  # [C, N, T]
    data = np.transpose(data, (1, 0, 2))                  # [N, C, T]

    windows = []
    for sample in data:
        sample = sample.astype(np.float32)
        sample = sample - sample.mean(axis=-1, keepdims=True)
        windows.append(sample)
    return windows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=str, required=True, help="Root directory of Mumtaz EDF files")
    parser.add_argument("--processed", type=str, required=True, help="Output LMDB directory")
    parser.add_argument("--target_fs", type=int, default=100, help="Final sampling rate")
    parser.add_argument("--window_seconds", type=float, default=5.0, help="Window length in seconds")
    parser.add_argument("--l_freq", type=float, default=0.3)
    parser.add_argument("--h_freq", type=float, default=45.0)
    parser.add_argument("--notch_freq", type=float, default=50.0)
    parser.add_argument("--scale_uv_divisor", type=float, default=None, help="Optional amplitude scaling divisor, e.g. 100. Use none to disable.")
    parser.add_argument("--map_size_gb", type=float, default=4.0)
    args = parser.parse_args()

    root_dir = Path(args.raw)
    out_dir = Path(args.processed)

    files_H, files_MDD = iter_files_original_style(root_dir)
    if len(files_H) == 0 and len(files_MDD) == 0:
        raise ValueError(f"No usable EDF files found under {root_dir}")

    files_dict = build_original_split(files_H, files_MDD)
    db = open_lmdb(out_dir, map_size=int(args.map_size_gb * (1024 ** 3)))
    dataset_keys = {"train": [], "val": [], "test": []}

    split_file_counts = {k: 0 for k in dataset_keys}
    split_window_counts = {k: 0 for k in dataset_keys}
    split_label_windows = {k: {0: 0, 1: 0} for k in dataset_keys}

    print("Original-style split:")
    print(f"H files   : {len(files_H)}")
    print(f"MDD files : {len(files_MDD)}")
    print(f"Selected channels: {len(SELECTED_CHANNELS)}")
    print(f"Final sampling rate: {args.target_fs}")
    print(f"Window length: {args.window_seconds}s")

    for split in ["train", "val", "test"]:
        print(f"\n=== {split} ===")
        for rec in files_dict[split]:
            split_file_counts[split] += 1
            try:
                windows = preprocess_recording(
                    edf_path=rec["path"],
                    target_fs=args.target_fs,
                    window_seconds=args.window_seconds,
                    l_freq=args.l_freq,
                    h_freq=args.h_freq,
                    notch_freq=args.notch_freq,
                    scale_uv_divisor=args.scale_uv_divisor,
                )
            except Exception as e:
                print(f"[skip] {rec['relpath']}: {e}")
                continue

            print(f"{rec['relpath']} -> {len(windows)} windows")
            for i, sample in enumerate(windows):
                key = f"{rec['path'].stem}__{i}__{split_file_counts[split]}"
                payload = {
                    "sample": sample,                     # [19, 500] when target_fs=100 and window_seconds=5
                    "label": int(rec["label"]),          # 0=H, 1=MDD
                    "condition": rec["condition"],       # EC / EO
                    "session": rec["session"],
                    "fs": int(args.target_fs),
                    "ch_names": list(SELECTED_CHANNELS),
                    "source_file": str(rec["path"]),
                }
                save_to_lmdb(db, key, payload)
                dataset_keys[split].append(key)
                split_window_counts[split] += 1
                split_label_windows[split][int(rec["label"])] += 1

    meta = {
        "split_mode": "original_author_style_file_split",
        "selected_channels": list(SELECTED_CHANNELS),
        "target_fs": int(args.target_fs),
        "window_seconds": float(args.window_seconds),
        "file_counts": split_file_counts,
        "window_counts": split_window_counts,
        "label_window_counts": split_label_windows,
        "files_h_sorted": [x["relpath"] for x in files_H],
        "files_mdd_sorted": [x["relpath"] for x in files_MDD],
    }

    save_to_lmdb(db, "__keys__", dataset_keys)
    save_to_lmdb(db, "__meta__", meta)
    db.close()

    print("\nDone.")
    for split in ("train", "val", "test"):
        print(
            f"{split}: files={split_file_counts[split]}, "
            f"windows={split_window_counts[split]}, "
            f"H={split_label_windows[split][0]}, "
            f"MDD={split_label_windows[split][1]}"
        )


if __name__ == "__main__":
    main()
