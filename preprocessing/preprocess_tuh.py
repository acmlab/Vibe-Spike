# preprocess_tuh_biot_style_100hz.py
# -*- coding: utf-8 -*-
"""
BIOT-style TUAB/TUEV preprocessing, but resampled to 100 Hz.

This script follows the two BIOT baseline preprocessing files you provided:

TUAB BIOT-style:
  - use official TUAB folders:
      train/abnormal/01_tcp_ar
      train/normal/01_tcp_ar
      eval/abnormal/01_tcp_ar
      eval/normal/01_tcp_ar
  - split official train subjects into train/val with 80/20 per class
  - keep official eval as test
  - convert REF channels to 16 TCP bipolar channels
  - cut non-overlapping 10 s windows
  - original BIOT uses 200 Hz and 2000 samples;
    this script uses 100 Hz and 1000 samples.

TUEV BIOT-style:
  - use official TUEV folders:
      train/
      eval/
  - official train -> train
  - official eval  -> val by default
  - convert REF channels to 16 TCP bipolar channels
  - extract event windows from .rec annotations
  - original BIOT uses 250 Hz and 5 s event windows;
    this script uses 100 Hz and 500 samples.

Output:
  - LMDB by default, compatible with simple Dataset readers.
  - Optional PKL folders that mimic BIOT sample dictionaries.

Sample keys in LMDB:
  sample: np.float32 [16, T]
  label : int
  fs    : 100
  ch_names: 16 standard single-electrode names used by the model
  bipolar_ch_names: original BIOT bipolar derivation names
  plus metadata such as source_file, split, subject_id, etc.

Examples:

TUAB:
  python preprocess_tuh_biot_style_100hz.py \\
    --dataset tuab \\
    --root /workspace/TUH/TUAB/v3.0.1/edf \\
    --processed /workspace/TUH_preprocessed/TUAB_BIOT_16ch_100hz_lmdb \\
    --overwrite

TUEV:
  python preprocess_tuh_biot_style_100hz.py \\
    --dataset tuev \\
    --root /workspace/TUH/TUEV/v2.0.1/edf \\
    --processed /workspace/TUH_preprocessed/TUEV_BIOT_16ch_100hz_lmdb \\
    --overwrite
"""

import argparse
import csv
import json
import pickle
import random
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lmdb
import mne
import numpy as np


# ============================================================
# BIOT 16-channel TCP bipolar montage
# ============================================================

BIPOLAR_PAIRS = [
    ("FP1", "F7"),
    ("F7", "T3"),
    ("T3", "T5"),
    ("T5", "O1"),

    ("FP2", "F8"),
    ("F8", "T4"),
    ("T4", "T6"),
    ("T6", "O2"),

    ("FP1", "F3"),
    ("F3", "C3"),
    ("C3", "P3"),
    ("P3", "O1"),

    ("FP2", "F4"),
    ("F4", "C4"),
    ("C4", "P4"),
    ("P4", "O2"),
]

BIPOLAR_NAMES = [f"{a}-{b}" for a, b in BIPOLAR_PAIRS]

# ---------------------------------------------------------------------
# IMPORTANT for your S2/resampling model:
# The signal is still BIOT-style bipolar difference, but the channel name
# exposed to the model is converted to a standard single-electrode name.
#
# This avoids names such as "FP1-F7" that cannot be matched by a standard
# 10-20 montage loader.
#
# We use a unique 16-channel 10-20 subset:
#   FP1-F7  -> Fp1
#   F7-T3   -> F7
#   T3-T5   -> T3
#   T5-O1   -> T5
#   FP2-F8  -> Fp2
#   F8-T4   -> F8
#   T4-T6   -> T4
#   T6-O2   -> T6
#   FP1-F3  -> F3
#   F3-C3   -> C3
#   C3-P3   -> P3
#   P3-O1   -> O1
#   FP2-F4  -> F4
#   F4-C4   -> C4
#   C4-P4   -> P4
#   P4-O2   -> O2
#
# This is not claiming the bipolar signal is a true monopolar signal.
# It is only a model-side spatial anchor so that the resampler can assign
# each bipolar derivation to a valid standard electrode coordinate.
# ---------------------------------------------------------------------
MODEL_STANDARD_16_NAMES = [
    "Fp1", "F7", "T3", "T5",
    "Fp2", "F8", "T4", "T6",
    "F3", "C3", "P3", "O1",
    "F4", "C4", "P4", "O2",
]

BIPOLAR_TO_MODEL_CHANNEL = {
    bipolar_name: model_name
    for bipolar_name, model_name in zip(BIPOLAR_NAMES, MODEL_STANDARD_16_NAMES)
}



def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def write_model_channel_mapping_csv(processed: Path):
    """
    Export the mapping between BIOT bipolar derivations and model-side
    standard single-electrode channel anchors.
    """
    ensure_dir(processed)
    out_csv = processed / "biot_16_to_model_standard_channels.csv"
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "bipolar_signal_name", "positive_electrode", "negative_electrode", "model_channel_name"])
        for i, ((pos_elec, neg_elec), bipolar_name, model_name) in enumerate(
            zip(BIPOLAR_PAIRS, BIPOLAR_NAMES, MODEL_STANDARD_16_NAMES)
        ):
            w.writerow([i, bipolar_name, pos_elec, neg_elec, model_name])
    return out_csv


def canonical_eeg_name(name: str) -> str:
    """
    Convert channel names such as:
      EEG FP1-REF, EEG FP1-LE, FP1-REF, FP1
    to canonical:
      FP1, F7, T3, ...
    """
    s = str(name).upper().strip()
    s = s.replace("EEG", "").strip()

    # Remove common reference suffixes.
    for suf in ["-REF", "-LE", "-AR", "_REF", "_LE", "_AR", ".REF", ".LE", ".AR"]:
        s = s.replace(suf, "")

    s = s.replace(" ", "").replace("_", "").replace(".", "")
    # If still contains a bipolar separator, keep the first token.
    if "-" in s:
        s = s.split("-")[0]

    alias = {
        "T7": "T3",
        "T8": "T4",
        "P7": "T5",
        "P8": "T6",
        "FPZ": "FPZ",
        "FZ": "FZ",
        "CZ": "CZ",
        "PZ": "PZ",
    }
    return alias.get(s, s)


def make_channel_index(raw_ch_names: List[str]) -> Dict[str, int]:
    idx = {}
    for i, ch in enumerate(raw_ch_names):
        key = canonical_eeg_name(ch)
        if key not in idx:
            idx[key] = i
    return idx


def convert_to_biot_16_bipolar(raw_data: np.ndarray, raw_ch_names: List[str]) -> np.ndarray:
    """
    raw_data: [C_raw, T]
    return:   [16, T]
    """
    ch_idx = make_channel_index(raw_ch_names)

    missing = []
    for a, b in BIPOLAR_PAIRS:
        if a not in ch_idx:
            missing.append(a)
        if b not in ch_idx:
            missing.append(b)
    if missing:
        missing_unique = sorted(set(missing))
        raise KeyError(f"Missing required BIOT TCP channels: {missing_unique}; raw_ch_names={raw_ch_names}")

    out = []
    for a, b in BIPOLAR_PAIRS:
        out.append(raw_data[ch_idx[a]] - raw_data[ch_idx[b]])
    return np.stack(out, axis=0).astype(np.float32)


def read_resampled_biot_16(edf_path: Path, target_fs: int, preload: bool = True) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Read EDF, resample to target_fs, convert to 16 bipolar TCP channels.

    Returns:
      signals_16: [16, T]
      times:      [T]
      fs:         float
    """
    raw = mne.io.read_raw_edf(str(edf_path), preload=preload, verbose="ERROR")
    if int(round(raw.info["sfreq"])) != int(target_fs):
        raw.resample(int(target_fs), verbose="ERROR")

    signals, times = raw[:]
    ch_names = list(raw.info["ch_names"])
    raw.close()

    signals_16 = convert_to_biot_16_bipolar(signals.astype(np.float32), ch_names)
    signals_16 = np.nan_to_num(signals_16, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    return signals_16, times.astype(np.float64), float(target_fs)


# ============================================================
# LMDB writer
# ============================================================

class LMDBWriter:
    def __init__(self, path: Path, map_size_gb: float = 128.0, commit_interval: int = 1000):
        ensure_dir(path)
        self.env = lmdb.open(
            str(path),
            map_size=int(float(map_size_gb) * (1024 ** 3)),
            subdir=True,
            lock=True,
            map_async=True,
        )
        self.txn = self.env.begin(write=True)
        self.commit_interval = int(commit_interval)
        self.count = 0

    def put(self, key: str, value):
        self.txn.put(key.encode("utf-8"), pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
        self.count += 1
        if self.count % self.commit_interval == 0:
            self.txn.commit()
            self.txn = self.env.begin(write=True)

    def close(self):
        self.txn.commit()
        self.env.sync()
        self.env.close()


class OutputWriter:
    def __init__(self, processed: Path, output_format: str, map_size_gb: float, commit_interval: int):
        self.processed = processed
        self.output_format = output_format
        self.keys = {}
        self.lmdb_writer = None

        if output_format in ["lmdb", "both"]:
            self.lmdb_writer = LMDBWriter(processed, map_size_gb=map_size_gb, commit_interval=commit_interval)

        if output_format in ["pkl", "both"]:
            ensure_dir(processed)

    def init_splits(self, splits: List[str]):
        self.keys = {s: [] for s in splits}
        if self.output_format in ["pkl", "both"]:
            for s in splits:
                ensure_dir(self.processed / s)

    def put(self, split: str, key_stem: str, lmdb_sample: dict, pkl_sample: dict):
        key = f"{split}/{key_stem}"
        if split not in self.keys:
            self.keys[split] = []
        self.keys[split].append(key)

        if self.lmdb_writer is not None:
            self.lmdb_writer.put(key, lmdb_sample)

        if self.output_format in ["pkl", "both"]:
            pkl_path = self.processed / split / f"{key_stem}.pkl"
            with open(pkl_path, "wb") as f:
                pickle.dump(pkl_sample, f, protocol=pickle.HIGHEST_PROTOCOL)

    def close(self, meta: dict):
        if self.lmdb_writer is not None:
            self.lmdb_writer.put("__keys__", self.keys)
            self.lmdb_writer.put("__meta__", meta)
            self.lmdb_writer.close()

        with open(self.processed / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        with open(self.processed / "keys_count.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["split", "n_samples"])
            for split, ks in self.keys.items():
                w.writerow([split, len(ks)])


# ============================================================
# TUAB BIOT-style preprocessing
# ============================================================

def subject_id_from_tuab_file(edf_path: Path) -> str:
    # BIOT baseline uses item.split("_")[0].
    return edf_path.name.split("_")[0]


def list_subjects(folder: Path) -> List[str]:
    edfs = sorted(folder.glob("*.edf"))
    return sorted(set(subject_id_from_tuab_file(p) for p in edfs))


def split_subjects_80_20(subjects: List[str], train_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    subjects = list(subjects)
    rng = random.Random(seed)
    rng.shuffle(subjects)
    n_train = int(len(subjects) * float(train_ratio))
    return subjects[:n_train], subjects[n_train:]


def find_tuab_folder(root: Path, split: str, label_name: str, channel_std: str) -> Path:
    p = root / split / label_name / channel_std
    if not p.exists():
        raise FileNotFoundError(f"TUAB folder not found: {p}")
    return p


def process_tuab_subjects_from_folder(
    folder: Path,
    subjects: List[str],
    split: str,
    label: int,
    writer: OutputWriter,
    target_fs: int,
    window_seconds: float,
) -> List[dict]:
    win = int(round(float(window_seconds) * int(target_fs)))
    summaries = []

    edfs = sorted(folder.glob("*.edf"))
    subject_set = set(subjects)

    for edf in edfs:
        sid = subject_id_from_tuab_file(edf)
        if sid not in subject_set:
            continue

        try:
            signals, times, fs = read_resampled_biot_16(edf, target_fs=target_fs)
        except Exception as e:
            print(f"[skip][TUAB] {edf}: {e}")
            summaries.append({
                "split": split,
                "source_file": str(edf),
                "subject_id": sid,
                "label": int(label),
                "error": str(e),
                "n_windows": 0,
            })
            continue

        n_win = int(signals.shape[1] // win)
        n_written = 0

        for i in range(n_win):
            start = i * win
            seg = signals[:, start:start + win].astype(np.float32)
            key_stem = f"{edf.stem}_{i:06d}"

            lmdb_sample = {
                "sample": seg,
                "X": seg,
                "label": int(label),
                "y": int(label),
                "fs": int(target_fs),
                "ch_names": list(MODEL_STANDARD_16_NAMES),
                "bipolar_ch_names": list(BIPOLAR_NAMES),
                "bipolar_to_model_channel": dict(BIPOLAR_TO_MODEL_CHANNEL),
                "subject_id": sid,
                "source_file": str(edf),
                "split": split,
                "window_index": int(i),
                "start_sample": int(start),
                "start_sec": float(start / target_fs),
                "window_seconds": float(window_seconds),
                "task": "tuab_normal_abnormal_biot_16ch",
            }
            pkl_sample = {
                "X": seg,
                "y": int(label),
            }
            writer.put(split, key_stem, lmdb_sample, pkl_sample)
            n_written += 1

        summaries.append({
            "split": split,
            "source_file": str(edf),
            "subject_id": sid,
            "label": int(label),
            "fs": float(fs),
            "n_samples": int(signals.shape[1]),
            "n_windows": int(n_written),
        })

        print(f"[TUAB] {split:5s} | label={label} | windows={n_written:5d} | {edf.name}")

    return summaries


def preprocess_tuab_biot_style(args):
    root = Path(args.root)
    processed = Path(args.processed)

    train_abn = find_tuab_folder(root, "train", "abnormal", args.channel_std)
    train_nor = find_tuab_folder(root, "train", "normal", args.channel_std)
    eval_abn = find_tuab_folder(root, "eval", "abnormal", args.channel_std)
    eval_nor = find_tuab_folder(root, "eval", "normal", args.channel_std)

    abn_subjects = list_subjects(train_abn)
    nor_subjects = list_subjects(train_nor)

    train_a, val_a = split_subjects_80_20(abn_subjects, args.tuab_train_ratio, args.seed)
    train_n, val_n = split_subjects_80_20(nor_subjects, args.tuab_train_ratio, args.seed + 1009)

    test_a = list_subjects(eval_abn)
    test_n = list_subjects(eval_nor)

    writer = OutputWriter(
        processed=processed,
        output_format=args.output_format,
        map_size_gb=args.map_size_gb,
        commit_interval=args.commit_interval,
    )
    writer.init_splits(["train", "val", "test"])

    file_summaries = []
    file_summaries += process_tuab_subjects_from_folder(train_abn, train_a, "train", 1, writer, args.target_fs, args.tuab_window_seconds)
    file_summaries += process_tuab_subjects_from_folder(train_nor, train_n, "train", 0, writer, args.target_fs, args.tuab_window_seconds)
    file_summaries += process_tuab_subjects_from_folder(train_abn, val_a, "val", 1, writer, args.target_fs, args.tuab_window_seconds)
    file_summaries += process_tuab_subjects_from_folder(train_nor, val_n, "val", 0, writer, args.target_fs, args.tuab_window_seconds)
    file_summaries += process_tuab_subjects_from_folder(eval_abn, test_a, "test", 1, writer, args.target_fs, args.tuab_window_seconds)
    file_summaries += process_tuab_subjects_from_folder(eval_nor, test_n, "test", 0, writer, args.target_fs, args.tuab_window_seconds)

    label_counts = {s: {0: 0, 1: 0} for s in ["train", "val", "test"]}
    for fsu in file_summaries:
        if "error" in fsu:
            continue
        split = fsu["split"]
        label = int(fsu["label"])
        label_counts[split][label] += int(fsu["n_windows"])

    meta = {
        "dataset": "TUAB",
        "style": "BIOT baseline style, resampled to 100 Hz",
        "task": "normal vs abnormal",
        "label_map": {"normal": 0, "abnormal": 1},
        "target_fs": int(args.target_fs),
        "channel_type": "BIOT 16 TCP bipolar signals anchored to standard single-electrode names",
        "model_channel_names": MODEL_STANDARD_16_NAMES,
        "bipolar_pairs": BIPOLAR_PAIRS,
        "bipolar_names": BIPOLAR_NAMES,
        "bipolar_to_model_channel": BIPOLAR_TO_MODEL_CHANNEL,
        "tuab_train_ratio": float(args.tuab_train_ratio),
        "tuab_val_ratio": float(1.0 - args.tuab_train_ratio),
        "window_seconds": float(args.tuab_window_seconds),
        "split_rule": {
            "train_val": "official train split is split subject-wise into train/val separately for normal and abnormal",
            "test": "official eval split is kept as test",
        },
        "subject_counts": {
            "train_abnormal": len(train_a),
            "val_abnormal": len(val_a),
            "test_abnormal": len(test_a),
            "train_normal": len(train_n),
            "val_normal": len(val_n),
            "test_normal": len(test_n),
        },
        "keys_count": {s: len(writer.keys.get(s, [])) for s in ["train", "val", "test"]},
        "label_counts": label_counts,
        "file_summaries": file_summaries,
    }

    writer.close(meta)
    return meta


# ============================================================
# TUEV BIOT-style preprocessing
# ============================================================

def find_annotation_file(edf_path: Path) -> Optional[Path]:
    for suffix in [".rec", ".csv", ".lbl", ".tse"]:
        p = edf_path.with_suffix(suffix)
        if p.exists():
            return p
    return None


def load_rec_event_data(rec_path: Path) -> np.ndarray:
    data = np.genfromtxt(str(rec_path), delimiter=",")
    if data.size == 0:
        return np.zeros((0, 4), dtype=np.float64)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] < 4:
        raise ValueError(f"Annotation file has <4 columns: {rec_path}, shape={data.shape}")
    return data.astype(np.float64)


def normalize_tuev_label(label_value: float, mode: str = "auto") -> int:
    raw = int(round(float(label_value)))

    if mode == "zero":
        if not (0 <= raw <= 5):
            raise ValueError(f"Expected zero-based TUEV label 0..5, got {raw}")
        return raw

    if mode == "one":
        if not (1 <= raw <= 6):
            raise ValueError(f"Expected one-based TUEV label 1..6, got {raw}")
        return raw - 1

    # auto
    if 1 <= raw <= 6:
        return raw - 1
    if 0 <= raw <= 5:
        return raw
    raise ValueError(f"Cannot normalize TUEV label: {label_value}")


def extract_biot_tuev_event_window(
    signals: np.ndarray,
    times: np.ndarray,
    start_sec: float,
    stop_sec: float,
    target_fs: int,
    window_seconds: float,
) -> np.ndarray:
    """
    Mimic BIOT BuildEvents:
      signals[:, offset + start - 2s : offset + end + 2s]
    but robustly crops/pads to exactly window_seconds * target_fs.

    With ordinary TUEV 1-second events and window_seconds=5,
    this is exactly 2s before + event + 2s after.
    """
    fs = int(target_fs)
    win = int(round(float(window_seconds) * fs))

    if len(times) == 0:
        raise ValueError("Empty times array")

    start_candidates = np.where(times >= float(start_sec))[0]
    end_candidates = np.where(times >= float(stop_sec))[0]

    if len(start_candidates) == 0:
        start_idx = len(times) - 1
    else:
        start_idx = int(start_candidates[0])

    if len(end_candidates) == 0:
        end_idx = len(times) - 1
    else:
        end_idx = int(end_candidates[0])

    offset = signals.shape[1]
    signals3 = np.concatenate([signals, signals, signals], axis=1)

    s = offset + start_idx - 2 * fs
    e = offset + end_idx + 2 * fs

    seg = signals3[:, s:e]

    if seg.shape[1] == win:
        return seg.astype(np.float32)

    # Robust fallback: center crop or edge pad to exactly win.
    if seg.shape[1] > win:
        extra = seg.shape[1] - win
        left = extra // 2
        seg = seg[:, left:left + win]
    elif seg.shape[1] < win:
        pad = win - seg.shape[1]
        if seg.shape[1] == 0:
            seg = np.zeros((signals.shape[0], win), dtype=np.float32)
        else:
            seg = np.pad(seg, ((0, 0), (pad // 2, pad - pad // 2)), mode="edge")

    return seg[:, :win].astype(np.float32)


def process_tuev_edf_folder(
    base_dir: Path,
    split: str,
    writer: OutputWriter,
    target_fs: int,
    event_window_seconds: float,
    label_base: str,
) -> List[dict]:
    summaries = []

    edfs = sorted(base_dir.rglob("*.edf"))
    for edf in edfs:
        ann = find_annotation_file(edf)
        if ann is None:
            print(f"[skip][TUEV] no annotation for {edf}")
            summaries.append({
                "split": split,
                "source_file": str(edf),
                "error": "no_annotation",
                "n_events_written": 0,
            })
            continue

        try:
            event_data = load_rec_event_data(ann)
            signals, times, fs = read_resampled_biot_16(edf, target_fs=target_fs)
        except Exception as e:
            print(f"[skip][TUEV] {edf}: {e}")
            summaries.append({
                "split": split,
                "source_file": str(edf),
                "annotation_file": str(ann),
                "error": str(e),
                "n_events_written": 0,
            })
            continue

        n_written = 0
        label_counts_local = {i: 0 for i in range(6)}

        for idx, row in enumerate(event_data):
            try:
                offending_channel = int(row[0])
                start_sec = float(row[1])
                stop_sec = float(row[2])
                label = normalize_tuev_label(row[3], mode=label_base)

                seg = extract_biot_tuev_event_window(
                    signals=signals,
                    times=times,
                    start_sec=start_sec,
                    stop_sec=stop_sec,
                    target_fs=target_fs,
                    window_seconds=event_window_seconds,
                )
            except Exception as e:
                print(f"[skip][TUEV event] {edf.name} event={idx}: {e}")
                continue

            key_stem = f"{edf.stem}-event{idx:06d}"

            lmdb_sample = {
                "sample": seg,
                "X": seg,
                "signal": seg,
                "label": int(label),
                "y": int(label),
                "offending_channel": int(offending_channel),
                "fs": int(target_fs),
                "ch_names": list(MODEL_STANDARD_16_NAMES),
                "bipolar_ch_names": list(BIPOLAR_NAMES),
                "bipolar_to_model_channel": dict(BIPOLAR_TO_MODEL_CHANNEL),
                "source_file": str(edf),
                "annotation_file": str(ann),
                "split": split,
                "event_index": int(idx),
                "event_channel": int(offending_channel),
                "event_start_sec": float(start_sec),
                "event_stop_sec": float(stop_sec),
                "window_seconds": float(event_window_seconds),
                "task": "tuev_event_classification_biot_16ch",
            }
            pkl_sample = {
                "signal": seg,
                "offending_channel": np.asarray([offending_channel], dtype=np.int64),
                "label": np.asarray([label], dtype=np.int64),
            }
            writer.put(split, key_stem, lmdb_sample, pkl_sample)
            n_written += 1
            label_counts_local[int(label)] += 1

        summaries.append({
            "split": split,
            "source_file": str(edf),
            "annotation_file": str(ann),
            "fs": float(fs),
            "n_samples": int(signals.shape[1]),
            "n_events_parsed": int(event_data.shape[0]),
            "n_events_written": int(n_written),
            "label_counts": label_counts_local,
        })
        print(f"[TUEV] {split:5s} | events={n_written:5d}/{event_data.shape[0]:5d} | {edf.name}")

    return summaries


def preprocess_tuev_biot_style(args):
    root = Path(args.root)
    processed = Path(args.processed)

    train_dir = root / "train"
    eval_dir = root / "eval"

    if not train_dir.exists():
        raise FileNotFoundError(f"TUEV train folder not found: {train_dir}")
    if not eval_dir.exists():
        raise FileNotFoundError(f"TUEV eval folder not found: {eval_dir}")

    eval_split_name = str(args.tuev_eval_split_name)
    splits = ["train", eval_split_name]

    writer = OutputWriter(
        processed=processed,
        output_format=args.output_format,
        map_size_gb=args.map_size_gb,
        commit_interval=args.commit_interval,
    )
    writer.init_splits(splits)

    file_summaries = []
    file_summaries += process_tuev_edf_folder(
        base_dir=train_dir,
        split="train",
        writer=writer,
        target_fs=args.target_fs,
        event_window_seconds=args.tuev_event_window_seconds,
        label_base=args.tuev_label_base,
    )
    file_summaries += process_tuev_edf_folder(
        base_dir=eval_dir,
        split=eval_split_name,
        writer=writer,
        target_fs=args.target_fs,
        event_window_seconds=args.tuev_event_window_seconds,
        label_base=args.tuev_label_base,
    )

    label_counts = {s: {i: 0 for i in range(6)} for s in splits}
    for fsu in file_summaries:
        if "error" in fsu:
            continue
        split = fsu["split"]
        for label, count in fsu.get("label_counts", {}).items():
            label_counts[split][int(label)] += int(count)

    meta = {
        "dataset": "TUEV",
        "style": "BIOT baseline style, resampled to 100 Hz",
        "task": "event classification",
        "label_map": {
            "SPSW": 0,
            "GPED": 1,
            "PLED": 2,
            "EYEM": 3,
            "ARTF": 4,
            "BCKG": 5,
        },
        "target_fs": int(args.target_fs),
        "channel_type": "BIOT 16 TCP bipolar signals anchored to standard single-electrode names",
        "model_channel_names": MODEL_STANDARD_16_NAMES,
        "bipolar_pairs": BIPOLAR_PAIRS,
        "bipolar_names": BIPOLAR_NAMES,
        "bipolar_to_model_channel": BIPOLAR_TO_MODEL_CHANNEL,
        "event_window_seconds": float(args.tuev_event_window_seconds),
        "split_rule": {
            "train": "official TUEV train folder",
            eval_split_name: "official TUEV eval folder",
            "note": "No random val_ratio is used for TUEV in BIOT-style preprocessing.",
        },
        "keys_count": {s: len(writer.keys.get(s, [])) for s in splits},
        "label_counts": label_counts,
        "file_summaries": file_summaries,
    }

    writer.close(meta)
    return meta


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="BIOT-style TUAB/TUEV preprocessing at 100 Hz")

    parser.add_argument("--dataset", type=str, required=True, choices=["tuab", "tuev"])
    parser.add_argument("--root", type=str, required=True, help="Path to the dataset edf root.")
    parser.add_argument("--processed", type=str, required=True, help="Output directory.")
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--target_fs", type=int, default=100)
    parser.add_argument("--output_format", type=str, default="lmdb", choices=["lmdb", "pkl", "both"])
    parser.add_argument("--map_size_gb", type=float, default=128.0)
    parser.add_argument("--commit_interval", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)

    # TUAB options.
    parser.add_argument("--channel_std", type=str, default="01_tcp_ar")
    parser.add_argument("--tuab_train_ratio", type=float, default=0.8)
    parser.add_argument("--tuab_window_seconds", type=float, default=10.0)

    # TUEV options.
    parser.add_argument("--tuev_eval_split_name", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--tuev_event_window_seconds", type=float, default=5.0)
    parser.add_argument("--tuev_label_base", type=str, default="auto", choices=["auto", "zero", "one"])

    args = parser.parse_args()

    root = Path(args.root)
    processed = Path(args.processed)

    if not root.exists():
        raise FileNotFoundError(f"Root does not exist: {root}")

    if processed.exists() and args.overwrite:
        shutil.rmtree(processed)
    ensure_dir(processed)
    mapping_csv = write_model_channel_mapping_csv(processed)

    print("========== BIOT-STYLE TUH PREPROCESS ==========")
    print(f"dataset        : {args.dataset}")
    print(f"root           : {root}")
    print(f"processed      : {processed}")
    print(f"target_fs      : {args.target_fs}")
    print(f"output_format  : {args.output_format}")
    print(f"channel map    : {mapping_csv}")
    print(f"channels       : 16 TCP bipolar signals, model-side standard electrode names")
    print(f"signal names   : {BIPOLAR_NAMES}\n    model names    : {MODEL_STANDARD_16_NAMES}")
    if args.dataset == "tuab":
        print(f"TUAB split     : official train -> subject 80/20 train/val, official eval -> test")
        print(f"TUAB window    : {args.tuab_window_seconds}s")
    else:
        print(f"TUEV split     : official train -> train, official eval -> {args.tuev_eval_split_name}")
        print(f"TUEV window    : {args.tuev_event_window_seconds}s")
    print("No bandpass / no z-score is applied, matching the BIOT preprocessing style.")
    print("The actual signals remain bipolar differences; only ch_names are standard 10-20 anchors for the model.")
    print("================================================")

    if args.dataset == "tuab":
        meta = preprocess_tuab_biot_style(args)
    else:
        meta = preprocess_tuev_biot_style(args)

    print("\n========== DONE ==========")
    print(f"Saved to  : {processed}")
    print(f"meta.json : {processed / 'meta.json'}")
    print("keys_count:")
    print(json.dumps(meta.get("keys_count", {}), indent=2, ensure_ascii=False))
    print("label_counts:")
    print(json.dumps(meta.get("label_counts", {}), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
