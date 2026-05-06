# preprocess_chbmit_biot_exact_100hz.py
# -*- coding: utf-8 -*-
"""
CHB-MIT preprocessing that follows the provided BIOT code as closely as possible,
with ONLY ONE intended change:

    final saved segments are downsampled from 256 Hz to 100 Hz.

What is kept the same as BIOT:
  1. Use pyedflib.highlevel.read_edf(..., digital=False), not MNE.
     This keeps the same physical-value scale as the BIOT pipeline.
  2. First build clean_signals/*.pkl for each recording.
  3. Channel selection follows each subject's summary-file reference channel list.
  4. Missing channels in the subject-specific valid channel set are filled with zeros.
  5. Segment stage uses the same 16 CHB-MIT bipolar channels.
  6. Segment stage uses 10-second windows at original 256 Hz.
  7. Normal non-overlap windows are labeled positive only if seizure start or end
     falls inside the window, matching the BIOT code.
  8. Extra positive seizure windows are generated from seizure_start - 1 s to
     seizure_end + 1 s with 5-second stride, matching the BIOT code.
  9. Split follows the uploaded BIOT process2.py:
       val  = chb21, chb22
       test = chb23, chb24
       train = all other processed subjects

Output:
  - BIOT-style pkl files:
      out_dir/train/*.pkl
      out_dir/val/*.pkl
      out_dir/test/*.pkl
    Each pkl contains:
      {"X": np.float32 [16, 1000], "y": 0/1}

  - Optional LMDB with exactly the same segment data:
      use --write_lmdb
    This is convenient for your current model training code.

Example:
  python preprocess_chbmit_biot_exact_100hz.py \
    --root /workspace/CHBMIT \
    --clean_dir /workspace/CHBMIT/clean_signals_biot \
    --out_dir /workspace/CHBMIT/clean_segments_biot_100hz \
    --overwrite \
    --n_jobs 8

With LMDB:
  python preprocess_chbmit_biot_exact_100hz.py \
    --root /workspace/CHBMIT \
    --clean_dir /workspace/CHBMIT/clean_signals_biot \
    --out_dir /workspace/CHBMIT/clean_segments_biot_100hz \
    --write_lmdb \
    --lmdb_path /workspace/CHBMIT/clean_segments_biot_100hz_lmdb \
    --overwrite \
    --n_jobs 8
"""

import argparse
import csv
import json
import os
import pickle
import random
import re
import shutil
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pyedflib.highlevel as hl

try:
    from scipy.signal import resample_poly
except Exception:
    resample_poly = None

try:
    import lmdb
except Exception:
    lmdb = None

from tqdm import tqdm


# ============================================================
# Same 16 channels as BIOT process2.py
# ============================================================

BIOT_16_CHANNELS = [
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FP2-F8",
    "F8-T8",
    "T8-P8",
    "P8-O2",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
]

# Optional model-side anchors, only used in LMDB metadata.
# The actual signals are still BIOT bipolar channels.
MODEL_STANDARD_16_NAMES = [
    "Fp1", "F7", "T3", "T5",
    "Fp2", "F8", "T4", "T6",
    "F3", "C3", "P3", "O1",
    "F4", "C4", "P4", "O2",
]

BIPOLAR_TO_MODEL_CHANNEL = {
    b: m for b, m in zip(BIOT_16_CHANNELS, MODEL_STANDARD_16_NAMES)
}

ORIGINAL_FS = 256
TARGET_FS = 100
WINDOW_SECONDS = 10
ORIGINAL_WINDOW_LEN = ORIGINAL_FS * WINDOW_SECONDS   # 2560
TARGET_WINDOW_LEN = TARGET_FS * WINDOW_SECONDS       # 1000


# ============================================================
# Same full-data parameters as BIOT process1.py
# ============================================================

BIOT_PARAMETERS = [
    ("01", "01", 2, 46, 0),
    ("02", "01", 2, 35, 0),
    ("03", "01", 2, 38, 0),
    ("05", "01", 2, 39, 0),
    ("06", "01", 2, 24, 0),
    ("07", "01", 2, 19, 0),
    ("08", "02", 3, 29, 0),
    ("10", "01", 2, 89, 0),
    ("11", "01", 2, 99, 0),
    ("14", "01", 2, 42, 0),
    ("20", "01", 2, 68, 0),
    ("21", "01", 2, 33, 0),
    ("22", "01", 2, 77, 0),
    ("23", "06", 7, 20, 0),
    ("24", "01", 3, 21, 0),
    ("04", "07", 1, 43, 1),
    ("09", "02", 1, 19, 1),
    ("15", "02", 1, 63, 1),
    ("16", "01", 2, 19, 0),
    ("18", "02", 1, 36, 1),
    ("19", "02", 1, 30, 1),
]

# Same split as uploaded process2.py.
BIOT_VAL_PATS = ["chb21", "chb22"]
BIOT_TEST_PATS = ["chb23", "chb24"]


# ============================================================
# Generic utilities
# ============================================================

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def compressed_pickle(path_no_ext: str, data):
    # BIOT name kept for compatibility; here we save ordinary pickle.
    pickle.dump(data, open(path_no_ext, "wb"), protocol=pickle.HIGHEST_PROTOCOL)


def fixed_length_2560(segment: np.ndarray) -> np.ndarray:
    """
    BIOT's extra seizure segments may be shorter near the file boundary.
    For 100 Hz output, we make the segment exactly 10 s before downsampling.
    This keeps training samples fixed-length.
    """
    segment = np.asarray(segment, dtype=np.float32)
    if segment.shape[1] == ORIGINAL_WINDOW_LEN:
        return segment
    if segment.shape[1] > ORIGINAL_WINDOW_LEN:
        return segment[:, :ORIGINAL_WINDOW_LEN].astype(np.float32)

    pad = ORIGINAL_WINDOW_LEN - segment.shape[1]
    if segment.shape[1] == 0:
        return np.zeros((segment.shape[0], ORIGINAL_WINDOW_LEN), dtype=np.float32)

    # Edge padding is less destructive than zero padding for boundary cases.
    return np.pad(segment, ((0, 0), (0, pad)), mode="edge").astype(np.float32)


def downsample_256_to_100(segment_256: np.ndarray) -> np.ndarray:
    """
    Downsample [16, 2560] to [16, 1000].

    Uses polyphase filtering when scipy is available:
      256 -> 100 = up 25 / down 64
    """
    x = fixed_length_2560(segment_256)

    if resample_poly is not None:
        y = resample_poly(x, up=25, down=64, axis=1)
    else:
        # Fallback: linear interpolation.
        old_t = np.linspace(0.0, 1.0, x.shape[1], endpoint=False)
        new_t = np.linspace(0.0, 1.0, TARGET_WINDOW_LEN, endpoint=False)
        y = np.stack([np.interp(new_t, old_t, ch) for ch in x], axis=0)

    y = np.asarray(y, dtype=np.float32)

    # Force exact length.
    if y.shape[1] > TARGET_WINDOW_LEN:
        y = y[:, :TARGET_WINDOW_LEN]
    elif y.shape[1] < TARGET_WINDOW_LEN:
        pad = TARGET_WINDOW_LEN - y.shape[1]
        y = np.pad(y, ((0, 0), (0, pad)), mode="edge")

    return y.astype(np.float32)


def write_channel_mapping_csv(out_dir: Path):
    ensure_dir(out_dir)
    path = out_dir / "biot_16_bipolar_to_model_channels.csv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "bipolar_channel", "model_channel_name"])
        for i, (b, m) in enumerate(zip(BIOT_16_CHANNELS, MODEL_STANDARD_16_NAMES)):
            w.writerow([i, b, m])
    return path


# ============================================================
# BIOT process1.py equivalent
# ============================================================

def process_metadata(summary: str, filename: str) -> dict:
    """
    Same logic as BIOT process1.py:
      - Parse summary file.
      - Store seizure times as sample indices at 256 Hz:
          start = second * 256 - 1
          end   = second * 256 - 1
    """
    with open(summary, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    metadata = {}
    times = []

    for i in range(len(lines)):
        line = lines[i].split()
        if len(line) == 3 and line[2] == filename:
            j = i + 1
            processed = False

            while not processed and j < len(lines):
                parts = lines[j].split()
                if len(parts) > 0 and parts[0] == "Number":
                    seizures = int(parts[-1])
                    processed = True
                j += 1

            if not processed:
                seizures = 0

            if seizures > 0:
                j = i + 1
                for _ in range(seizures):
                    processed = False
                    while not processed and j < len(lines) - 1:
                        l = lines[j].split()
                        if len(l) > 0 and l[0] == "Seizure" and "Start" in l:
                            start = int(l[-2]) * ORIGINAL_FS - 1
                            end = int(lines[j + 1].split()[-2]) * ORIGINAL_FS - 1
                            processed = True
                        j += 1
                    if processed:
                        times.append((start, end))

            metadata["seizures"] = seizures
            metadata["times"] = times

    if "seizures" not in metadata:
        metadata["seizures"] = 0
        metadata["times"] = []

    return metadata


def drop_channels(edf_source: str, to_keep: List[int]) -> dict:
    """
    Same as BIOT: hl.read_edf(..., digital=False), then keep selected channels.
    """
    signals, signal_headers, header = hl.read_edf(
        edf_source,
        ch_nrs=to_keep,
        digital=False,
    )
    clean_file = {}

    for signal, header_i in zip(signals, signal_headers):
        channel = header_i.get("label")
        if channel in clean_file.keys():
            channel = channel + "-2"
        clean_file[channel] = np.asarray(signal, dtype=np.float32)

    return clean_file


def move_channels(clean_dict: dict, channels: dict, target_no_ext: str):
    """
    Same as BIOT:
      - delete channels not in the subject-specific valid channel dict
      - missing expected channels are zero-filled
      - save .pkl
    """
    keys_to_delete = []
    for key in clean_dict:
        if key != "metadata" and key not in channels.keys():
            keys_to_delete.append(key)

    for key in keys_to_delete:
        del clean_dict[key]

    size = 0
    for item in clean_dict.keys():
        if item != "metadata":
            size = len(clean_dict.get(item))
            break

    for k in channels.keys():
        if k not in clean_dict.keys():
            clean_dict[k] = np.zeros(size, dtype=np.float32)

    compressed_pickle(target_no_ext + ".pkl", clean_dict)


def process_files(pacient: str, valid_channels: list, channels: dict, start: int, end: int, signals_path: Path, clean_path: Path):
    """
    Same role as BIOT process_files().
    """
    for num_int in range(start, end + 1):
        to_keep = []

        num = ("0" + str(num_int))[-2:]
        filename = str(signals_path / f"chb{pacient}" / f"chb{pacient}_{num}.edf")

        try:
            signals, signal_headers, header = hl.read_edf(filename, digital=False)
            n = 0
            for h in signal_headers:
                if h.get("label") in valid_channels:
                    if n not in to_keep:
                        to_keep.append(n)
                n += 1
        except (OSError, FileNotFoundError):
            print("****************************************")
            print("WARNING - File does not exist. Processing next file.")
            print(filename)
            print("****************************************")
            continue

        if len(to_keep) <= 0:
            print(f"[skip] no valid channels found in {filename}")
            continue

        try:
            print(
                "Removing",
                len(signal_headers) - len(to_keep),
                "channels from file",
                f"chb{pacient}_{num}.edf",
            )
            clean_dict = drop_channels(filename, to_keep=to_keep)
            print("Processing file", filename)
        except AssertionError:
            print("****************************************")
            print("WARNING - File could not be read. Processing next file.")
            print(filename)
            print("****************************************")
            continue

        metadata = process_metadata(
            str(signals_path / f"chb{pacient}" / f"chb{pacient}-summary.txt"),
            f"chb{pacient}_{num}.edf",
        )
        metadata["channels"] = valid_channels
        clean_dict["metadata"] = metadata

        target = str(clean_path / f"chb{pacient}" / f"chb{pacient}_{num}.edf")
        move_channels(clean_dict, channels, target)


def start_process(param_tuple):
    """
    Wrapper for multiprocessing.

    param_tuple:
      (pacient, num, start, end, sum_ind, signals_path, clean_path)
    """
    pacient, num, start, end, sum_ind, signals_path_str, clean_path_str = param_tuple
    signals_path = Path(signals_path_str)
    clean_path = Path(clean_path_str)

    subject_clean_dir = clean_path / f"chb{pacient}"
    ensure_dir(subject_clean_dir)

    summary_path = signals_path / f"chb{pacient}" / f"chb{pacient}-summary.txt"

    with open(summary_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    channels = defaultdict(list)
    valid_channels = []
    to_keep = []

    channel_index = 1
    summary_index = 0

    # Same summary parsing as BIOT.
    for line in lines:
        line = line.split()
        if len(line) == 0:
            continue

        if line[0] == "Channels" and line[1] == "changed:":
            summary_index += 1

        if (
            line[0] == "Channel"
            and summary_index == sum_ind
            and (line[2] != "-" and line[2] != ".")
        ):
            if line[2] in channels.keys():
                name = line[2] + "-2"
            else:
                name = line[2]

            channels[name].append(str(channel_index))
            channel_index += 1
            valid_channels.append(name)
            to_keep.append(int(line[1][:-1]) - 1)

    filename = str(signals_path / f"chb{pacient}" / f"chb{pacient}_{num}.edf")
    target = str(clean_path / f"chb{pacient}" / f"chb{pacient}_{num}.edf")

    print(f"[clean] subject=chb{pacient}, ref_file=chb{pacient}_{num}.edf, summary_index={sum_ind}")

    clean_dict = drop_channels(filename, to_keep=to_keep)
    metadata = process_metadata(
        str(summary_path),
        f"chb{pacient}_{num}.edf",
    )
    metadata["channels"] = valid_channels
    clean_dict["metadata"] = metadata

    compressed_pickle(target + ".pkl", clean_dict)

    # Process the rest of files using the same channel reference.
    process_files(pacient, valid_channels, channels, start, end, signals_path, clean_path)

    return f"chb{pacient}"


# ============================================================
# Segment writer
# ============================================================

class OptionalLMDBWriter:
    def __init__(self, lmdb_path: Optional[Path], map_size_gb: float = 128.0, commit_interval: int = 1000):
        self.enabled = lmdb_path is not None
        self.keys = {"train": [], "val": [], "test": []}
        self.count = 0
        self.commit_interval = int(commit_interval)

        if not self.enabled:
            self.env = None
            self.txn = None
            return

        if lmdb is None:
            raise RuntimeError("lmdb is not installed. Install with: pip install lmdb")

        ensure_dir(lmdb_path)
        self.env = lmdb.open(
            str(lmdb_path),
            map_size=int(float(map_size_gb) * (1024 ** 3)),
            subdir=True,
            lock=True,
            map_async=True,
            meminit=False,
            max_readers=256,
        )
        self.txn = self.env.begin(write=True)

    def put(self, split: str, key: str, sample: dict):
        if split not in self.keys:
            self.keys[split] = []
        self.keys[split].append(key)

        if not self.enabled:
            return

        self.txn.put(key.encode("utf-8"), pickle.dumps(sample, protocol=pickle.HIGHEST_PROTOCOL))
        self.count += 1
        if self.count % self.commit_interval == 0:
            self.txn.commit()
            self.txn = self.env.begin(write=True)

    def close(self, meta: dict):
        if not self.enabled:
            return
        self.txn.put(b"__keys__", pickle.dumps(self.keys, protocol=pickle.HIGHEST_PROTOCOL))
        self.txn.put(b"__meta__", pickle.dumps(meta, protocol=pickle.HIGHEST_PROTOCOL))
        self.txn.commit()
        self.env.sync()
        self.env.close()


def infer_split(folder_name: str) -> str:
    if folder_name in BIOT_TEST_PATS:
        return "test"
    if folder_name in BIOT_VAL_PATS:
        return "val"
    return "train"


def save_segment(
    segment_256: np.ndarray,
    label: int,
    out_file: Path,
    split: str,
    lmdb_writer: OptionalLMDBWriter,
    lmdb_key: str,
    source_file: str,
    source_start_index_256: int,
    is_augmented_positive: bool,
):
    """
    Save BIOT-style pkl with final 100 Hz data.
    """
    segment_100 = downsample_256_to_100(segment_256)

    # BIOT-style pkl content.
    pkl_obj = {
        "X": segment_100,
        "y": int(label),
    }
    with open(out_file, "wb") as f:
        pickle.dump(pkl_obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Optional LMDB content.
    lmdb_sample = {
        "sample": segment_100,
        "X": segment_100,
        "label": int(label),
        "y": int(label),
        "fs": TARGET_FS,
        "original_fs": ORIGINAL_FS,
        "ch_names": list(MODEL_STANDARD_16_NAMES),
        "bipolar_ch_names": list(BIOT_16_CHANNELS),
        "bipolar_to_model_channel": dict(BIPOLAR_TO_MODEL_CHANNEL),
        "source_file": source_file,
        "source_start_index_256": int(source_start_index_256),
        "source_start_sec": float(source_start_index_256 / ORIGINAL_FS),
        "window_seconds": WINDOW_SECONDS,
        "is_augmented_positive": bool(is_augmented_positive),
        "task": "chbmit_seizure_detection_biot_exact_100hz",
    }
    lmdb_writer.put(split, lmdb_key, lmdb_sample)


def sub_to_segments(args_tuple):
    """
    Same role as BIOT process2.py sub_to_segments(), but final segment X is 100 Hz.
    """
    folder, root_str, out_str, lmdb_path_str, map_size_gb, commit_interval = args_tuple

    root = Path(root_str)
    out = Path(out_str)
    split = infer_split(folder)
    out_folder = out / split
    ensure_dir(out_folder)

    lmdb_writer = OptionalLMDBWriter(None)  # multiprocessing-safe pkl-only in worker
    # LMDB is written in a single-process path below; worker mode uses pkl only.

    stats = {
        "folder": folder,
        "split": split,
        "n_files": 0,
        "n_segments": 0,
        "n_non_seizure": 0,
        "n_seizure": 0,
        "n_augmented_positive": 0,
        "errors": [],
    }

    print(f"Processing {folder} -> {split}...")

    folder_path = root / folder
    if not folder_path.exists():
        stats["errors"].append(f"folder not found: {folder_path}")
        return stats

    for f_name in tqdm(sorted(os.listdir(folder_path)), desc=f"{folder}->{split}", leave=False):
        if not f_name.endswith(".pkl"):
            continue

        stats["n_files"] += 1
        f_path = folder_path / f_name

        try:
            record = pickle.load(open(f_path, "rb"))
            signal = []
            for channel in BIOT_16_CHANNELS:
                if channel in record:
                    signal.append(record[channel])
                else:
                    raise ValueError(f"Channel {channel} not found in record {f_path}")
            signal = np.asarray(signal, dtype=np.float32)

            if "times" in record["metadata"]:
                seizure_times = record["metadata"]["times"]
            else:
                seizure_times = []

            stem = f_name.split(".")[0]

            # Same normal 10-s non-overlap segmentation and label rule as BIOT.
            for i in range(0, signal.shape[1], ORIGINAL_FS * WINDOW_SECONDS):
                segment = signal[:, i : i + ORIGINAL_FS * WINDOW_SECONDS]

                if segment.shape[1] == ORIGINAL_FS * WINDOW_SECONDS:
                    label = 0
                    for seizure_time in seizure_times:
                        if (
                            i < seizure_time[0] < i + ORIGINAL_FS * WINDOW_SECONDS
                            or i < seizure_time[1] < i + ORIGINAL_FS * WINDOW_SECONDS
                        ):
                            label = 1
                            break

                    out_file = out_folder / f"{stem}-{i}.pkl"
                    lmdb_key = f"{split}/{stem}-{i}"
                    save_segment(
                        segment,
                        label,
                        out_file,
                        split,
                        lmdb_writer,
                        lmdb_key,
                        source_file=str(f_path),
                        source_start_index_256=i,
                        is_augmented_positive=False,
                    )

                    stats["n_segments"] += 1
                    if label == 1:
                        stats["n_seizure"] += 1
                    else:
                        stats["n_non_seizure"] += 1

            # Same positive augmentation as BIOT.
            for idx, seizure_time in enumerate(seizure_times):
                for i in range(
                    max(0, seizure_time[0] - ORIGINAL_FS),
                    min(seizure_time[1] + ORIGINAL_FS, signal.shape[1]),
                    5 * ORIGINAL_FS,
                ):
                    segment = signal[:, i : i + ORIGINAL_FS * WINDOW_SECONDS]
                    label = 1

                    out_file = out_folder / f"{stem}-s-{idx}-add-{i}.pkl"
                    lmdb_key = f"{split}/{stem}-s-{idx}-add-{i}"

                    save_segment(
                        segment,
                        label,
                        out_file,
                        split,
                        lmdb_writer,
                        lmdb_key,
                        source_file=str(f_path),
                        source_start_index_256=i,
                        is_augmented_positive=True,
                    )

                    stats["n_segments"] += 1
                    stats["n_seizure"] += 1
                    stats["n_augmented_positive"] += 1

        except Exception as e:
            msg = f"{f_path}: {repr(e)}"
            print("[segment skip]", msg)
            stats["errors"].append(msg)

    return stats


def build_lmdb_from_pkl_segments(out_dir: Path, lmdb_path: Path, map_size_gb: float, commit_interval: int):
    """
    Build LMDB from already saved BIOT-style pkl segments.
    This keeps multiprocessing simple and avoids multiple writers to one LMDB.
    """
    if lmdb_path.exists():
        shutil.rmtree(lmdb_path)
    writer = OptionalLMDBWriter(lmdb_path, map_size_gb=map_size_gb, commit_interval=commit_interval)

    label_counts = {s: {0: 0, 1: 0} for s in ["train", "val", "test"]}

    for split in ["train", "val", "test"]:
        split_dir = out_dir / split
        if not split_dir.exists():
            continue

        for pkl_path in tqdm(sorted(split_dir.glob("*.pkl")), desc=f"lmdb:{split}"):
            obj = pickle.load(open(pkl_path, "rb"))
            x = np.asarray(obj["X"], dtype=np.float32)
            y = int(obj["y"])

            key = f"{split}/{pkl_path.stem}"
            sample = {
                "sample": x,
                "X": x,
                "label": y,
                "y": y,
                "fs": TARGET_FS,
                "original_fs": ORIGINAL_FS,
                "ch_names": list(MODEL_STANDARD_16_NAMES),
                "bipolar_ch_names": list(BIOT_16_CHANNELS),
                "bipolar_to_model_channel": dict(BIPOLAR_TO_MODEL_CHANNEL),
                "source_file": str(pkl_path),
                "task": "chbmit_seizure_detection_biot_exact_100hz",
            }
            writer.put(split, key, sample)
            label_counts[split][y] += 1

    meta = {
        "dataset": "CHB-MIT",
        "style": "BIOT exact process1/process2, final saved segment downsampled to 100 Hz",
        "target_fs": TARGET_FS,
        "original_fs": ORIGINAL_FS,
        "window_seconds": WINDOW_SECONDS,
        "segment_shape": [16, TARGET_WINDOW_LEN],
        "label_map": {"non_seizure": 0, "seizure": 1},
        "channels": BIOT_16_CHANNELS,
        "model_channel_anchors": MODEL_STANDARD_16_NAMES,
        "split": {
            "train": "all processed folders except chb21/chb22/chb23/chb24",
            "val": BIOT_VAL_PATS,
            "test": BIOT_TEST_PATS,
        },
        "label_counts": label_counts,
    }
    writer.close(meta)
    return meta, writer.keys


def write_summary(out_dir: Path, stats_list: list, lmdb_meta: Optional[dict] = None, lmdb_keys: Optional[dict] = None):
    rows = []
    total = {
        "split": "TOTAL",
        "n_files": 0,
        "n_segments": 0,
        "n_non_seizure": 0,
        "n_seizure": 0,
        "n_augmented_positive": 0,
    }

    for s in stats_list:
        rows.append({
            "folder": s["folder"],
            "split": s["split"],
            "n_files": s["n_files"],
            "n_segments": s["n_segments"],
            "n_non_seizure": s["n_non_seizure"],
            "n_seizure": s["n_seizure"],
            "n_augmented_positive": s["n_augmented_positive"],
            "n_errors": len(s["errors"]),
        })
        for k in ["n_files", "n_segments", "n_non_seizure", "n_seizure", "n_augmented_positive"]:
            total[k] += int(s[k])

    with open(out_dir / "segment_summary_by_subject.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["folder", "split", "n_files", "n_segments", "n_non_seizure", "n_seizure", "n_augmented_positive", "n_errors"])
        w.writeheader()
        w.writerows(rows)

    split_counts = {}
    for split in ["train", "val", "test"]:
        split_rows = [r for r in rows if r["split"] == split]
        split_counts[split] = {
            "n_segments": int(sum(r["n_segments"] for r in split_rows)),
            "n_non_seizure": int(sum(r["n_non_seizure"] for r in split_rows)),
            "n_seizure": int(sum(r["n_seizure"] for r in split_rows)),
            "n_augmented_positive": int(sum(r["n_augmented_positive"] for r in split_rows)),
        }

    meta = {
        "dataset": "CHB-MIT",
        "style": "BIOT exact process1/process2 with final segment downsampled to 100 Hz",
        "target_fs": TARGET_FS,
        "original_fs": ORIGINAL_FS,
        "window_seconds": WINDOW_SECONDS,
        "segment_shape": [16, TARGET_WINDOW_LEN],
        "pkl_format": {"X": "[16,1000] float32", "y": "0/1"},
        "channels": BIOT_16_CHANNELS,
        "model_channel_anchors_for_optional_lmdb": MODEL_STANDARD_16_NAMES,
        "split": {
            "train": "all processed folders except chb21/chb22/chb23/chb24",
            "val": BIOT_VAL_PATS,
            "test": BIOT_TEST_PATS,
        },
        "biot_parameters": BIOT_PARAMETERS,
        "split_counts": split_counts,
        "lmdb_meta": lmdb_meta,
        "lmdb_keys_count": {k: len(v) for k, v in lmdb_keys.items()} if lmdb_keys is not None else None,
    }

    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    with open(out_dir / "keys_count.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "n_segments", "n_non_seizure", "n_seizure", "n_augmented_positive"])
        for split, c in split_counts.items():
            w.writerow([split, c["n_segments"], c["n_non_seizure"], c["n_seizure"], c["n_augmented_positive"]])


def main():
    parser = argparse.ArgumentParser(description="BIOT-exact CHB-MIT preprocessing with final 100 Hz segments")
    parser.add_argument("--root", type=str, required=True, help="CHB-MIT root containing chb01/, chb02/, ...")
    parser.add_argument("--clean_dir", type=str, default=None, help="Output clean_signals directory. Default: <root>/clean_signals_biot")
    parser.add_argument("--out_dir", type=str, default=None, help="Output clean_segments directory. Default: <root>/clean_segments_biot_100hz")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip_clean", action="store_true", help="Skip EDF->clean_signals stage if clean_dir already exists.")
    parser.add_argument("--skip_segments", action="store_true", help="Skip clean_signals->segments stage.")
    parser.add_argument("--n_jobs", type=int, default=1)

    parser.add_argument("--write_lmdb", action="store_true")
    parser.add_argument("--lmdb_path", type=str, default=None)
    parser.add_argument("--lmdb_map_size_gb", type=float, default=128.0)
    parser.add_argument("--lmdb_commit_interval", type=int, default=1000)

    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"root does not exist: {root}")

    clean_dir = Path(args.clean_dir) if args.clean_dir is not None else (root / "clean_signals_biot")
    out_dir = Path(args.out_dir) if args.out_dir is not None else (root / "clean_segments_biot_100hz")

    if args.overwrite:
        if clean_dir.exists() and not args.skip_clean:
            shutil.rmtree(clean_dir)
        if out_dir.exists() and not args.skip_segments:
            shutil.rmtree(out_dir)
        if args.write_lmdb:
            lmdb_path = Path(args.lmdb_path) if args.lmdb_path is not None else (root / "clean_segments_biot_100hz_lmdb")
            if lmdb_path.exists():
                shutil.rmtree(lmdb_path)

    ensure_dir(clean_dir)
    ensure_dir(out_dir)
    for split in ["train", "val", "test"]:
        ensure_dir(out_dir / split)

    mapping_csv = write_channel_mapping_csv(out_dir)

    print("========== CHB-MIT BIOT-EXACT 100Hz PREPROCESS ==========")
    print(f"root       : {root}")
    print(f"clean_dir  : {clean_dir}")
    print(f"out_dir    : {out_dir}")
    print(f"target fs  : {TARGET_FS} Hz")
    print(f"orig fs    : {ORIGINAL_FS} Hz")
    print(f"segment    : 10 s, [16, {TARGET_WINDOW_LEN}]")
    print(f"split      : val={BIOT_VAL_PATS}, test={BIOT_TEST_PATS}, train=others")
    print(f"mapping csv: {mapping_csv}")
    print("IMPORTANT: this uses pyedflib digital=False like BIOT, so EEG scale should stay in physical units.")
    print("=========================================================")

    # ------------------------------------------------------------
    # Stage 1: EDF -> clean_signals
    # ------------------------------------------------------------
    if not args.skip_clean:
        params = []
        for p in BIOT_PARAMETERS:
            pacient, num, start, end, sum_ind = p
            params.append((pacient, num, start, end, sum_ind, str(root), str(clean_dir)))

        print(f"\n[stage 1] EDF -> clean_signals, jobs={args.n_jobs}")
        if args.n_jobs > 1:
            with mp.Pool(int(args.n_jobs)) as pool:
                list(tqdm(pool.imap_unordered(start_process, params), total=len(params), desc="clean_signals"))
        else:
            for param in tqdm(params, desc="clean_signals"):
                start_process(param)
    else:
        print("\n[stage 1] skipped clean_signals stage")

    # ------------------------------------------------------------
    # Stage 2: clean_signals -> 100Hz pkl segments
    # ------------------------------------------------------------
    stats_list = []
    if not args.skip_segments:
        folders = sorted([p.name for p in clean_dir.iterdir() if p.is_dir()])
        print(f"\n[stage 2] clean_signals -> 100Hz segments, folders={folders}, jobs={args.n_jobs}")

        worker_args = [
            (folder, str(clean_dir), str(out_dir), None, args.lmdb_map_size_gb, args.lmdb_commit_interval)
            for folder in folders
        ]

        if args.n_jobs > 1:
            with mp.Pool(int(args.n_jobs)) as pool:
                stats_list = list(tqdm(pool.imap_unordered(sub_to_segments, worker_args), total=len(worker_args), desc="segments"))
        else:
            for wa in tqdm(worker_args, desc="segments"):
                stats_list.append(sub_to_segments(wa))
    else:
        print("\n[stage 2] skipped segment stage")

    # ------------------------------------------------------------
    # Optional Stage 3: pkl segments -> LMDB
    # ------------------------------------------------------------
    lmdb_meta = None
    lmdb_keys = None
    if args.write_lmdb:
        lmdb_path = Path(args.lmdb_path) if args.lmdb_path is not None else (root / "clean_segments_biot_100hz_lmdb")
        print(f"\n[stage 3] pkl segments -> LMDB: {lmdb_path}")
        lmdb_meta, lmdb_keys = build_lmdb_from_pkl_segments(
            out_dir,
            lmdb_path,
            map_size_gb=args.lmdb_map_size_gb,
            commit_interval=args.lmdb_commit_interval,
        )
        print(f"[lmdb] saved to {lmdb_path}")

    # If skip_segments was used, collect rough split counts from existing pkl.
    if args.skip_segments:
        stats_list = []
        for split in ["train", "val", "test"]:
            files = list((out_dir / split).glob("*.pkl"))
            n0, n1 = 0, 0
            for p in files:
                try:
                    y = int(pickle.load(open(p, "rb"))["y"])
                    if y == 1:
                        n1 += 1
                    else:
                        n0 += 1
                except Exception:
                    pass
            stats_list.append({
                "folder": f"existing_{split}",
                "split": split,
                "n_files": 0,
                "n_segments": len(files),
                "n_non_seizure": n0,
                "n_seizure": n1,
                "n_augmented_positive": 0,
                "errors": [],
            })

    write_summary(out_dir, stats_list, lmdb_meta=lmdb_meta, lmdb_keys=lmdb_keys)

    print("\n========== DONE ==========")
    print(f"BIOT-style 100Hz pkl segments: {out_dir}")
    print(f"meta.json                   : {out_dir / 'meta.json'}")
    print(f"keys_count.csv              : {out_dir / 'keys_count.csv'}")
    print(f"summary csv                 : {out_dir / 'segment_summary_by_subject.csv'}")
    if args.write_lmdb:
        print(f"LMDB                         : {Path(args.lmdb_path) if args.lmdb_path else (root / 'clean_segments_biot_100hz_lmdb')}")


if __name__ == "__main__":
    main()
