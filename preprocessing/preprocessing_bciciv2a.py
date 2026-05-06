"""
BCI Competition IV 2a preprocessing for S2/SADT-style downstream training.

Key behavior
------------
1. Split samples by motor imagery cue onset.
   - Training session (*.T.gdf): cue codes 769/770/771/772 -> labels 0/1/2/3.
   - Evaluation session (*.E.gdf): unknown cue code 783; labels are read from AxxE.mat
     when --eval_label_dir is provided.
2. Cut cue-onset 0--4 s, i.e. the motor-imagery period.
3. Resample each trial to target_fs, default 100 Hz.
   - default output shape for --type reve: [22, 400]
   - optional --type cbramod: [22, 4, 100] when window_sec=4 and target_fs=100
4. Save to LMDB with __keys__ and __meta__.

Example
-------
python preprocessing_bciciv2a.py \
  --raw data/BCICIV_2a_gdf \
  --processed ./bciciv2a_lmdb_100hz \
  --eval_label_dir data/BCICIV_2a_gdf/true_labels \
  --type reve \
  --target_fs 100 \
  --overwrite
"""

import argparse
import os
import pickle
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lmdb
import mne
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, lfilter, resample_poly


ELECTRODES_22: Tuple[str, ...] = (
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz",
)

# BCI Competition IV 2a motor-imagery cue codes.
CUE_CODE_TO_LABEL: Dict[int, int] = {
    769: 0,  # left hand
    770: 1,  # right hand
    771: 2,  # feet
    772: 3,  # tongue
}
UNKNOWN_EVAL_CUE_CODE = 783

DEFAULT_FILES_DICT = {
    "train": [
        "A01E.gdf", "A01T.gdf",
        "A02E.gdf", "A02T.gdf",
        "A03E.gdf", "A03T.gdf",
        "A04E.gdf", "A04T.gdf",
        "A05E.gdf", "A05T.gdf",
    ],
    "val": [
        "A06E.gdf", "A06T.gdf",
        "A07E.gdf", "A07T.gdf",
    ],
    "test": [
        "A08E.gdf", "A08T.gdf",
        "A09E.gdf", "A09T.gdf",
    ],
}


def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 5):
    nyq = 0.5 * float(fs)
    low = float(lowcut) / nyq
    high = float(highcut) / nyq
    if not (0.0 < low < high < 1.0):
        raise ValueError(
            f"Invalid bandpass range: lowcut={lowcut}, highcut={highcut}, fs={fs}. "
            f"Need 0 < lowcut < highcut < fs/2."
        )
    return butter(order, [low, high], btype="band")


def apply_bandpass(
    x: np.ndarray,
    fs: float,
    lowcut: float,
    highcut: float,
    order: int,
    method: str,
) -> np.ndarray:
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    if method == "lfilter":
        return lfilter(b, a, x, axis=-1)
    if method == "filtfilt":
        return filtfilt(b, a, x, axis=-1)
    raise ValueError(f"Unsupported filter method: {method}")


def resample_to_target_fs(x: np.ndarray, original_fs: int, target_fs: int) -> np.ndarray:
    """Resample [C, T] from original_fs to target_fs using rational polyphase filtering."""
    from math import gcd

    original_fs = int(round(original_fs))
    target_fs = int(round(target_fs))
    g = gcd(original_fs, target_fs)
    up = target_fs // g
    down = original_fs // g
    return resample_poly(x, up=up, down=down, axis=-1)


def decode_event_descriptions(events: np.ndarray, event_id: Dict[str, int]) -> List[Tuple[int, int]]:
    """
    Convert MNE events into [(sample_index, original_annotation_code), ...].
    MNE maps annotation descriptions to internal integer IDs; this reverses that map.
    """
    inv_event_id = {int(v): str(k) for k, v in event_id.items()}
    decoded = []
    for e in events:
        sample_idx = int(e[0])
        internal_id = int(e[2])
        desc = inv_event_id.get(internal_id, None)
        if desc is None:
            continue
        try:
            original_code = int(desc)
        except ValueError:
            continue
        decoded.append((sample_idx, original_code))
    return decoded


def find_eval_label_file(eval_label_dir: str, gdf_filename: str) -> Optional[Path]:
    """Find AxxE.mat for AxxE.gdf, with a few common filename variants."""
    stem = Path(gdf_filename).stem  # e.g., A01E
    root = Path(eval_label_dir)
    candidates = [
        root / f"{stem}.mat",
        root / f"{stem.lower()}.mat",
        root / f"{stem}_labels.mat",
        root / f"{stem.lower()}_labels.mat",
    ]
    for p in candidates:
        if p.exists():
            return p
    # Fallback: recursive search, useful when true_labels.zip creates subfolders.
    matches = list(root.rglob(f"{stem}.mat")) + list(root.rglob(f"{stem.lower()}.mat"))
    if len(matches) > 0:
        return matches[0]
    return None


def load_eval_labels(eval_label_dir: Optional[str], gdf_filename: str, expected_n: int) -> Optional[np.ndarray]:
    """
    Load true labels for AxxE.gdf from AxxE.mat.

    Common official true-label files store labels as 1..4 under key 'classlabel'.
    This function maps them to 0..3. It also supports already-zero-based 0..3 labels.
    """
    if eval_label_dir is None:
        return None

    label_file = find_eval_label_file(eval_label_dir, gdf_filename)
    if label_file is None:
        print(f"[warn] {gdf_filename}: no matching .mat label file found in {eval_label_dir}")
        return None

    mat = loadmat(label_file)

    preferred_keys = ["classlabel", "labels", "label", "y", "Y"]
    arrays = []
    for key in preferred_keys:
        if key in mat:
            arrays.append((key, mat[key]))
    for key, value in mat.items():
        if key.startswith("__"):
            continue
        if key not in preferred_keys:
            arrays.append((key, value))

    labels = None
    used_key = None
    for key, value in arrays:
        arr = np.asarray(value).reshape(-1)
        if arr.size < expected_n:
            continue
        if np.issubdtype(arr.dtype, np.number):
            cand = arr[:expected_n].astype(np.int64)
            unique = set(np.unique(cand).tolist())
            if unique.issubset({1, 2, 3, 4}):
                labels = cand - 1
                used_key = key
                break
            if unique.issubset({0, 1, 2, 3}):
                labels = cand
                used_key = key
                break

    if labels is None:
        raise ValueError(
            f"Could not find a valid label vector in {label_file}. "
            f"Expected at least {expected_n} labels with values 1..4 or 0..3. "
            f"Available keys: {[k for k in mat.keys() if not k.startswith('__')]}"
        )

    print(f"  loaded eval labels from {label_file} | key={used_key} | n={len(labels)}")
    return labels.astype(np.int64)


def extract_cue_onsets_and_labels(
    decoded_events: List[Tuple[int, int]],
    gdf_filename: str,
    eval_label_dir: Optional[str],
) -> Tuple[List[int], List[int], Dict[str, int]]:
    """
    Return cue onsets and zero-based labels.

    For T files, labels come from event codes 769..772.
    For E files, cue onsets come from code 783 and labels come from AxxE.mat.
    """
    motor_onsets = []
    motor_labels = []
    unknown_eval_onsets = []

    for sample_idx, code in decoded_events:
        if code in CUE_CODE_TO_LABEL:
            motor_onsets.append(sample_idx)
            motor_labels.append(CUE_CODE_TO_LABEL[code])
        elif code == UNKNOWN_EVAL_CUE_CODE:
            unknown_eval_onsets.append(sample_idx)

    info = {
        "motor_cues": len(motor_onsets),
        "unknown_eval_cues": len(unknown_eval_onsets),
    }

    if len(motor_onsets) > 0:
        return motor_onsets, motor_labels, info

    if len(unknown_eval_onsets) > 0:
        labels = load_eval_labels(eval_label_dir, gdf_filename, expected_n=len(unknown_eval_onsets))
        if labels is None:
            print(
                f"[warn] {gdf_filename}: found {len(unknown_eval_onsets)} unknown evaluation cues "
                f"(code {UNKNOWN_EVAL_CUE_CODE}) but no true-label .mat was provided. This file will add 0 samples."
            )
            return [], [], info
        return unknown_eval_onsets, labels.tolist(), info

    return [], [], info


def preprocess_one_trial(
    raw_data_uV: np.ndarray,
    cue_sample: int,
    original_fs: int,
    target_fs: int,
    window_sec: float,
    lowcut: float,
    highcut: float,
    filter_order: int,
    filter_method: str,
    do_car: bool,
    expected_channels: int,
) -> Optional[np.ndarray]:
    """Cut cue-onset 0--window_sec and return [22, window_sec * target_fs]."""
    src_len = int(round(float(window_sec) * float(original_fs)))
    target_len = int(round(float(window_sec) * float(target_fs)))

    start = int(cue_sample)
    end = start + src_len
    if start < 0 or end > raw_data_uV.shape[-1]:
        return None

    sample = raw_data_uV[:expected_channels, start:end].astype(np.float64, copy=True)
    if sample.shape[0] != expected_channels or sample.shape[-1] != src_len:
        return None

    # Replace NaN/Inf before filtering. Some GDF files contain NaN blocks at run boundaries.
    sample = np.nan_to_num(sample, nan=0.0, posinf=0.0, neginf=0.0)

    # Original script used common-average removal across EEG channels at each time point.
    if do_car:
        sample = sample - np.mean(sample, axis=0, keepdims=True)

    sample = apply_bandpass(sample, fs=original_fs, lowcut=lowcut, highcut=highcut, order=filter_order, method=filter_method)
    sample = resample_to_target_fs(sample, original_fs=original_fs, target_fs=target_fs)

    # resample_poly should give exactly target_len for 250->100 and 4 s, but trim/pad defensively.
    if sample.shape[-1] > target_len:
        sample = sample[:, :target_len]
    elif sample.shape[-1] < target_len:
        pad = target_len - sample.shape[-1]
        sample = np.pad(sample, ((0, 0), (0, pad)), mode="constant")

    sample = np.nan_to_num(sample, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return sample


def label_distribution(labels: List[int], num_classes: int = 4) -> Dict[int, int]:
    if len(labels) == 0:
        return {i: 0 for i in range(num_classes)}
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes)
    return {int(i): int(counts[i]) for i in range(num_classes)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=str, required=True, help="Raw BCICIV 2a GDF directory")
    parser.add_argument("--processed", type=str, required=True, help="Output LMDB directory")
    parser.add_argument("--eval_label_dir", type=str, default=None, help="Directory containing A01E.mat ... A09E.mat true labels")
    parser.add_argument("--type", type=str, choices=["reve", "cbramod"], default="reve")
    parser.add_argument("--target_fs", type=int, default=100)
    parser.add_argument("--window_sec", type=float, default=4.0, help="Seconds to cut after cue onset")
    parser.add_argument("--lowcut", type=float, default=0.3)
    parser.add_argument("--highcut", type=float, default=45.0)
    parser.add_argument("--filter_order", type=int, default=5)
    parser.add_argument("--filter_method", type=str, choices=["lfilter", "filtfilt"], default="lfilter")
    parser.add_argument("--no_car", action="store_true", help="Disable common-average removal across EEG channels")
    parser.add_argument("--map_size_gb", type=float, default=4.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    raw_dir = Path(args.raw)
    processed_dir = Path(args.processed)

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    if processed_dir.exists() and args.overwrite:
        shutil.rmtree(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    expected_channels = len(ELECTRODES_22)
    target_len = int(round(float(args.window_sec) * float(args.target_fs)))

    if args.type == "cbramod":
        if abs(args.window_sec - round(args.window_sec)) > 1e-6:
            raise ValueError("--type cbramod requires integer window_sec so [C, seconds, fs] reshape is valid.")
        expected_shape = (expected_channels, int(round(args.window_sec)), int(args.target_fs))
    else:
        expected_shape = (expected_channels, target_len)

    map_size = int(float(args.map_size_gb) * (1024 ** 3))
    db = lmdb.open(str(processed_dir), map_size=map_size, subdir=True, lock=True, readahead=False, meminit=False)

    dataset_keys: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    dataset_labels: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
    file_summaries = []

    try:
        for split, files_list in DEFAULT_FILES_DICT.items():
            for filename in files_list:
                gdf_path = raw_dir / filename
                if not gdf_path.exists():
                    print(f"[warn] File {filename} not found in {raw_dir}. Skipping.")
                    continue

                print(f"Processing {filename}...")
                raw = mne.io.read_raw_gdf(str(gdf_path), preload=True, verbose="ERROR")
                # Keep the same spirit as the original code: only EEG channels, then first 22 channels.
                try:
                    raw.pick_types(eeg=True, verbose="ERROR")
                except TypeError:
                    raw.pick_types(eeg=True)

                original_fs = int(round(float(raw.info["sfreq"])))
                if args.highcut >= 0.5 * original_fs:
                    raise ValueError(
                        f"highcut={args.highcut} is not below Nyquist for {filename} with fs={original_fs}."
                    )

                raw_data = raw.get_data(units="uV")
                if raw_data.shape[0] < expected_channels:
                    raise ValueError(f"{filename}: expected at least {expected_channels} EEG channels, got {raw_data.shape[0]}")

                events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
                decoded_events = decode_event_descriptions(events, event_id)
                cue_onsets, cue_labels, cue_info = extract_cue_onsets_and_labels(decoded_events, filename, args.eval_label_dir)

                added = 0
                skipped_short = 0
                skipped_bad = 0

                for i, (cue_sample, label) in enumerate(zip(cue_onsets, cue_labels)):
                    sample = preprocess_one_trial(
                        raw_data_uV=raw_data,
                        cue_sample=cue_sample,
                        original_fs=original_fs,
                        target_fs=args.target_fs,
                        window_sec=args.window_sec,
                        lowcut=args.lowcut,
                        highcut=args.highcut,
                        filter_order=args.filter_order,
                        filter_method=args.filter_method,
                        do_car=not args.no_car,
                        expected_channels=expected_channels,
                    )
                    if sample is None:
                        skipped_short += 1
                        continue

                    if args.type == "cbramod":
                        sample = sample.reshape(expected_channels, int(round(args.window_sec)), int(args.target_fs))

                    if tuple(sample.shape) != tuple(expected_shape):
                        skipped_bad += 1
                        print(f"[warn] {filename}-{i}: bad shape {sample.shape}, expected {expected_shape}; skipped")
                        continue

                    sample_key = f"{Path(filename).stem}-{i:03d}"
                    data_dict = {
                        "sample": sample.astype(np.float32),
                        "label": int(label),
                        "ch_names": list(ELECTRODES_22),
                        "sfreq": int(args.target_fs),
                        "window_sec": float(args.window_sec),
                        "cue_onset_original_sample": int(cue_sample),
                        "source_file": filename,
                    }
                    with db.begin(write=True) as txn:
                        txn.put(key=sample_key.encode("utf-8"), value=pickle.dumps(data_dict, protocol=pickle.HIGHEST_PROTOCOL))

                    dataset_keys[split].append(sample_key)
                    dataset_labels[split].append(int(label))
                    added += 1

                summary = {
                    "file": filename,
                    "split": split,
                    "events": int(events.shape[0]),
                    "motor_cues": int(cue_info.get("motor_cues", 0)),
                    "unknown_eval_cues": int(cue_info.get("unknown_eval_cues", 0)),
                    "added": int(added),
                    "skipped_short": int(skipped_short),
                    "skipped_bad": int(skipped_bad),
                }
                file_summaries.append(summary)

                print(
                    f"  events={summary['events']}, motor_cues={summary['motor_cues']}, "
                    f"unknown_eval_cues={summary['unknown_eval_cues']}"
                )
                print(f"  added={added}, skipped_short={skipped_short}, skipped_bad={skipped_bad}")

        meta = {
            "dataset": "BCICIV2A",
            "split_files": DEFAULT_FILES_DICT,
            "electrodes": list(ELECTRODES_22),
            "num_channels": expected_channels,
            "num_classes": 4,
            "label_map": {
                "769_left_hand": 0,
                "770_right_hand": 1,
                "771_feet": 2,
                "772_tongue": 3,
            },
            "sample_timing": "cue onset 0-4s by default",
            "window_sec": float(args.window_sec),
            "target_fs": int(args.target_fs),
            "expected_shape": tuple(expected_shape),
            "lowcut": float(args.lowcut),
            "highcut": float(args.highcut),
            "filter_order": int(args.filter_order),
            "filter_method": args.filter_method,
            "car": not args.no_car,
            "type": args.type,
            "file_summaries": file_summaries,
        }

        with db.begin(write=True) as txn:
            txn.put(key=b"__keys__", value=pickle.dumps(dataset_keys, protocol=pickle.HIGHEST_PROTOCOL))
            txn.put(key=b"__meta__", value=pickle.dumps(meta, protocol=pickle.HIGHEST_PROTOCOL))

    finally:
        db.sync()
        db.close()

    print("\n========== Done ==========")
    print(f"Output LMDB: {processed_dir}")
    total = 0
    for split in ["train", "val", "test"]:
        n = len(dataset_keys[split])
        total += n
        print(f"{split}: {n} samples | label_dist={label_distribution(dataset_labels[split], num_classes=4)}")
    print(f"total: {total} samples")
    print(f"Expected sample shape: {list(expected_shape)}")

    if args.eval_label_dir is None:
        print(
            "\n[important] --eval_label_dir was not provided. "
            "Evaluation files AxxE.gdf contain unknown cue code 783, so they are skipped. "
            "To reach 5184 samples, provide A01E.mat ... A09E.mat via --eval_label_dir."
        )


if __name__ == "__main__":
    main()
