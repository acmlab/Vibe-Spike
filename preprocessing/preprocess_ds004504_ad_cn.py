# preprocess_ds004504_ad_cn_30s_100hz.py
# -*- coding: utf-8 -*-
"""
Preprocess OpenNeuro ds004504 preprocessed EEG data for AD-vs-CN classification.

Target task
-----------
  - Dataset: OpenNeuro ds004504
  - Use the already processed/preprocessed EEG files, preferably under derivatives/
  - Only keep AD and CN subjects:
      CN -> label 0
      AD -> label 1
  - Subject-level split:
      for each class, the LAST 3 subjects after sorting by subject id are used as VAL
      all remaining AD/CN subjects are used as TRAIN
      no TEST split is created
  - Segment length: 30 seconds
  - Final saved sampling rate: 100 Hz
  - Final saved sample shape: [19, 3000] by default

Expected input after download, for example:
  aws s3 sync --no-sign-request s3://openneuro.org/ds004504 ds004504-download/

Example:
  python preprocess_ds004504_ad_cn_30s_100hz.py \
    --root ds004504-download \
    --out_dir ds004504_ad_cn_30s_100hz \
    --write_lmdb \
    --lmdb_path ds004504_ad_cn_30s_100hz_lmdb \
    --overwrite

Output:
  ds004504_ad_cn_30s_100hz/
    train/*.pkl
    val/*.pkl
    meta.json
    keys_count.csv
    segment_summary_by_subject.csv
    channel_mapping.csv

Each pkl contains at least:
  {
    "X": np.float32 [C, 3000],
    "y": 0/1,
  }

Optional LMDB contains the same samples and __keys__/__meta__.

Notes:
  1. This script uses MNE to read preprocessed files. It supports .set/.edf/.vhdr/.fif.
  2. It preferentially searches derivatives/ first. If no files are found there and
     --allow_raw_fallback is set, it falls back to root sub-*/eeg files.
  3. The split is purely subject-level and deterministic. No test set is written.
"""

import argparse
import csv
import json
import os
import pickle
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import mne
except Exception as e:
    mne = None

try:
    import lmdb
except Exception:
    lmdb = None


# ============================================================
# Dataset/task constants
# ============================================================

DATASET_NAME = "ds004504"
TASK_NAME = "ds004504_ad_cn_30s_100hz"

TARGET_FS = 100
WINDOW_SECONDS = 30
TARGET_WINDOW_LEN = TARGET_FS * WINDOW_SECONDS

# The paper/dataset uses 19 scalp electrodes.
# Keep this stable order for model input.
STANDARD_19_CHANNELS = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T3", "C3", "Cz", "C4", "T4",
    "T5", "P3", "Pz", "P4", "T6", "O1", "O2",
]

# Some readers/datasets may use modern names. Keep aliases conservative.
CHANNEL_ALIASES = {
    "FP1": "Fp1",
    "FP2": "Fp2",
    "FPZ": "Fpz",
    "FZ": "Fz",
    "CZ": "Cz",
    "PZ": "Pz",
    "OZ": "Oz",
    "T7": "T3",
    "T8": "T4",
    "P7": "T5",
    "P8": "T6",
}

GROUP_TO_LABEL = {
    "C": 0,
    "CN": 0,
    "CONTROL": 0,
    "HC": 0,
    "HEALTHY": 0,
    "A": 1,
    "AD": 1,
    "ALZHEIMER": 1,
    "ALZHEIMERS": 1,
    "ALZHEIMER'S": 1,
}

LABEL_TO_NAME = {
    0: "CN",
    1: "AD",
}


# ============================================================
# Basic IO helpers
# ============================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def conv(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.integer, np.floating, np.bool_)):
            return x.item()
        if isinstance(x, Path):
            return str(x)
        if isinstance(x, dict):
            return {str(k): conv(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [conv(v) for v in x]
        return x

    path.write_text(json.dumps(conv(data), ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: List[dict], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for k in row.keys():
                if k not in fieldnames:
                    fieldnames.append(k)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def natural_subject_key(sub_id: str) -> Tuple[int, str]:
    m = re.search(r"sub-(\d+)", str(sub_id))
    if m is not None:
        return int(m.group(1)), str(sub_id)
    return 10**9, str(sub_id)


def extract_subject_id(path_or_text: str) -> Optional[str]:
    m = re.search(r"(sub-[A-Za-z0-9]+)", str(path_or_text))
    if m is None:
        return None
    return m.group(1)


def safe_stage_name(s: str) -> str:
    return str(s).replace("/", "_").replace(" ", "_")


# ============================================================
# participants.tsv and split
# ============================================================

def read_participants(participants_tsv: Path) -> pd.DataFrame:
    """
    Robust reader for participants.tsv.

    Normal BIDS file is tab-separated with columns:
      participant_id, Gender, Age, Group, MMSE

    Some display tools collapse it into whitespace, so this function has a fallback
    parser that reads whitespace tokens in groups of 5.
    """
    if not participants_tsv.exists():
        raise FileNotFoundError(f"participants.tsv not found: {participants_tsv}")

    # First try normal BIDS TSV.
    try:
        df = pd.read_csv(participants_tsv, sep="\t")
        if "participant_id" in df.columns and "Group" in df.columns and len(df) > 0:
            return df
    except Exception:
        pass

    # Fallback: whitespace-token parser.
    text = participants_tsv.read_text(encoding="utf-8", errors="ignore")
    tokens = text.split()
    if len(tokens) < 10:
        raise ValueError(f"Cannot parse participants.tsv: too few tokens in {participants_tsv}")

    header = tokens[:5]
    if header[0] != "participant_id" or "Group" not in header:
        raise ValueError(f"Unexpected participants.tsv header: {header}")

    body = tokens[5:]
    if len(body) % 5 != 0:
        raise ValueError(
            f"Cannot parse participants.tsv by groups of 5. "
            f"body token count={len(body)} is not divisible by 5."
        )

    rows = []
    for i in range(0, len(body), 5):
        rows.append(dict(zip(header, body[i:i + 5])))
    return pd.DataFrame(rows)


def normalize_group_value(x: Any) -> str:
    s = str(x).strip()
    s = s.replace(" ", "")
    return s.upper()


def build_subject_table(root: Path) -> pd.DataFrame:
    df = read_participants(root / "participants.tsv")
    required = ["participant_id", "Group"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"participants.tsv missing required column {c!r}; columns={list(df.columns)}")

    rows = []
    for _, row in df.iterrows():
        sub_id = str(row["participant_id"]).strip()
        group_raw = row["Group"]
        group = normalize_group_value(group_raw)
        if group not in GROUP_TO_LABEL:
            # FTD and any other group are excluded for AD-vs-CN.
            continue
        label = int(GROUP_TO_LABEL[group])
        rows.append({
            "subject": sub_id,
            "group_raw": str(group_raw),
            "group": LABEL_TO_NAME[label],
            "label": label,
            "sort_key": natural_subject_key(sub_id)[0],
        })

    out = pd.DataFrame(rows).sort_values(["label", "sort_key", "subject"]).reset_index(drop=True)
    if out.empty:
        raise ValueError("No AD/CN subjects found in participants.tsv")
    return out


def build_last_k_val_split(subject_table: pd.DataFrame, val_last_k_per_class: int = 3) -> Dict[str, str]:
    """
    For each class separately, sort subjects by subject id and use the last K as val.
    No test split is created.
    """
    split = {}
    for label in sorted(subject_table["label"].unique().tolist()):
        sub_df = subject_table[subject_table["label"] == label].copy()
        sub_df = sub_df.sort_values(["sort_key", "subject"])
        subjects = sub_df["subject"].tolist()
        if len(subjects) <= val_last_k_per_class:
            raise ValueError(
                f"Class {LABEL_TO_NAME[int(label)]} has only {len(subjects)} subjects, "
                f"cannot reserve last {val_last_k_per_class} for val."
            )
        val_subjects = set(subjects[-int(val_last_k_per_class):])
        for s in subjects:
            split[s] = "val" if s in val_subjects else "train"
    return split


# ============================================================
# EEG file discovery / reading
# ============================================================

def _candidate_score(path: Path) -> Tuple[int, int, str]:
    """
    Smaller score is preferred.
    Prefer files that look preprocessed/cleaned and standard EEG formats.
    """
    p = str(path).lower()
    name = path.name.lower()
    # Prefer derivatives and preprocessed-looking files.
    score = 0
    if "derivatives" not in p:
        score += 20
    if any(k in name for k in ["preproc", "preprocessed", "clean", "cleaned", "processed"]):
        score -= 10
    if name.endswith(".set"):
        score += 0
    elif name.endswith(".fif") or name.endswith(".fif.gz"):
        score += 1
    elif name.endswith(".vhdr"):
        score += 2
    elif name.endswith(".edf"):
        score += 3
    else:
        score += 10
    return score, len(str(path)), str(path)


def discover_eeg_files(root: Path, allow_raw_fallback: bool = False) -> Dict[str, Path]:
    """
    Discover one EEG file per subject.

    By default, search derivatives/ only because the user requested the already processed dataset.
    If no derivative files are found and allow_raw_fallback=True, also search raw BIDS folders.
    """
    patterns = ["*.set", "*.edf", "*.vhdr", "*.fif", "*.fif.gz"]
    candidates: List[Path] = []

    deriv = root / "derivatives"
    if deriv.exists():
        for pat in patterns:
            candidates.extend(deriv.rglob(pat))

    if len(candidates) == 0 and allow_raw_fallback:
        for pat in patterns:
            candidates.extend(root.glob(f"sub-*/eeg/{pat}"))
            candidates.extend(root.rglob(pat))
        candidates = [p for p in candidates if "derivatives" not in str(p).lower()]

    by_sub: Dict[str, List[Path]] = defaultdict(list)
    for p in candidates:
        sub_id = extract_subject_id(str(p))
        if sub_id is None:
            continue
        # Ignore non-EEG or hidden files.
        if any(part.startswith(".") for part in p.parts):
            continue
        by_sub[sub_id].append(p)

    selected: Dict[str, Path] = {}
    for sub_id, files in by_sub.items():
        files_sorted = sorted(files, key=_candidate_score)
        selected[sub_id] = files_sorted[0]

    return selected


def canonicalize_channel_name(ch_name: str) -> str:
    s = str(ch_name).strip()
    s = s.replace("EEG ", "")
    s = s.replace("-Ref", "")
    s = s.replace("-REF", "")
    s = s.replace(".", "")
    s = s.strip()

    # Remove common suffix after dash only when it looks like a reference suffix.
    # Do NOT break bipolar names generally; ds004504 should be monopolar.
    for suffix in ["-LE", "-RE", "-A1", "-A2", "-AVG", "-M1", "-M2"]:
        if s.upper().endswith(suffix):
            s = s[: -len(suffix)]
            break

    upper = s.upper()
    if upper in CHANNEL_ALIASES:
        return CHANNEL_ALIASES[upper]

    # Standard capitalization for channels like FP1, FZ, CZ, PZ.
    standard_map = {c.upper(): c for c in STANDARD_19_CHANNELS}
    if upper in standard_map:
        return standard_map[upper]
    return s


def load_raw_with_mne(eeg_path: Path):
    if mne is None:
        raise RuntimeError("mne is not installed. Install with: pip install mne")

    suffixes = "".join(eeg_path.suffixes).lower()
    suffix = eeg_path.suffix.lower()

    if suffix == ".set":
        raw = mne.io.read_raw_eeglab(str(eeg_path), preload=True, verbose="ERROR")
    elif suffix == ".edf":
        raw = mne.io.read_raw_edf(str(eeg_path), preload=True, verbose="ERROR")
    elif suffix == ".vhdr":
        raw = mne.io.read_raw_brainvision(str(eeg_path), preload=True, verbose="ERROR")
    elif suffix == ".fif" or suffixes.endswith(".fif.gz"):
        raw = mne.io.read_raw_fif(str(eeg_path), preload=True, verbose="ERROR")
    else:
        raise ValueError(f"Unsupported EEG file type: {eeg_path}")

    # Canonicalize channel names conservatively.
    rename = {}
    for ch in raw.ch_names:
        new = canonicalize_channel_name(ch)
        if new != ch and new not in raw.ch_names:
            rename[ch] = new
    if rename:
        raw.rename_channels(rename)

    return raw


def pick_or_fill_channels(raw, desired_channels: List[str], missing_policy: str = "error") -> Tuple[np.ndarray, float, List[str], List[str]]:
    """
    Return data [C,T] in desired channel order.

    missing_policy:
      - error: raise if any desired channel is missing
      - zero: fill missing channels with zeros
    """
    sfreq = float(raw.info["sfreq"])
    ch_to_idx = {canonicalize_channel_name(ch): i for i, ch in enumerate(raw.ch_names)}

    data_rows = []
    used = []
    missing = []
    n_times = int(raw.n_times)

    full_data = None
    for ch in desired_channels:
        if ch in ch_to_idx:
            if full_data is None:
                full_data = raw.get_data().astype(np.float32)
            data_rows.append(full_data[ch_to_idx[ch]].astype(np.float32))
            used.append(ch)
        else:
            missing.append(ch)
            if missing_policy == "zero":
                data_rows.append(np.zeros(n_times, dtype=np.float32))
                used.append(ch)
            else:
                raise ValueError(
                    f"Missing channel {ch!r}. Available channels: {raw.ch_names}. "
                    f"Use --missing_channel_policy zero if you want zero-fill."
                )

    data = np.stack(data_rows, axis=0).astype(np.float32)
    return data, sfreq, used, missing


def resample_to_100_with_mne_array(data: np.ndarray, orig_sfreq: float, target_sfreq: int = TARGET_FS) -> np.ndarray:
    """
    Resample [C,T] to target_sfreq using MNE's filter/resample.

    Important fix:
      Some MNE/SciPy combinations raise:
        ValueError("Data to be resampled must be real floating, got float32")
      when the input is float32. Therefore, resampling is done in float64 and
      the saved segment is cast back to float32 afterwards.
    """
    data = np.asarray(data)

    if abs(float(orig_sfreq) - float(target_sfreq)) < 1e-6:
        return data.astype(np.float32, copy=False)

    # MNE/SciPy resampling path: use float64 to avoid the float32 dtype error.
    data64 = np.asarray(data, dtype=np.float64)

    if mne is None:
        # Conservative fallback: linear interpolation.
        old_len = data64.shape[1]
        new_len = int(round(old_len * float(target_sfreq) / float(orig_sfreq)))
        old_t = np.linspace(0.0, 1.0, old_len, endpoint=False)
        new_t = np.linspace(0.0, 1.0, new_len, endpoint=False)
        out = np.stack([np.interp(new_t, old_t, row) for row in data64], axis=0)
        return np.asarray(out, dtype=np.float32)

    out = mne.filter.resample(
        data64,
        down=float(orig_sfreq),
        up=float(target_sfreq),
        axis=1,
        verbose="ERROR",
    )
    return np.asarray(out, dtype=np.float32)


# ============================================================
# Optional LMDB writer
# ============================================================

class OptionalLMDBWriter:
    def __init__(self, lmdb_path: Optional[Path], map_size_gb: float = 128.0, commit_interval: int = 1000):
        self.enabled = lmdb_path is not None
        self.keys: Dict[str, List[str]] = {"train": [], "val": []}
        self.count = 0
        self.commit_interval = int(commit_interval)
        self.env = None
        self.txn = None

        if not self.enabled:
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

    def put(self, split: str, key: str, sample: dict) -> None:
        if split not in self.keys:
            self.keys[split] = []
        self.keys[split].append(key)
        if not self.enabled:
            return
        assert self.txn is not None
        self.txn.put(key.encode("utf-8"), pickle.dumps(sample, protocol=pickle.HIGHEST_PROTOCOL))
        self.count += 1
        if self.count % self.commit_interval == 0:
            self.txn.commit()
            self.txn = self.env.begin(write=True)

    def close(self, meta: dict) -> None:
        if not self.enabled:
            return
        assert self.txn is not None and self.env is not None
        self.txn.put(b"__keys__", pickle.dumps(self.keys, protocol=pickle.HIGHEST_PROTOCOL))
        self.txn.put(b"__meta__", pickle.dumps(meta, protocol=pickle.HIGHEST_PROTOCOL))
        self.txn.commit()
        self.env.sync()
        self.env.close()


# ============================================================
# Segmenting/writing
# ============================================================

def write_channel_mapping_csv(out_dir: Path, channels: List[str]) -> Path:
    path = out_dir / "channel_mapping.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "channel_name"])
        for i, ch in enumerate(channels):
            w.writerow([i, ch])
    return path


def save_segment(
    out_file: Path,
    split: str,
    lmdb_writer: OptionalLMDBWriter,
    key: str,
    x: np.ndarray,
    y: int,
    subject: str,
    group: str,
    source_file: Path,
    start_sample_100hz: int,
    channels: List[str],
) -> None:
    obj = {
        "X": np.asarray(x, dtype=np.float32),
        "y": int(y),
        "subject": subject,
        "group": group,
        "fs": TARGET_FS,
        "window_seconds": WINDOW_SECONDS,
        "ch_names": list(channels),
        "source_file": str(source_file),
        "source_start_sample_100hz": int(start_sample_100hz),
        "source_start_sec": float(start_sample_100hz / TARGET_FS),
        "task": TASK_NAME,
    }
    with out_file.open("wb") as f:
        pickle.dump({"X": obj["X"], "y": int(y)}, f, protocol=pickle.HIGHEST_PROTOCOL)

    # LMDB keeps richer metadata while preserving X/y compatibility.
    lmdb_obj = dict(obj)
    lmdb_obj["sample"] = obj["X"]
    lmdb_obj["label"] = int(y)
    lmdb_writer.put(split, key, lmdb_obj)


def segment_subject(
    subject: str,
    label: int,
    group: str,
    split: str,
    eeg_path: Path,
    out_dir: Path,
    lmdb_writer: OptionalLMDBWriter,
    channels: List[str],
    missing_channel_policy: str = "error",
    drop_last: bool = True,
    min_std: float = 1e-8,
) -> dict:
    stats = {
        "subject": subject,
        "group": group,
        "label": int(label),
        "split": split,
        "source_file": str(eeg_path),
        "orig_sfreq": None,
        "n_channels": len(channels),
        "n_times_100hz": 0,
        "duration_sec_100hz": 0.0,
        "n_segments": 0,
        "n_skipped_nonfinite": 0,
        "n_skipped_low_std": 0,
        "missing_channels": "",
        "error": "",
    }

    try:
        raw = load_raw_with_mne(eeg_path)
        data, sfreq, _, missing = pick_or_fill_channels(raw, channels, missing_policy=missing_channel_policy)
        stats["orig_sfreq"] = float(sfreq)
        stats["missing_channels"] = ";".join(missing)

        data_100 = resample_to_100_with_mne_array(data, orig_sfreq=sfreq, target_sfreq=TARGET_FS)
        data_100 = np.asarray(data_100, dtype=np.float32)

        if data_100.ndim != 2:
            raise ValueError(f"Expected data [C,T], got shape={data_100.shape}")
        if data_100.shape[0] != len(channels):
            raise ValueError(f"Expected C={len(channels)}, got shape={data_100.shape}")

        n_times = int(data_100.shape[1])
        stats["n_times_100hz"] = n_times
        stats["duration_sec_100hz"] = float(n_times / TARGET_FS)

        win = TARGET_WINDOW_LEN
        if drop_last:
            starts = list(range(0, max(n_times - win + 1, 0), win))
        else:
            starts = list(range(0, n_times, win))

        split_dir = out_dir / split
        ensure_dir(split_dir)

        for seg_idx, start in enumerate(starts):
            seg = data_100[:, start:start + win]
            if seg.shape[1] < win:
                if drop_last:
                    continue
                pad = win - seg.shape[1]
                if seg.shape[1] == 0:
                    seg = np.zeros((len(channels), win), dtype=np.float32)
                else:
                    seg = np.pad(seg, ((0, 0), (0, pad)), mode="edge")

            seg = np.asarray(seg, dtype=np.float32)
            if not np.isfinite(seg).all():
                stats["n_skipped_nonfinite"] += 1
                continue
            if float(np.nanstd(seg)) < float(min_std):
                stats["n_skipped_low_std"] += 1
                continue

            file_stem = f"{subject}_{LABEL_TO_NAME[int(label)]}_seg{seg_idx:04d}_start{start:08d}"
            out_file = split_dir / f"{file_stem}.pkl"
            key = f"{split}/{file_stem}"
            save_segment(
                out_file=out_file,
                split=split,
                lmdb_writer=lmdb_writer,
                key=key,
                x=seg,
                y=label,
                subject=subject,
                group=group,
                source_file=eeg_path,
                start_sample_100hz=start,
                channels=channels,
            )
            stats["n_segments"] += 1

    except Exception as e:
        stats["error"] = repr(e)
        print(f"[skip subject] {subject} {eeg_path}: {repr(e)}")

    return stats


def count_labels_from_pkl(out_dir: Path) -> Dict[str, Dict[int, int]]:
    counts = {"train": {0: 0, 1: 0}, "val": {0: 0, 1: 0}}
    for split in ["train", "val"]:
        for p in sorted((out_dir / split).glob("*.pkl")):
            try:
                obj = pickle.load(open(p, "rb"))
                y = int(obj["y"])
                counts[split][y] = counts[split].get(y, 0) + 1
            except Exception:
                pass
    return counts


def write_summary(
    out_dir: Path,
    stats_rows: List[dict],
    subject_table: pd.DataFrame,
    subject_split: Dict[str, str],
    eeg_files: Dict[str, Path],
    channels: List[str],
    lmdb_path: Optional[Path],
    lmdb_keys: Optional[Dict[str, List[str]]] = None,
) -> None:
    write_csv(
        out_dir / "segment_summary_by_subject.csv",
        stats_rows,
        fieldnames=[
            "subject", "group", "label", "split", "source_file", "orig_sfreq",
            "n_channels", "n_times_100hz", "duration_sec_100hz", "n_segments",
            "n_skipped_nonfinite", "n_skipped_low_std", "missing_channels", "error",
        ],
    )

    split_counts = count_labels_from_pkl(out_dir)
    keys_rows = []
    for split in ["train", "val"]:
        n_cn = int(split_counts[split].get(0, 0))
        n_ad = int(split_counts[split].get(1, 0))
        keys_rows.append({
            "split": split,
            "n_segments": n_cn + n_ad,
            "n_CN": n_cn,
            "n_AD": n_ad,
        })
    write_csv(out_dir / "keys_count.csv", keys_rows, fieldnames=["split", "n_segments", "n_CN", "n_AD"])

    split_subject_rows = []
    for _, row in subject_table.iterrows():
        sub = row["subject"]
        split_subject_rows.append({
            "subject": sub,
            "group": row["group"],
            "label": int(row["label"]),
            "split": subject_split.get(sub, "excluded"),
            "eeg_file": str(eeg_files.get(sub, "")),
        })
    write_csv(out_dir / "subject_split.csv", split_subject_rows, fieldnames=["subject", "group", "label", "split", "eeg_file"])

    meta = {
        "dataset": DATASET_NAME,
        "task": TASK_NAME,
        "description": "AD-vs-CN preprocessing from ds004504 already-processed EEG files; 30s non-overlap windows; final 100 Hz; no test split.",
        "target_fs": TARGET_FS,
        "window_seconds": WINDOW_SECONDS,
        "segment_shape": [len(channels), TARGET_WINDOW_LEN],
        "pkl_format": {"X": f"[{len(channels)},{TARGET_WINDOW_LEN}] float32", "y": "0/1"},
        "label_map": {"CN": 0, "AD": 1},
        "channels": list(channels),
        "split_rule": "For each of AD and CN separately, sorted by subject id, the last K subjects are val; all remaining AD/CN are train; no test.",
        "subject_split": split_subject_rows,
        "split_counts": keys_rows,
        "lmdb_path": str(lmdb_path) if lmdb_path is not None else None,
        "lmdb_keys_count": {k: len(v) for k, v in lmdb_keys.items()} if lmdb_keys is not None else None,
    }
    save_json(out_dir / "meta.json", meta)


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess OpenNeuro ds004504 AD/CN to 30s 100Hz pkl/LMDB segments.")
    parser.add_argument("--root", type=str, required=True, help="Root of ds004504 download, e.g. ds004504-download")
    parser.add_argument("--out_dir", type=str, default=None, help="Output pkl dir. Default: <root>/ds004504_ad_cn_30s_100hz")
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--val_last_k_per_class", type=int, default=3, help="Use last K subjects of each class as val.")
    parser.add_argument("--allow_raw_fallback", action="store_true", help="If no derivatives EEG files are found, fall back to raw BIDS EEG files.")
    parser.add_argument("--missing_channel_policy", choices=["error", "zero"], default="error")
    parser.add_argument("--drop_last", action="store_true", default=True, help="Drop incomplete final window. Default True.")
    parser.add_argument("--keep_last", action="store_false", dest="drop_last", help="Pad incomplete final window instead of dropping it.")
    parser.add_argument("--min_std", type=float, default=1e-8, help="Skip windows whose global std is below this value.")

    parser.add_argument("--channels", type=str, default=",".join(STANDARD_19_CHANNELS), help="Comma-separated channel order to save.")

    parser.add_argument("--write_lmdb", action="store_true")
    parser.add_argument("--lmdb_path", type=str, default=None)
    parser.add_argument("--lmdb_map_size_gb", type=float, default=64.0)
    parser.add_argument("--lmdb_commit_interval", type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"root not found: {root}")

    out_dir = Path(args.out_dir) if args.out_dir is not None else (root / "ds004504_ad_cn_30s_100hz")
    lmdb_path = None
    if args.write_lmdb:
        lmdb_path = Path(args.lmdb_path) if args.lmdb_path is not None else (root / "ds004504_ad_cn_30s_100hz_lmdb")

    if args.overwrite:
        if out_dir.exists():
            shutil.rmtree(out_dir)
        if lmdb_path is not None and lmdb_path.exists():
            shutil.rmtree(lmdb_path)

    ensure_dir(out_dir)
    ensure_dir(out_dir / "train")
    ensure_dir(out_dir / "val")

    channels = [c.strip() for c in str(args.channels).split(",") if c.strip()]
    if len(channels) == 0:
        raise ValueError("No channels specified")

    mapping_csv = write_channel_mapping_csv(out_dir, channels)

    subject_table = build_subject_table(root)
    subject_split = build_last_k_val_split(subject_table, val_last_k_per_class=int(args.val_last_k_per_class))
    eeg_files = discover_eeg_files(root, allow_raw_fallback=bool(args.allow_raw_fallback))

    missing_file_subjects = [s for s in subject_table["subject"].tolist() if s not in eeg_files]
    if len(missing_file_subjects) > 0:
        print("⚠️ These AD/CN subjects have no discovered EEG file and will be skipped:")
        print(missing_file_subjects)

    lmdb_writer = OptionalLMDBWriter(lmdb_path, map_size_gb=args.lmdb_map_size_gb, commit_interval=args.lmdb_commit_interval)

    print("========== ds004504 AD/CN 30s 100Hz PREPROCESS ==========")
    print(f"root          : {root}")
    print(f"out_dir       : {out_dir}")
    print(f"target fs     : {TARGET_FS} Hz")
    print(f"window        : {WINDOW_SECONDS} s -> {TARGET_WINDOW_LEN} samples")
    print(f"channels      : {len(channels)} {channels}")
    print(f"split rule    : last {args.val_last_k_per_class} subjects per class -> val; no test")
    print(f"mapping csv   : {mapping_csv}")
    print(f"write_lmdb    : {args.write_lmdb}")
    if lmdb_path is not None:
        print(f"lmdb_path     : {lmdb_path}")
    print("=========================================================")

    print("\nSubject split:")
    for label in [0, 1]:
        cls = LABEL_TO_NAME[label]
        subs = subject_table[subject_table["label"] == label]["subject"].tolist()
        val_subs = [s for s in subs if subject_split.get(s) == "val"]
        train_subs = [s for s in subs if subject_split.get(s) == "train"]
        print(f"  {cls}: train={len(train_subs)} {train_subs[:3]}...{train_subs[-3:] if train_subs else []}; val={len(val_subs)} {val_subs}")

    stats_rows = []
    rows_iter = list(subject_table.sort_values(["label", "sort_key", "subject"]).to_dict("records"))
    for row in tqdm(rows_iter, desc="subjects"):
        subject = row["subject"]
        if subject not in eeg_files:
            stats_rows.append({
                "subject": subject,
                "group": row["group"],
                "label": int(row["label"]),
                "split": subject_split.get(subject, "excluded"),
                "source_file": "",
                "orig_sfreq": None,
                "n_channels": len(channels),
                "n_times_100hz": 0,
                "duration_sec_100hz": 0.0,
                "n_segments": 0,
                "n_skipped_nonfinite": 0,
                "n_skipped_low_std": 0,
                "missing_channels": "",
                "error": "EEG file not found",
            })
            continue

        stats = segment_subject(
            subject=subject,
            label=int(row["label"]),
            group=row["group"],
            split=subject_split[subject],
            eeg_path=eeg_files[subject],
            out_dir=out_dir,
            lmdb_writer=lmdb_writer,
            channels=channels,
            missing_channel_policy=args.missing_channel_policy,
            drop_last=bool(args.drop_last),
            min_std=float(args.min_std),
        )
        stats_rows.append(stats)

    meta_for_lmdb = {
        "dataset": DATASET_NAME,
        "task": TASK_NAME,
        "target_fs": TARGET_FS,
        "window_seconds": WINDOW_SECONDS,
        "segment_shape": [len(channels), TARGET_WINDOW_LEN],
        "label_map": {"CN": 0, "AD": 1},
        "channels": channels,
        "split_rule": f"last {args.val_last_k_per_class} subjects per class -> val; no test",
    }
    lmdb_writer.close(meta_for_lmdb)

    write_summary(
        out_dir=out_dir,
        stats_rows=stats_rows,
        subject_table=subject_table,
        subject_split=subject_split,
        eeg_files=eeg_files,
        channels=channels,
        lmdb_path=lmdb_path,
        lmdb_keys=lmdb_writer.keys if args.write_lmdb else None,
    )

    print("\n========== DONE ==========")
    print(f"pkl segments : {out_dir}")
    print(f"meta.json    : {out_dir / 'meta.json'}")
    print(f"keys_count   : {out_dir / 'keys_count.csv'}")
    print(f"summary csv  : {out_dir / 'segment_summary_by_subject.csv'}")
    print(f"subject split: {out_dir / 'subject_split.csv'}")
    if args.write_lmdb:
        print(f"LMDB         : {lmdb_path}")


if __name__ == "__main__":
    main()
