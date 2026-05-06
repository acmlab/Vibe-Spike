import argparse
import math
import pickle
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import lmdb
import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly


SEEDVIG_CHANNELS = [
    "FT7",
    "FT8",
    "T7",
    "T8",
    "TP7",
    "TP8",
    "CP1",
    "CP2",
    "P1",
    "PZ",
    "P2",
    "PO3",
    "POZ",
    "PO4",
    "O1",
    "OZ",
    "O2",
]

MODEL_PROFILE_SUFFIXES = {
    "EEGNet": "lmdb_100hz",
    "EEGConformer": "lmdb_100hz",
    "SPaRCNet": "lmdb_100hz",
    "ContraWR": "lmdb_100hz",
    "CNN-Transformer": "lmdb_100hz",
    "FFCL": "lmdb_100hz",
    "ST-Transformer": "lmdb_100hz",
    "BIOT": "lmdb_biot_16ch_200hz",
    "LaBraM-Base": "lmdb_labram_30ch_200hz",
    "CbraMod": "lmdb_cbramod_30ch_200hz",
    "REVE-Base": "lmdb_reve_30ch_200hz",
}


@dataclass(frozen=True)
class Profile:
    name: str
    target_fs: int
    channels: Tuple[str, ...]
    pad_to_channels: int | None = None

    @property
    def selected_channels(self) -> List[str]:
        names = list(self.channels)
        if self.pad_to_channels is not None and self.pad_to_channels > len(names):
            names.extend([f"PAD{i:02d}" for i in range(1, self.pad_to_channels - len(names) + 1)])
        return names


BASE_PROFILE = Profile(name="lmdb_100hz", target_fs=100, channels=tuple(SEEDVIG_CHANNELS))
ALL_PROFILES = [
    BASE_PROFILE,
    Profile(name="lmdb_biot_16ch_200hz", target_fs=200, channels=tuple(SEEDVIG_CHANNELS[:16])),
    Profile(name="lmdb_labram_30ch_200hz", target_fs=200, channels=tuple(SEEDVIG_CHANNELS), pad_to_channels=30),
    Profile(name="lmdb_cbramod_30ch_200hz", target_fs=200, channels=tuple(SEEDVIG_CHANNELS), pad_to_channels=30),
    Profile(name="lmdb_reve_30ch_200hz", target_fs=200, channels=tuple(SEEDVIG_CHANNELS)),
]


def split_into_sessions(n_samples: int, n_sessions: int) -> List[int]:
    base = n_samples // n_sessions
    remainder = n_samples % n_sessions
    return [base + (1 if i < remainder else 0) for i in range(n_sessions)]


def session_id_for_index(index: int, session_sizes: List[int]) -> int:
    end = 0
    for session_id, size in enumerate(session_sizes):
        end += size
        if index < end:
            return session_id
    return len(session_sizes) - 1


def make_recording_split(mat_files: List[Path], seed: int) -> Dict[str, set[str]]:
    files = list(mat_files)
    rng = random.Random(seed)
    rng.shuffle(files)
    n = len(files)
    n_train = max(1, int(round(n * 0.70)))
    n_val = max(1, int(round(n * 0.15)))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    train = {p.stem for p in files[:n_train]}
    val = {p.stem for p in files[n_train : n_train + n_val]}
    test = {p.stem for p in files[n_train + n_val :]}
    return {"train": train, "val": val, "test": test}


def split_name_for_window(
    mat_stem: str,
    window_idx: int,
    n_windows: int,
    split_mode: str,
    recording_split: Dict[str, set[str]],
    n_sessions: int,
    fold: int,
    val_fold: int,
) -> str:
    if split_mode == "recording":
        for split, stems in recording_split.items():
            if mat_stem in stems:
                return split
        raise KeyError(f"Recording {mat_stem} not found in split map.")

    session_sizes = split_into_sessions(n_windows, n_sessions)
    session_id = session_id_for_index(window_idx, session_sizes)
    if session_id == fold:
        return "test"
    if session_id == val_fold:
        return "val"
    return "train"


def transform_segment(segment_txc: np.ndarray, source_fs: int, profile: Profile) -> np.ndarray:
    x = segment_txc.astype(np.float32, copy=False)
    if profile.target_fs != source_fs:
        gcd = math.gcd(source_fs, profile.target_fs)
        up = profile.target_fs // gcd
        down = source_fs // gcd
        x = resample_poly(x, up=up, down=down, axis=0).astype(np.float32, copy=False)

    channel_to_idx = {name: idx for idx, name in enumerate(SEEDVIG_CHANNELS)}
    selected = x[:, [channel_to_idx[ch] for ch in profile.channels]].T

    if profile.pad_to_channels is not None and profile.pad_to_channels > selected.shape[0]:
        pad = np.zeros((profile.pad_to_channels - selected.shape[0], selected.shape[1]), dtype=np.float32)
        selected = np.concatenate([selected, pad], axis=0)
    return np.ascontiguousarray(selected, dtype=np.float32)


def load_seedvig_pair(raw_path: Path, label_path: Path) -> Tuple[np.ndarray, np.ndarray, int, List[str]]:
    mat = loadmat(raw_path, squeeze_me=True, struct_as_record=False)
    eeg_struct = mat["EEG"]
    eeg = np.asarray(eeg_struct.data, dtype=np.float32)
    ch_names = [str(ch) for ch in np.asarray(eeg_struct.chn).tolist()]
    sample_rate = int(eeg_struct.sample_rate)
    if ch_names != SEEDVIG_CHANNELS:
        raise ValueError(f"{raw_path.name}: unexpected channels {ch_names}")

    labels_mat = loadmat(label_path, squeeze_me=True, struct_as_record=False)
    perclos = np.asarray(labels_mat["perclos"], dtype=np.float32).reshape(-1)
    return eeg, perclos, sample_rate, ch_names


def put_record(txn, key: str, sample: np.ndarray, label: int, perclos: float, raw_path: Path, window_idx: int, split: str, channels: List[str], target_fs: int, window_sec: float) -> None:
    row = {
        "sample": sample,
        "label": int(label),
        "label_bin_clean": int(label),
        "label_bin_loose": int(label),
        "label_3cls": int(label),
        "perclos": float(perclos),
        "source_file": raw_path.name,
        "subject_id": raw_path.stem,
        "session_id": split,
        "window_index": int(window_idx),
        "ch_names": channels,
        "target_fs": int(target_fs),
        "window_sec": float(window_sec),
    }
    txn.put(key.encode("utf-8"), pickle.dumps(row, protocol=pickle.HIGHEST_PROTOCOL))


def build_lmdb_for_profile(
    seedvig_root: Path,
    out_dir: Path,
    profile: Profile,
    split_mode: str,
    n_sessions: int,
    fold: int,
    val_fold: int,
    threshold: float,
    window_sec: float,
    seed: int,
    max_files: int | None,
    map_size_gb: float,
    overwrite: bool,
) -> None:
    raw_dir = seedvig_root / "Raw_Data"
    label_dir = seedvig_root / "perclos_labels"
    raw_files = sorted(raw_dir.glob("*.mat"))
    if max_files is not None:
        raw_files = raw_files[:max_files]
    if not raw_files:
        raise FileNotFoundError(f"No raw .mat files found under {raw_dir}")

    missing_labels = [p.name for p in raw_files if not (label_dir / p.name).exists()]
    if missing_labels:
        raise FileNotFoundError(f"Missing perclos label files: {missing_labels[:5]}")

    if out_dir.exists():
        if not overwrite:
            print(f"[skip] {out_dir} exists. Use --overwrite to rebuild.")
            return
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    recording_split = make_recording_split(raw_files, seed)
    keys = {"train": [], "val": [], "test": []}
    label_counts = {split: {0: 0, 1: 0} for split in keys}
    meta = {
        "dataset": "SEED-VIG",
        "profile": profile.name,
        "selected_channels": profile.selected_channels,
        "target_fs": int(profile.target_fs),
        "window_sec": float(window_sec),
        "label_field": "label_bin_clean",
        "perclos_threshold": float(threshold),
        "split_mode": split_mode,
        "n_sessions": int(n_sessions),
        "fold": int(fold),
        "val_fold": int(val_fold),
        "source_root": str(seedvig_root),
        "model_profile_suffixes": MODEL_PROFILE_SUFFIXES,
    }

    env = lmdb.open(str(out_dir), map_size=int(map_size_gb * 1024**3), subdir=True, lock=True)
    txn = env.begin(write=True)
    writes_since_commit = 0
    try:
        for rec_idx, raw_path in enumerate(raw_files):
            eeg, perclos, source_fs, _ = load_seedvig_pair(raw_path, label_dir / raw_path.name)
            samples_per_window = int(round(source_fs * window_sec))
            max_windows = min(len(perclos), eeg.shape[0] // samples_per_window)
            if max_windows <= 0:
                raise ValueError(f"{raw_path.name}: no complete windows found.")

            for window_idx in range(max_windows):
                start = window_idx * samples_per_window
                end = start + samples_per_window
                split = split_name_for_window(
                    raw_path.stem,
                    window_idx,
                    max_windows,
                    split_mode,
                    recording_split,
                    n_sessions,
                    fold,
                    val_fold,
                )
                sample = transform_segment(eeg[start:end], source_fs, profile)
                label = int(float(perclos[window_idx]) >= threshold)
                key = f"{profile.name}__{raw_path.stem}__{window_idx:04d}"
                put_record(
                    txn,
                    key,
                    sample,
                    label,
                    float(perclos[window_idx]),
                    raw_path,
                    window_idx,
                    split,
                    profile.selected_channels,
                    profile.target_fs,
                    window_sec,
                )
                keys[split].append(key)
                label_counts[split][label] += 1
                writes_since_commit += 1
                if writes_since_commit >= 1000:
                    txn.commit()
                    txn = env.begin(write=True)
                    writes_since_commit = 0

            print(f"[{profile.name}] {rec_idx + 1}/{len(raw_files)} {raw_path.name} windows={max_windows}")

        meta["num_records"] = {split: len(split_keys) for split, split_keys in keys.items()}
        meta["label_counts"] = label_counts
        txn.put(b"__keys__", pickle.dumps(keys, protocol=pickle.HIGHEST_PROTOCOL))
        txn.put(b"__meta__", pickle.dumps(meta, protocol=pickle.HIGHEST_PROTOCOL))
        txn.commit()
    except Exception:
        txn.abort()
        raise
    finally:
        env.sync()
        env.close()

    print(f"[done] {out_dir} train={len(keys['train'])} val={len(keys['val'])} test={len(keys['test'])}")


def parse_args():
    parser = argparse.ArgumentParser(description="Create SEED-VIG LMDB datasets for train_sadt_benchmarks.py.")
    parser.add_argument("--seedvig_root", type=str, default="SEED-VIG/SEED-VIG")
    parser.add_argument("--out_root", type=str, default="seedvig_benchmark_lmdb")
    parser.add_argument("--profiles", choices=["base", "all", "reve"], default="base")
    parser.add_argument("--split_mode", choices=["session_fold", "recording"], default="session_fold")
    parser.add_argument("--folds", type=int, default=5, help="Number of session folds to materialize. Use 1 for a single fold.")
    parser.add_argument("--fold", type=int, default=0, help="Fold index used when --folds 1.")
    parser.add_argument("--val_fold_offset", type=int, default=1)
    parser.add_argument("--n_sessions", type=int, default=5)
    parser.add_argument("--perclos_threshold", type=float, default=0.35)
    parser.add_argument("--window_sec", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--map_size_gb", type=float, default=8.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seedvig_root = Path(args.seedvig_root)
    if not (seedvig_root / "Raw_Data").exists():
        raise FileNotFoundError(f"Missing Raw_Data under {seedvig_root}")
    if not (seedvig_root / "perclos_labels").exists():
        raise FileNotFoundError(f"Missing perclos_labels under {seedvig_root}")
    if args.n_sessions < 2:
        raise ValueError("--n_sessions must be >= 2")
    if args.folds < 1:
        raise ValueError("--folds must be >= 1")

    if args.profiles == "base":
        profiles = [BASE_PROFILE]
    elif args.profiles == "reve":
        profiles = [p for p in ALL_PROFILES if p.name == "lmdb_reve_30ch_200hz"]
    else:
        profiles = ALL_PROFILES
    fold_indices = [args.fold] if args.folds == 1 else list(range(args.folds))
    out_root = Path(args.out_root)

    for fold in fold_indices:
        if fold < 0 or fold >= args.n_sessions:
            raise ValueError(f"fold must be in [0, {args.n_sessions - 1}], got {fold}")
        val_fold = (fold + args.val_fold_offset) % args.n_sessions
        fold_root = out_root / f"fold_{fold:02d}" if args.folds > 1 else out_root
        for profile in profiles:
            build_lmdb_for_profile(
                seedvig_root=seedvig_root,
                out_dir=fold_root / profile.name,
                profile=profile,
                split_mode=args.split_mode,
                n_sessions=args.n_sessions,
                fold=fold,
                val_fold=val_fold,
                threshold=args.perclos_threshold,
                window_sec=args.window_sec,
                seed=args.seed,
                max_files=args.max_files,
                map_size_gb=args.map_size_gb,
                overwrite=args.overwrite,
            )


if __name__ == "__main__":
    main()
