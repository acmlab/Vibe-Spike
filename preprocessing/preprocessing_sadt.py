
import argparse
import pickle
import random
import re
from pathlib import Path
from typing import List, Tuple

import lmdb
import mne
import numpy as np


PAPER_EEG32 = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "FT7", "FC3", "FCz", "FC4", "FT8",
    "T3", "C3", "Cz", "C4", "T4",
    "TP7", "CP3", "CPz", "CP4", "TP8",
    "A1", "T5", "P3", "Pz", "P4", "T6", "A2",
    "O1", "Oz", "O2",
]

SELECTED_CHANNELS = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "FT7", "FC3", "FCz", "FC4", "FT8",
    "T7", "C3", "Cz", "C4", "T8",
    "TP7", "CP3", "CPz", "CP4", "TP8",
    "P7", "P3", "Pz", "P4", "P8",
    "O1", "Oz", "O2",
]

PAPER_TO_CANONICAL = {
    "T3": "T7",
    "T4": "T8",
    "T5": "P7",
    "T6": "P8",
    "FCZ": "FCz",
    "CPZ": "CPz",
    "PZ": "Pz",
    "CZ": "Cz",
    "OZ": "Oz",
}

DEVIATION_CODES = {"251", "252"}
RESPONSE_ONSET_CODE = "253"
RESPONSE_OFFSET_CODE = "254"


def canonicalize_name(name: str) -> str:
    name = str(name).strip()
    name = name.replace("EEG ", "").replace("-REF", "").replace("-LE", "").replace("-A1", "").replace("-A2", "")
    return PAPER_TO_CANONICAL.get(name, name)


def parse_subject_and_session(file_path: Path) -> Tuple[str, str]:
    stem = file_path.stem
    m = re.match(r"^(s\d+)_", stem, flags=re.IGNORECASE)
    if m is None:
        raise ValueError(f"Cannot parse subject id from file name: {file_path.name}")
    return m.group(1).lower(), stem


def split_subjects(subject_ids: List[str], seed: int, train_ratio: float, val_ratio: float, test_ratio: float):
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-8:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")
    ids = sorted(set(subject_ids))
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    if n_train + n_val > n:
        n_val = max(0, n - n_train)
    return {
        "train": set(ids[:n_train]),
        "val": set(ids[n_train:n_train + n_val]),
        "test": set(ids[n_train + n_val:]),
    }


def open_lmdb(path: Path, map_size: int):
    path.mkdir(parents=True, exist_ok=True)
    return lmdb.open(str(path), map_size=map_size, subdir=True)


def save_to_lmdb(db, key: str, value: dict):
    with db.begin(write=True) as txn:
        txn.put(key.encode(), pickle.dumps(value))


def pick_and_standardize_eeg(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    ch_names_raw = [canonicalize_name(ch) for ch in raw.ch_names]
    idx_map = {}
    for idx, ch in enumerate(ch_names_raw):
        if ch in {"A1", "A2"}:
            continue
        if ch in SELECTED_CHANNELS and ch not in idx_map:
            idx_map[ch] = idx

    if len(idx_map) == len(SELECTED_CHANNELS):
        picks = [idx_map[ch] for ch in SELECTED_CHANNELS]
        raw.pick(picks)
        raw.rename_channels({old: canonicalize_name(old) for old in raw.ch_names})
        raw.reorder_channels(SELECTED_CHANNELS)
        return raw

    if len(raw.ch_names) >= 32:
        raw.pick(raw.ch_names[:32])
        raw.rename_channels({old: PAPER_EEG32[i] for i, old in enumerate(raw.ch_names)})
        raw.drop_channels(["A1", "A2"])
        raw.rename_channels({old: canonicalize_name(old) for old in raw.ch_names})
        raw.reorder_channels(SELECTED_CHANNELS)
        return raw

    raise ValueError(f"Cannot identify SADT EEG channels: {raw.ch_names}")


def parse_events_from_annotations(raw: mne.io.BaseRaw):
    events = []
    for ann in raw.annotations:
        desc = str(ann["description"]).strip()
        if desc in DEVIATION_CODES or desc == RESPONSE_ONSET_CODE or desc == RESPONSE_OFFSET_CODE:
            sample = int(round(ann["onset"] * raw.info["sfreq"]))
            events.append((sample, desc))
    events.sort(key=lambda x: x[0])
    return events


def build_trials_from_events(events, sfreq: float):
    trials = []
    i = 0
    n = len(events)
    while i < n:
        sample_i, code_i = events[i]
        if code_i not in DEVIATION_CODES:
            i += 1
            continue

        deviation_sample = sample_i
        deviation_code = code_i
        response_onset_sample = None
        response_offset_sample = None
        j = i + 1
        while j < n:
            sample_j, code_j = events[j]
            if code_j in DEVIATION_CODES:
                break
            if code_j == RESPONSE_ONSET_CODE and response_onset_sample is None:
                response_onset_sample = sample_j
            elif code_j == RESPONSE_OFFSET_CODE and response_onset_sample is not None:
                response_offset_sample = sample_j
                break
            j += 1

        if response_onset_sample is not None and response_onset_sample > deviation_sample:
            trials.append(
                {
                    "deviation_sample": int(deviation_sample),
                    "response_onset_sample": int(response_onset_sample),
                    "response_offset_sample": None if response_offset_sample is None else int(response_offset_sample),
                    "rt_sec": float((response_onset_sample - deviation_sample) / sfreq),
                    "deviation_code": deviation_code,
                }
            )
        i = j
    return trials


def assign_labels_from_rt(trials, sfreq: float, baseline_minutes: float = 10.0):
    baseline_limit = int(round(baseline_minutes * 60.0 * sfreq))
    baseline_rts = [t["rt_sec"] for t in trials if t["deviation_sample"] <= baseline_limit]
    if len(baseline_rts) < 5:
        baseline_rts = [t["rt_sec"] for t in trials]
    if len(baseline_rts) == 0:
        raise ValueError("No valid RTs found to build baseline distribution.")

    mu = float(np.mean(baseline_rts))
    std = float(np.std(baseline_rts))
    if std < 1e-6:
        std = 1e-6

    for t in trials:
        rt_z = (t["rt_sec"] - mu) / std
        t["rt_z"] = float(rt_z)
        t["baseline_mu_rt"] = mu
        t["baseline_std_rt"] = std
        if rt_z <= 1.0:
            label_3cls = 0
        elif rt_z <= 2.0:
            label_3cls = 1
        else:
            label_3cls = 2
        t["label_3cls"] = int(label_3cls)
        t["label_bin_loose"] = 0 if label_3cls == 0 else 1
        t["label_bin_clean"] = 0 if label_3cls == 0 else (1 if label_3cls == 2 else -1)

    return trials, mu, std


def extract_trial_windows(signals, trials, sfreq: float, pre_onset_sec: float, post_onset_sec: float):
    samples = []
    total_len = signals.shape[1]
    pre_n = int(round(pre_onset_sec * sfreq))
    post_n = int(round(post_onset_sec * sfreq))
    win_len = pre_n + post_n
    for t in trials:
        onset = t["deviation_sample"]
        start = onset - pre_n
        end = onset + post_n
        if start < 0 or end > total_len or end <= start:
            continue
        x = signals[:, start:end]
        if x.shape[1] != win_len:
            continue
        rec = dict(t)
        rec["sample"] = x.astype(np.float32)
        rec["window_start_sample"] = int(start)
        rec["window_end_sample"] = int(end)
        samples.append(rec)
    return samples


def preprocess_recording(set_path: Path, target_fs: int, l_freq: float, h_freq: float, pre_onset_sec: float, post_onset_sec: float, baseline_minutes: float):
    raw = mne.io.read_raw_eeglab(str(set_path), preload=True, verbose="ERROR")
    raw = pick_and_standardize_eeg(raw)
    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose="ERROR")
    if int(round(raw.info["sfreq"])) != int(target_fs):
        raw.resample(target_fs, verbose="ERROR")

    signals = raw.get_data(units="uV").astype(np.float32)
    events = parse_events_from_annotations(raw)
    trials = build_trials_from_events(events, float(raw.info["sfreq"]))
    trials, baseline_mu_rt, baseline_std_rt = assign_labels_from_rt(trials, float(raw.info["sfreq"]), baseline_minutes)
    samples = extract_trial_windows(signals, trials, float(raw.info["sfreq"]), pre_onset_sec, post_onset_sec)
    raw.close()
    return samples, baseline_mu_rt, baseline_std_rt


def main():
    parser = argparse.ArgumentParser(description="Preprocess SADT .set files for downstream vigilance classification")
    parser.add_argument("--raw", type=str, required=True)
    parser.add_argument("--processed", type=str, required=True)
    parser.add_argument("--target_fs", type=int, default=100)
    parser.add_argument("--l_freq", type=float, default=1.0)
    parser.add_argument("--h_freq", type=float, default=45.0)
    parser.add_argument("--pre_onset_sec", type=float, default=5.0)
    parser.add_argument("--post_onset_sec", type=float, default=0.0)
    parser.add_argument("--baseline_minutes", type=float, default=10.0)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--map_size_gb", type=float, default=8.0)
    args = parser.parse_args()

    root_dir = Path(args.raw)
    out_dir = Path(args.processed)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_files = sorted(root_dir.glob("*.set"))
    if len(set_files) == 0:
        raise ValueError(f"No .set files found in {root_dir}")

    subject_ids = [parse_subject_and_session(p)[0] for p in set_files]
    split_subject_map = split_subjects(subject_ids, args.seed, args.train_ratio, args.val_ratio, args.test_ratio)

    db = open_lmdb(out_dir, map_size=int(args.map_size_gb * (1024 ** 3)))
    dataset_keys = {"train": [], "val": [], "test": []}
    split_file_counts = {k: 0 for k in dataset_keys}
    split_window_counts = {k: 0 for k in dataset_keys}
    split_label3_counts = {k: {0: 0, 1: 0, 2: 0} for k in dataset_keys}
    split_bin_loose_counts = {k: {0: 0, 1: 0} for k in dataset_keys}
    split_bin_clean_counts = {k: {0: 0, 1: 0, -1: 0} for k in dataset_keys}

    print("SADT preprocessing configuration")
    print(f"Files found          : {len(set_files)}")
    print(f"Subjects found       : {len(set(subject_ids))}")
    print(f"Selected channels    : {len(SELECTED_CHANNELS)}")
    print(f"Final sampling rate  : {args.target_fs} Hz")
    print(f"Window               : [{-args.pre_onset_sec:.1f}, +{args.post_onset_sec:.1f}] sec around deviation onset")
    print("Labels:")
    print("  label_3cls     : 0=optimal, 1=suboptimal, 2=poor")
    print("  label_bin_loose: 0=optimal, 1=suboptimal_or_poor")
    print("  label_bin_clean: 0=optimal, 1=poor, -1=suboptimal_ignore")

    for set_path in set_files:
        subject_id, session_id = parse_subject_and_session(set_path)
        split = "train" if subject_id in split_subject_map["train"] else ("val" if subject_id in split_subject_map["val"] else "test")
        split_file_counts[split] += 1

        try:
            samples, baseline_mu_rt, baseline_std_rt = preprocess_recording(
                set_path=set_path,
                target_fs=args.target_fs,
                l_freq=args.l_freq,
                h_freq=args.h_freq,
                pre_onset_sec=args.pre_onset_sec,
                post_onset_sec=args.post_onset_sec,
                baseline_minutes=args.baseline_minutes,
            )
        except Exception as e:
            print(f"[skip] {set_path.name}: {e}")
            continue

        print(f"{set_path.name} -> split={split}, trials={len(samples)}, baseline_mu={baseline_mu_rt:.3f}s, baseline_std={baseline_std_rt:.3f}s")

        for i, rec in enumerate(samples):
            key = f"{session_id}__{i}"
            payload = {
                "sample": rec["sample"],
                "label": int(rec["label_3cls"]),
                "label_3cls": int(rec["label_3cls"]),
                "label_bin_loose": int(rec["label_bin_loose"]),
                "label_bin_clean": int(rec["label_bin_clean"]),
                "rt_sec": float(rec["rt_sec"]),
                "rt_z": float(rec["rt_z"]),
                "deviation_code": rec["deviation_code"],
                "subject_id": subject_id,
                "session_id": session_id,
                "fs": int(args.target_fs),
                "ch_names": list(SELECTED_CHANNELS),
                "source_file": str(set_path),
                "window_start_sample": int(rec["window_start_sample"]),
                "window_end_sample": int(rec["window_end_sample"]),
                "baseline_mu_rt": float(rec["baseline_mu_rt"]),
                "baseline_std_rt": float(rec["baseline_std_rt"]),
            }
            save_to_lmdb(db, key, payload)
            dataset_keys[split].append(key)
            split_window_counts[split] += 1
            split_label3_counts[split][int(rec["label_3cls"])] += 1
            split_bin_loose_counts[split][int(rec["label_bin_loose"])] += 1
            split_bin_clean_counts[split][int(rec["label_bin_clean"])] += 1

    meta = {
        "dataset_name": "SADT / Multi-channel EEG recordings during a sustained-attention driving task",
        "split_mode": "subject_wise",
        "selected_channels": list(SELECTED_CHANNELS),
        "target_fs": int(args.target_fs),
        "bandpass": [float(args.l_freq), float(args.h_freq)],
        "window_sec": {"pre_onset_sec": float(args.pre_onset_sec), "post_onset_sec": float(args.post_onset_sec)},
        "baseline_minutes": float(args.baseline_minutes),
        "label_definition": {
            "label_3cls": {
                "0": "optimal",
                "1": "suboptimal",
                "2": "poor",
                "rule": "session-specific RT z-score relative to first 10 min baseline; z<=1 optimal, 1<z<=2 suboptimal, z>2 poor",
            },
            "label_bin_loose": {"0": "optimal", "1": "suboptimal_or_poor"},
            "label_bin_clean": {"0": "optimal", "1": "poor", "-1": "suboptimal_ignore"},
        },
        "subject_split": {k: sorted(list(v)) for k, v in split_subject_map.items()},
        "file_counts": split_file_counts,
        "window_counts": split_window_counts,
        "label3_counts": split_label3_counts,
        "label_bin_loose_counts": split_bin_loose_counts,
        "label_bin_clean_counts": split_bin_clean_counts,
    }

    save_to_lmdb(db, "__keys__", dataset_keys)
    save_to_lmdb(db, "__meta__", meta)
    db.close()

    print("\\nDone.")
    for split in ("train", "val", "test"):
        print(
            f"{split}: files={split_file_counts[split]}, windows={split_window_counts[split]}, "
            f"3cls={split_label3_counts[split]}, "
            f"bin_loose={split_bin_loose_counts[split]}, "
            f"bin_clean={split_bin_clean_counts[split]}"
        )


if __name__ == "__main__":
    main()
