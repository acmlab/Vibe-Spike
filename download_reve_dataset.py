"""Utilities for downloading the REVE EEG dataset from Hugging Face.

Dataset source:
    https://huggingface.co/datasets/brain-bzh/reve-dataset

The ``output_dir`` argument is the dataset root directory. Files keep the
relative paths used in the Hugging Face repository. Therefore, when the REVE
repository contains files under ``data/``, setting ``--output-dir reve_dataset``
will create a local layout such as::

    reve_dataset/
    ├── data/
    │   ├── recording_-_eeg_-_79.npy
    │   └── ...
    ├── positions/
    ├── stats/
    └── csv_recordings/ or metadata CSV files

Recommended usage:
    python download_reve_dataset.py --mode small --output-dir ./reve_dataset
    python download_reve_dataset.py --mode all --output-dir ./reve_dataset
    python download_reve_dataset.py --mode indices --indices 82 121 127 145 --output-dir ./reve_dataset
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

from huggingface_hub import HfApi, snapshot_download


REVE_REPO_ID = "brain-bzh/reve-dataset"

# A minimal subset for quick code-path testing.
SMALL_TRAIN_IDS = [74, 79, 80]
SMALL_VAL_IDS: List[int] = []
SMALL_IDS = sorted(set(SMALL_TRAIN_IDS + SMALL_VAL_IDS))

METADATA_BASENAMES = {
    "df_big.csv",
    "df_corrected.csv",
    "df_stats.csv",
    "df_stats_tmp.csv",
}

PathLike = Union[str, Path]
BasenameIndex = Dict[str, List[str]]


def eeg_basename(index: int) -> str:
    return f"recording_-_eeg_-_{index}.npy"


def positions_basename(index: int) -> str:
    return f"recording_-_positions_-_{index}.npy"


def stats_basename(index: int) -> str:
    return f"recording_-_stats_-_{index}.npy"


def build_basename_index(repo_files: Sequence[str]) -> BasenameIndex:
    basename_index: BasenameIndex = defaultdict(list)
    for remote_path in repo_files:
        basename_index[os.path.basename(remote_path)].append(remote_path)
    return dict(basename_index)


def resolve_preferred_path(
    basename_index: BasenameIndex,
    basename: str,
    preferred_prefixes: Optional[Sequence[str]] = None,
) -> str:
    candidates = basename_index.get(basename, [])
    if not candidates:
        raise FileNotFoundError(f"Remote file not found: {basename}")

    if preferred_prefixes is not None:
        for prefix in preferred_prefixes:
            for remote_path in candidates:
                if remote_path.startswith(prefix):
                    return remote_path

    return candidates[0]


def collect_metadata_paths(basename_index: BasenameIndex) -> List[str]:
    paths: List[str] = []
    for basename in sorted(METADATA_BASENAMES):
        if basename in basename_index:
            paths.append(
                resolve_preferred_path(
                    basename_index=basename_index,
                    basename=basename,
                    preferred_prefixes=("csv_recordings/", ""),
                )
            )
    return paths


def collect_paths_for_indices(indices: Iterable[int], basename_index: BasenameIndex) -> List[str]:
    selected_paths = set(collect_metadata_paths(basename_index))

    for index in sorted(set(int(i) for i in indices)):
        selected_paths.add(
            resolve_preferred_path(
                basename_index=basename_index,
                basename=eeg_basename(index),
                preferred_prefixes=("data/", "recordings/", "dataloader/", ""),
            )
        )
        selected_paths.add(
            resolve_preferred_path(
                basename_index=basename_index,
                basename=positions_basename(index),
                preferred_prefixes=("positions/", "data/", ""),
            )
        )
        selected_paths.add(
            resolve_preferred_path(
                basename_index=basename_index,
                basename=stats_basename(index),
                preferred_prefixes=("stats/", "data/", ""),
            )
        )

    return sorted(selected_paths)


def list_repo_files(repo_id: str = REVE_REPO_ID, token: Optional[str] = None) -> List[str]:
    api = HfApi(token=token)
    return api.list_repo_files(repo_id=repo_id, repo_type="dataset")


def _print_download_plan(
    repo_id: str,
    mode: str,
    output_path: Path,
    selected_count: int,
    indices: Optional[Sequence[int]] = None,
) -> None:
    print("=" * 80)
    print(f"Repository       : {repo_id}")
    print(f"Mode             : {mode}")
    print(f"Dataset root     : {output_path.resolve()}")
    print(f"Selected files   : {selected_count}")
    if mode == "small":
        print(f"Small subset IDs : {SMALL_IDS}")
    if mode == "indices":
        print(f"Requested IDs    : {sorted(set(int(i) for i in indices or []))}")
    if output_path.name.lower() == "data":
        print("[warning] --output-dir is the dataset root, not the recordings subfolder.")
        print("[warning] If the remote repository contains files under data/, this will create data/data/.")
        print("[warning] Recommended: --output-dir ./reve_dataset")
    print("=" * 80)


def download_reve_dataset(
    output_dir: PathLike = "reve_dataset",
    mode: str = "small",
    repo_id: str = REVE_REPO_ID,
    indices: Optional[Sequence[int]] = None,
    max_workers: int = 8,
    token: Optional[str] = None,
) -> List[str]:
    """Download REVE files from Hugging Face.

    Args:
        output_dir: Dataset root directory. Remote relative paths are preserved
            under this directory. Use ``reve_dataset`` rather than ``data`` if
            you want the downloaded recordings to appear under
            ``reve_dataset/data``.
        mode: One of ``small``, ``indices``, or ``all``.
        repo_id: Hugging Face dataset repository ID.
        indices: Recording indices used when ``mode='indices'``.
        max_workers: Number of parallel download workers.
        token: Optional Hugging Face access token.

    Returns:
        Selected remote file paths. For ``mode='all'``, this is the full
        repository file list.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    mode = mode.lower().strip()
    selected_paths: Optional[List[str]]

    repo_files = list_repo_files(repo_id=repo_id, token=token)

    if mode == "all":
        selected_paths = None
        selected_count = len(repo_files)
    elif mode in {"small", "indices"}:
        basename_index = build_basename_index(repo_files)
        if mode == "small":
            selected_paths = collect_paths_for_indices(SMALL_IDS, basename_index)
        else:
            if indices is None or len(indices) == 0:
                raise ValueError("`indices` must be provided when mode='indices'.")
            selected_paths = collect_paths_for_indices(indices, basename_index)
        selected_count = len(selected_paths)
    else:
        raise ValueError("mode must be one of: 'small', 'indices', or 'all'.")

    _print_download_plan(
        repo_id=repo_id,
        mode=mode,
        output_path=output_path,
        selected_count=selected_count,
        indices=indices,
    )

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(output_path),
        allow_patterns=selected_paths,
        max_workers=max_workers,
        token=token,
    )

    print("Download finished.")
    return repo_files if mode == "all" else list(selected_paths or [])


def demo_download_small(
    output_dir: PathLike = "reve_dataset",
    repo_id: str = REVE_REPO_ID,
    max_workers: int = 8,
    token: Optional[str] = None,
) -> List[str]:
    """Download the predefined small REVE subset to a dataset root directory."""
    return download_reve_dataset(
        output_dir=output_dir,
        mode="small",
        repo_id=repo_id,
        indices=None,
        max_workers=max_workers,
        token=token,
    )


def demo_download_all(
    output_dir: PathLike = "reve_dataset",
    repo_id: str = REVE_REPO_ID,
    max_workers: int = 8,
    token: Optional[str] = None,
) -> List[str]:
    """Download the full REVE dataset repository to a dataset root directory."""
    return download_reve_dataset(
        output_dir=output_dir,
        mode="all",
        repo_id=repo_id,
        indices=None,
        max_workers=max_workers,
        token=token,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the REVE EEG dataset from Hugging Face Hub."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=("small", "indices", "all"),
        required=True,
        help="Download mode.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reve_dataset",
        help="Dataset root directory. Recommended: ./reve_dataset",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=REVE_REPO_ID,
        help="Hugging Face dataset repository ID.",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="*",
        default=None,
        help="Recording indices used when mode='indices'.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Number of parallel download workers.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Optional Hugging Face token.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download_reve_dataset(
        output_dir=args.output_dir,
        mode=args.mode,
        repo_id=args.repo_id,
        indices=args.indices,
        max_workers=args.max_workers,
        token=args.token,
    )


if __name__ == "__main__":
    main()
