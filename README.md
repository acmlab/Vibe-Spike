# Downloading the REVE Dataset

This repository provides a helper script for downloading the REVE EEG dataset from Hugging Face:

- Dataset page: `https://huggingface.co/datasets/brain-bzh/reve-dataset`
- Download script: `download_reve_dataset_clean.py`

The script supports three download modes:

1. `small`: download a predefined small subset for quick tests and debugging.
2. `indices`: download selected recording IDs.
3. `all`: download the full REVE dataset repository.

## Installation

Install the required Python package:

```bash
pip install huggingface_hub
```

If the dataset requires authentication in your environment, log in to Hugging Face first:

```bash
huggingface-cli login
```

Alternatively, pass a token directly with `--token`.

## Quick Start

### 1. Download the small subset

Use this mode for testing the data loader or checking the directory structure.

```bash
python download_reve_dataset.py \
  --mode small \
  --output-dir ./reve_dataset_small
```

This downloads the predefined small subset used by the script, including EEG data, electrode positions, statistics files, and metadata CSV files.

### 2. Download selected recordings

Use this mode when only specific recording indices are needed.

```bash
python download_reve_dataset.py \
  --mode indices \
  --indices 82 121 127 145 \
  --output-dir ./reve_dataset_subset
```

For each requested recording index, the script attempts to download:

- `recording_-_eeg_-_<ID>.npy`
- `recording_-_positions_-_<ID>.npy`
- `recording_-_stats_-_<ID>.npy`
- available metadata CSV files

### 3. Download the full REVE dataset

Use this mode to download all files in the Hugging Face dataset repository.

```bash
python download_reve_dataset.py \
  --mode all \
  --output-dir ./reve_dataset_all
```

The full dataset is large. Make sure the target disk has sufficient free space before running this command.

## Optional Arguments

```bash
python download_reve_dataset.py \
  --mode small \
  --output-dir ./reve_dataset_small \
  --max-workers 8
```

Available arguments:

| Argument | Description | Default |
|---|---|---|
| `--mode` | Download mode: `small`, `indices`, or `all`. | Required |
| `--output-dir` | Local directory for downloaded files. | `reve_dataset` |
| `--repo-id` | Hugging Face dataset repository ID. | `brain-bzh/reve-dataset` |
| `--indices` | Recording IDs used when `--mode indices` is selected. | `None` |
| `--max-workers` | Number of parallel download workers. | `8` |
| `--token` | Optional Hugging Face access token. | `None` |

## Python API Usage

The script can also be imported as a Python module.

### Download the small subset

```python
from download_reve_dataset import demo_download_small

demo_download_small(output_dir="./reve_dataset_small")
```

### Download the full dataset

```python
from download_reve_dataset import demo_download_all

demo_download_all(output_dir="./reve_dataset_all")
```

### Download selected indices

```python
from download_reve_dataset import download_reve_dataset

download_reve_dataset(
    output_dir="./reve_dataset_subset",
    mode="indices",
    indices=[82, 121, 127, 145],
)
```

## Expected Output Structure

The exact directory structure follows the Hugging Face repository layout. Depending on the selected mode, the downloaded directory may include files such as:

```text
reve_dataset_small/
├── csv_recordings/
│   ├── df_big.csv
│   ├── df_corrected.csv
│   ├── df_stats.csv
│   └── df_stats_tmp.csv
├── recordings/
│   ├── recording_-_eeg_-_72.npy
│   └── ...
├── positions/
│   ├── recording_-_positions_-_72.npy
│   └── ...
└── stats/
    ├── recording_-_stats_-_72.npy
    └── ...
```

## Notes

- The `small` mode is intended for quick testing and does not represent the full dataset.
- The `all` mode downloads the complete Hugging Face dataset repository and may require substantial storage.
- The script uses `huggingface_hub.snapshot_download` internally, so previously downloaded files are cached by Hugging Face and reused when possible.
- If download fails due to authentication or network issues, verify your Hugging Face login status or provide `--token`.

## Citation / Dataset Acknowledgement

If you use REVE in your work, please cite the original REVE dataset/model source and follow the usage terms provided on the Hugging Face dataset page:

```text
https://huggingface.co/datasets/brain-bzh/reve-dataset
```
