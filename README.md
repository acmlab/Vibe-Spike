# Vibe-Spike

Vibe-Spike is an EEG foundation-model pretraining codebase built around time-frequency masked reconstruction and a biologically inspired neural coherence encoder. The repository includes utilities for downloading the open REVE EEG dataset subset from Hugging Face, building REVE-style dataloaders, and launching masked pretraining.

> REVE dataset source: <https://huggingface.co/datasets/brain-bzh/reve-dataset>

## Repository structure

```text
.
├── configs/
│   └── train_vibe_spike.yaml          # Training configuration
├── data/
│   └── build_train_val_dataloader.py  # REVE-style dataloader utilities
├── models/
│   └── ...                            # Vibe-Spike model components
├── utils/
│   └── ...                            # Utility functions
├── download_reve_dataset.py           # REVE dataset download helper
├── train_vibe_spike.py                # Main pretraining script
├── requirements.txt                   # Python dependencies excluding PyTorch
└── setup_env.sh                       # Example environment setup script
```

Large datasets, model checkpoints, logs, and generated outputs should not be committed to this repository.

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

Install PyTorch according to your CUDA version. For example, for CUDA 12.8:

```bash
python -m pip install --upgrade pip setuptools wheel
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

The provided `setup_env.sh` contains the same basic setup flow and can be adapted to your system.

## Download the REVE dataset

The downloader preserves the relative paths used in the Hugging Face dataset repository. Therefore, `--output-dir` should be the **dataset root directory**, not the internal `data/` subfolder.

For a quick test, download the small subset:

```bash
python download_reve_dataset.py --mode small --output-dir ./reve_dataset
```

Expected local layout:

```text
reve_dataset/
├── data/
│   ├── recording_-_eeg_-_74.npy
│   ├── recording_-_eeg_-_79.npy
│   └── recording_-_eeg_-_80.npy
├── positions/
│   ├── recording_-_positions_-_74.npy
│   ├── recording_-_positions_-_79.npy
│   └── recording_-_positions_-_80.npy
├── stats/
│   ├── recording_-_stats_-_74.npy
│   ├── recording_-_stats_-_79.npy
│   └── recording_-_stats_-_80.npy
└── csv_recordings/
    ├── df_big.csv
    ├── df_corrected.csv
    └── df_stats.csv or df_stats_tmp.csv
```

Do **not** use `--output-dir ./data` unless you intentionally want a nested layout such as `data/data/`.

### Download modes

Download a small subset for testing:

```bash
python download_reve_dataset.py --mode small --output-dir ./reve_dataset
```

Download selected recording IDs:

```bash
python download_reve_dataset.py \
  --mode indices \
  --indices 82 121 127 145 \
  --output-dir ./reve_dataset
```

Download the full REVE repository:

```bash
python download_reve_dataset.py --mode all --output-dir ./reve_dataset
```

Optional arguments:

```bash
python download_reve_dataset.py \
  --mode small \
  --output-dir ./reve_dataset \
  --max-workers 8 \
  --token YOUR_HF_TOKEN
```

## Configure the dataset path

After downloading the dataset, set the dataloader path in `configs/train_vibe_spike.yaml` to the dataset root:

```yaml
dataloader:
  path: reve_dataset
```

The path should point to the directory that contains `data/`, `positions/`, `stats/`, and the metadata CSV files or `csv_recordings/`.

## Pretraining

Launch pretraining with:

```bash
python train_vibe_spike.py
```

The script loads:

```text
configs/train_vibe_spike.yaml
```

The training pipeline uses Accelerate, builds train loaders through `data.build_train_val_dataloader.get_train_val_loaders`, loads target montage coordinates from the configured Excel file, and trains `VibeSpikePretrainTask` with the configured optimizer, scheduler, masking, and checkpoint settings.

For multi-GPU training, configure Accelerate first and then launch:

```bash
accelerate config
accelerate launch train_vibe_spike.py
```

Training logs and checkpoints are written to the directory specified by:

```yaml
save:
  dir: ckpt
  name: vibe-spike
```

Typical outputs include:

```text
step_metrics.csv
epoch_metrics.csv
*_step_XXXXXXXX.pt
*_epoch_XXXX.pt
*_last.pt
*_final_task.pt
*_final_model.pt
```

## Notes

- The `small` download mode is intended for code-path testing only.
- Full REVE pretraining requires substantially more disk space and compute resources.
- Dataset files, checkpoints, and generated outputs should be excluded from Git.
- If only metadata CSV files are downloaded, the dataloader will not be able to construct training samples until the corresponding EEG, position, and statistics files are available.
