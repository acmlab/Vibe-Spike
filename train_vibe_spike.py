import csv
import math
import random
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration
from omegaconf import DictConfig, OmegaConf
from openpyxl import load_workbook
from torch.optim import AdamW
from tqdm.auto import tqdm

from data.build_train_val_dataloader import get_train_val_loaders
from models.vibe_spike_pretrain_task import VibeSpikePretrainTask


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_target_coords_from_xlsx(xlsx_path: str, sheet_name=0) -> torch.Tensor:
    wb = load_workbook(xlsx_path, data_only=True)
    ws = wb[wb.sheetnames[sheet_name]] if isinstance(sheet_name, int) else wb[sheet_name]

    rows = list(ws.iter_rows(values_only=True))
    if len(rows) < 2:
        raise ValueError(f"No valid rows found in {xlsx_path}")

    header = [str(x).strip() if x is not None else "" for x in rows[0]]
    col_map = {name: idx for idx, name in enumerate(header)}

    required = ["Channel", "X", "Y", "Z"]
    for key in required:
        if key not in col_map:
            raise ValueError(f"Missing required column '{key}' in {xlsx_path}. Found header={header}")

    coords = []
    for row in rows[1:]:
        if row is None:
            continue
        ch = row[col_map["Channel"]]
        if ch is None:
            continue
        coords.append([
            float(row[col_map["X"]]),
            float(row[col_map["Y"]]),
            float(row[col_map["Z"]]),
        ])

    return torch.tensor(coords, dtype=torch.float32)


def ensure_bct_and_bcp(batch: Any, device: torch.device):
    if not isinstance(batch, (tuple, list)) or len(batch) < 3:
        raise ValueError(f"Expected batch like (x, pos, b_m, ...), got {type(batch)}")

    x = batch[0].to(device, non_blocking=True)
    pos = batch[1].to(device, non_blocking=True)
    b_m = batch[2]

    if x.ndim != 3:
        raise ValueError(f"Expected x 3D, got {tuple(x.shape)}")
    if pos.ndim != 3 or pos.shape[-1] != 3:
        raise ValueError(f"Expected pos [B, C, 3], got {tuple(pos.shape)}")

    B, C_pos, _ = pos.shape
    if x.shape[0] != B:
        raise ValueError(f"Batch mismatch: x.shape={tuple(x.shape)}, pos.shape={tuple(pos.shape)}")

    if x.shape[1] == C_pos:
        x_bct = x
    elif x.shape[2] == C_pos:
        x_bct = x.permute(0, 2, 1).contiguous()
    else:
        raise ValueError(f"Cannot align x with pos. x.shape={tuple(x.shape)}, pos.shape={tuple(pos.shape)}")

    return x_bct, pos, b_m


def zero_all_dropout(model: torch.nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0


class WarmupStableDecay:
    def __init__(self, optimizer, total_steps: int, warmup_steps: int, stable_steps: int, min_lr: float):
        self.optimizer = optimizer
        self.total_steps = max(int(total_steps), 1)
        self.warmup_steps = max(int(warmup_steps), 0)
        self.stable_steps = max(int(stable_steps), 0)
        self.min_lr = float(min_lr)
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.step_num = 0

    def state_dict(self):
        return {
            "step_num": self.step_num,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "stable_steps": self.stable_steps,
            "min_lr": self.min_lr,
            "base_lrs": self.base_lrs,
        }

    def load_state_dict(self, state):
        self.step_num = state["step_num"]
        self.total_steps = state["total_steps"]
        self.warmup_steps = state["warmup_steps"]
        self.stable_steps = state["stable_steps"]
        self.min_lr = state["min_lr"]
        self.base_lrs = state["base_lrs"]

    def get_last_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]

    def step(self):
        self.step_num += 1
        decay_start = self.warmup_steps + self.stable_steps
        for i, pg in enumerate(self.optimizer.param_groups):
            base_lr = self.base_lrs[i]
            if self.step_num <= self.warmup_steps and self.warmup_steps > 0:
                lr = base_lr * self.step_num / self.warmup_steps
            elif self.step_num <= decay_start:
                lr = base_lr
            else:
                remain_steps = max(self.total_steps - decay_start, 1)
                progress = min((self.step_num - decay_start) / remain_steps, 1.0)
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                lr = self.min_lr + (base_lr - self.min_lr) * cosine
            pg["lr"] = lr


def append_csv_row(csv_path: Path, row: Dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(path: Path, accelerator: Accelerator, task, optimizer, scheduler, epoch: int, step: int, cfg: DictConfig):
    path.parent.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        state = {
            "epoch": epoch,
            "step": step,
            "task_state_dict": accelerator.unwrap_model(task).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "configs": OmegaConf.to_container(cfg, resolve=True),
        }
        torch.save(state, path)
        print(f"[checkpoint] saved: {path}", flush=True)
    accelerator.wait_for_everyone()


def main(cfg: DictConfig):
    set_seed(cfg.seed)
    mixed_precision = "fp16" if cfg.train.use_amp else "no"
    accum_steps = int(getattr(cfg.train, "accumulate_grad_batches", 1))

    accelerator = Accelerator(
        gradient_accumulation_steps=accum_steps,
        mixed_precision=mixed_precision,
        dataloader_config=DataLoaderConfiguration(even_batches=False),
    )

    if accelerator.is_main_process:
        print("========== CONFIG ==========")
        print(OmegaConf.to_yaml(cfg, resolve=True))
        print("============================")

    train_loader, _, len_train, _, len_train_sampler, _ = get_train_val_loaders(
        cfg,
        return_val=False,
        n_gpu=accelerator.num_processes,
    )

    target_coords = load_target_coords_from_xlsx(cfg.model.target_coords_path)
    task = VibeSpikePretrainTask(cfg, target_coords)

    if cfg.train.zero_dropout_for_debug:
        zero_all_dropout(task)

    optimizer = AdamW(
        task.parameters(),
        lr=cfg.optimizer.lr,
        betas=tuple(cfg.optimizer.betas),
        eps=cfg.optimizer.eps,
        weight_decay=cfg.optimizer.weight_decay,
    )

    updates_per_epoch = math.ceil(len(train_loader) / max(accum_steps, 1))
    total_steps = updates_per_epoch * cfg.train.max_epochs
    warmup_steps = int(getattr(cfg.scheduler, "warmup_steps", 0))
    if warmup_steps <= 0 and hasattr(cfg.scheduler, "warmup_epochs"):
        warmup_steps = int(cfg.scheduler.warmup_epochs * updates_per_epoch)

    scheduler = WarmupStableDecay(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        stable_steps=int(getattr(cfg.scheduler, "stable_steps", 0)),
        min_lr=cfg.scheduler.min_lr,
    )

    task, optimizer, train_loader = accelerator.prepare(task, optimizer, train_loader)

    save_dir = Path(cfg.save.dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    step_csv = save_dir / "step_metrics.csv"
    epoch_csv = save_dir / "epoch_metrics.csv"
    last_ckpt = save_dir / f"{cfg.save.name}_last.pt"

    if accelerator.is_main_process:
        base_task = accelerator.unwrap_model(task)
        total_params_m = sum(p.numel() for p in base_task.parameters()) / 1e6

        encoder_params = 0
        encoder_params += sum(p.numel() for p in base_task.model.project.parameters())
        encoder_params += sum(p.numel() for p in base_task.model.s2block.parameters())
        encoder_params += base_task.model.channel_missing_token.numel()
        encoder_params_m = encoder_params / 1e6

        decoder_params = 0
        decoder_params += sum(p.numel() for p in base_task.model.decoder.parameters())
        decoder_params += base_task.model.decoder_block_token.numel()
        decoder_params_m = decoder_params / 1e6

        print(f"device={accelerator.device}")
        print(f"accelerator.num_processes={accelerator.num_processes}")
        print(f"accelerator.distributed_type={accelerator.distributed_type}")
        print(f"train segments={len_train}, train batches={len(train_loader)}, train sampler batches={len_train_sampler}")
        print(f"updates/epoch={updates_per_epoch}, total_steps={total_steps}")
        print(f"model params={total_params_m:.4f} M")
        print(f"model encoder params={encoder_params_m:.4f} M")
        print(f"model decoder params={decoder_params_m:.4f} M")
        print(f"task.graph_type={getattr(base_task, 'graph_type', 'NA')}")

    global_update_step = 0
    loss_ema = None
    log_window = {"loss_total": [], "loss_masked": [], "loss_consistency": []}

    for epoch in range(1, cfg.train.max_epochs + 1):
        task.train()
        epoch_losses = {"loss_total": [], "loss_masked": [], "loss_consistency": []}
        epoch_start = time.time()

        progress = tqdm(train_loader, disable=not accelerator.is_local_main_process, desc=f"epoch {epoch:03d}", leave=False)

        for batch_idx, batch in enumerate(progress, start=1):
            x_src, input_coords, b_m = ensure_bct_and_bcp(batch, accelerator.device)
            if epoch == 1 and batch_idx == 1 and accelerator.is_main_process:
                print(f"x_src.shape={tuple(x_src.shape)}")
                print(f"input_coords.shape={tuple(input_coords.shape)}")

            with accelerator.accumulate(task):
                with accelerator.autocast():
                    loss, stats = task(x_src, input_coords, b_m)

                accelerator.backward(loss)

                if accelerator.sync_gradients and cfg.train.grad_clip is not None:
                    accelerator.clip_grad_norm_(task.parameters(), cfg.train.grad_clip)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    scheduler.step()
                    global_update_step += 1

                    current_loss_total = float(accelerator.gather_for_metrics(stats["loss_total"]).mean().item())
                    current_loss_masked = float(accelerator.gather_for_metrics(stats["loss_masked"]).mean().item())
                    current_loss_consistency = float(accelerator.gather_for_metrics(stats["loss_consistency"]).mean().item())
                    current_lr = scheduler.get_last_lr()[0]

                    loss_ema = current_loss_total if loss_ema is None else 0.95 * loss_ema + 0.05 * current_loss_total
                    epoch_losses["loss_total"].append(current_loss_total)
                    epoch_losses["loss_masked"].append(current_loss_masked)
                    epoch_losses["loss_consistency"].append(current_loss_consistency)

                    log_window["loss_total"].append(current_loss_total)
                    log_window["loss_masked"].append(current_loss_masked)
                    log_window["loss_consistency"].append(current_loss_consistency)

                    if (cfg.save.save_every_steps is not None) and (global_update_step % cfg.save.save_every_steps == 0):
                        step_ckpt = save_dir / f"{cfg.save.name}_step_{global_update_step:08d}.pt"
                        save_checkpoint(step_ckpt, accelerator, task, optimizer, scheduler, epoch, global_update_step, cfg)

                    if global_update_step % cfg.train.log_every_steps == 0:
                        avg_loss_total = float(np.mean(log_window["loss_total"])) if log_window["loss_total"] else 0.0
                        avg_loss_masked = float(np.mean(log_window["loss_masked"])) if log_window["loss_masked"] else 0.0
                        avg_loss_consistency = float(np.mean(log_window["loss_consistency"])) if log_window["loss_consistency"] else 0.0

                        log_row = {
                            "epoch": epoch,
                            "update_step": global_update_step,
                            "batch_idx": batch_idx,
                            "lr": current_lr,
                            "loss_total_step": current_loss_total,
                            "loss_masked_step": current_loss_masked,
                            "loss_consistency_step": current_loss_consistency,
                            "loss_ema_step": float(loss_ema),
                            "loss_total_avg": avg_loss_total,
                            "loss_masked_avg": avg_loss_masked,
                            "loss_consistency_avg": avg_loss_consistency,
                            "log_window_size": len(log_window["loss_total"]),
                        }

                        if accelerator.is_main_process:
                            append_csv_row(step_csv, log_row)

                        progress.set_postfix(step=global_update_step, lr=f"{current_lr:.2e}", loss=f"{avg_loss_total:.4f}")
                        log_window["loss_total"].clear()
                        log_window["loss_masked"].clear()
                        log_window["loss_consistency"].clear()

        epoch_row = {
            "epoch": epoch,
            "update_step_end": global_update_step,
            "lr_end": scheduler.get_last_lr()[0],
            "loss_total": float(np.mean(epoch_losses["loss_total"])) if epoch_losses["loss_total"] else 0.0,
            "loss_masked": float(np.mean(epoch_losses["loss_masked"])) if epoch_losses["loss_masked"] else 0.0,
            "loss_consistency": float(np.mean(epoch_losses["loss_consistency"])) if epoch_losses["loss_consistency"] else 0.0,
            "epoch_time_sec": time.time() - epoch_start,
        }

        if accelerator.is_main_process:
            append_csv_row(epoch_csv, epoch_row)

        if (cfg.save.save_every_epochs is not None) and (epoch % cfg.save.save_every_epochs == 0):
            epoch_ckpt = save_dir / f"{cfg.save.name}_epoch_{epoch:04d}.pt"
            save_checkpoint(epoch_ckpt, accelerator, task, optimizer, scheduler, epoch, global_update_step, cfg)

        if cfg.save.save_last:
            save_checkpoint(last_ckpt, accelerator, task, optimizer, scheduler, epoch, global_update_step, cfg)

        if accelerator.is_main_process:
            print(
                f"[Epoch {epoch:03d}] "
                f"loss_total={epoch_row['loss_total']:.4f} "
                f"loss_masked={epoch_row['loss_masked']:.4f} "
                f"loss_consistency={epoch_row['loss_consistency']:.4f} "
                f"lr={epoch_row['lr_end']:.2e} "
                f"time={epoch_row['epoch_time_sec']:.1f}s"
            )

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        base_task = accelerator.unwrap_model(task)
        final_task_path = save_dir / f"{cfg.save.name}_final_task.pt"
        final_model_path = save_dir / f"{cfg.save.name}_final_model.pt"
        torch.save(base_task.state_dict(), final_task_path)
        torch.save(base_task.model.state_dict(), final_model_path)
        print(f"Saved final task to {final_task_path}")
        print(f"Saved final model to {final_model_path}")

    accelerator.end_training()


if __name__ == "__main__":
    cfg = OmegaConf.load("configs/train_vibe_spike.yaml")
    main(cfg)
