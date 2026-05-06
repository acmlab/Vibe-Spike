import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _to_tensor(x, dtype=None, device=None):
    if isinstance(x, torch.Tensor):
        y = x
        if dtype is not None:
            y = y.to(dtype=dtype)
        if device is not None:
            y = y.to(device=device)
        return y
    return torch.as_tensor(x, dtype=dtype, device=device)


class ElectrodeResampler(nn.Module):
    def __init__(
        self,
        tgt_pos,
        thickness_scale=0.50,
        grid_n=52,
        cache_dir=".cache/electrode_resampler",
    ):
        super().__init__()

        tgt_pos_np = _to_numpy(tgt_pos).astype(np.float64)
        if tgt_pos_np.ndim != 2 or tgt_pos_np.shape[1] != 3:
            raise ValueError(f"tgt_pos must be (C_tgt, 3), got {tuple(tgt_pos_np.shape)}")

        self.C_tgt = int(tgt_pos_np.shape[0])
        self.thickness_scale = float(thickness_scale)
        self.grid_n = int(grid_n)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.tgt_pos = torch.as_tensor(tgt_pos_np, dtype=torch.float32)
        self._cache_key = self._make_cache_key(tgt_pos_np, self.thickness_scale, self.grid_n)
        self._cache_path = self.cache_dir / f"{self._cache_key}.pt"

        if self._cache_path.exists():
            partition = torch.load(self._cache_path, map_location="cpu", weights_only=False)
        else:
            partition = self._build_partition(tgt_pos_np, self.thickness_scale, self.grid_n)
            torch.save(partition, self._cache_path)

        self.register_buffer("partition_center", torch.as_tensor(partition["center"], dtype=torch.float32))
        self.register_buffer("partition_R", torch.as_tensor(partition["R"], dtype=torch.float32))
        self.register_buffer("partition_dirs", torch.as_tensor(partition["dirs"], dtype=torch.float32))
        self.register_buffer("partition_radii", torch.as_tensor(partition["radii"], dtype=torch.float32))
        self.register_buffer("partition_tgt_pos", torch.as_tensor(partition["tgt_pos"], dtype=torch.float32))
        self.shell_half_thickness = float(partition["shell_half_thickness"])

    @staticmethod
    def _make_cache_key(tgt_pos_np, thickness_scale, grid_n):
        h = hashlib.sha1()
        h.update(np.ascontiguousarray(tgt_pos_np, dtype=np.float64).tobytes())
        h.update(str(float(thickness_scale)).encode("utf-8"))
        h.update(str(int(grid_n)).encode("utf-8"))
        return h.hexdigest()

    @staticmethod
    def _fit_tgt_shell(pos_tgt_np, thickness_scale):
        center = pos_tgt_np.mean(axis=0)
        q = pos_tgt_np - center

        radii = np.linalg.norm(q, axis=1)
        radii = np.maximum(radii, 1e-8)
        dirs = q / radii[:, None]

        cov = np.cov(q.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        R = eigvecs[:, order]

        q_local = q @ R
        if q_local[:, 2].mean() < 0:
            R[:, 2] *= -1

        d = np.linalg.norm(pos_tgt_np[:, None, :] - pos_tgt_np[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        nn = d.min(axis=1)
        shell_half_thickness = float(np.median(nn) * thickness_scale)

        return {
            "center": center,
            "R": R,
            "dirs": dirs,
            "radii": radii,
            "shell_half_thickness": shell_half_thickness,
            "tgt_pos": pos_tgt_np,
        }

    @classmethod
    def _build_partition(cls, pos_tgt_np, thickness_scale, grid_n):
        partition = cls._fit_tgt_shell(pos_tgt_np, thickness_scale)
        partition["grid_n"] = int(grid_n)
        return partition

    def _lookup_shell_radius(self, query_dirs):
        ref_dirs = self.partition_dirs
        ref_radii = self.partition_radii
        sim = query_dirs @ ref_dirs.T
        idx = sim.argmax(dim=1)
        r = ref_radii[idx]
        return r, idx

    def _project_points_to_shell(self, points):
        center = self.partition_center
        R = self.partition_R

        q = points - center
        rr = torch.linalg.norm(q, dim=1)
        rr_safe = torch.clamp(rr, min=1e-8)
        dirs = q / rr_safe[:, None]

        shell_r, _ = self._lookup_shell_radius(dirs)
        world_proj = center[None, :] + dirs * shell_r[:, None]
        sphere_proj = dirs @ R

        return {
            "rho": rr,
            "dirs": dirs,
            "world_proj": world_proj,
            "sphere_proj": sphere_proj,
        }

    def _assign_src_to_tgt(self, src_pos):
        src_proj = self._project_points_to_shell(src_pos)
        tgt_pos = self.partition_tgt_pos

        d_src_tgt = torch.cdist(src_proj["world_proj"], tgt_pos)
        src_region_idx = d_src_tgt.argmin(dim=1)
        src_dist_to_tgt = d_src_tgt[torch.arange(src_pos.shape[0], device=src_pos.device), src_region_idx]

        chosen_src_idx_per_tgt = torch.full(
            (self.C_tgt,),
            -1,
            dtype=torch.long,
            device=src_pos.device,
        )
        chosen_dist_per_tgt = torch.full(
            (self.C_tgt,),
            float("inf"),
            dtype=src_pos.dtype,
            device=src_pos.device,
        )

        for s in range(src_pos.shape[0]):
            t = int(src_region_idx[s].item())
            d = src_dist_to_tgt[s]
            if d < chosen_dist_per_tgt[t]:
                chosen_dist_per_tgt[t] = d
                chosen_src_idx_per_tgt[t] = s

        empty_tgt_mask = chosen_src_idx_per_tgt < 0

        return {
            "src_region_idx": src_region_idx,
            "src_dist_to_tgt": src_dist_to_tgt,
            "chosen_src_idx_per_tgt": chosen_src_idx_per_tgt,
            "chosen_dist_per_tgt": chosen_dist_per_tgt,
            "empty_tgt_mask": empty_tgt_mask,
            "src_proj": src_proj,
        }

    def forward(self, x, src_pos):
        x = _to_tensor(x, dtype=torch.float32, device=self.partition_center.device)
        src_pos = _to_tensor(src_pos, dtype=torch.float32, device=self.partition_center.device)

        if x.ndim != 2:
            raise ValueError(f"x must be (C_src, T), got {tuple(x.shape)}")
        if src_pos.ndim != 2 or src_pos.shape[1] != 3:
            raise ValueError(f"src_pos must be (C_src, 3), got {tuple(src_pos.shape)}")
        if x.shape[0] != src_pos.shape[0]:
            raise ValueError(f"x.shape[0] must equal src_pos.shape[0], got {x.shape[0]} and {src_pos.shape[0]}")

        assign_result = self._assign_src_to_tgt(src_pos)

        T = x.shape[1]
        x_resample = torch.zeros((self.C_tgt, T), dtype=x.dtype, device=x.device)
        m_c = torch.zeros((self.C_tgt, 1), dtype=x.dtype, device=x.device)

        chosen_src_idx_per_tgt = assign_result["chosen_src_idx_per_tgt"]
        valid_tgt = chosen_src_idx_per_tgt >= 0

        if valid_tgt.any():
            x_resample[valid_tgt] = x[chosen_src_idx_per_tgt[valid_tgt]]
            m_c[valid_tgt, 0] = 1.0

        projection_table = {
            "chosen_src_idx_per_tgt": chosen_src_idx_per_tgt.detach().cpu(),
            "chosen_dist_per_tgt": assign_result["chosen_dist_per_tgt"].detach().cpu(),
            "src_region_idx": assign_result["src_region_idx"].detach().cpu(),
            "src_dist_to_tgt": assign_result["src_dist_to_tgt"].detach().cpu(),
            "empty_tgt_mask": assign_result["empty_tgt_mask"].detach().cpu(),
        }

        return x_resample, m_c, projection_table
