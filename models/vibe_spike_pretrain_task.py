from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import KDTree

from models.model import S2NetBlock
from models.electrode_resampler import ElectrodeResampler


def spatial_masking(C, masking_ratio, radius, precomp_masked_indices=None):
    n_channels = C.shape[0]
    if n_channels == 0:
        return np.array([], dtype=np.int64)
    n_masked_channels = max(1, int(masking_ratio * n_channels))
    mask = np.zeros(n_channels, dtype=bool)
    if precomp_masked_indices is not None and len(precomp_masked_indices) > 0:
        mask[np.asarray(precomp_masked_indices, dtype=np.int64)] = True
    kdtree = KDTree(C)
    while np.sum(mask) < n_masked_channels:
        unmasked_indices = np.where(~mask)[0]
        if len(unmasked_indices) == 0:
            break
        _index = np.random.choice(unmasked_indices)
        _channel = C[_index]
        _neighbors = kdtree.query_ball_point(_channel, radius)
        mask[_neighbors] = True
    masked_indices = np.where(mask)[0]
    return masked_indices[:n_masked_channels]


def create_block_mask_indices(
    n_chans,
    masking_ratio,
    radius_spat_mask,
    radius_temp_mask,
    num_patches,
    pos,
    dropout_ratio,
    dropout_ratio_radius,
):
    if n_chans == 0 or num_patches == 0:
        return np.zeros((0, 0), dtype=np.int64)

    num_block_patches = num_patches // radius_temp_mask
    num_isolated_patches = num_patches % radius_temp_mask
    num_masked_chans = max(1, int(masking_ratio * n_chans))

    if dropout_ratio > 0:
        dropout_channels = spatial_masking(pos, dropout_ratio, dropout_ratio_radius)[: max(1, int(dropout_ratio * n_chans))]
    else:
        dropout_channels = None

    block_masks = [
        spatial_masking(pos, masking_ratio, radius_spat_mask, dropout_channels)[:num_masked_chans]
        for _ in range(num_block_patches)
    ]
    if len(block_masks) > 0:
        idx_block = np.repeat(np.array(block_masks)[:, np.newaxis, :], radius_temp_mask, axis=1).reshape(-1, num_masked_chans)
    else:
        idx_block = np.zeros((0, num_masked_chans), dtype=np.int64)

    isolated_masks = [
        spatial_masking(pos, masking_ratio, radius_spat_mask)[:num_masked_chans]
        for _ in range(num_isolated_patches)
    ]
    if len(isolated_masks) > 0:
        idx_isolated = np.array(isolated_masks, dtype=np.int64)
    else:
        idx_isolated = np.zeros((0, num_masked_chans), dtype=np.int64)

    if idx_block.shape[0] == 0 and idx_isolated.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.int64)
    if idx_block.shape[0] == 0:
        mask = idx_isolated
    elif idx_isolated.shape[0] == 0:
        mask = idx_block
    else:
        mask = np.concatenate([idx_block, idx_isolated], axis=0)

    mask = mask[:num_patches]
    return mask


class ComplexSTFTFrontEnd(nn.Module):
    def __init__(
        self,
        num_channels: int,
        sample_rate: int,
        n_fft: int,
        win_length: int,
        hop_length: int,
        center: bool = True,
        min_freq_hz: float = 0.0,
        max_freq_hz: Optional[float] = None,
    ):
        super().__init__()
        self.num_channels = int(num_channels)
        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.win_length = int(win_length)
        self.hop_length = int(hop_length)
        self.center = bool(center)
        self.min_freq_hz = float(min_freq_hz)
        self.max_freq_hz = max_freq_hz

        self.register_buffer("window", torch.hann_window(self.win_length))

        freqs = torch.fft.rfftfreq(self.n_fft, d=1.0 / float(self.sample_rate))
        keep = freqs >= self.min_freq_hz
        if self.max_freq_hz is not None:
            keep = keep & (freqs <= float(self.max_freq_hz))
        keep_idx = torch.where(keep)[0]
        if keep_idx.numel() == 0:
            raise ValueError("No STFT bins left after frequency selection.")
        self.register_buffer("keep_idx", keep_idx)
        self.out_freq_bins = int(keep_idx.numel())

    def forward(self, x_bct: torch.Tensor):
        B, C, T = x_bct.shape
        if C != self.num_channels:
            raise ValueError(f"Expected {self.num_channels} channels, got {C}")

        x = x_bct.reshape(B * C, T).float()
        X = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            return_complex=True,
        )
        X = X[:, self.keep_idx, :]
        T_tf = X.shape[-1]

        X = X.reshape(B, C, self.out_freq_bins, T_tf)
        real = X.real
        imag = X.imag
        X_ri = torch.stack([real, imag], dim=2)
        return X_ri


class EEGTemporalStemNoPool(nn.Module):
    def __init__(self, channels: int, kernel_sizes: Tuple[int, int, int] = (3, 5, 9), dropout: float = 0.10):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=k, padding=k // 2, groups=channels, bias=False),
                nn.ELU(inplace=True),
            )
            for k in kernel_sizes
        ])
        self.fuse = nn.Sequential(
            nn.Conv1d(channels * len(kernel_sizes), channels, kernel_size=1, bias=False),
            nn.ELU(inplace=True),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_bct: torch.Tensor) -> torch.Tensor:
        feats = [branch(x_bct) for branch in self.branches]
        x = torch.cat(feats, dim=1)
        x = self.fuse(x)
        x = self.dropout(x)
        return x


class TFProjectStem(nn.Module):
    def __init__(self, in_channels: int, proj_dim: int = 64, stem_kernel_sizes: Tuple[int, int, int] = (3, 5, 9), dropout: float = 0.10):
        super().__init__()
        self.preproj = nn.Sequential(
            nn.Conv1d(in_channels, proj_dim, kernel_size=1, bias=False),
            nn.GELU(),
        )
        self.stem = EEGTemporalStemNoPool(channels=proj_dim, kernel_sizes=stem_kernel_sizes, dropout=dropout)

    def forward(self, x_bft: torch.Tensor) -> torch.Tensor:
        x = self.preproj(x_bft)
        x = self.stem(x)
        return x


class TemporalConvDecoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels, out_channels, kernel_size=1, bias=True),
        )

    def forward(self, x_bct: torch.Tensor) -> torch.Tensor:
        return self.net(x_bct)


def build_correlation_graph_batch(z_bnt: torch.Tensor, diag_value: float = 0.15, clamp_negative: bool = True) -> torch.Tensor:
    x = z_bnt - z_bnt.mean(dim=-1, keepdim=True)
    cov = torch.einsum("bnt,bmt->bnm", x, x) / max(x.shape[-1] - 1, 1)
    var = torch.diagonal(cov, dim1=1, dim2=2)
    std = torch.sqrt(var.clamp_min(1e-8))
    corr = cov / (std.unsqueeze(2) * std.unsqueeze(1) + 1e-8)
    if clamp_negative:
        corr = corr.clamp(min=0.0)
    B, N, _ = corr.shape
    eye = torch.eye(N, device=corr.device, dtype=corr.dtype).unsqueeze(0)
    off = corr * (1.0 - eye)
    off_sum = off.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    off = off / off_sum
    sc = off * (1.0 - diag_value) + eye * diag_value
    sc = sc / (sc.sum(dim=-1, keepdim=True) + 1e-8)
    return sc


def apply_train_augment(
    x: torch.Tensor,
    channel_dropout_p: float = 0.0,
    time_mask_p: float = 0.10,
    time_mask_ratio: float = 0.05,
    noise_std: float = 0.01,
) -> torch.Tensor:
    if channel_dropout_p > 0:
        keep = (torch.rand(x.shape[0], x.shape[1], 1, device=x.device) > channel_dropout_p).float()
        x = x * keep
    if time_mask_p > 0:
        if torch.rand(1, device=x.device).item() < time_mask_p:
            T = x.shape[-1]
            span = max(1, int(T * time_mask_ratio))
            start = int(torch.randint(0, max(T - span + 1, 1), (1,), device=x.device).item())
            x = x.clone()
            x[:, :, start:start + span] = 0.0
    if noise_std > 0:
        x = x + noise_std * torch.randn_like(x)
    return x


class TFTMAEPretrainModel(nn.Module):
    def __init__(
        self,
        num_channels: int,
        tgt_pos: torch.Tensor,
        sample_rate: int = 100,
        n_fft: int = 128,
        win_length: int = 128,
        hop_length: int = 16,
        min_freq_hz: float = 0.0,
        max_freq_hz: Optional[float] = None,
        proj_dim: int = 128,
        stem_kernel_sizes: Tuple[int, int, int] = (3, 5, 9),
        stem_dropout: float = 0.10,
        s2_out_dim: int = 128,
        num_s2_layers: int = 3,
        osc_dim: int = 2,
        is_input_encode: bool = True,
        encoder_hidden_dim: int = 32,
        k: float = 1.0,
        dt: float = 1.0,
        phase_delay_steps: int = 1,
        low_n: float = 0.0,
        high_n: float = 4.0,
        branch: int = 8,
        low_m: float = 0.0,
        high_m: float = 4.0,
        vth_snn: float = 0.5,
        dendritic_branch_constraint: bool = False,
        mem_input: bool = False,
        vth_readout: float = 0.15,
        graph_diag_value: float = 0.5,
        masking_window: int = 200,
        masking_overlap: int = 20,
        masking_ratio: float = 0.55,
        radius_spat_mask: float = 0.03,
        radius_temp_mask: int = 3,
        dropout_ratio: float = 0.10,
        dropout_radius: float = 0.04,
        decoder_hidden: int = 1024,
        decoder_dropout: float = 0.00,
        use_decoder_block_token: bool = True,
        device: str = "cuda",
    ):
        super().__init__()
        self.graph_diag_value = float(graph_diag_value)
        self.masking_window = int(masking_window)
        self.masking_overlap = int(masking_overlap)
        self.masking_ratio = float(masking_ratio)
        self.radius_spat_mask = float(radius_spat_mask)
        self.radius_temp_mask = int(radius_temp_mask)
        self.dropout_ratio = float(dropout_ratio)
        self.dropout_radius = float(dropout_radius)

        self.resampler = ElectrodeResampler(tgt_pos=tgt_pos)
        self.register_buffer("tgt_pos", tgt_pos.float())

        self.tf = ComplexSTFTFrontEnd(
            num_channels=num_channels,
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=True,
            min_freq_hz=min_freq_hz,
            max_freq_hz=max_freq_hz,
        )

        self.channel_missing_token = nn.Parameter(torch.zeros(1, num_channels, 2, self.tf.out_freq_bins, 1))

        # changed: direct channel-time valid-mask concat, no frequency-broadcast mask concat
        in_ch = 2 * num_channels * self.tf.out_freq_bins + num_channels
        self.project = TFProjectStem(
            in_channels=in_ch,
            proj_dim=proj_dim,
            stem_kernel_sizes=stem_kernel_sizes,
            dropout=stem_dropout,
        )

        neuron_dims = [proj_dim] * int(num_s2_layers)
        self.s2block = S2NetBlock(
            num_regions=proj_dim,
            neuron_dims=neuron_dims,
            output_dim=s2_out_dim,
            osc_dim=osc_dim,
            is_input_encode=is_input_encode,
            encoder_hidden_dim=encoder_hidden_dim,
            k=k,
            dt=dt,
            phase_delay_steps=phase_delay_steps,
            low_n=low_n,
            high_n=high_n,
            branch=branch,
            low_m=low_m,
            high_m=high_m,
            vth_snn=vth_snn,
            dendritic_branch_constraint=dendritic_branch_constraint,
            mem_input=mem_input,
            vth_readout=vth_readout,
            device=device,
        )

        self.use_decoder_block_token = bool(use_decoder_block_token)
        self.decoder_block_token = nn.Parameter(torch.zeros(1, s2_out_dim, 1))

        # decoder-side concat already has no frequency broadcast
        decoder_in_ch = s2_out_dim + 2 * num_channels
        decoder_out_ch = num_channels * self.tf.out_freq_bins
        self.decoder = TemporalConvDecoder(
            in_channels=decoder_in_ch,
            hidden_channels=decoder_hidden,
            out_channels=decoder_out_ch,
            dropout=decoder_dropout,
        )

    def _batch_resample(self, x_bct: torch.Tensor, pos_bc3: torch.Tensor):
        xs = []
        ms = []
        tables = []
        for b in range(x_bct.shape[0]):
            x_resample, m_c, table = self.resampler(x_bct[b], pos_bc3[b])
            xs.append(x_resample)
            ms.append(m_c)
            tables.append(table)
        return torch.stack(xs, dim=0), torch.stack(ms, dim=0), tables

    def _build_block_mask(self, valid_mask_bc1: torch.Tensor, waveform_len: int, tf_len: int) -> torch.Tensor:
        B, C, _ = valid_mask_bc1.shape
        block_mask = torch.zeros((B, C, tf_len), dtype=torch.bool, device=valid_mask_bc1.device)

        patch_step = self.masking_window - self.masking_overlap
        if waveform_len <= self.masking_window:
            patch_starts = [0]
        else:
            patch_starts = list(range(0, waveform_len - self.masking_window + 1, patch_step))
        num_patches = len(patch_starts)

        tgt_pos_np = self.tgt_pos.detach().cpu().numpy()
        for b in range(B):
            valid_idx = torch.where(valid_mask_bc1[b, :, 0] > 0.5)[0].detach().cpu().numpy()
            if len(valid_idx) == 0:
                continue
            pos_valid = tgt_pos_np[valid_idx]
            mask_idx = create_block_mask_indices(
                n_chans=len(valid_idx),
                masking_ratio=self.masking_ratio,
                radius_spat_mask=self.radius_spat_mask,
                radius_temp_mask=self.radius_temp_mask,
                num_patches=num_patches,
                pos=pos_valid,
                dropout_ratio=self.dropout_ratio,
                dropout_ratio_radius=self.dropout_radius,
            )
            if mask_idx.size == 0:
                continue
            for p, start_sample in enumerate(patch_starts):
                end_sample = min(start_sample + self.masking_window, waveform_len)
                start_tf = int(round(start_sample / self.tf.hop_length))
                end_tf = int(round(end_sample / self.tf.hop_length))
                start_tf = max(0, min(start_tf, tf_len - 1))
                end_tf = max(start_tf + 1, min(end_tf, tf_len))
                masked_valid_local = mask_idx[p]
                masked_global = valid_idx[masked_valid_local]
                block_mask[b, masked_global, start_tf:end_tf] = True
        return block_mask

    def _loss_l1_masked(self, recon_tf_mag: torch.Tensor, target_tf_mag: torch.Tensor, block_mask_bct: torch.Tensor) -> torch.Tensor:
        # changed: average over number of masked (channel, time) patches, not masked scalar elements
        # L = (1 / |P_m|) * sum_{i in P_m} ||recon_i - target_i||_1
        patch_l1 = torch.abs(recon_tf_mag - target_tf_mag).sum(dim=2)  # [B, C, T_tf]
        mask = block_mask_bct.float()
        denom = mask.sum().clamp_min(1.0)
        return (patch_l1 * mask).sum() / denom

    def forward(self, x_bct: torch.Tensor, pos_bc3: torch.Tensor):
        x_resample, m_c, projection_tables = self._batch_resample(x_bct, pos_bc3)
        B, C, T = x_resample.shape

        tf_ri_target = self.tf(x_resample)
        _, _, _, F, T_tf = tf_ri_target.shape
        tf_mag_target = torch.sqrt(tf_ri_target[:, :, 0].pow(2) + tf_ri_target[:, :, 1].pow(2) + 1e-8)

        if m_c.dim() == 2:
            m_c = m_c.unsqueeze(-1)

        tf_valid = m_c[:, :, None, None, :].expand(-1, -1, 1, F, T_tf).to(tf_ri_target.dtype)

        block_mask_bct = self._build_block_mask(m_c, T, T_tf)
        block_mask_tf = block_mask_bct[:, :, None, None, :].expand(-1, -1, 2, F, -1)

        tf_ri_encoder = tf_ri_target + (1.0 - tf_valid).expand(-1, -1, 2, -1, -1) * self.channel_missing_token
        tf_ri_encoder = tf_ri_encoder.masked_fill(block_mask_tf, 0.0)
        tf_mag_encoder = torch.sqrt(tf_ri_encoder[:, :, 0].pow(2) + tf_ri_encoder[:, :, 1].pow(2) + 1e-8)

        tf_flat_encoder = tf_ri_encoder.reshape(B, C * 2 * F, T_tf)
        valid_mask_bt = m_c.expand(-1, -1, T_tf).to(tf_ri_target.dtype)
        encoder_in = torch.cat([tf_flat_encoder, valid_mask_bt], dim=1)

        proj_x = self.project(encoder_in)
        sc = build_correlation_graph_batch(proj_x, diag_value=self.graph_diag_value)
        s2_in = proj_x.permute(0, 2, 1).contiguous()
        s2_feat, spikes = self.s2block(s2_in, sc)

        if self.use_decoder_block_token:
            block_any_bt = block_mask_bct.any(dim=1, keepdim=True).float()
            dec_feat = s2_feat + block_any_bt * self.decoder_block_token
        else:
            dec_feat = s2_feat

        decoder_valid_bt = m_c.expand(-1, -1, T_tf).float()
        decoder_in = torch.cat([dec_feat, block_mask_bct.float(), decoder_valid_bt], dim=1)
        recon_flat = self.decoder(decoder_in)
        recon_tf_mag = recon_flat.reshape(B, C, F, T_tf)

        loss = self._loss_l1_masked(recon_tf_mag, tf_mag_target, block_mask_bct)

        return {
            "loss": loss,
            "x_resample": x_resample,
            "m_c": m_c,
            "projection_tables": projection_tables,
            "tf_ri_target": tf_ri_target,
            "tf_ri_encoder": tf_ri_encoder,
            "tf_mag_target": tf_mag_target,
            "tf_mag_encoder": tf_mag_encoder,
            "block_mask_bct": block_mask_bct,
            "proj_x": proj_x,
            "s2_feat": s2_feat,
            "recon_tf_mag": recon_tf_mag,
            "spikes": spikes,
            "sc": sc,
        }


class VibeSpikePretrainTask(nn.Module):
    def __init__(self, cfg, target_coords: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        self.graph_type = "internal_correlation_from_proj_x"
        self.graph_normalize = False

        self.model = TFTMAEPretrainModel(
            num_channels=int(cfg.model.num_channels),
            tgt_pos=target_coords,
            sample_rate=int(cfg.model.sample_rate),
            n_fft=int(cfg.model.n_fft),
            win_length=int(cfg.model.win_length),
            hop_length=int(cfg.model.hop_length),
            min_freq_hz=float(cfg.model.min_freq_hz),
            max_freq_hz=float(cfg.model.max_freq_hz) if cfg.model.max_freq_hz is not None else None,
            proj_dim=int(cfg.model.proj_dim),
            stem_kernel_sizes=tuple(cfg.model.stem_kernel_sizes),
            stem_dropout=float(cfg.model.stem_dropout),
            s2_out_dim=int(cfg.model.s2_out_dim),
            num_s2_layers=int(cfg.model.num_s2_layers),
            osc_dim=int(cfg.model.osc_dim),
            is_input_encode=bool(cfg.model.is_input_encode),
            encoder_hidden_dim=int(cfg.model.encoder_hidden_dim),
            k=float(cfg.model.k),
            dt=float(cfg.model.dt),
            phase_delay_steps=int(cfg.model.phase_delay_steps),
            low_n=float(cfg.model.low_n),
            high_n=float(cfg.model.high_n),
            branch=int(cfg.model.branch),
            low_m=float(cfg.model.low_m),
            high_m=float(cfg.model.high_m),
            vth_snn=float(cfg.model.vth_snn),
            dendritic_branch_constraint=bool(cfg.model.dendritic_branch_constraint),
            mem_input=bool(cfg.model.mem_input),
            vth_readout=float(cfg.model.vth_readout),
            graph_diag_value=float(cfg.model.graph_diag_value),
            masking_window=int(cfg.preprocessing.masking.masking_window),
            masking_overlap=int(cfg.preprocessing.masking.masking_overlap),
            masking_ratio=float(cfg.preprocessing.masking.ratio),
            radius_spat_mask=float(cfg.preprocessing.masking.radius_spat_mask),
            radius_temp_mask=int(cfg.preprocessing.masking.radius_temp_mask),
            dropout_ratio=float(cfg.preprocessing.masking.dropout_ratio),
            dropout_radius=float(cfg.preprocessing.masking.dropout_radius),
            decoder_hidden=int(cfg.model.decoder_hidden),
            decoder_dropout=float(cfg.model.decoder_dropout),
            use_decoder_block_token=bool(cfg.model.use_decoder_block_token),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    def forward(self, x_src, input_coords, b_m=None):
        if self.training:
            x_src = apply_train_augment(
                x_src,
                channel_dropout_p=float(getattr(self.cfg.train, "train_channel_dropout", 0.0)),
                time_mask_p=float(getattr(self.cfg.train, "train_time_mask_p", 0.10)),
                time_mask_ratio=float(getattr(self.cfg.train, "train_time_mask_ratio", 0.05)),
                noise_std=float(getattr(self.cfg.train, "train_noise_std", 0.01)),
            )

        outputs = self.model(x_src, input_coords)
        loss = outputs["loss"]

        stats = {
            "loss_total": loss.detach(),
            "loss_masked": loss.detach(),
            "loss_consistency": torch.zeros((), device=loss.device, dtype=loss.dtype),
            "spike_rate": outputs["spikes"].float().mean().detach(),
            "block_ratio": outputs["block_mask_bct"].float().mean().detach(),
            "valid_ratio": outputs["m_c"].float().mean().detach(),
        }
        return loss, stats
