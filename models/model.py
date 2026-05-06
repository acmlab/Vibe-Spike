import torch
import torch.nn as nn
from torch.autograd import Variable
from utils.surrograte import *
import torch.nn.functional as F


class graphVectorKuramoto(nn.Module):
    """
    references: https://anonymous.4open.science/r/S2Net-7956/SNN/SNN_layers/vector_kuramoto.py
    Graph-Aware Vector Kuramoto layer with OT-derived Phase Lags.
    Strictly aligns with Eq. (5) and OT Surrogate mechanics.
    args:
    - N: Number of oscillators (brain regions)
    - D: Dimensionality of the phase space (default 2 for 2D phases)
    - K: Global coupling strength
    - dt: Time step for Euler integration
    - alpha_scale: Scaling factor for OT-derived phase lags
    - device: 'cuda' or 'cpu' for computation

    internal parameters:
    - omega: Natural frequency of each oscillator (learnable)
    - kappa: Control stiffness for sensory drive (learnable)
    - direction_learner: Learnable parameters to determine directionality of phase lags (N x N matrix)

    inputs:
    - theta_prev: [B, N, D] previous phase states of oscillators
    - gamma: [B, N] External sensory drive/gating signal for each oscillator
    - A: [B, N, N] Optional connectivity matrix (structural priors). If None, assumes fully connected.
    """
    def __init__(
        self,
        N,
        D=2,
        K=1.0,
        dt=1.0,
        alpha_scale=1.0,
        device="cuda"
    ):
        super().__init__()
        # Model parameters
        self.N = N
        self.D = D
        self.K = K
        self.dt = dt
        self.alpha_scale = alpha_scale
        self.device = device

        # Natural frequency ω_i (aligns with revised Eq. 5)
        self.omega = nn.Parameter(torch.randn(N, D) * 0.1)

        # Control stiffness κ_i
        self.kappa = nn.Parameter(torch.ones(N, D))
        self.direction_learner = nn.Parameter(torch.randn(N, N) * 0.01)

    def forward(self, theta_prev, gamma, A=None):
        B, H, D = theta_prev.shape
        device = theta_prev.device

        # 1. Handle Graph Structure & OT Surrogate
        if A is None:
            # Fallback if no graph provided
            A_lat = torch.ones(B, H, H, device=device)
            alpha = torch.zeros(B, H, H, 1, device=device)
        else:
            # Symmetrize connectivity
            A_lat = 0.5 * (A + A.transpose(1, 2))
            A_lat = torch.relu(A_lat) + 1e-6  # Avoid div by zero

            # --- OT-Derived Phase Lag (Section 3.3 in Paper) ---
            # Cost C_ij = 1 / A_ij
            cost_matrix = 1.0 / A_lat

            # Normalize cost to [0, 1] per batch to stabilize
            c_min = cost_matrix.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
            c_max = cost_matrix.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
            norm_cost = (cost_matrix - c_min) / (c_max - c_min + 1e-6)
            direction = self.direction_learner - self.direction_learner.transpose(0, 1)  # [N, N]
            direction_mask = torch.tanh(direction)
            alpha_matrix = direction_mask.unsqueeze(0) * norm_cost  # [B, N, N]
            # alpha_ij = alpha_0 * norm(C_ij)
            # Expand to [B, H, H, 1] to broadcast over D dim
            # alpha = (self.alpha_scale * norm_cost).unsqueeze(-1)
            alpha = (self.alpha_scale * alpha_matrix).unsqueeze(-1)

        # 2. Kuramoto Dynamics
        theta_i = theta_prev.unsqueeze(2)  # [B, H, 1, D]
        theta_j = theta_prev.unsqueeze(1)  # [B, 1, H, D]

        # Interaction term: sin(theta_j - theta_i - alpha_ij)
        phase_diff = theta_j - theta_i - alpha

        # Weighted sum by adjacency A_ij
        interaction = torch.sum(A_lat.unsqueeze(-1) * torch.sin(phase_diff), dim=2)
        coupling = (self.K / float(H)) * interaction

        # 3. Sensory Drive (Corrected to Sinusoidal)
        # kappa * sin(gamma - theta)
        gamma_exp = gamma.unsqueeze(-1)
        drive_term = self.kappa * torch.sin(gamma_exp - theta_prev)

        # 4. Euler Integration
        theta_dot = self.omega + coupling + drive_term
        theta_new = theta_prev + self.dt * theta_dot

        return theta_new


class RhythmSnnLayer(nn.Module):
    """
    A rhythmic SNN layer that integrates a rhythmic mask.
    references paper: https://www.nature.com/articles/s41467-025-63771-x

    Input Data Format: [B, C] where B is batch size, C is the number of channels (e.g., brain regions/eeg channels)

    :args:
    - input_dim: Number of neurons in the input layer (e.g., number/channel of brain regions for first layer)
    - c_min, c_max: Range for cycle lengths in the rhythmic mask
    - min_dc, max_dc: Range for duty cycles in the rhythmic mask
    - phase_shift_max: Maximum phase shift for the rhythmic mask
    - vth: Base firing threshold for neurons
    - dt: Time step for simulation
    - branch: Number of dendritic branches for integration
    - low_m, high_m: Range for initializing membrane time constants (tau_m)
    - low_n, high_n: Range for initializing dendritic time constants (tau_n)
    - device: 'cuda' or 'cpu' for computation
    - bias: Whether to include bias in the dense layer
    """
    def __init__(
            self,
            input_dim,
            output_dim,
            vth,
            dt,
            branch,
            low_m,
            high_m,
            low_n,
            high_n,
            device,
            dendritic_branch_constraint=True,
            bias=True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.vth = vth
        self.dt = dt

        # Dendritic Parameters
        self.dendritic_branch_constraint = dendritic_branch_constraint
        if dendritic_branch_constraint:
            self.pad = ((input_dim + output_dim) // branch * branch + branch - (input_dim + output_dim)) % branch
            self.dense = nn.Linear(input_dim + output_dim + self.pad, output_dim * branch, bias=bias)

            self.tau_m = nn.Parameter(torch.Tensor(self.output_dim))
            self.tau_n = nn.Parameter(torch.Tensor(self.output_dim, branch))
            self.branch = branch

            nn.init.uniform_(self.tau_m, low_m, high_m)
            nn.init.uniform_(self.tau_n, low_n, high_n)
        else:
            self.dense = nn.Linear(input_dim, output_dim, bias=bias)
            self.tau_m = nn.Parameter(torch.Tensor(self.output_dim))
            nn.init.uniform_(self.tau_m, low_m, high_m)

        # internal states
        self.mem = None
        self.spike = None
        self.d_input = None
        self.v_th = None

    def set_neuron_state(self, batch_size):
        # self.mem = torch.rand(batch_size, self.output_dim).to(self.device)
        # self.spike = torch.rand(batch_size, self.output_dim).to(self.device)
        self.mem = torch.zeros(batch_size, self.output_dim).to(self.device)
        self.spike = torch.zeros(batch_size, self.output_dim).to(self.device)
        if self.dendritic_branch_constraint:
            self.d_input = torch.zeros(batch_size, self.output_dim, self.branch).to(self.device)
        self.v_th = torch.ones(batch_size, self.output_dim).to(self.device) * self.vth

    def forward(self, input_seq, rhythmic_mask=None):
        batch_size, input_dim = input_seq.size(0), input_seq.size(1)

        if rhythmic_mask is None:
            raise ValueError("Rhythmic mask must be provided for the forward pass.")

        if not self.dendritic_branch_constraint:
            # If no dendritic branch constraint, simply apply the dense layer and then the rhythm modulation
             l_input = self.dense(input_seq.float())
        else:
            raise NotImplementedError("Dendritic branch constraint version is not implemented in this snippet.")

        # update membrane potential and spike with rhythm modulation
        self.mem, self.spike = mem_update_pra_rhythm(
            l_input, self.mem, self.spike, self.v_th, self.tau_m,
            rhythmic_mask,
            self.dt, device=self.device
        )

        return self.mem, self.spike


class ReadoutLayer(nn.Module):
    """
    A readout layer that takes the output from the rhythmic SNN layer and produces class predictions.
    This class is from the original spike_dense_test_origin.
    Args:
    - input_dim: Number of neurons in the input layer (should match output_dim of RhythmSnnLayer)
    - output_dim: Number of classes for classification
    - vth: Firing threshold for the readout neurons
    - dt: Time step for simulation
    - device: 'cuda' or 'cpu' for computation
    - bias: Whether to include bias in the dense layer
    """
    def __init__(
            self,
            input_dim,
            output_dim,
            low_m=0,
            high_m=4,
            vth=0.5,
            dt=4,
            device='cpu',
            bias=True
    ):
        super(ReadoutLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.vth = vth
        self.dt = dt

        self.dense = nn.Linear(input_dim, output_dim, bias=bias)
        self.tau_m = nn.Parameter(torch.Tensor(self.output_dim))
        nn.init.uniform_(self.tau_m, low_m, high_m)

        # internal states
        self.mem = None
        self.spike = None
        self.v_th = None

    def set_neuron_state(self, batch_size):
        #
        # self.mem = Variable(torch.rand(batch_size, self.output_dim)).to(self.device)
        # self.spike = Variable(torch.rand(batch_size, self.output_dim)).to(self.device)
        # self.v_th = Variable(torch.ones(batch_size, self.output_dim)*self.vth).to(self.device)
        self.mem = Variable(torch.zeros(batch_size, self.output_dim)).to(self.device)
        self.spike = Variable(torch.zeros(batch_size, self.output_dim)).to(self.device)
        self.v_th = Variable(torch.ones(batch_size, self.output_dim) * self.vth).to(self.device)

    def forward(self, input_spike):
        k_input = input_spike.float()

        d_input = self.dense(k_input)
        self.mem, self.spike = mem_update_pra(d_input, self.mem, self.spike, self.v_th, self.tau_m, self.dt, device=self.device)
        return self.mem, self.spike


class TemporalDownsampleLayer_Conv(nn.Module):
    """
    A temporal down sampling layer that reduces the sequence length by a factor of `downsample_factor`.
    This can be used between RhythmSnnLayers to reduce computational load and capture longer-term dependencies.

    Input Data Format: [B, C, T]
        - B: batch size
        - T: sequence length (time steps)
        - C: feature / channel dimension

    Output Data Format: [B, C, T_down]
        - T_down is approximately T // downsample_factor
        - C is kept unchanged

    :args:
    - downsample_factor: Factor used to reduce the temporal dimension
    - kernel_size: Temporal kernel size for Conv1d. If None, use downsample_factor
    - bias: Whether to include bias in Conv1d
    - use_bn: Whether to use BatchNorm1d after Conv1d
    - activation: Whether to use GELU activation after Conv1d
    - dropout: Dropout probability after activation
    - depthwise: If True, perform depthwise temporal convolution (groups = channels),
                 so each channel is filtered independently along time
    """

    def __init__(
            self,
            in_channels,
            downsample_factor,
            kernel_size=None,
            bias=False,
            use_bn=True,
            activation=True,
            dropout=0.0,
            depthwise=True,
    ):
        super(TemporalDownsampleLayer_Conv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.downsample_factor = downsample_factor
        self.kernel_size = kernel_size if kernel_size is not None else downsample_factor
        self.bias = bias
        self.use_bn = use_bn
        self.activation = activation
        self.dropout_p = dropout
        self.depthwise = depthwise

        groups = in_channels if self.depthwise else 1
        padding = self.kernel_size // 2

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=self.kernel_size,
            stride=self.downsample_factor,
            padding=padding,
            groups=groups,
            bias=self.bias,
        )
        self.bn = nn.BatchNorm1d(in_channels) if self.use_bn else None
        self.act = nn.GELU() if activation else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def _build_layer(self, in_channels, device):
        self.in_channels = in_channels
        self.out_channels = in_channels   # keep channel dimension unchanged

        groups = in_channels if self.depthwise else 1
        padding = self.kernel_size // 2

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=self.kernel_size,
            stride=self.downsample_factor,
            padding=padding,
            groups=groups,
            bias=self.bias,
        ).to(device)

        if self.use_bn:
            self.bn = nn.BatchNorm1d(in_channels).to(device)

    def forward(self, input_seq):
        """
        Args:
            input_seq: [B, C, T]

        Returns:
            output_seq: [B, C, T_down]
        """
        if input_seq.dim() != 3:
            raise ValueError(f"Expected input_seq with shape [B, T, C], but got {tuple(input_seq.shape)}")

        batch_size, seq_len, channel_num = input_seq.size()

        # lazy initialization according to input channel dimension
        # if self.conv is None:
        #     self._build_layer(channel_num, input_seq.device)

        # [B, T, C] -> [B, C, T]
        # x = input_seq.permute(0, 2, 1).contiguous()
        x = input_seq
        # temporal convolution down sampling
        x = self.conv(x)

        if self.bn is not None:
            x = self.bn(x)

        if self.act is not None:
            x = self.act(x)

        if self.dropout is not None:
            x = self.dropout(x)

        # [B, C, T_down] -> [B, T_down, C]
        # output_seq = x.permute(0, 2, 1).contiguous()
        output_seq = x

        return output_seq  # [B, C, T_down]


class TemporalDownsampleLayer_Pool(nn.Module):
    """
    A temporal down sampling layer based on temporal pooling.

    This layer reduces the sequence length by applying pooling along the temporal axis,
    while keeping the channel / feature dimension unchanged.

    Input Data Format: [B, T, C]
        - B: batch size
        - T: sequence length (time steps)
        - C: feature / channel dimension

    Output Data Format: [B, T_down, C]
        - T_down is approximately T // downsample_factor
        - C is kept unchanged

    :args:
    - downsample_factor: Factor used to reduce the temporal dimension
    - pool_type: Type of pooling, either 'avg' or 'max'
    """

    def __init__(
            self,
            downsample_factor,
            pool_type='avg',
    ):
        super(TemporalDownsampleLayer_Pool, self).__init__()

        self.downsample_factor = downsample_factor
        self.pool_type = pool_type.lower()

        if self.pool_type == 'avg':
            self.pool = nn.AvgPool1d(
                kernel_size=self.downsample_factor,
                stride=self.downsample_factor,
            )
        elif self.pool_type == 'max':
            self.pool = nn.MaxPool1d(
                kernel_size=self.downsample_factor,
                stride=self.downsample_factor,
            )
        else:
            raise ValueError(f"Unsupported pool_type: {pool_type}. Choose from ['avg', 'max'].")

    def forward(self, input_seq):
        """
        Args:
            input_seq: [B, T, C]

        Returns:
            output_seq: [B, T_down, C]
        """
        if input_seq.dim() != 3:
            raise ValueError(f"Expected input_seq with shape [B, T, C], but got {tuple(input_seq.shape)}")

        batch_size, seq_len, channel_num = input_seq.size()

        # [B, T, C] -> [B, C, T]
        x = input_seq.permute(0, 2, 1).contiguous()

        # temporal pooling down sampling
        x = self.pool(x)

        # [B, C, T_down] -> [B, T_down, C]
        output_seq = x.permute(0, 2, 1).contiguous()

        return output_seq


class TemporalDownsampleLayer_Patch(nn.Module):
    def __init__(
            self,
            in_channels,
            patch_size,
            stride=None,
            embed_dim=None,
            bias=True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.patch_size = patch_size
        self.stride = stride if stride is not None else patch_size
        self.embed_dim = embed_dim

        self.patch_dim = self.in_channels * self.patch_size

        if self.embed_dim is not None:
            self.proj = nn.Linear(self.patch_dim, self.embed_dim, bias=bias)
            self.out_channels = self.embed_dim
        else:
            self.proj = None
            self.out_channels = self.patch_dim

    def forward(self, input_seq):
        """
        input_seq: [B, T, C]
        return:
            [B, N, patch_dim] or [B, N, embed_dim]
        """
        if input_seq.dim() != 3:
            raise ValueError(f"Expected input_seq with shape [B, T, C], but got {tuple(input_seq.shape)}")

        B, T, C = input_seq.size()

        if C != self.in_channels:
            raise ValueError(f"Input channel mismatch: expected {self.in_channels}, got {C}")

        if T < self.patch_size:
            raise ValueError(f"Sequence length {T} is smaller than patch_size {self.patch_size}")

        # [B, T, C] -> [B, C, T]
        x = input_seq.permute(0, 2, 1).contiguous()

        # [B, C, T] -> [B, C, N, P]
        x = x.unfold(dimension=2, size=self.patch_size, step=self.stride)

        # [B, C, N, P] -> [B, N, C, P]
        x = x.permute(0, 2, 1, 3).contiguous()

        # [B, N, C, P] -> [B, N, C*P]
        x = x.view(B, x.size(1), -1)

        if self.proj is not None:
            x = self.proj(x)

        return x


class TemporalUpsampleLayer_Patch(nn.Module):
    def __init__(
            self,
            out_channels,
            patch_size,
            stride=None,
            embed_dim=None,
            bias=True,
    ):
        super().__init__()

        self.out_channels = out_channels
        self.patch_size = patch_size
        self.stride = stride if stride is not None else patch_size
        self.embed_dim = embed_dim
        self.patch_dim = self.out_channels * self.patch_size

        if self.embed_dim is not None:
            self.unproj = nn.Linear(self.embed_dim, self.patch_dim, bias=bias)
        else:
            self.unproj = None

    def forward(self, patch_seq, target_len=None):
        """
        patch_seq: [B, N, patch_dim] or [B, N, embed_dim]
        return:   [B, T, C]
        """
        if patch_seq.dim() != 3:
            raise ValueError(f"Expected patch_seq with shape [B, N, D], but got {tuple(patch_seq.shape)}")

        B, N, D = patch_seq.shape

        if self.unproj is not None:
            patch_seq = self.unproj(patch_seq)

        if patch_seq.size(-1) != self.patch_dim:
            raise ValueError(
                f"Patch dim mismatch: expected {self.patch_dim}, got {patch_seq.size(-1)}"
            )

        full_len = (N - 1) * self.stride + self.patch_size
        T = full_len if target_len is None else target_len

        # [B, N, C*P] -> [B, N, C, P]
        x = patch_seq.view(B, N, self.out_channels, self.patch_size)

        recon = patch_seq.new_zeros(B, self.out_channels, full_len)
        count = patch_seq.new_zeros(B, self.out_channels, full_len)

        for i in range(N):
            start = i * self.stride
            end = start + self.patch_size
            recon[:, :, start:end] += x[:, i, :, :]
            count[:, :, start:end] += 1.0

        recon = recon / count.clamp_min(1.0)
        recon = recon[:, :, :T]

        # [B, C, T] -> [B, T, C]
        return recon.permute(0, 2, 1).contiguous()


class S2NetBlock(nn.Module):
    def __init__(
            self,
            num_regions,
            neuron_dims,
            output_dim,
            osc_dim,
            is_input_encode,
            encoder_hidden_dim,
            k,
            dt,
            phase_delay_steps,
            low_n,
            high_n,
            branch,
            low_m,
            high_m,
            vth_snn,
            dendritic_branch_constraint,
            mem_input,
            vth_readout,
            device
    ):
        super().__init__()

        # model parameters
        self.num_regions = num_regions
        self.neuron_dims = neuron_dims
        self.osc_dim = osc_dim
        self.k = k
        self.dt = dt
        self.phase_delay_steps = phase_delay_steps
        self.device = device
        self.mem_input = mem_input

        # ===== original-style top-down encoder =====
        self.input_encode = is_input_encode
        self.latent_dim = encoder_hidden_dim

        if self.input_encode:
            self.f_proj = nn.Sequential(
                nn.Linear(num_regions, self.latent_dim),
                nn.LayerNorm(self.latent_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.1)
            ).to(device)

            self.enc = nn.Linear(self.latent_dim, num_regions).to(device)

        # Graph-Aware Vector Kuramoto Layer
        self.kuramoto = graphVectorKuramoto(
            N=self.num_regions,
            D=osc_dim,
            K=k,
            dt=dt,
            alpha_scale=1.0,
            device=device
        )
        self.mask_proj = nn.Linear(num_regions, num_regions).to(device)

        # Rhythm SNN Layer parameters
        self.low_n, self.high_n = low_n, high_n
        self.vth_snn = vth_snn

        # dendritic branch parameters
        self.dendritic_branch_constraint = dendritic_branch_constraint
        self.branch = branch
        self.low_m, self.high_m = low_m, high_m

        # Rhythm SNN Layer
        self.snn_dim_adapter = nn.Linear(osc_dim, 1).to(device)

        for i in range(len(self.neuron_dims)):
            setattr(
                self,
                f'rhythm_snn_layer_{i}',
                RhythmSnnLayer(
                    input_dim=self.num_regions if i == 0 else self.neuron_dims[i - 1],
                    output_dim=self.neuron_dims[i],
                    vth=self.vth_snn,
                    dt=self.dt,
                    branch=self.branch,
                    low_m=self.low_m,
                    high_m=self.high_m,
                    low_n=self.low_n,
                    high_n=self.high_n,
                    device=device,
                    dendritic_branch_constraint=self.dendritic_branch_constraint,
                    bias=True,
                ).to(device)
            )

        # Readout Layer
        self.vth_readout = vth_readout
        self.output_dim = output_dim
        self.readout_layer = ReadoutLayer(
            input_dim=self.num_regions,
            output_dim=self.output_dim,
            low_m=low_m,
            high_m=high_m,
            vth=vth_readout,
            dt=dt,
            device=device,
            bias=True
        )

    def forward(self, x, sc):
        B, T, N = x.shape
        x = x.to(self.device)
        sc = sc.to(self.device)

        theta_hist = []
        theta = torch.zeros(B, self.num_regions, self.osc_dim, device=self.device)
        feats_list = []
        mask_hidden_list = []

        for t in range(T):
            x_t = x[:, t, :]

            if self.input_encode:
                z_t = self.f_proj(x_t)
                gamma_t = self.enc(z_t)
            else:
                gamma_t = x_t

            theta = self.kuramoto(theta, gamma_t, A=sc)
            theta_hist.append(theta)

            idx = max(0, t - self.phase_delay_steps)
            theta_mean = theta_hist[idx].mean(dim=-1)
            mask_ = 0.5 * (1.0 + torch.sin(theta_mean))

            phase_feat = torch.sin(theta)
            phase_feat_gated = phase_feat * mask_.unsqueeze(-1)
            feats_list.append(phase_feat_gated.unsqueeze(1))

            mask_hidden = torch.sigmoid(self.mask_proj(mask_))
            mask_hidden_list.append(mask_hidden)

        kuramoto_out = torch.cat(feats_list, dim=1)
        masks = torch.stack(mask_hidden_list, dim=1)

        batch_size, seq_len, num_regions, osc_dim = kuramoto_out.size()

        for i in range(len(self.neuron_dims)):
            getattr(self, f'rhythm_snn_layer_{i}').set_neuron_state(batch_size)
        self.readout_layer.set_neuron_state(batch_size)

        outputs = []
        spikes = []

        for t in range(seq_len):
            kuramoto_out_t = kuramoto_out[:, t, :, :]
            currents = self.snn_dim_adapter(kuramoto_out_t).squeeze(-1)

            g_t = masks[:, t, :]

            x = currents
            mem = None
            spike = None
            for i in range(len(self.neuron_dims)):
                mem, spike = getattr(self, f'rhythm_snn_layer_{i}')(x, rhythmic_mask=g_t)
                if self.mem_input:
                    x = mem
                else:
                    x = spike

            spikes.append(spike)
            if self.mem_input:
                mem_readout, spike_readout = self.readout_layer(mem)
            else:
                mem_readout, spike_readout = self.readout_layer(spike)
            outputs.append(mem_readout)

        outputs = torch.stack(outputs).permute(1, 2, 0)
        spikes = torch.stack(spikes).permute(1, 2, 0)

        return outputs, spikes


class S2NetClassifier(nn.Module):
    """
    A complete S2Net classifier that integrates the S2NetBlock and classification head.
    """
    def __init__(
            self,
            num_regions,
            class_num,
            osc_dim,
            neuron_dims,
            is_input_encode,
            encoder_hidden_dim,
            temporal_downsample_type,
            downsample_factor,
            k,
            dt,
            phase_delay_steps,
            low_n,
            high_n,
            branch,
            low_m,
            high_m,
            vth_snn,
            dendritic_branch_constraint,
            mem_input,
            vth_readout,
            device
    ):
        super().__init__()

        self.temporal_downsample_type = temporal_downsample_type
        self.downsample_factor = downsample_factor   # [5, 10, 20] for example

        if self.temporal_downsample_type == 'conv':
            for i in range(len(self.downsample_factor)):
                setattr(
                    self,
                    f'temporal_downsample_{i}',
                    TemporalDownsampleLayer_Conv(
                        in_channels=num_regions,
                        downsample_factor=self.downsample_factor[i],
                        kernel_size=2 * self.downsample_factor[i] - 1,
                        bias=False,
                        use_bn=True,
                        activation=True,
                        dropout=0.0,
                        depthwise=True
                    ).to(device)
                )

        self.s2net_block = S2NetBlock(
            num_regions=num_regions,
            output_dim=class_num,
            neuron_dims=neuron_dims,
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
            device=device
        )

        # classification head
        self.logsoft = nn.LogSoftmax(dim=-1)

    def forward(self, x, sc):
        # x: [B, C, T]
        if self.temporal_downsample_type == 'conv':
            for i in range(len(self.downsample_factor)):
                temporal_downsample_layer = getattr(self, f'temporal_downsample_{i}')
                x = temporal_downsample_layer(x)  # [B, C, T] -> [B, C, T_down]
        elif self.temporal_downsample_type is None:
            pass

        x = x.permute(0, 2, 1).contiguous()  # [B, C, T] -> [B, T, C]
        out, spikes = self.s2net_block(x, sc)
        logits_pool = out.mean(dim=2)  # global average pooling over time dimension, [B, class_num]
        return self.logsoft(logits_pool), spikes


class SAIE_Learned_Global(nn.Module):
    """
    this code is from https://github.com/Jiazhen-Hong/SAMBA/blob/main/Models/SAMBA.py
    paper: https://arxiv.org/pdf/2511.18571v1
    Spatio-Adaptive Input Embedding (SAIE)

    args:
    - hidden_dim: hidden dimension of the MLP used to compute spatially adaptive weights

    static method:
    - _coords_to_tensor:
        Convert input coordinates to a tensor of shape [C, 3]. Supports both tensor and dict formats.
        Supported input formats:
        - torch.Tensor with shape [C, 3]
        - dict / OrderedDict: {channel_name: (x, y, z), ...}

    forward method:
    - forward(x, input_coords, target_coords):
        Compute the spatio-adaptive input embedding.
        Args:
        - x: Tensor of shape [B, C_in, T], where C_in is the number of input channels.
        - input_coords:
            - Tensor of shape [C_in, 3], or
            - dict / ordered mapping from channel name to 3D coordinates
        - target_coords:
            - Tensor of shape [C_out, 3], or
            - dict / ordered mapping from channel name to 3D coordinates

        Returns:
        - x_out: Tensor of shape [B, C_out, T], projected into the target coordinate space.
        - weights: Tensor of shape [C_out, C_in], representing learned spatial projection weights.
    """
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    @staticmethod
    def _coords_to_tensor(coords, device):
        if isinstance(coords, torch.Tensor):
            if coords.ndim != 2 or coords.shape[1] != 3:
                raise ValueError(f"Expected coords tensor shape [C, 3], but got {tuple(coords.shape)}")
            return coords.to(device=device, dtype=torch.float32)

        if isinstance(coords, dict):
            return torch.tensor(list(coords.values()), dtype=torch.float32, device=device)

        raise TypeError(
            f"Unsupported coords type: {type(coords)}. "
            f"Expected torch.Tensor or dict / OrderedDict."
        )

    def forward(self, x, input_coords, target_coords):
        B, C_in, T = x.shape
        device = x.device

        in_pos = self._coords_to_tensor(input_coords, device)
        out_pos = self._coords_to_tensor(target_coords, device)

        if in_pos.shape[0] != C_in:
            raise ValueError(
                f"Input coords channel count mismatch: x has C_in={C_in}, "
                f"but input_coords has {in_pos.shape[0]} rows."
            )

        # Compute pairwise 3D differences for all (out, in) channel pairs → [C_out, C_in, 3]
        diff = out_pos[:, None, :] - in_pos[None, :, :]

        # Predict weights via MLP and normalize → [C_out, C_in]
        weights = self.net(diff).squeeze(-1)
        weights = torch.softmax(weights, dim=1)

        # Apply spatial projection → [B, C_out, T]
        x_out = torch.einsum('oc,bct->bot', weights, x)
        return x_out, weights


class FixedGeometricResampler(nn.Module):
    """
    Fixed geometric resampler with batch-wise input coordinates.

    Input:
        x_src:       [B, C_in, T]
        input_coords:[B, C_in, 3]

    Fixed buffer:
        target_coords:[C_out, 3]

    Output:
        z_tgt:       [B, C_out, T]
        W:           [B, C_out, C_in]

    No learnable parameters.
    """

    def __init__(
        self,
        target_coords: torch.Tensor,
        sigma: float = 0.15,
        eps: float = 1e-8,
    ):
        super().__init__()

        if not isinstance(target_coords, torch.Tensor):
            raise TypeError(f"target_coords must be torch.Tensor, got {type(target_coords)}")

        if target_coords.ndim != 2 or target_coords.shape[1] != 3:
            raise ValueError(
                f"Expected target_coords shape [C_out, 3], got {tuple(target_coords.shape)}"
            )

        self.register_buffer("target_coords", target_coords.to(dtype=torch.float32))
        self.sigma = float(sigma)
        self.eps = float(eps)

    def build_weights(self, input_coords: torch.Tensor) -> torch.Tensor:
        """
        Build batch-wise fixed geometric weights.

        Args:
            input_coords: [B, C_in, 3]

        Returns:
            W: [B, C_out, C_in]
        """
        if not isinstance(input_coords, torch.Tensor):
            raise TypeError(f"input_coords must be torch.Tensor, got {type(input_coords)}")

        if input_coords.ndim != 3 or input_coords.shape[-1] != 3:
            raise ValueError(
                f"Expected input_coords shape [B, C_in, 3], got {tuple(input_coords.shape)}"
            )

        device = self.target_coords.device
        input_coords = input_coords.to(device=device, dtype=torch.float32)

        # target_coords: [C_out, 3]
        # input_coords:  [B, C_in, 3]
        # diff:          [B, C_out, C_in, 3]
        pos_out = self.target_coords.unsqueeze(0).unsqueeze(2)   # [1, C_out, 1, 3]
        pos_in = input_coords.unsqueeze(1)                       # [B, 1, C_in, 3]

        diff = pos_out - pos_in
        dist2 = (diff ** 2).sum(dim=-1)                          # [B, C_out, C_in]

        denom = max(2.0 * self.sigma * self.sigma, self.eps)
        logits = -dist2 / denom                                  # [B, C_out, C_in]

        W = torch.softmax(logits, dim=-1)                        # [B, C_out, C_in]
        return W

    def forward(self, x_src: torch.Tensor, input_coords: torch.Tensor):
        """
        Args:
            x_src:        [B, C_in, T]
            input_coords: [B, C_in, 3]

        Returns:
            z_tgt: [B, C_out, T]
            W:     [B, C_out, C_in]
        """
        if not isinstance(x_src, torch.Tensor):
            raise TypeError(f"x_src must be torch.Tensor, got {type(x_src)}")

        if x_src.ndim != 3:
            raise ValueError(f"Expected x_src shape [B, C_in, T], got {tuple(x_src.shape)}")

        if input_coords.ndim != 3 or input_coords.shape[-1] != 3:
            raise ValueError(
                f"Expected input_coords shape [B, C_in, 3], got {tuple(input_coords.shape)}"
            )

        B, C_in, T = x_src.shape
        B_pos, C_pos, D_pos = input_coords.shape

        if B_pos != B:
            raise ValueError(
                f"Batch size mismatch: x_src has B={B}, input_coords has B={B_pos}"
            )
        if C_pos != C_in:
            raise ValueError(
                f"Channel mismatch: x_src has C_in={C_in}, input_coords has C_in={C_pos}"
            )
        if D_pos != 3:
            raise ValueError(f"Last dim of input_coords must be 3, got {D_pos}")

        x_src = x_src.to(device=self.target_coords.device)
        W = self.build_weights(input_coords)                     # [B, C_out, C_in]

        # z_tgt[b, o, t] = sum_c W[b, o, c] * x_src[b, c, t]
        z_tgt = torch.einsum("boc,bct->bot", W, x_src)

        return z_tgt, W


def build_source_time_mask_from_flat_indices(
    batch_mask,
    batch_size: int,
    n_chans: int,
    seq_len: int,
    patch_size: int,
    patch_stride: int,
    device,
):
    """
    Args:
        batch_mask:
            - Tensor [B, N_mask], or
            - list/tuple of length B, each item is 1D tensor of flat masked indices

    Returns:
        m_src: [B, C_in, T] in {0,1}
    """
    m_src = torch.zeros((batch_size, n_chans, seq_len), dtype=torch.float32, device=device)

    if isinstance(batch_mask, torch.Tensor):
        if batch_mask.ndim == 1:
            batch_mask = batch_mask.unsqueeze(0).expand(batch_size, -1)
        elif batch_mask.ndim != 2:
            raise ValueError(f"batch_mask tensor should be [B, N_mask], got {tuple(batch_mask.shape)}")
        masks_iter = [batch_mask[b] for b in range(batch_size)]
    else:
        masks_iter = batch_mask

    for b, flat_ids in enumerate(masks_iter):
        if flat_ids is None:
            continue

        flat_ids = torch.as_tensor(flat_ids, device=device, dtype=torch.long).view(-1)
        chan_idx = flat_ids % n_chans
        patch_idx = flat_ids // n_chans

        for c, p in zip(chan_idx.tolist(), patch_idx.tolist()):
            start = p * patch_stride
            end = min(start + patch_size, seq_len)
            if start < seq_len:
                m_src[b, c, start:end] = 1.0

    return m_src


def project_mask_to_target_space(m_src: torch.Tensor, W: torch.Tensor):
    """
    Args:
        m_src: [B, C_in, T]
        W:     [B, C_out, C_in]

    Returns:
        m_tar: [B, C_out, T], soft mask in [0, 1]
    """
    if not isinstance(m_src, torch.Tensor):
        raise TypeError(f"m_src must be torch.Tensor, got {type(m_src)}")
    if not isinstance(W, torch.Tensor):
        raise TypeError(f"W must be torch.Tensor, got {type(W)}")

    if m_src.ndim != 3:
        raise ValueError(f"Expected m_src shape [B, C_in, T], got {tuple(m_src.shape)}")
    if W.ndim != 3:
        raise ValueError(f"Expected W shape [B, C_out, C_in], got {tuple(W.shape)}")

    Bm, C_in_m, _ = m_src.shape
    Bw, _, C_in_w = W.shape

    if Bm != Bw:
        raise ValueError(f"Batch mismatch: m_src has B={Bm}, W has B={Bw}")
    if C_in_m != C_in_w:
        raise ValueError(f"Channel mismatch: m_src has C_in={C_in_m}, W has C_in={C_in_w}")

    m_src = m_src.to(dtype=torch.float32, device=W.device)
    m_tar = torch.einsum("boc,bct->bot", W, m_src)
    return m_tar.clamp(0.0, 1.0)


class CanonicalTargetPreprocessor(nn.Module):
    """
    Fixed preprocessing inside model:
        source-space -> target-space
        source mask  -> target mask
    """

    def __init__(
        self,
        target_coords: torch.Tensor,
        sigma: float,
        patch_size: int,
        patch_stride: int,
    ):
        super().__init__()
        self.resampler = FixedGeometricResampler(
            target_coords=target_coords,
            sigma=sigma,
        )
        self.patch_size = int(patch_size)
        self.patch_stride = int(patch_stride)

    def forward(
        self,
        x_src: torch.Tensor,
        input_coords: torch.Tensor,
        batch_mask=None,
    ):
        """
        Args:
            x_src:        [B, C_in, T]
            input_coords: [B, C_in, 3]
            batch_mask:
                - Tensor [B, N_mask], or
                - list/tuple length B, each item 1D masked flat indices, or
                - None

        Returns:
            z_clean: [B, C_out, T]
            z_vis:   [B, C_out, T]
            m_src:   [B, C_in, T] or None
            m_tar:   [B, C_out, T] or None
            W:       [B, C_out, C_in]
        """
        if not isinstance(x_src, torch.Tensor):
            raise TypeError(f"x_src must be torch.Tensor, got {type(x_src)}")
        if not isinstance(input_coords, torch.Tensor):
            raise TypeError(f"input_coords must be torch.Tensor, got {type(input_coords)}")

        if x_src.ndim != 3:
            raise ValueError(f"Expected x_src [B, C_in, T], got {tuple(x_src.shape)}")
        if input_coords.ndim != 3 or input_coords.shape[-1] != 3:
            raise ValueError(
                f"Expected input_coords [B, C_in, 3], got {tuple(input_coords.shape)}"
            )

        B, C_in, T = x_src.shape
        device = x_src.device

        z_clean, W = self.resampler(x_src, input_coords)

        if batch_mask is None:
            return z_clean, z_clean, None, None, W

        m_src = build_source_time_mask_from_flat_indices(
            batch_mask=batch_mask,
            batch_size=B,
            n_chans=C_in,
            seq_len=T,
            patch_size=self.patch_size,
            patch_stride=self.patch_stride,
            device=device,
        )

        x_masked = (1.0 - m_src) * x_src
        z_vis, _ = self.resampler(x_masked, input_coords)
        m_tar = project_mask_to_target_space(m_src, W)

        return z_clean, z_vis, m_src, m_tar, W


# =========================================================
# Target-space learnable mask token
# =========================================================

class TargetSpaceMaskToken(nn.Module):
    """
    Learnable mask token in target-space.
    """

    def __init__(self, num_channels: int, mode: str = "replace", init_std: float = 0.02):
        super().__init__()
        if mode not in {"replace", "add"}:
            raise ValueError(f"Unsupported mode: {mode}")

        self.mode = mode
        self.mask_token = nn.Parameter(torch.zeros(1, num_channels, 1))
        nn.init.trunc_normal_(self.mask_token, std=init_std)

    def forward(self, z_vis: torch.Tensor, m_tar: torch.Tensor):
        """
        Args:
            z_vis: [B, C_out, T]
            m_tar: [B, C_out, T]
        """
        if z_vis.shape != m_tar.shape:
            raise ValueError(f"Shape mismatch: z_vis={tuple(z_vis.shape)}, m_tar={tuple(m_tar.shape)}")

        m_tar = m_tar.to(dtype=z_vis.dtype, device=z_vis.device).clamp(0.0, 1.0)

        if self.mode == "replace":
            return (1.0 - m_tar) * z_vis + m_tar * self.mask_token
        return z_vis + m_tar * self.mask_token


class SAOE_ResidualCorrection(nn.Module):
    """
    Learnable residual correction:
    delta_source = scale * R(target_coords, source_coords) @ x_target
    """
    def __init__(self, hidden_dim=32, init_scale=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))

    @staticmethod
    def _coords_to_tensor(coords, device):
        if isinstance(coords, torch.Tensor):
            return coords.to(device=device, dtype=torch.float32)
        if isinstance(coords, dict):
            return torch.tensor(list(coords.values()), dtype=torch.float32, device=device)
        raise TypeError(f"Unsupported coords type: {type(coords)}")

    def forward(self, x, target_coords, source_coords):
        """
        x: [B, C_target, T]
        returns:
            delta_source: [B, C_source, T]
            delta_w: [C_source, C_target]
        """
        device = x.device
        target_pos = self._coords_to_tensor(target_coords, device)   # [C_tgt, 3]
        source_pos = self._coords_to_tensor(source_coords, device)   # [C_src, 3]

        diff = source_pos[:, None, :] - target_pos[None, :, :]       # [C_src, C_tgt, 3]
        delta_w = self.net(diff).squeeze(-1)                         # [C_src, C_tgt]

        delta_w = delta_w / (delta_w.abs().mean(dim=1, keepdim=True) + 1e-6)

        alpha = torch.tanh(self.scale)   # scale in (-1, 1)
        delta_source = alpha * torch.einsum('sc,bct->bst', delta_w, x)

        return delta_source, delta_w


class SharedTemporalResidualRefiner(nn.Module):
    """
    Channel-count-agnostic residual refiner.
    Apply the same temporal refiner to every source channel independently.

    Input:  x_base [B, C_src, T]
    Output: delta  [B, C_src, T]
    """
    def __init__(self, hidden_dim=16, kernel_size=7, init_scale=0.0):
        super().__init__()
        pad = kernel_size // 2

        self.net = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=kernel_size, padding=pad, bias=True),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=pad, bias=True),
            nn.GELU(),
            nn.Conv1d(hidden_dim, 1, kernel_size=1, bias=True),
        )
        self.scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))

    def forward(self, x_base):
        # x_base: [B, C_src, T]
        B, C, T = x_base.shape

        # [B, C, T] -> [B*C, 1, T]
        x = x_base.reshape(B * C, 1, T)

        delta = self.net(x)  # [B*C, 1, T]
        delta = delta.reshape(B, C, T)

        alpha = torch.tanh(self.scale)
        return alpha * delta


class TemporalUpsampleLayer_Conv(nn.Module):
    """
    A temporal up sampling layer that increases the sequence length by a factor of `upsample_factor`.

    Input Data Format: [B, C, T]
        - B: batch size
        - T: sequence length
        - C: feature / channel dimension

    Output Data Format: [B, C, T_up]

    Notes:
    - Main upsampling is done by ConvTranspose1d.
    - Because exact inverse length is not always guaranteed, you can pass target_len
      to force the final temporal length to match a desired value exactly.
    """

    def __init__(
            self,
            in_channels,
            upsample_factor,
            kernel_size=None,
            bias=False,
            use_bn=True,
            activation=True,
            dropout=0.0,
            depthwise=True,
            align_mode="linear",   # "linear" or "nearest"
    ):
        super(TemporalUpsampleLayer_Conv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.upsample_factor = upsample_factor
        self.kernel_size = kernel_size if kernel_size is not None else upsample_factor
        self.bias = bias
        self.use_bn = use_bn
        self.activation = activation
        self.dropout_p = dropout
        self.depthwise = depthwise
        self.align_mode = align_mode

        groups = in_channels if self.depthwise else 1
        padding = self.kernel_size // 2

        self.deconv = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=self.kernel_size,
            stride=self.upsample_factor,
            padding=padding,
            output_padding=max(0, self.upsample_factor - 1),
            groups=groups,
            bias=self.bias,
        )

        self.bn = nn.BatchNorm1d(in_channels) if self.use_bn else None
        self.act = nn.GELU() if activation else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def _match_target_length(self, x, target_len):
        cur_len = x.size(-1)

        if cur_len == target_len:
            return x

        if abs(cur_len - target_len) >= 2 * self.upsample_factor:
            raise ValueError(
                f"Current length {cur_len} and target length {target_len} "
                f"differ by more than 2x upsample_factor {self.upsample_factor}. "
                f"Consider adjusting the upsample_factor or"
                f" checking the input sequence length and target_len for consistency."
            )

        if self.align_mode in ["linear", "nearest"]:
            if self.align_mode == "linear":
                x = F.interpolate(x, size=target_len, mode="linear", align_corners=False)
            else:
                x = F.interpolate(x, size=target_len, mode="nearest")
            return x

        # fallback to manually cropping/padding if align_mode is not recognized
        if cur_len > target_len:
            x = x[..., :target_len]
        else:
            pad_len = target_len - cur_len
            x = F.pad(x, (0, pad_len))
        return x

    def forward(self, input_seq, target_len=None):
        if input_seq.dim() != 3:
            raise ValueError(f"Expected input_seq with shape [B, C, T], but got {tuple(input_seq.shape)}")

        batch_size, channel_num, seq_len = input_seq.size()

        if channel_num != self.in_channels:
            raise ValueError(
                f"Input channel mismatch: expected {self.in_channels}, but got {channel_num}"
            )

        # [B, T, C] -> [B, C, T]
        # x = input_seq.permute(0, 2, 1).contiguous()
        x = input_seq
        # transposed convolution upsampling
        x = self.deconv(x)

        if self.bn is not None:
            x = self.bn(x)

        if self.act is not None:
            x = self.act(x)

        if self.dropout is not None:
            x = self.dropout(x)

        # force exact temporal length if needed
        if target_len is not None:
            x = self._match_target_length(x, target_len)

        # [B, C, T_up] -> [B, T_up, C]
        # output_seq = x.permute(0, 2, 1).contiguous()
        output_seq = x
        return output_seq  # [B, C, T_up]


class S2NetRecon(nn.Module):
    """
    A complete S2Net model for sequence reconstruction.
    Includes the SAIE layer for spatially adaptive input embedding, kuramoto layer for spatio-temporal dynamics,
    Rhythm SNN layers for temporal processing, and a readout layer for reconstruction.

    Inputs:
    - input_seq: [B, T, C] where B is batch size, T is sequence length, C is number of channels (e.g., brain regions)
    - sc: [B, C, C] structural connectivity matrix for the kuramoto layer
    - target_coords: coordinates of the target channels for SAIE and rhythm modulation
    - input_coords: coordinates of the input channels for SAIE
    """
    def __init__(
            self,
            target_coords,
            saie_hidden_dim,
            neuron_dims,
            osc_dim,
            is_input_encode,
            encoder_hidden_dim,
            temporal_downsample_type,
            downsample_factor,
            k,
            dt,
            phase_delay_steps,
            low_n,
            high_n,
            branch,
            low_m,
            high_m,
            vth_snn,
            dendritic_branch_constraint,
            mem_input,
            vth_readout,
            device
    ):
        super().__init__()

        self.target_coords = target_coords
        num_regions = len(target_coords)
        self.saie_layer = SAIE_Learned_Global(hidden_dim=saie_hidden_dim).to(device)
        self.temporal_downsample_type = temporal_downsample_type
        self.downsample_factor = downsample_factor  # [5, 10, 20] for example

        if self.temporal_downsample_type == 'conv':
            for i in range(len(self.downsample_factor)):
                setattr(
                    self,
                    f'temporal_downsample_{i}',
                    TemporalDownsampleLayer_Conv(
                        in_channels=num_regions,
                        downsample_factor=self.downsample_factor[i],
                        kernel_size=2 * self.downsample_factor[i] - 1,
                        bias=True,
                        use_bn=True,
                        activation=True,
                        dropout=0.0,
                        depthwise=True
                    ).to(device)
                )

                setattr(
                    self,
                    f'temporal_upsample_{i}',
                    TemporalUpsampleLayer_Conv(
                        in_channels=num_regions,
                        upsample_factor=self.downsample_factor[i],
                        kernel_size=2 * self.downsample_factor[i] - 1,
                        bias=False,
                        use_bn=True,
                        activation=True,
                        dropout=0.0,
                        depthwise=True,
                        align_mode="linear"
                    ).to(device)
                )
        elif self.temporal_downsample_type == 'patch':
            if len(self.downsample_factor) != 1:
                raise ValueError("For patch-based downsampling, downsample_factor should"
                                 " be a list with a single integer representing the patch size.")

            patch_size = self.downsample_factor[0]
            # downsample
            setattr(
                self,
                f'temporal_downsample_0',
                TemporalDownsampleLayer_Patch(
                    in_channels=num_regions,
                    patch_size=patch_size,
                    stride=patch_size,
                    embed_dim=None,
                    bias=True
                ).to(device)
            )
            # upsample
            setattr(
                self,
                f'temporal_upsample_0',
                TemporalUpsampleLayer_Patch(
                    out_channels=num_regions,
                    patch_size=patch_size,
                    stride=patch_size,
                    embed_dim=None,
                    bias=True
                ).to(device)
            )

        if self.temporal_downsample_type == 'patch':
            num_regions = num_regions * self.downsample_factor[0]

        self.s2net_block = S2NetBlock(
            num_regions=num_regions,
            output_dim=num_regions,
            osc_dim=osc_dim,
            neuron_dims=neuron_dims,
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
            device=device
        )

        #  MLP decoder
        #self.decoder = nn.Sequential(
        #    nn.Linear(num_regions, num_regions * 2),
        #    nn.LayerNorm(num_regions * 2),
        #    nn.LeakyReLU(0.1),
        #    nn.Dropout(0.1),
        #    nn.Linear(num_regions * 2, num_regions * 2),
        #    nn.LayerNorm(num_regions * 2),
        #    nn.LeakyReLU(0.1),
        #    nn.Linear(num_regions * 2, num_regions)
        #)

        self.decoder = nn.Sequential(
            nn.Conv1d(num_regions, num_regions * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(num_regions * 2, num_regions * 2, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(num_regions * 2, num_regions, kernel_size=1)
        )

    def unpatchify(self, patch_seq):
        """
        Convert patch-level reconstruction back to temporal sequence.

        Args:
            patch_seq: [B, N, C_out * patch_size]

        Returns:
            recon_seq: [B, T, C_out]
        """
        if self.temporal_downsample_type != 'patch':
            raise ValueError("unpatchify is only valid when temporal_downsample_type == 'patch'")

        B, N, D = patch_seq.shape
        C = self.target_dim
        P = self.patch_size

        if D != C * P:
            raise ValueError(
                f"Patch reconstruction dim mismatch: expected {C * P}, but got {D}"
            )

        # [B, N, C*P] -> [B, N, C, P]
        x = patch_seq.view(B, N, C, P)

        # [B, N, C, P] -> [B, N, P, C]
        x = x.permute(0, 1, 3, 2).contiguous()

        # [B, N, P, C] -> [B, N*P, C]
        recon_seq = x.view(B, N * P, C)

        return recon_seq

    def forward(self, input_seq, input_coords, sc):
        # input_seq: [B, C_in, T]

        # SAIE projection, [B, C_in, T] → [B, C_out, T]
        input_seq, ch_weights = self.saie_layer(input_seq, input_coords, self.target_coords)
        target_seq = input_seq.detach()  # [B, C_out, T], used as target for reconstruction loss, detached from gradient flow

        # temporal downsample
        x = input_seq
        downsampled_lens = []

        if self.temporal_downsample_type == 'conv':
            for i in range(len(self.downsample_factor)):
                x = getattr(self, f'temporal_downsample_{i}')(x)
                downsampled_lens.append(x.size(2))

            # [B, C_out, T_down] -> [B, T_down, C_out]
            x = x.permute(0, 2, 1).contiguous()

        elif self.temporal_downsample_type == 'patch':
            # [B, C_out, T] -> [B, T, C_out]
            x = x.permute(0, 2, 1).contiguous()

            # patch downsample: [B, T, C_out] -> [B, N, C_out * patch_size]
            x = getattr(self, f'temporal_downsample_0')(x)

        else:
            # no downsample
            x = x.permute(0, 2, 1).contiguous()  # [B, C_out, T] -> [B, T, C_out]

        # S2Net block
        out, spikes = self.s2net_block(x, sc)

        # reconstruction path
        if self.temporal_downsample_type == 'conv':
            x = out  # [B, C_out, T_down]

            # upsample back to original temporal resolution
            for i in reversed(range(len(self.downsample_factor))):
                target_len = downsampled_lens[i - 1] if i > 0 else target_seq.size(2)
                x = getattr(self, f'temporal_upsample_{i}')(x, target_len=target_len)

            # decoder
            # x = x.permute(0, 2, 1).contiguous()  # [B, C_out, T] -> [B, T, C_out]
            recon_target_seq = self.decoder(x)  # [B, T, C_out]
            # recon_target_seq = recon_target_seq.permute(0, 2, 1).contiguous()  # [B, C_out, T]

        elif self.temporal_downsample_type == 'patch':
            x = out.permute(0, 2, 1).contiguous()  # [B, C_out * patch_size, N] -> [B, N, C_out * patch_size]

            # decoder in patch space
            x = self.decoder(x)  # [B, N, C_out * patch_size]

            # unpatch back to temporal sequence
            x = getattr(self, f'temporal_upsample_0')(x, target_len=target_seq.size(2))  # [B, T, C_out]
            recon_target_seq = x.permute(0, 2, 1).contiguous()  # [B, C_out, T]

        else:
            x = out.permute(0, 2, 1).contiguous()  # [B, C_out, T] -> [B, T, C_out]
            recon_target_seq = self.decoder(x)  # [B, T, C_out]
            recon_target_seq = recon_target_seq.permute(0, 2, 1).contiguous()  # [B, C_out, T]

        eps = 1e-4
        Wt = ch_weights.transpose(0, 1)  # [C_source, C_target]
        W_out = torch.linalg.inv(Wt @ ch_weights + eps * torch.eye(Wt.size(0), device=Wt.device)) @ Wt

        recon_source_seq_base = torch.einsum('sc,bct->bst', W_out, recon_target_seq)
        recon_source_seq = recon_source_seq_base

        return recon_source_seq, recon_target_seq, target_seq, spikes, ch_weights


class S2NetReconTargetSpace(nn.Module):
    """
    New reconstruction model:
        source-space input -> fixed preprocessing -> target-space masking -> S2 backbone

    This version:
    - removes internal SAIE
    - removes pseudo-inverse source reconstruction
    - works only in target-space
    - supports fixed preprocessing + learnable target-space mask token
    """

    def __init__(
            self,
            target_coords,
            sigma,
            patch_size,
            patch_stride,
            neuron_dims,
            osc_dim,
            is_input_encode,
            encoder_hidden_dim,
            temporal_downsample_type,
            downsample_factor,
            k,
            dt,
            phase_delay_steps,
            low_n,
            high_n,
            branch,
            low_m,
            high_m,
            vth_snn,
            dendritic_branch_constraint,
            mem_input,
            vth_readout,
            device,
            use_mask_token: bool = True,
            mask_token_mode: str = "replace",
    ):
        super().__init__()

        if temporal_downsample_type != "conv":
            raise NotImplementedError(
                "This target-space version is currently written for temporal_downsample_type='conv' only."
            )

        if not isinstance(target_coords, torch.Tensor):
            target_coords = torch.tensor(target_coords, dtype=torch.float32)

        self.target_coords = target_coords.to(dtype=torch.float32)
        self.target_dim = self.target_coords.shape[0]
        self.temporal_downsample_type = temporal_downsample_type
        self.downsample_factor = list(downsample_factor)
        self.device = device
        self.use_mask_token = bool(use_mask_token)

        # fixed preprocessing inside model
        self.preprocessor = CanonicalTargetPreprocessor(
            target_coords=self.target_coords,
            sigma=sigma,
            patch_size=patch_size,
            patch_stride=patch_stride,
        )

        # learnable target-space mask token
        if self.use_mask_token:
            self.mask_token = TargetSpaceMaskToken(
                num_channels=self.target_dim,
                mode=mask_token_mode,
                init_std=0.02,
            )

        # temporal conv down / up
        for i in range(len(self.downsample_factor)):
            setattr(
                self,
                f"temporal_downsample_{i}",
                TemporalDownsampleLayer_Conv(
                    in_channels=self.target_dim,
                    downsample_factor=self.downsample_factor[i],
                    kernel_size=2 * self.downsample_factor[i] - 1,
                    bias=True,
                    use_bn=True,
                    activation=True,
                    dropout=0.1,
                    depthwise=True,
                ).to(device)
            )

            setattr(
                self,
                f"temporal_upsample_{i}",
                TemporalUpsampleLayer_Conv(
                    in_channels=self.target_dim,
                    upsample_factor=self.downsample_factor[i],
                    kernel_size=2 * self.downsample_factor[i] - 1,
                    bias=False,
                    use_bn=True,
                    activation=True,
                    dropout=0.1,
                    depthwise=True,
                    align_mode="linear",
                ).to(device)
            )

        self.s2net_block = S2NetBlock(
            num_regions=self.target_dim,
            output_dim=self.target_dim,
            osc_dim=osc_dim,
            neuron_dims=neuron_dims,
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

        self.decoder = nn.Sequential(
            nn.Conv1d(self.target_dim, self.target_dim * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(self.target_dim * 2, self.target_dim * 2, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(self.target_dim * 2, self.target_dim, kernel_size=1),
        )

    def _prepare_sc(self, sc: torch.Tensor, batch_size: int, device):
        """
        Accept:
            sc: [C, C] or [1, C, C] or [B, C, C]
        Return:
            [B, C, C]
        """
        if not isinstance(sc, torch.Tensor):
            raise TypeError(f"sc must be torch.Tensor, got {type(sc)}")

        sc = sc.to(device=device, dtype=torch.float32)

        if sc.ndim == 2:
            sc = sc.unsqueeze(0).expand(batch_size, -1, -1)
        elif sc.ndim == 3:
            if sc.shape[0] == 1:
                sc = sc.expand(batch_size, -1, -1)
            elif sc.shape[0] != batch_size:
                raise ValueError(f"sc batch mismatch: expected {batch_size}, got {sc.shape[0]}")
        else:
            raise ValueError(f"Expected sc [C,C] or [B,C,C], got {tuple(sc.shape)}")

        return sc

    def forward(
            self,
            x_src: torch.Tensor,
            input_coords: torch.Tensor,
            sc: torch.Tensor,
            batch_mask=None,
    ):
        """
        Args:
            x_src:        [B, C_in, T]
            input_coords: [B, C_in, 3]
            sc:           [C_out, C_out] or [B, C_out, C_out]
            batch_mask:   source patch flat mask indices or None

        Returns:
            recon_target_seq: [B, C_out, T]
            target_seq:       [B, C_out, T]
            m_tar:            [B, C_out, T] or None
            spikes:           [B, ..., ...]
            W:                [B, C_out, C_in]
        """
        # fixed preprocessing
        target_seq, z_vis, m_src, m_tar, W = self.preprocessor(
            x_src=x_src,
            input_coords=input_coords,
            batch_mask=batch_mask,
        )

        # optional learnable mask token injection in target space
        if self.use_mask_token and (m_tar is not None):
            x = self.mask_token(z_vis, m_tar)
        else:
            x = z_vis

        B = x.shape[0]
        sc = self._prepare_sc(sc, batch_size=B, device=x.device)

        # temporal downsample
        downsampled_lens = []
        for i in range(len(self.downsample_factor)):
            x = getattr(self, f"temporal_downsample_{i}")(x)  # [B, C_out, T_down]
            downsampled_lens.append(x.size(2))

        # [B, C_out, T_down] -> [B, T_down, C_out]
        x = x.permute(0, 2, 1).contiguous()

        # S2 block
        out, spikes = self.s2net_block(x, sc)  # out: [B, C_out, T_down]

        # upsample back
        x = out
        for i in reversed(range(len(self.downsample_factor))):
            target_len = downsampled_lens[i - 1] if i > 0 else target_seq.size(2)
            x = getattr(self, f"temporal_upsample_{i}")(x, target_len=target_len)

        recon_target_seq = self.decoder(x)  # [B, C_out, T]

        return recon_target_seq, target_seq, m_tar, spikes, W


import torch
import matplotlib.pyplot as plt

from models.model import S2NetReconTargetSpace


def compute_sc_from_pos(pos: torch.Tensor, sigma: float = 0.3) -> torch.Tensor:
    """
    pos: [C, 3]
    return: [C, C]
    """
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"Expected pos [C, 3], got {tuple(pos.shape)}")

    diff = pos.unsqueeze(1) - pos.unsqueeze(0)   # [C, C, 3]
    dist2 = (diff ** 2).sum(dim=-1)              # [C, C]
    sc = torch.exp(-dist2 / (2 * sigma * sigma))
    return sc


def build_random_batch_mask(
    batch_size: int,
    n_chans: int,
    seq_len: int,
    patch_size: int,
    patch_stride: int,
    num_mask_per_sample: int,
    device,
) -> torch.Tensor:
    """
    构造 source patch flatten mask:
        flat_idx = n_chans * patch_idx + chan_idx
    return: [B, N_mask]
    """
    num_patches = 1 + (seq_len - patch_size) // patch_stride
    total_tokens = n_chans * num_patches

    masks = []
    for _ in range(batch_size):
        perm = torch.randperm(total_tokens, device=device)
        flat_ids = perm[:num_mask_per_sample]
        masks.append(flat_ids)

    return torch.stack(masks, dim=0)  # [B, N_mask]


def count_params_in_m(model) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def main():
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device)

    # --------------------------------------------------
    # random settings
    # --------------------------------------------------
    B = 2
    C_in = 16
    C_out = 64
    T = 256

    patch_size = 32
    patch_stride = 16
    num_mask_per_sample = 40

    # --------------------------------------------------
    # random source data
    # --------------------------------------------------
    x_src = torch.randn(B, C_in, T, device=device)            # [B, C_in, T]
    input_coords = torch.randn(B, C_in, 3, device=device)     # [B, C_in, 3]
    target_coords = torch.randn(C_out, 3, device=device)      # [C_out, 3]

    sc_64 = compute_sc_from_pos(target_coords, sigma=0.3).to(device)   # [64, 64]

    batch_mask = build_random_batch_mask(
        batch_size=B,
        n_chans=C_in,
        seq_len=T,
        patch_size=patch_size,
        patch_stride=patch_stride,
        num_mask_per_sample=num_mask_per_sample,
        device=device,
    )  # [B, N_mask]

    # --------------------------------------------------
    # build model
    # --------------------------------------------------
    model = S2NetReconTargetSpace(
        target_coords=target_coords,
        sigma=0.2,
        patch_size=patch_size,
        patch_stride=patch_stride,
        neuron_dims=[64, 64, 64, 64],
        osc_dim=2,
        is_input_encode=True,
        encoder_hidden_dim=32,
        temporal_downsample_type="conv",
        downsample_factor=[4, 4],
        k=1.0,
        dt=1,
        phase_delay_steps=1,
        low_n=0.0,
        high_n=4.0,
        branch=8,
        low_m=0.0,
        high_m=4.0,
        vth_snn=0.5,
        dendritic_branch_constraint=False,
        mem_input=False,
        vth_readout=0.5,
        device=str(device),
        use_mask_token=True,
        mask_token_mode="replace",
    ).to(device)

    total_params_m = count_params_in_m(model)
    trainable_params_m = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

    print(f"Total params     : {total_params_m:.6f} M")
    print(f"Trainable params : {trainable_params_m:.6f} M")

    # --------------------------------------------------
    # inspect preprocessing + forward
    # --------------------------------------------------
    model.eval()
    with torch.no_grad():
        # 先看前处理结果
        z_clean, z_vis, m_src, m_tar, W = model.preprocessor(
            x_src=x_src,
            input_coords=input_coords,
            batch_mask=batch_mask,
        )

        # 看加了 target-space learnable mask token 之后的输入
        if model.use_mask_token and m_tar is not None:
            z_with_token = model.mask_token(z_vis, m_tar)
        else:
            z_with_token = z_vis

        # 完整 forward
        recon_target_seq, target_seq, m_tar_fwd, spikes, W_fwd = model(
            x_src=x_src,
            input_coords=input_coords,
            sc=sc_64,
            batch_mask=batch_mask,
        )

    print("x_src shape            :", tuple(x_src.shape))
    print("input_coords shape     :", tuple(input_coords.shape))
    print("batch_mask shape       :", tuple(batch_mask.shape))
    print("z_clean shape          :", tuple(z_clean.shape))
    print("z_vis shape            :", tuple(z_vis.shape))
    print("m_src shape            :", tuple(m_src.shape) if m_src is not None else None)
    print("m_tar shape            :", tuple(m_tar.shape) if m_tar is not None else None)
    print("recon_target_seq shape :", tuple(recon_target_seq.shape))
    print("W shape                :", tuple(W.shape))
    print("spikes shape           :", tuple(spikes.shape))

    # --------------------------------------------------
    # plot sample 0
    # --------------------------------------------------
    b = 0
    src_mask_img = m_src[b].detach().cpu().numpy()                # [C_in, T]
    tar_mask_img = m_tar[b].detach().cpu().numpy()                # [C_out, T]
    z_vis_img = z_vis[b].detach().cpu().numpy()                   # [C_out, T]
    z_with_token_img = z_with_token[b].detach().cpu().numpy()     # [C_out, T]

    # 1) source-space dense mask
    plt.figure(figsize=(10, 4))
    plt.imshow(src_mask_img, aspect="auto", origin="lower")
    plt.title("Source-space dense mask (sample 0)")
    plt.xlabel("Time")
    plt.ylabel("Source channel")
    plt.tight_layout()
    plt.savefig("source_mask_sample0.png")

    # 2) target-space soft mask
    plt.figure(figsize=(10, 4))
    plt.imshow(tar_mask_img, aspect="auto", origin="lower")
    plt.title("Target-space soft mask (sample 0)")
    plt.xlabel("Time")
    plt.ylabel("Target channel")
    plt.tight_layout()
    plt.savefig("target_mask_sample0.png")

    # 3) source mask 后投到 target-space 的数据
    plt.figure(figsize=(10, 4))
    plt.imshow(z_vis_img, aspect="auto", origin="lower")
    plt.title("Target-space projected data after source masking (sample 0)")
    plt.xlabel("Time")
    plt.ylabel("Target channel")
    plt.tight_layout()
    plt.savefig("target_space_after_mask_sample0.png")

    # 4) 加了 learnable mask token 之后的 target-space 输入
    plt.figure(figsize=(10, 4))
    plt.imshow(z_with_token_img, aspect="auto", origin="lower")
    plt.title("Target-space input after learnable mask token injection (sample 0)")
    plt.xlabel("Time")
    plt.ylabel("Target channel")
    plt.tight_layout()
    plt.savefig("target_space_after_mask_token_sample0.png")

    # 5) 一个目标通道的 clean / visible / recon 波形
    ch = 10
    plt.figure(figsize=(10, 4))
    plt.plot(target_seq[b, ch].detach().cpu().numpy(), label="target clean")
    plt.plot(z_vis[b, ch].detach().cpu().numpy(), label="target visible")
    plt.plot(recon_target_seq[b, ch].detach().cpu().numpy(), label="model recon")
    plt.title(f"Waveforms at target channel {ch} (sample 0)")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig("waveforms_sample0.png")


if __name__ == "__main__":
    main()






