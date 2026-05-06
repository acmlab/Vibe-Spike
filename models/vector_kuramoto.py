import torch
import torch.nn as nn
import math


class VectorKuramoto(nn.Module):
    def __init__(
        self,
        N,              # H: number of oscillators (latent dims)
        D=2,            # osc_dim: internal dimension per oscillator
        K=1.0,
        dt=1.0,
        device="cuda",
    ):
        super().__init__()
        self.N = N
        self.D = D
        self.K = K
        self.dt = dt
        self.device = device

        # Learnable natural frequency ω_i,k
        self.omega = nn.Parameter(torch.zeros(N, D))

        # Learnable stiffness κ_i,k (control strength)
        self.kappa = nn.Parameter(torch.ones(N, D))

        # Learnable phase offsets α_ij 
        self.alpha = nn.Parameter(torch.zeros(N, N))

    def forward(self, theta_prev, gamma, A=None):

        B, H, D = theta_prev.shape
        assert H == self.N and D == self.D, \
            f"theta_prev shape {theta_prev.shape} does not match (N={self.N}, D={self.D})"

        device = theta_prev.device

        if A is None or A.dim() != 3 or A.size(1) != H or A.size(2) != H:
            A_lat = torch.ones(B, H, H, device=device)
        else:
            A_lat = 0.5 * (A + A.transpose(1, 2))
            A_lat = torch.relu(A_lat)

        #   theta_i: [B, H, 1, D]
        #   theta_j: [B, 1, H, D]
        theta_i = theta_prev.unsqueeze(2)
        theta_j = theta_prev.unsqueeze(1)

        # α_ij: [1, H, H, 1]， D  broadcast
        alpha = self.alpha.view(1, H, H, 1).to(device)

        phase_diff = theta_j - theta_i - alpha          # [B, H, H, D]
        coupling = (self.K / float(H)) * torch.sum(
            A_lat.unsqueeze(-1) * torch.sin(phase_diff), dim=2
        )                                               # [B, H, D]

        omega = self.omega.view(1, H, D).to(device)     # [1,H,D]
        kappa = self.kappa.view(1, H, D).to(device)     # [1,H,D]
        gamma_exp = gamma.unsqueeze(-1)                 # [B,H,1]

        theta_dot = omega + coupling + kappa * (gamma_exp - theta_prev)
        theta_new = theta_prev + self.dt * theta_dot
        return theta_new


class graphVectorKuramoto(nn.Module):
    """
    Graph-Aware Vector Kuramoto with OT-derived Phase Lags.
    Strictly aligns with Eq. (5) and OT Surrogate mechanics.
    """
    def __init__(self, N, D=2, K=1.0, dt=1.0, alpha_scale=1.0, device="cuda"):
        super().__init__()
        self.N = N
        self.D = D
        self.K = K
        self.dt = dt
        self.alpha_scale = alpha_scale # alpha_0 in paper
        self.device = device

        # Natural frequency ω_i (aligns with revised Eq. 5)
        self.omega = nn.Parameter(torch.randn(N, D) * 0.1)

        # Control stiffness κ_i
        self.kappa = nn.Parameter(torch.ones(N, D))
        self.direction_learner = nn.Parameter(torch.randn(N, N) * 0.01)
        # REMOVED: self.alpha = nn.Parameter(...) 
        # Reason: alpha must be derived from A, not learned freely.

    def forward(self, theta_prev, gamma, A=None):
        """
        A : [B, H, H] Connectivity matrix (Structural priors)
        """
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
            A_lat = torch.relu(A_lat) + 1e-6 # Avoid div by zero

            # --- OT-Derived Phase Lag (Section 3.3 in Paper) ---
            # Cost C_ij = 1 / A_ij
            cost_matrix = 1.0 / A_lat 
            
            # Normalize cost to [0, 1] per batch to stabilize
            c_min = cost_matrix.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
            c_max = cost_matrix.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
            norm_cost = (cost_matrix - c_min) / (c_max - c_min + 1e-6)
            direction = self.direction_learner - self.direction_learner.transpose(0, 1) # [N, N]
            direction_mask = torch.tanh(direction)
            alpha_matrix = direction_mask.unsqueeze(0) * norm_cost # [B, N, N]
            # alpha_ij = alpha_0 * norm(C_ij)
            # Expand to [B, H, H, 1] to broadcast over D dim
            # alpha = (self.alpha_scale * norm_cost).unsqueeze(-1)
            alpha = (self.alpha_scale * alpha_matrix).unsqueeze(-1)

        # 2. Kuramoto Dynamics
        theta_i = theta_prev.unsqueeze(2) # [B, H, 1, D]
        theta_j = theta_prev.unsqueeze(1) # [B, 1, H, D]

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