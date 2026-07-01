"""
models/mamba.py
Pure-PyTorch vectorised 2-D State Space Model (SS2D) bottleneck.

Key design choices
──────────────────
• No C extension, no selective-scan CUDA kernel — runs on any PyTorch >= 2.0.
• The recurrence h_t = exp(A) h_{t-1} + B_t x_t is solved in closed form via
  torch.cumsum (a single CUDA kernel launch), giving O(L) time but O(1) kernel
  calls; no Python loop over sequence length.
• A is shared (not input-dependent) to enable the parallel scan.
• Four scan directions (H→, H←, W→, W←) are merged by summation so the
  bottleneck captures global spatial dependencies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SS2D(nn.Module):
    """
    2-D Selective Scan Module (vectorised, no Python loop).

    Recurrence (closed form):
        h_t = exp(A·t) · cumsum_t( exp(-A·s) · B_s ⊗ x_s )
        y_t = C_t · h_t  +  D · x_t

    Scans in 4 directions: H→, H←, W→, W←. Outputs are summed.

    Parameters
    ----------
    d_model  : number of input and output channels
    d_state  : SSM state dimension (default 16)
    expand   : inner-dimension expansion factor (default 2)
    """

    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2):
        super().__init__()
        self.d_state = d_state
        self.d_inner = int(d_model * expand)
        d = self.d_inner

        self.in_proj  = nn.Linear(d_model, d * 2, bias=False)
        self.conv2d   = nn.Conv2d(d, d, 3, padding=1, groups=d, bias=True)
        self.act      = nn.SiLU()

        # Shared SSM parameters (constant A enables parallel scan)
        self.A_log    = nn.Parameter(torch.zeros(d, d_state).uniform_(-0.5, -0.1))
        self.D        = nn.Parameter(torch.ones(d))
        self.B_proj   = nn.Linear(d, d_state, bias=False)
        self.C_proj   = nn.Linear(d, d_state, bias=False)
        self.out_norm = nn.LayerNorm(d, eps=1e-6)
        self.out_proj = nn.Linear(d, d_model, bias=False)

    def _parallel_scan(self,
                       x_seq: torch.Tensor,
                       B_seq: torch.Tensor,
                       C_seq: torch.Tensor) -> torch.Tensor:
        """
        Vectorised causal SSM scan — single torch.cumsum call, no Python loop.

        Parameters
        ----------
        x_seq : (B, L, d_inner)
        B_seq : (B, L, d_state)
        C_seq : (B, L, d_state)

        Returns
        -------
        (B, L, d_inner)
        """
        _, L, _ = x_seq.shape
        A       = -torch.exp(self.A_log)                            # (d, d_state)

        # Outer product: v_s = x_s ⊗ B_s
        v       = x_seq.unsqueeze(-1) * B_seq.unsqueeze(2)         # (B, L, d, d_state)

        # Position-dependent decay weights: exp(A · t)
        t       = torch.arange(L, device=x_seq.device, dtype=x_seq.dtype)
        w_pos   = torch.exp(A.unsqueeze(0) * t.view(L, 1, 1))      # (L, d, d_state)

        # Normalise → prefix-sum → re-weight
        v_norm  = v / (w_pos.unsqueeze(0) + 1e-8)
        acc     = torch.cumsum(v_norm, dim=1)
        h       = w_pos.unsqueeze(0) * acc                         # (B, L, d, d_state)

        # Output: y_t = C_t · h_t (sum over d_state) + D · x_t
        y = (h * C_seq.unsqueeze(2)).sum(-1) + \
            self.D.unsqueeze(0).unsqueeze(0) * x_seq
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, H, W, C)   channel-last layout

        Returns
        -------
        (B, H, W, C)
        """
        B, H, W, C = x.shape
        xz           = self.in_proj(x)                              # (B,H,W,2d)
        xs, z        = xz.chunk(2, dim=-1)                          # each (B,H,W,d)

        xs_c = self.act(
            self.conv2d(xs.permute(0, 3, 1, 2)).permute(0, 2, 3, 1))

        B_all = self.B_proj(xs_c)    # (B, H, W, d_state)
        C_all = self.C_proj(xs_c)    # (B, H, W, d_state)

        # Rasterise for H-direction scans
        xs_hw = xs_c.reshape(B, H * W, -1)
        B_hw  = B_all.reshape(B, H * W, -1)
        C_hw  = C_all.reshape(B, H * W, -1)

        # Rasterise for W-direction scans (transpose H and W)
        xs_wh = xs_c.permute(0, 2, 1, 3).reshape(B, W * H, -1)
        B_wh  = B_all.permute(0, 2, 1, 3).reshape(B, W * H, -1)
        C_wh  = C_all.permute(0, 2, 1, 3).reshape(B, W * H, -1)

        # Four parallel scans (all use torch.cumsum — no loops)
        y0 = self._parallel_scan(xs_hw,          B_hw,          C_hw)
        y1 = self._parallel_scan(xs_hw.flip(1),  B_hw.flip(1),  C_hw.flip(1)).flip(1)
        y2 = self._parallel_scan(xs_wh,          B_wh,          C_wh)
        y3 = self._parallel_scan(xs_wh.flip(1),  B_wh.flip(1),  C_wh.flip(1)).flip(1)

        y0 = y0.reshape(B, H, W, -1)
        y1 = y1.reshape(B, H, W, -1)
        y2 = y2.reshape(B, W, H, -1).permute(0, 2, 1, 3)
        y3 = y3.reshape(B, W, H, -1).permute(0, 2, 1, 3)

        y  = y0 + y1 + y2 + y3                                     # (B,H,W,d_inner)
        y  = self.out_norm(y) * self.act(z)                         # gated
        return self.out_proj(y)                                     # (B,H,W,C)


class VSSBlock(nn.Module):
    """Single VSS residual block with pre-norm LayerNorm."""

    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
        self.ss2d = SS2D(d_model, d_state=d_state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ss2d(self.norm(x))


class MambaBottleneck(nn.Module):
    """
    Drop-in Mamba bottleneck applied at the lowest spatial resolution
    (16×16 for 256×256 input → L=256 tokens, fast and memory-efficient).

    Parameters
    ----------
    in_ch    : number of input channels (= deepest encoder channel count)
    d_state  : SSM state dimension (default 16)
    n_blocks : number of stacked VSSBlock layers (default 2)
    """

    def __init__(self, in_ch: int, d_state: int = 16, n_blocks: int = 2):
        super().__init__()
        self.proj_in  = nn.Conv2d(in_ch, in_ch, 1)
        self.vss      = nn.Sequential(
            *[VSSBlock(in_ch, d_state) for _ in range(n_blocks)])
        self.norm_out = nn.LayerNorm(in_ch, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, C, H, W)

        Returns
        -------
        (B, C, H, W)
        """
        z = self.proj_in(x)
        z = z.permute(0, 2, 3, 1)     # (B, H, W, C)
        z = self.vss(z)
        z = self.norm_out(z)
        return z.permute(0, 3, 1, 2)  # (B, C, H, W)
