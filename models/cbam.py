"""
models/cbam.py
Convolutional Block Attention Module (CBAM).
"""

import torch
import torch.nn as nn


class CBAM(nn.Module):
    """
    Channel attention followed by spatial attention.

    Channel attention
        AvgPool + MaxPool → shared MLP → element-wise sum → sigmoid → scale

    Spatial attention
        channel-avg concat channel-max → 7×7 conv → sigmoid → scale

    Parameters
    ----------
    ch   : number of input channels
    r    : channel reduction ratio for the MLP (default 16)
    sp_k : spatial attention kernel size (default 7)
    """

    def __init__(self, ch: int, r: int = 16, sp_k: int = 7):
        super().__init__()
        mid = max(ch // r, 4)

        self.avg  = nn.AdaptiveAvgPool2d(1)
        self.maxp = nn.AdaptiveMaxPool2d(1)
        self.mlp  = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ch, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, ch),
        )
        self.sp = nn.Conv2d(2, 1, sp_k, padding=sp_k // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, C, H, W)

        Returns
        -------
        (B, C, H, W) – attention-refined features
        """
        # ── Channel attention ──────────────────────────────────────────────────
        ca = torch.sigmoid(self.mlp(self.avg(x)) + self.mlp(self.maxp(x)))
        x  = x * ca.view(x.size(0), x.size(1), 1, 1)

        # ── Spatial attention ──────────────────────────────────────────────────
        sa = torch.sigmoid(
            self.sp(torch.cat([x.mean(1, keepdim=True),
                               x.max(1,  keepdim=True).values], dim=1)))
        return x * sa
