"""
models/mednext.py
MedNeXt-style bridge block used on every encoder skip connection.
"""

import torch
import torch.nn as nn


class MedNeXtBridge2D(nn.Module):
    """
    2-D ConvNeXt-style bridge block for skip-connection refinement.

    Architecture:
        DW-Conv(k×k) → LayerNorm → Linear(expand) → GELU → Linear(contract) → residual

    Large-kernel warm-start: kernel_size=7 blocks receive a bicubic-upsampled
    copy of a kernel_size=5 initialisation, giving a warm start for broader
    receptive fields without requiring pre-trained MedNeXt weights.

    Parameters
    ----------
    in_ch       : number of input (and output) channels
    exp_r       : linear expansion ratio (default 4 → mid = 4 × in_ch)
    kernel_size : depthwise-conv kernel (5 for shallow, 7 for deep levels)
    """

    def __init__(self, in_ch: int, exp_r: int = 4, kernel_size: int = 5):
        super().__init__()
        mid       = in_ch * exp_r
        self.dw   = nn.Conv2d(in_ch, in_ch, kernel_size,
                               padding=kernel_size // 2,
                               groups=in_ch, bias=False)
        self.norm = nn.LayerNorm(in_ch, eps=1e-6)
        self.pw1  = nn.Linear(in_ch, mid)
        self.act  = nn.GELU()
        self.pw2  = nn.Linear(mid, in_ch)
        nn.init.trunc_normal_(self.dw.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, C, H, W)

        Returns
        -------
        (B, C, H, W) – refined features with residual connection
        """
        r = x
        x = self.dw(x)                   # (B, C, H, W)
        x = x.permute(0, 2, 3, 1)        # (B, H, W, C)
        x = self.norm(x)
        x = self.pw2(self.act(self.pw1(x)))
        x = x.permute(0, 3, 1, 2)        # (B, C, H, W)
        return x + r
