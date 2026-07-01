"""
models/deform.py
Modulated deformable convolution refinement layer for irregular tumour shapes.
Wraps torchvision.ops.deform_conv2d with a learned 3×3 kernel.
"""

import torch
import torch.nn as nn
from torchvision.ops import deform_conv2d


class DeformableRefinement(nn.Module):
    """
    Learns per-position spatial offsets to adapt the receptive field to
    irregular tumour boundaries.

    Internals
    ---------
    offset_conv : predicts 2×9 offsets (Δx, Δy) for each of the 9 kernel
                  positions of a 3×3 deformable kernel.
    mask_conv   : predicts 9 modulation scalars (sigmoid-gated).
    value_conv  : the deformable weight tensor (learned 3×3 conv weights).

    Dropout2d before offset/mask prediction regularises the offset network,
    preventing overfitting of spatial positions on the small FigShare dataset.

    Initialisation
    --------------
    offset weights ~ N(0, 0.001): seeds a directional prior without
        collapsing all offsets to zero.
    offset bias   = 0          : network starts near identity warp.
    mask   weights ~ N(0, 0.001)
    mask   bias    = 0.5       : sigmoid(0.5) ≈ 0.62, soft start.

    Parameters
    ----------
    ch        : number of input (and output) channels
    dropout_p : Dropout2d probability applied before offset/mask heads
    """

    def __init__(self, ch: int, dropout_p: float = 0.1):
        super().__init__()
        self.drop        = nn.Dropout2d(p=dropout_p)
        self.offset_conv = nn.Conv2d(ch, 2 * 9, 3, padding=1)
        self.mask_conv   = nn.Conv2d(ch, 9,     3, padding=1)
        self.value_conv  = nn.Conv2d(ch, ch,    3, padding=1, bias=False)
        self.bn          = nn.BatchNorm2d(ch)
        self.act         = nn.ReLU(inplace=True)

        nn.init.normal_(self.offset_conv.weight, std=0.001)
        nn.init.zeros_(self.offset_conv.bias)
        nn.init.normal_(self.mask_conv.weight,   std=0.001)
        nn.init.constant_(self.mask_conv.bias, 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, C, H, W)

        Returns
        -------
        (B, C, H, W) – deformably refined features with residual connection
        """
        x_drop = self.drop(x)
        offset = self.offset_conv(x_drop)               # (B, 18, H, W)
        mask   = torch.sigmoid(self.mask_conv(x_drop))  # (B,  9, H, W)
        out    = deform_conv2d(
            input=x, offset=offset,
            weight=self.value_conv.weight,
            bias=None, padding=1, mask=mask)
        return self.act(self.bn(out)) + x               # residual
