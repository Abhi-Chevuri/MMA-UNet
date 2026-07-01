"""
models/decoder.py
CBAM-enhanced decoder block used in all four decoder stages of MMA-UNet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cbam import CBAM


class DecoderBlock(nn.Module):
    """
    One decoder stage of MMA-UNet.

    Forward pass:
        1. Bilinear upsample incoming feature map (×2)
        2. Interpolate skip and Mamba latent z_up to the new resolution
        3. Concatenate  [x_up ‖ skip ‖ z_up]
        4. Dilated 3×3 conv (d=2) → BN → ReLU
        5. CBAM (channel + spatial attention)
        6. 3×3 conv → BN → ReLU

    Channel convention (matches MMA-UNet architecture table):
        in_ch   : channels coming from the previous decoder stage
        skip_ch : channels of the MedNeXt bridge feature Mᵢ
        z_ch    : channels of the z-projection at this scale
        out_ch  : output channels of this block

    Parameters
    ----------
    in_ch   : int
    skip_ch : int
    z_ch    : int
    out_ch  : int
    """

    def __init__(self, in_ch: int, skip_ch: int, z_ch: int, out_ch: int):
        super().__init__()
        fused        = in_ch + skip_ch + z_ch
        self.up      = nn.Upsample(scale_factor=2,
                                    mode='bilinear', align_corners=True)
        self.dilconv = nn.Conv2d(fused, out_ch, 3,
                                  padding=2, dilation=2, bias=False)
        self.bn1     = nn.BatchNorm2d(out_ch)
        self.cbam    = CBAM(out_ch)
        self.refine  = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2     = nn.BatchNorm2d(out_ch)
        self.act     = nn.ReLU(inplace=True)

    def forward(self,
                incoming: torch.Tensor,
                skip:     torch.Tensor,
                z_up:     torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        incoming : (B, in_ch, H, W)      – previous decoder output
        skip     : (B, skip_ch, H', W')  – MedNeXt bridge feature (any H', W')
        z_up     : (B, z_ch, H'', W'')   – z projection (any H'', W'')

        Returns
        -------
        (B, out_ch, 2H, 2W)
        """
        # 1. Upsample incoming
        upsampled   = self.up(incoming)
        target_size = upsampled.shape[2:]

        # 2. Align skip and z_up to new resolution
        if skip.shape[2:] != target_size:
            skip = F.interpolate(skip, size=target_size,
                                 mode='bilinear', align_corners=True)
        if z_up.shape[2:] != target_size:
            z_up = F.interpolate(z_up, size=target_size,
                                 mode='bilinear', align_corners=True)

        # 3. Fuse
        fused = torch.cat([upsampled, skip, z_up], dim=1)

        # 4-6. Dilated conv → CBAM → refine
        x = self.act(self.bn1(self.dilconv(fused)))
        x = self.cbam(x)
        x = self.act(self.bn2(self.refine(x)))
        return x
