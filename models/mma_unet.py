"""
models/mma_unet.py
MMA-UNet: MedMamba-CBAM Attention U-Net for brain tumour segmentation.

Architecture overview
─────────────────────
Encoder  : EfficientNet-B5 (pretrained, hook-based feature extraction)
           Stem + blocks.0 frozen; blocks.1-4 fine-tuned at 1/10 LR.

Bridge   : MedNeXtBridge2D on each of the four encoder skip features.

Bottleneck: MambaBottleneck (2 × VSSBlock, pure-PyTorch parallel scan)
            applied at the deepest feature map (16×16 for 256×256 input).
            Fused with encoder feature via learnable α: z_fused = α·F4 + (1-α)·z.

z-projections: z_fused projected to each decoder channel count via 1×1 conv.
               All projections stay at bottleneck resolution; the decoder
               upsamples them as needed (resolves the "upsampled in the wrong
               direction" bug present in naïve implementations).

Decoder  : Four DecoderBlock stages (dilated conv + CBAM + refine).
           Each stage receives [x_up ‖ Mᵢ ‖ z_i] as input.

Refinement: DeformableRefinement after the final decoder stage, before the
            1×1 output head. Learns per-position offsets for irregular shapes.

Auxiliary heads: 1×1 conv output at 64×64 (dec3) and 128×128 (dec2)
                 for deep supervision during training.

Encoder feature shapes at input 256×256
----------------------------------------
    blocks.0 →  24ch @ 128×128  (F1, M1)
    blocks.1 →  40ch @  64×64   (F2, M2)
    blocks.2 →  64ch @  32×32   (F3, M3)
    blocks.4 → 176ch @  16×16   (F4, M4, bottleneck)

Decoder fused channel counts
-----------------------------
    dec4 : 176 + 176 + 128 = 480 → 128ch @ 32×32
    dec3 : 128 +  64 +  96 = 288 →  96ch @ 64×64
    dec2 :  96 +  40 +  64 = 200 →  64ch @ 128×128
    dec1 :  64 +  24 +  32 = 120 →  32ch @ 256×256
"""

import torch
import torch.nn as nn
import timm

from models.mednext import MedNeXtBridge2D
from models.mamba   import MambaBottleneck
from models.decoder import DecoderBlock
from models.deform  import DeformableRefinement


class MMAUNet(nn.Module):
    """
    Proposed MMA-UNet.

    Parameters
    ----------
    alpha_init : float
        Initial value of the learnable encoder–Mamba fusion weight α ∈ (0,1).
        α=1 means full encoder; α=0 means full Mamba output.  Default: 0.7.
    dropout_p  : float
        Dropout2d probability in the DeformableRefinement offset heads.
        Default: 0.1.
    """

    ENC_CHS      = [24, 40, 64, 176]
    DEC_CHS      = [128, 96, 64, 32]
    _HOOK_LAYERS = ["blocks.0", "blocks.1", "blocks.2", "blocks.4"]

    def __init__(self, alpha_init: float = 0.7, dropout_p: float = 0.1):
        super().__init__()

        # ── Encoder ───────────────────────────────────────────────────────────
        self.encoder = timm.create_model("efficientnet_b5", pretrained=True)
        self.encoder.classifier  = nn.Identity()
        self.encoder.global_pool = nn.Identity()

        # Freeze stem + blocks.0
        for name, p in self.encoder.named_parameters():
            if any(n in name for n in ["conv_stem", "bn1", "blocks.0"]):
                p.requires_grad = False

        # Hook-based feature extraction (timm-version independent)
        self._features = {}
        for layer_name in self._HOOK_LAYERS:
            module = dict(self.encoder.named_modules())[layer_name]
            module.register_forward_hook(self._make_hook(layer_name))

        # ── MedNeXt bridge blocks ──────────────────────────────────────────────
        self.bridge = nn.ModuleList([
            MedNeXtBridge2D(self.ENC_CHS[0], exp_r=4, kernel_size=5),  # 24ch
            MedNeXtBridge2D(self.ENC_CHS[1], exp_r=4, kernel_size=5),  # 40ch
            MedNeXtBridge2D(self.ENC_CHS[2], exp_r=4, kernel_size=7),  # 64ch
            MedNeXtBridge2D(self.ENC_CHS[3], exp_r=4, kernel_size=7),  # 176ch
        ])

        # ── Mamba bottleneck ───────────────────────────────────────────────────
        self.bottleneck = MambaBottleneck(
            in_ch=self.ENC_CHS[3], d_state=16, n_blocks=2)

        # ── Learnable fusion weight α ──────────────────────────────────────────
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

        # ── z projections at bottleneck resolution (decoder upsamples them) ────
        self.z_proj = nn.ModuleList([
            nn.Conv2d(self.ENC_CHS[3], self.DEC_CHS[0], 1),  # → 128ch
            nn.Conv2d(self.ENC_CHS[3], self.DEC_CHS[1], 1),  # →  96ch
            nn.Conv2d(self.ENC_CHS[3], self.DEC_CHS[2], 1),  # →  64ch
            nn.Conv2d(self.ENC_CHS[3], self.DEC_CHS[3], 1),  # →  32ch
        ])

        # ── CBAM Decoder blocks ────────────────────────────────────────────────
        self.dec4 = DecoderBlock(
            in_ch=self.ENC_CHS[3], skip_ch=self.ENC_CHS[3],
            z_ch=self.DEC_CHS[0],  out_ch=self.DEC_CHS[0])
        self.dec3 = DecoderBlock(
            in_ch=self.DEC_CHS[0], skip_ch=self.ENC_CHS[2],
            z_ch=self.DEC_CHS[1],  out_ch=self.DEC_CHS[1])
        self.dec2 = DecoderBlock(
            in_ch=self.DEC_CHS[1], skip_ch=self.ENC_CHS[1],
            z_ch=self.DEC_CHS[2],  out_ch=self.DEC_CHS[2])
        self.dec1 = DecoderBlock(
            in_ch=self.DEC_CHS[2], skip_ch=self.ENC_CHS[0],
            z_ch=self.DEC_CHS[3],  out_ch=self.DEC_CHS[3])

        # ── Deformable refinement + output head ───────────────────────────────
        self.deform   = DeformableRefinement(self.DEC_CHS[3], dropout_p=dropout_p)
        self.out_head = nn.Conv2d(self.DEC_CHS[3], 1, 1)

        # ── Auxiliary deep supervision heads ──────────────────────────────────
        self.aux_head_64  = nn.Conv2d(self.DEC_CHS[1], 1, 1)  # dec3 @  64×64
        self.aux_head_128 = nn.Conv2d(self.DEC_CHS[2], 1, 1)  # dec2 @ 128×128

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _make_hook(self, name: str):
        def hook(module, input, output):
            self._features[name] = output
        return hook

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : (B, 3, 256, 256) – ImageNet-normalised MRI tensor

        Returns
        -------
        pred    : (B, 1, 256, 256) – main segmentation logits
        aux_128 : (B, 1, 128, 128) – auxiliary logits for deep supervision
        aux_64  : (B, 1,  64,  64) – auxiliary logits for deep supervision
        """
        # ── Encoder (feature maps captured via hooks) ─────────────────────────
        self._features.clear()
        _ = self.encoder(x)

        F1 = self._features["blocks.0"]   # (B,  24, 128, 128)
        F2 = self._features["blocks.1"]   # (B,  40,  64,  64)
        F3 = self._features["blocks.2"]   # (B,  64,  32,  32)
        F4 = self._features["blocks.4"]   # (B, 176,  16,  16)

        # ── MedNeXt bridge ────────────────────────────────────────────────────
        M1 = self.bridge[0](F1)
        M2 = self.bridge[1](F2)
        M3 = self.bridge[2](F3)
        M4 = self.bridge[3](F4)

        # ── Mamba bottleneck + α-fusion ───────────────────────────────────────
        z       = self.bottleneck(F4)
        a       = torch.clamp(self.alpha, 0.0, 1.0)
        z_fused = a * F4 + (1.0 - a) * z            # (B, 176, 16, 16)

        # ── z projections at bottleneck resolution ────────────────────────────
        z4 = self.z_proj[0](z_fused)   # (B, 128, 16, 16)
        z3 = self.z_proj[1](z_fused)   # (B,  96, 16, 16)
        z2 = self.z_proj[2](z_fused)   # (B,  64, 16, 16)
        z1 = self.z_proj[3](z_fused)   # (B,  32, 16, 16)

        # ── Decoder (decoder upsamples z projections internally) ──────────────
        d4 = self.dec4(z_fused, M4, z4)   # (B, 128,  32,  32)
        d3 = self.dec3(d4,      M3, z3)   # (B,  96,  64,  64)
        d2 = self.dec2(d3,      M2, z2)   # (B,  64, 128, 128)
        d1 = self.dec1(d2,      M1, z1)   # (B,  32, 256, 256)

        # ── Auxiliary heads ───────────────────────────────────────────────────
        aux_64  = self.aux_head_64(d3)    # (B, 1,  64,  64)
        aux_128 = self.aux_head_128(d2)   # (B, 1, 128, 128)

        # ── Deformable refinement + final output ──────────────────────────────
        d1   = self.deform(d1)
        pred = self.out_head(d1)          # (B, 1, 256, 256)

        return pred, aux_128, aux_64
