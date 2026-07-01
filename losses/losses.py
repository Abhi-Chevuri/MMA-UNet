"""
losses/losses.py
All loss functions used in MMA-UNet training.

TotalLoss (L3) = DiceLoss + TverskyLoss + λ_bd · BoundaryLoss
                 + λ_a2 · aux_128_loss + λ_a1 · aux_64_loss

where auxiliary losses are Dice + Tversky computed at reduced resolutions
(128×128 and 64×64) for deep supervision on the two auxiliary decoder heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft Dice loss for binary segmentation.

        L = 1 - (2·TP + ε) / (2·TP + FP + FN + ε)

    Parameters
    ----------
    smooth : float
        Laplace smoothing constant to avoid division by zero (default 1e-6).
    """

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.s = smooth

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        p     = torch.sigmoid(logits)
        p     = p.view(p.size(0), -1)
        t     = targets.view(targets.size(0), -1)
        inter = (p * t).sum(1)
        return 1.0 - ((2 * inter + self.s) /
                       (p.sum(1) + t.sum(1) + self.s)).mean()


class TverskyLoss(nn.Module):
    """
    Tversky loss — a generalisation of Dice that independently weights
    False Positives (α) and False Negatives (β).

        TI = (TP + ε) / (TP + α·FP + β·FN + ε)
        L  = 1 - mean(TI)

    Setting α=β=0.5 recovers Dice loss.
    The paper uses α=0.3, β=0.7 to penalise false negatives more heavily.

    Parameters
    ----------
    alpha  : float – FP weight (default 0.3)
    beta   : float – FN weight (default 0.7)
    smooth : float – smoothing constant (default 1e-6)
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7,
                 smooth: float = 1e-6):
        super().__init__()
        self.a, self.b, self.s = alpha, beta, smooth

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        p  = torch.sigmoid(logits)
        p  = p.view(p.size(0), -1)
        t  = targets.view(targets.size(0), -1)
        TP = (p * t).sum(1)
        FP = (p * (1 - t)).sum(1)
        FN = ((1 - p) * t).sum(1)
        ti = (TP + self.s) / (TP + self.a * FP + self.b * FN + self.s)
        return 1.0 - ti.mean()


class BoundaryLoss(nn.Module):
    """
    Boundary-aware BCE loss.

    Upweights the standard binary cross-entropy near tumour boundaries by
    a factor of (1 + w · boundary_mask), where boundary_mask is obtained via
    morphological dilation minus the original mask:

        boundary = MaxPool(mask, k) − mask

    This focuses training on boundary voxels where the model's uncertainty
    is highest and boundary precision matters most for clinical use.

    Parameters
    ----------
    k : int   – dilation kernel size (default 5)
    w : float – boundary weight factor (default 3.0)
    """

    def __init__(self, k: int = 5, w: float = 3.0):
        super().__init__()
        self.pool = nn.MaxPool2d(k, stride=1, padding=k // 2)
        self.w    = w

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        bnd = self.pool(targets) - targets   # boundary region ∈ [0, 1]
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none')
        return (bce * (1.0 + self.w * bnd)).mean()


class TotalLoss(nn.Module):
    """
    Full multi-term loss used to train the proposed MMA-UNet (L3 in the
    loss ablation study).

        L = (Dice + Tversky)(main)
          + λ_bd · Boundary(main)
          + λ_a2 · (Dice + Tversky)(aux_128)
          + λ_a1 · (Dice + Tversky)(aux_64)

    Auxiliary masks are down-sampled to match aux head resolutions via
    nearest-neighbour interpolation before loss computation.

    Parameters
    ----------
    cfg : dict
        Must contain:
            TVERSKY_A  – Tversky α (FP weight)
            TVERSKY_B  – Tversky β (FN weight)
            LAMBDA_AUX1 – weight for the 64×64 auxiliary head loss
            LAMBDA_AUX2 – weight for the 128×128 auxiliary head loss
            LAMBDA_BD   – weight for the boundary-aware BCE term
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.dice    = DiceLoss()
        self.tversky = TverskyLoss(cfg["TVERSKY_A"], cfg["TVERSKY_B"])
        self.bnd     = BoundaryLoss()
        self.la1     = cfg["LAMBDA_AUX1"]   # aux_64  weight
        self.la2     = cfg["LAMBDA_AUX2"]   # aux_128 weight
        self.lb      = cfg["LAMBDA_BD"]     # boundary weight

    def forward(self,
                pred:    torch.Tensor,
                aux_128: torch.Tensor,
                aux_64:  torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pred    : (B, 1, 256, 256) – main logits
        aux_128 : (B, 1, 128, 128) – auxiliary logits
        aux_64  : (B, 1,  64,  64) – auxiliary logits
        targets : (B, 1, 256, 256) – binary ground-truth masks

        Returns
        -------
        scalar loss tensor
        """
        t128 = F.interpolate(targets, aux_128.shape[-2:], mode='nearest')
        t064 = F.interpolate(targets, aux_64.shape[-2:],  mode='nearest')

        main  = self.dice(pred, targets) + self.tversky(pred, targets)
        bd    = self.bnd(pred, targets)
        a128  = self.dice(aux_128, t128) + self.tversky(aux_128, t128)
        a064  = self.dice(aux_64,  t064) + self.tversky(aux_64,  t064)

        return main + self.lb * bd + self.la2 * a128 + self.la1 * a064
