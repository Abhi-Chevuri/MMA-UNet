"""
utils/training_utils.py
========================
Optimizer construction (3 parameter groups: encoder / deform module / rest)
and the single-epoch train/val loops used by scripts/train.py.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from utils.metrics import metrics, avg_metrics


def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    """
    Three parameter groups:
        group 0 - encoder (slow, pretrained):         LR = ENC_LR
        group 1 - deform module (slow, regularised):  LR = DEFORM_LR
        group 2 - everything else (normal speed):     LR = LR
    """
    enc_params = [p for n, p in model.encoder.named_parameters() if p.requires_grad]

    deform_params = [p for n, p in model.named_parameters()
                      if "deform" in n and p.requires_grad]

    new_params = [p for n, p in model.named_parameters()
                  if "encoder" not in n and "deform" not in n and p.requires_grad]

    return torch.optim.AdamW([
        {"params": enc_params, "lr": cfg["ENC_LR"], "weight_decay": cfg["WEIGHT_DECAY"]},
        {"params": deform_params, "lr": cfg["DEFORM_LR"], "weight_decay": cfg["DEFORM_WD"]},
        {"params": new_params, "lr": cfg["LR"], "weight_decay": cfg["WEIGHT_DECAY"]},
    ])


def train_epoch(model, loader, opt, criterion, scaler, device, cfg):
    model.train()
    losses, mets = [], []
    pbar = tqdm(loader, desc="  Train", leave=True, ncols=100)
    for batch in pbar:
        imgs = batch[0].to(device, non_blocking=True)
        masks = batch[1].to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=cfg["AMP"]):
            pred, a128, a64 = model(imgs)
            loss = criterion(pred, a128, a64, masks)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), cfg["GRAD_CLIP"])
        scaler.step(opt)
        scaler.update()
        losses.append(loss.item())

        # BN fix: report metrics in eval mode (running stats, not batch stats).
        model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=cfg["AMP"]):
                pred_eval, _, _ = model(imgs)
        model.train()
        mets.append(metrics(pred_eval.detach(), masks))
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return float(np.mean(losses)), avg_metrics(mets)


@torch.no_grad()
def val_epoch(model, loader, criterion, device, cfg):
    model.eval()
    losses, mets = [], []
    for batch in tqdm(loader, desc="  Val  ", leave=True, ncols=100):
        imgs = batch[0].to(device, non_blocking=True)
        masks = batch[1].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=cfg["AMP"]):
            pred, a128, a64 = model(imgs)
            loss = criterion(pred, a128, a64, masks)
        losses.append(loss.item())
        mets.append(metrics(pred, masks))
    return float(np.mean(losses)), avg_metrics(mets)
