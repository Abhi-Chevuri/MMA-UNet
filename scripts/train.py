"""
scripts/train.py
=================
Train MMA-UNet on the FigShare brain-tumor dataset.

Example
-------
    python scripts/train.py --config configs/config.yaml
    python scripts/train.py --config configs/config.yaml --set BATCH_SIZE=8 NUM_EPOCHS=50

Outputs (paths controlled by configs/config.yaml -> `paths:`)
---------------------------------------------------------------
    outputs/checkpoints/mma_unet_best.pth   best checkpoint (by 0.5*(val IoU + val Dice))
    outputs/splits/dataset_split_indices.json   persisted 80/10/10 file-path split
    per-epoch train/val metrics
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn

# ── make the repo root importable when running `python scripts/train.py` ────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.figshare_dataset import build_dataloaders          # noqa: E402
from models.mma_unet import MMAUNet                               # noqa: E402
from losses.losses import TotalLoss                                # noqa: E402
from utils.config import load_config                               # noqa: E402
from utils.metrics import metrics, avg_metrics                     # noqa: E402
from utils.training_utils import build_optimizer, train_epoch, val_epoch  # noqa: E402

warnings.filterwarnings("ignore")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_scheduler(opt: torch.optim.Optimizer, cfg: dict) -> torch.optim.lr_scheduler.LambdaLR:
    warmup = cfg.get("WARMUP_EPOCHS", 5)

    def lr_lambda(ep: int) -> float:
        if ep < warmup:
            return (ep + 1) / warmup
        prog = (ep - warmup) / max(cfg["NUM_EPOCHS"] - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def train(cfg: dict, device: torch.device):
    train_dl, val_dl, test_dl = build_dataloaders(cfg)

    model = MMAUNet(alpha_init=cfg["ALPHA_INIT"], dropout_p=cfg["DEFORM_DROP"]).to(device)
    criterion = TotalLoss(cfg).to(device)
    opt = build_optimizer(model, cfg)
    scheduler = build_scheduler(opt, cfg)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["AMP"])

    total_p = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params    : {total_p / 1e6:.2f}M")
    print(f"Trainable params: {trainable_p / 1e6:.2f}M")

    best_score, no_imp = 0.0, 0
    history = {k: [] for k in
               ["tr_loss", "vl_loss", "tr_iou", "vl_iou", "vl_dice", "vl_prec", "vl_rec", "vl_spec"]}

    for ep in range(1, cfg["NUM_EPOCHS"] + 1):
        tr_loss, tr_m = train_epoch(model, train_dl, opt, criterion, scaler, device, cfg)
        # val_dl is used ONLY for early-stopping / checkpoint selection.
        vl_loss, vl_m = val_epoch(model, val_dl, criterion, device, cfg)
        scheduler.step()

        for k, v in zip(
            ["tr_loss", "vl_loss", "tr_iou", "vl_iou", "vl_dice", "vl_prec", "vl_rec", "vl_spec"],
            [tr_loss, vl_loss, tr_m["iou"], vl_m["iou"], vl_m["dice"],
             vl_m["precision"], vl_m["recall"], vl_m["specificity"]],
        ):
            history[k].append(v)

        lr_now = opt.param_groups[-1]["lr"]
        print(f"[{ep:03d}/{cfg['NUM_EPOCHS']}]  "
              f"Train_loss={tr_loss:.4f}  Train_IoU={tr_m['iou']:.4f}  "
              f"Val_loss={vl_loss:.4f}  Val_IoU={vl_m['iou']:.4f}  "
              f"Dice={vl_m['dice']:.4f}  Prec={vl_m['precision']:.4f}  "
              f"Rec={vl_m['recall']:.4f}  Spec={vl_m['specificity']:.4f}  lr={lr_now:.2e}  "
              f"[early-stop monitor: val only — test set is unseen]")

        score = (vl_m["iou"] + vl_m["dice"]) / 2.0
        if score > best_score:
            best_score = score
            no_imp = 0
            torch.save(
                {"epoch": ep, "state_dict": model.state_dict(),
                 "best_score": best_score, "metrics": vl_m, "cfg": cfg},
                cfg["SAVE_PATH"],
            )
            print(f" \u2713 Best checkpoint saved (val IoU={vl_m['iou']:.4f}  val Dice={vl_m['dice']:.4f})")
        else:
            no_imp += 1
            if no_imp >= cfg["PATIENCE"]:
                print(f"Early stopping at epoch {ep}.")
                break

    history_path = os.path.join(cfg["OUTPUT_DIR"], "history.json")
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved -> {history_path}")

    return history, test_dl


def parse_args():
    p = argparse.ArgumentParser(description="Train MMA-UNet")
    p.add_argument("--config", type=str, default="configs/config.yaml")
    p.add_argument("--set", nargs="*", default=None,
                    help="Override config values, e.g. --set BATCH_SIZE=32 LR=5e-5")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config, overrides=args.set)

    seed_everything(cfg.get("SEED", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()
    print(f"Device : {device}  |  GPUs : {n_gpus}")

    train(cfg, device)
    print("\nTraining completed.")


if __name__ == "__main__":
    main()
