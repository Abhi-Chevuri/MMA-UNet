xz"""
utils/visualization.py
=======================
Qualitative plotting helpers:
  - visualise(): MRI | GT contour | Prediction contour, for a batch from any loader.
  - failure_case_analysis(): per-class failure rate bar chart + worst-IoU grid.
"""
from __future__ import annotations

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


@torch.no_grad()
def visualise(model, loader, device, cfg, save_path: str, n: int = 8):
    """MRI | GT contour | Prediction contour — side by side, for `n` samples."""
    model.eval()
    imgs, masks = next(iter(loader))[:2]
    imgs, masks = imgs[:n].to(device), masks[:n]

    with torch.cuda.amp.autocast(enabled=cfg["AMP"]):
        pred, _, _ = model(imgs)
    pred_bin = (torch.sigmoid(pred) > 0.5).cpu().float()

    imgs_dn = (imgs.cpu() * _STD + _MEAN).clamp(0, 1)

    fig, axes = plt.subplots(n, 3, figsize=(11, 3.5 * n))
    fig.suptitle("MRI  |  Ground Truth (green)  |  Prediction (red)", fontsize=12)
    for i in range(n):
        img_np = imgs_dn[i].permute(1, 2, 0).numpy()
        gt_np = (masks[i, 0].numpy() * 255).astype(np.uint8)
        pd_np = (pred_bin[i, 0].numpy() * 255).astype(np.uint8)

        gt_ov, pd_ov = img_np.copy(), img_np.copy()
        for contours, overlay, colour in [
            (cv2.findContours(gt_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], gt_ov, (0, 1, 0)),
            (cv2.findContours(pd_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], pd_ov, (1, 0, 0)),
        ]:
            cv2.drawContours(overlay, contours, -1, colour, 2)

        for j, panel in enumerate([img_np, gt_ov, pd_ov]):
            axes[i][j].imshow(panel)
            axes[i][j].axis("off")
            if i == 0:
                axes[i][j].set_title(["MRI", "Ground Truth", "Prediction"][j])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {save_path}")


@torch.no_grad()
def failure_case_analysis(model, test_dl, device, cfg,
                           n_show: int = 12, iou_threshold: float = 0.5,
                           save_path: str = "outputs/plots/failure_cases.png"):
    """A 'failure' is a TEST-set image whose IoU < iou_threshold."""
    model.eval()
    label_names = {1: "Meningioma", 2: "Glioma", 3: "Pituitary"}

    print("Analysing failure cases on test set...")
    records = []
    for batch in tqdm(test_dl, desc="  scanning", leave=False):
        imgs, masks, labels, fps = batch
        imgs, masks = imgs.to(device), masks.to(device)
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=cfg["AMP"]):
                pred, _, _ = model(imgs)
        probs = (torch.sigmoid(pred) > 0.5).float()
        for i in range(imgs.size(0)):
            p, t = probs[i].view(-1), masks[i].view(-1)
            iou = ((p * t).sum() + 1e-6) / (p.sum() + t.sum() - (p * t).sum() + 1e-6)
            records.append({
                "fp": fps[i], "label": int(labels[i]), "iou": float(iou.item()),
                "img": imgs[i].cpu(), "mask": masks[i].cpu(), "pred": pred[i].detach().cpu(),
            })

    failures = [r for r in records if r["iou"] < iou_threshold]
    print(f"\nTotal TEST samples    : {len(records)}")
    print(f"Failures (IoU<{iou_threshold}) : {len(failures)} "
          f"({100 * len(failures) / max(len(records), 1):.1f}%)")
    for lbl in [1, 2, 3]:
        cls_all = [r for r in records if r["label"] == lbl]
        cls_fail = [r for r in failures if r["label"] == lbl]
        rate = 100 * len(cls_fail) / len(cls_all) if cls_all else 0
        print(f"  {label_names[lbl]:<14}: {len(cls_fail)}/{len(cls_all)} failures ({rate:.1f}%)")

    failures_sorted = sorted(failures, key=lambda x: x["iou"])
    show = failures_sorted[:n_show]

    n_grid_cols = 4
    n_grid_rows = max(len(show), 1)
    fig = plt.figure(figsize=(14, 3 + 3.2 * n_grid_rows))
    gs = fig.add_gridspec(n_grid_rows + 1, n_grid_cols,
                           height_ratios=[2.5] + [1] * n_grid_rows, hspace=0.35, wspace=0.08)

    ax_bar = fig.add_subplot(gs[0, :])
    bar_labels, bar_vals = [], []
    for lbl in [1, 2, 3]:
        cls_all = [r for r in records if r["label"] == lbl]
        cls_fail = [r for r in failures if r["label"] == lbl]
        rate = 100 * len(cls_fail) / len(cls_all) if cls_all else 0
        bar_labels.append(label_names[lbl])
        bar_vals.append(rate)

    colors = ["#5DCAA5", "#7F77DD", "#D85A30"]
    bars = ax_bar.bar(bar_labels, bar_vals, color=colors, width=0.5)
    ax_bar.set_ylabel("Failure rate (%)")
    ax_bar.set_title(f"Failure cases (IoU < {iou_threshold}) by tumor type — TEST set", fontsize=11)
    ax_bar.set_ylim(0, max(bar_vals) * 1.3 + 5 if bar_vals else 10)
    for bar, val in zip(bars, bar_vals):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=10)

    for j, hdr in enumerate(["MRI", "Ground Truth", "Prediction", "Info"]):
        ax = fig.add_subplot(gs[1, j])
        ax.set_title(hdr, fontsize=9)
        ax.axis("off")

    for i, rec in enumerate(show):
        row_gs = i + 1
        img_np = (rec["img"] * _STD + _MEAN).clamp(0, 1).permute(1, 2, 0).numpy()
        mask_np = rec["mask"][0].numpy()
        pred_np = (torch.sigmoid(rec["pred"][0]) > 0.5).float().numpy()

        gt_ov = img_np.copy()
        ct, _ = cv2.findContours((mask_np * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(gt_ov, ct, -1, (0, 1, 0), 2)

        pd_ov = img_np.copy()
        cp, _ = cv2.findContours((pred_np * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(pd_ov, cp, -1, (1, 0, 0), 2)

        for j, panel in enumerate([img_np, gt_ov, pd_ov]):
            ax = fig.add_subplot(gs[row_gs, j])
            ax.imshow(panel)
            ax.axis("off")

        ax_info = fig.add_subplot(gs[row_gs, 3])
        ax_info.axis("off")
        info_txt = (f"Class: {label_names[rec['label']]}\n"
                    f"IoU:   {rec['iou']:.4f}\n"
                    f"File:  {os.path.basename(rec['fp'])}")
        ax_info.text(0.05, 0.5, info_txt, transform=ax_info.transAxes, fontsize=7.5,
                     verticalalignment="center", fontfamily="monospace",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {save_path}")
    return failures
