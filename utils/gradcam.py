"""
utils/gradcam.py
=================
Pure-PyTorch GradCAM for MMA-UNet, targeting `model.dec1` (last decoder
block, 32ch @ 256x256), plus a plotting helper that picks the best /
median / worst IoU sample per tumor class from the test set and renders
MRI | Ground Truth | GradCAM | Overlay panels.
"""
from __future__ import annotations

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from datasets.figshare_dataset import FigShareDataset

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class GradCAM:
    """
    Usage:
        gcam = GradCAM(model, target_layer=model.dec1)
        cam  = gcam(img_tensor)   # returns HxW numpy array in [0,1]
        gcam.remove()             # always clean up hooks
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self._act = None
        self._grad = None
        self._hooks = []
        self._hooks.append(target_layer.register_forward_hook(self._save_act))
        self._hooks.append(target_layer.register_full_backward_hook(self._save_grad))

    def _save_act(self, module, inp, out):
        self._act = out.detach()

    def _save_grad(self, module, grad_in, grad_out):
        self._grad = grad_out[0].detach()

    def __call__(self, img_tensor: torch.Tensor) -> np.ndarray:
        """img_tensor: 1 x 3 x H x W on the model's device -> H x W in [0,1]."""
        self.model.eval()
        img_tensor = img_tensor.requires_grad_(False)

        pred, _, _ = self.model(img_tensor)
        self.model.zero_grad()
        score = torch.sigmoid(pred).mean()
        score.backward()

        weights = self._grad.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self._act).sum(dim=1, keepdim=True)
        cam = F.relu(cam).squeeze().cpu().numpy()

        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam

    def remove(self):
        for h in self._hooks:
            h.remove()


def visualise_gradcam(model, test_dl, device, cfg, save_path: str):
    """
    For each tumor type, pick 3 representative TEST samples (best / median /
    worst by IoU). Columns: MRI | Ground Truth | GradCAM heatmap | Overlay.
    """
    model.eval()
    gcam = GradCAM(model, target_layer=model.dec1)
    label_names = {1: "Meningioma", 2: "Glioma", 3: "Pituitary"}

    print("Computing per-image IoU for GradCAM sample selection (test set)...")
    records = []
    for batch in tqdm(test_dl, desc="  scanning test set", leave=False):
        imgs, masks, labels, fps = batch
        imgs, masks = imgs.to(device), masks.to(device)
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=cfg["AMP"]):
                pred, _, _ = model(imgs)
        probs = (torch.sigmoid(pred) > 0.5).float()
        for i in range(imgs.size(0)):
            p, t = probs[i].view(-1), masks[i].view(-1)
            iou = ((p * t).sum() + 1e-6) / (p.sum() + t.sum() - (p * t).sum() + 1e-6)
            records.append({"fp": fps[i], "label": int(labels[i]), "iou": float(iou.item())})

    cases_per_class = {}
    for lbl in [1, 2, 3]:
        cls = sorted([r for r in records if r["label"] == lbl], key=lambda x: x["iou"])
        n = len(cls)
        if n == 0:
            continue
        cases_per_class[lbl] = {
            "worst": cls[max(0, n // 10)],
            "median": cls[n // 2],
            "best": cls[min(n - 1, 9 * n // 10)],
        }

    case_labels = ["worst", "median", "best"]
    n_rows = 3 * len(cases_per_class)
    n_cols = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    fig.suptitle("GradCAM on TEST set — Best / Median / Worst per tumor type", fontsize=12, y=1.01)

    col_titles = ["MRI", "Ground Truth", "GradCAM", "CAM overlay"]
    for j, t in enumerate(col_titles):
        axes[0][j].set_title(t, fontsize=10)

    row = 0
    for lbl, cases in cases_per_class.items():
        for case_name in case_labels:
            rec = cases[case_name]
            tmp = FigShareDataset([rec["fp"]], cfg["IMG_SIZE"], augment=False)
            img_t, mask_t, _, _ = tmp[0]
            img_np = (img_t * _STD + _MEAN).clamp(0, 1).permute(1, 2, 0).numpy()
            mask_np = mask_t[0].numpy()

            img_gpu = img_t.unsqueeze(0).to(device)
            cam = gcam(img_gpu)
            cam_rs = cv2.resize(cam, (cfg["IMG_SIZE"], cfg["IMG_SIZE"]))
            cam_rgb = plt.cm.jet(cam_rs)[:, :, :3]
            overlay = np.clip(0.55 * img_np + 0.45 * cam_rgb, 0, 1)

            gt_ov = img_np.copy()
            ct, _ = cv2.findContours((mask_np * 255).astype(np.uint8),
                                      cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(gt_ov, ct, -1, (0, 1, 0), 2)

            row_label = f"{label_names[lbl]}\n{case_name}\nIoU={rec['iou']:.3f}"
            axes[row][0].set_ylabel(row_label, fontsize=8, labelpad=4)
            axes[row][0].imshow(img_np); axes[row][0].axis("off")
            axes[row][1].imshow(gt_ov); axes[row][1].axis("off")
            axes[row][2].imshow(cam_rgb); axes[row][2].axis("off")
            axes[row][3].imshow(overlay); axes[row][3].axis("off")
            row += 1

    gcam.remove()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {save_path}")
