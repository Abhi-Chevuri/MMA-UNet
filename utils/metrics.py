"""
utils/metrics.py
=================
Segmentation metrics: per-batch Dice/IoU/Precision/Recall/Specificity/F1,
averaging helper, and a per-tumor-class breakdown used by scripts/test.py.
"""
from __future__ import annotations

import torch
from tqdm import tqdm


@torch.no_grad()
def metrics(pred_logits: torch.Tensor, targets: torch.Tensor, thresh: float = 0.5) -> dict:
    """Dice / IoU / Precision / Recall / Specificity / F1 for one batch."""
    p = (torch.sigmoid(pred_logits) > thresh).float()
    t = targets.float()
    s = 1e-6
    TP = (p * t).sum().item()
    FP = (p * (1 - t)).sum().item()
    FN = ((1 - p) * t).sum().item()
    TN = ((1 - p) * (1 - t)).sum().item()

    dice = (2 * TP + s) / (2 * TP + FP + FN + s)
    iou = (TP + s) / (TP + FP + FN + s)
    prec = (TP + s) / (TP + FP + s)
    rec = (TP + s) / (TP + FN + s)
    spec = (TN + s) / (TN + FP + s)
    f1 = 2 * prec * rec / (prec + rec + s)
    return dict(dice=dice, iou=iou, precision=prec, recall=rec, specificity=spec, f1=f1)


def avg_metrics(mlist: list) -> dict:
    keys = mlist[0].keys()
    return {k: sum(m[k] for m in mlist) / len(mlist) for k in keys}


@torch.no_grad()
def final_metrics(model, test_dl, device, cfg) -> dict:
    """Evaluate on the held-out test set (never seen during training)."""
    model.eval()
    all_m = []
    for batch in tqdm(test_dl, desc="Evaluating on test set"):
        imgs, masks = batch[0].to(device), batch[1].to(device)
        with torch.cuda.amp.autocast(enabled=cfg["AMP"]):
            pred, _, _ = model(imgs)
        all_m.append(metrics(pred, masks))
    agg = avg_metrics(all_m)
    print("\n" + "\u2550" * 52)
    print("  Final TEST Metrics — MMA-UNet")
    print("\u2550" * 52)
    for k, v in agg.items():
        print(f"  {k:<14}: {v:.4f}")
    print("\u2550" * 52)
    return agg


@torch.no_grad()
def per_class_metrics(model, test_dl, device, cfg) -> dict:
    """Per-tumor-class metrics on the held-out test set."""
    model.eval()
    label_names = {1: "meningioma", 2: "glioma", 3: "pituitary"}
    all_records = {1: [], 2: [], 3: []}

    for batch in tqdm(test_dl, desc="Per-class metrics (test)", leave=False):
        imgs, masks, labels, _ = (batch[0].to(device), batch[1].to(device), batch[2], batch[3])
        with torch.cuda.amp.autocast(enabled=cfg["AMP"]):
            pred, _, _ = model(imgs)
        for i in range(imgs.size(0)):
            m = metrics(pred[i:i + 1], masks[i:i + 1])
            lbl = int(labels[i])
            if lbl in all_records:
                all_records[lbl].append(m)

    results = {}
    all_flat = []
    for lbl in [1, 2, 3]:
        recs = all_records[lbl]
        if recs:
            results[label_names[lbl]] = avg_metrics(recs)
            all_flat.extend(recs)
    results["overall"] = avg_metrics(all_flat) if all_flat else {}

    metric_keys = ["dice", "iou", "precision", "recall", "specificity", "f1"]
    col_w = 14
    header = f"{'Metric':<14}" + "".join(
        f"{n.capitalize():>{col_w}}" for n in ["Overall", "Meningioma", "Glioma", "Pituitary"])
    print()
    print("=" * 70)
    print("  Per-Class TEST Metrics — MMA-UNet")
    print("=" * 70)
    print(header)
    print("-" * 70)
    for k in metric_keys:
        row = f"{k:<14}"
        for grp in ["overall", "meningioma", "glioma", "pituitary"]:
            v = results.get(grp, {}).get(k, 0.0)
            row += f"{v:>{col_w}.4f}"
        print(row)
    print("=" * 70)
    print(f"  Test samples — "
          f"Meningioma: {len(all_records[1])}  Glioma: {len(all_records[2])}  "
          f"Pituitary: {len(all_records[3])}")
    print("=" * 70)
    return results
