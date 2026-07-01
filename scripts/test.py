"""
scripts/test.py
================
Evaluate a trained MMA-UNet checkpoint on the held-out TEST split.

Reports overall metrics + per-class metrics, and (optionally) saves
GradCAM and failure-case figures — this script is single-model, single-pass evaluation.

Example
-------
    python scripts/test.py --config configs/config.yaml
    python scripts/test.py --config configs/config.yaml --checkpoint outputs/checkpoints/mma_unet_best.pth
    python scripts/test.py --config configs/config.yaml --gradcam --failures
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.figshare_dataset import FigShareDataset, build_dataloaders  # noqa: E402
from models.mma_unet import MMAUNet                                       # noqa: E402
from utils.config import load_config                                      # noqa: E402
from utils.metrics import final_metrics, per_class_metrics                # noqa: E402

warnings.filterwarnings("ignore")


def get_test_loader(cfg: dict) -> DataLoader:
    """Rebuild the TEST loader from the persisted split file (preferred, so it
    exactly matches what was held out during training). Falls back to
    re-splitting from scratch if no split file exists yet.
    """
    if os.path.exists(cfg["SPLIT_PATH"]):
        with open(cfg["SPLIT_PATH"]) as fp:
            split = json.load(fp)
        test_ds = FigShareDataset(split["test"], cfg["IMG_SIZE"], augment=False)
        test_dl = DataLoader(test_ds, batch_size=cfg["BATCH_SIZE"], shuffle=False,
                              num_workers=cfg["NUM_WORKERS"], pin_memory=True)
        print(f"Loaded TEST split from {cfg['SPLIT_PATH']} ({len(split['test'])} slices)")
        return test_dl

    print(f"No split file at {cfg['SPLIT_PATH']} — rebuilding an 80/10/10 split from DATA_DIR.")
    _, _, test_dl = build_dataloaders(cfg)
    return test_dl


def load_model(cfg: dict, checkpoint_path: str, device: torch.device) -> MMAUNet:
    model = MMAUNet(alpha_init=cfg["ALPHA_INIT"]).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"Loaded checkpoint: epoch {ckpt['epoch']}  val_score={ckpt['best_score']:.4f}")
    print(f"Val metrics at checkpoint: {ckpt['metrics']}")
    return model


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate MMA-UNet on the held-out test set")
    p.add_argument("--config", type=str, default="configs/config.yaml")
    p.add_argument("--checkpoint", type=str, default=None,
                    help="Overrides paths.SAVE_PATH from the config")
    p.add_argument("--gradcam", action="store_true", help="Also save a GradCAM figure")
    p.add_argument("--failures", action="store_true", help="Also save a failure-case figure")
    p.add_argument("--set", nargs="*", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config, overrides=args.set)
    checkpoint_path = args.checkpoint or cfg["SAVE_PATH"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    test_dl = get_test_loader(cfg)
    model = load_model(cfg, checkpoint_path, device)

    print("\n── Overall TEST metrics (held-out, unseen during training) ──")
    overall = final_metrics(model, test_dl, device, cfg)

    print("\n── Per-class TEST metrics ──")
    per_class = per_class_metrics(model, test_dl, device, cfg)

    results_path = os.path.join(cfg["OUTPUT_DIR"], "test_results.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({"overall": overall, "per_class": per_class}, f, indent=2)
    print(f"\nResults saved -> {results_path}")

    if args.gradcam:
        from utils.gradcam import visualise_gradcam
        print("\n── GradCAM visualisation (test set) ──")
        visualise_gradcam(model, test_dl, device, cfg, save_path=cfg["GRADCAM_PATH"])

    if args.failures:
        from utils.visualization import failure_case_analysis
        print("\n── Failure case analysis (test set) ──")
        failure_case_analysis(model, test_dl, device, cfg,
                               n_show=cfg.get("N_FAILURE_SHOW", 12),
                               iou_threshold=cfg.get("IOU_THRESHOLD", 0.5),
                               save_path=cfg["FAILURE_CASES_PATH"])


if __name__ == "__main__":
    main()
