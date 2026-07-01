"""
scripts/predict.py
===================
Run inference with a trained MMA-UNet checkpoint on new MRI slice(s).
No ground truth is required. This is plain single-pass inference.

Accepts:
  - a single file (.mat in the FigShare `cjdata` format, or a plain image:
    .png/.jpg/.jpeg/.tif/.tiff)
  - a directory, in which case every supported file inside is processed

For each input, saves:
  outputs/predictions/<stem>_mask.png      binary predicted mask
  outputs/predictions/<stem>_overlay.png   MRI with predicted contour (red)
                                            [+ ground-truth contour (green)
                                             if the source .mat has a tumorMask]

Example
-------
    python scripts/predict.py --config configs/config.yaml --input path/to/slice.mat
    python scripts/predict.py --config configs/config.yaml --input path/to/folder/
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import scipy.io as sio
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mma_unet import MMAUNet          # noqa: E402
from utils.config import load_config          # noqa: E402

warnings.filterwarnings("ignore")

_MAT_EXT = {".mat"}
_IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _preprocess(u8_gray: np.ndarray, img_size: int) -> np.ndarray:
    """Same pipeline as training: CLAHE -> gamma -> blur -> resize -> normalise."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    u8 = clahe.apply(u8_gray)
    lut = np.array([((i / 255.0) ** 0.8) * 255 for i in range(256)], dtype=np.uint8)
    u8 = cv2.LUT(u8, lut)
    u8 = cv2.GaussianBlur(u8, (3, 3), 0)
    u8 = cv2.resize(u8, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    f32 = np.stack([u8] * 3, axis=-1).astype(np.float32) / 255.0
    return (f32 - _MEAN) / _STD


def load_mat_slice(path: str):
    """Returns (raw_gray_uint8, gt_mask_or_None)."""
    import h5py
    try:
        with h5py.File(path, "r") as f:
            data = f["cjdata"]
            img = np.array(data["image"], dtype=np.float32).T
            mask = np.array(data["tumorMask"], dtype=np.float32).T if "tumorMask" in data else None
    except Exception:
        mat = sio.loadmat(path, simplify_cells=True)["cjdata"]
        img = np.array(mat["image"], dtype=np.float32)
        mask = np.array(mat["tumorMask"], dtype=np.float32) if "tumorMask" in mat else None
    u8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return u8, mask


def load_image_file(path: str):
    """Returns (raw_gray_uint8, None) for a plain image file."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img, None


def find_inputs(input_path: str) -> list[str]:
    p = Path(input_path)
    if p.is_file():
        return [str(p)]
    if p.is_dir():
        exts = _MAT_EXT | _IMG_EXT
        return sorted(str(f) for f in p.rglob("*") if f.suffix.lower() in exts)
    raise FileNotFoundError(f"Input not found: {input_path}")


@torch.no_grad()
def predict_one(model, path: str, cfg: dict, device: torch.device, out_dir: str, threshold: float):
    ext = Path(path).suffix.lower()
    if ext in _MAT_EXT:
        u8_gray, gt_mask = load_mat_slice(path)
    else:
        u8_gray, gt_mask = load_image_file(path)

    img_size = cfg["IMG_SIZE"]
    norm = _preprocess(u8_gray, img_size)
    img_t = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    with torch.cuda.amp.autocast(enabled=cfg.get("AMP", True)):
        pred, _, _ = model(img_t)
    prob = torch.sigmoid(pred)[0, 0].cpu().numpy()
    pred_mask = (prob > threshold).astype(np.uint8) * 255

    stem = Path(path).stem
    os.makedirs(out_dir, exist_ok=True)

    mask_path = os.path.join(out_dir, f"{stem}_mask.png")
    cv2.imwrite(mask_path, pred_mask)

    # Overlay: display-sized MRI with predicted contour (red) [+ GT contour (green)]
    display = cv2.resize(u8_gray, (img_size, img_size))
    overlay = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)  # red, BGR

    if gt_mask is not None:
        gt_rs = cv2.resize(gt_mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        gt_bin = ((gt_rs > 0.5).astype(np.uint8)) * 255
        gt_contours, _ = cv2.findContours(gt_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, gt_contours, -1, (0, 255, 0), 2)  # green, BGR

    overlay_path = os.path.join(out_dir, f"{stem}_overlay.png")
    cv2.imwrite(overlay_path, overlay)

    tumor_px = int((pred_mask > 0).sum())
    print(f"{stem:<30} tumor_px={tumor_px:<7} mask -> {mask_path}   overlay -> {overlay_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Run MMA-UNet inference on new MRI slice(s)")
    p.add_argument("--config", type=str, default="configs/config.yaml")
    p.add_argument("--checkpoint", type=str, default=None, help="Overrides paths.SAVE_PATH")
    p.add_argument("--input", type=str, required=True, help="File or directory to run inference on")
    p.add_argument("--out-dir", type=str, default=None, help="Overrides paths.PREDICT_OUT_DIR")
    p.add_argument("--threshold", type=float, default=None, help="Overrides predict.THRESHOLD")
    p.add_argument("--set", nargs="*", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config, overrides=args.set)
    checkpoint_path = args.checkpoint or cfg["SAVE_PATH"]
    out_dir = args.out_dir or cfg.get("PREDICT_OUT_DIR", "outputs/predictions")
    threshold = args.threshold if args.threshold is not None else cfg.get("THRESHOLD", 0.5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    model = MMAUNet(alpha_init=cfg["ALPHA_INIT"]).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"Loaded checkpoint: epoch {ckpt['epoch']}  val_score={ckpt['best_score']:.4f}")

    inputs = find_inputs(args.input)
    if not inputs:
        print(f"No supported files found under: {args.input}")
        return
    print(f"Found {len(inputs)} file(s) to run inference on.\n")

    for path in inputs:
        predict_one(model, path, cfg, device, out_dir, threshold)

    print(f"\nDone. Predictions saved under: {out_dir}")


if __name__ == "__main__":
    main()
