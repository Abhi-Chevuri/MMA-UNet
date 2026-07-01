"""
datasets/dataset.py
FigShare brain-tumour dataset loader + stratified 80/10/10 dataloader builder.
"""

import os
import glob
import json
import random
from collections import defaultdict

import cv2
import h5py
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class FigShareDataset(Dataset):
    """
    Reads the FigShare brain-tumour dataset.

    Each .mat file contains a MATLAB struct under key 'cjdata' with fields:
        image      – 2-D float64 MRI slice (typically 512×512)
        tumorMask  – 2-D binary mask (same shape)
        label      – 1=meningioma  2=glioma  3=pituitary
        PID        – patient ID

    Preprocessing pipeline (image):
        CLAHE → gamma correction (γ=0.8) → Gaussian blur (3×3)
        → resize to img_size × img_size → 3-channel stack
        → ImageNet normalise (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

    Preprocessing pipeline (mask):
        resize (nearest-neighbour) → binarise at 0.5

    Augmentation (train only):
        horizontal flip · vertical flip · rotation ±15°
        · intensity scale (×[0.85,1.15]) · elastic deformation

    __getitem__ returns a 4-tuple:
        img_t    : (3, H, W)  float32 tensor – normalised MRI
        mask_t   : (1, H, W)  float32 tensor – binary tumour mask
        label    : int – 1=meningioma  2=glioma  3=pituitary
        filepath : str – absolute path to the source .mat file
    """

    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, file_list: list, img_size: int = 256, augment: bool = False):
        self.files    = file_list
        self.img_size = img_size
        self.augment  = augment

    def __len__(self) -> int:
        return len(self.files)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _load(self, path: str):
        """Load image, mask and label from a .mat file (HDF5 or legacy)."""
        try:
            with h5py.File(path, "r") as f:
                data = f["cjdata"]
                img  = np.array(data["image"],     dtype=np.float32).T
                mask = np.array(data["tumorMask"], dtype=np.float32).T
                lbl  = int(np.array(data["label"]).item())
        except Exception:
            mat  = sio.loadmat(path, simplify_cells=True)["cjdata"]
            img  = np.array(mat["image"],     dtype=np.float32)
            mask = np.array(mat["tumorMask"], dtype=np.float32)
            lbl  = int(mat["label"])
        return img, mask, lbl

    def _preprocess_img(self, img: np.ndarray) -> np.ndarray:
        """CLAHE → gamma → Gaussian → resize → 3-ch → normalise."""
        u8    = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        u8    = clahe.apply(u8)
        lut   = np.array([((i / 255.0) ** 0.8) * 255 for i in range(256)],
                          dtype=np.uint8)
        u8    = cv2.LUT(u8, lut)
        u8    = cv2.GaussianBlur(u8, (3, 3), 0)
        u8    = cv2.resize(u8, (self.img_size, self.img_size),
                            interpolation=cv2.INTER_LINEAR)
        f32   = np.stack([u8] * 3, axis=-1).astype(np.float32) / 255.0
        return (f32 - self._MEAN) / self._STD

    def _augment(self, img: np.ndarray, mask: np.ndarray):
        """Apply random spatial and intensity augmentation in-place."""
        if random.random() > 0.5:
            img  = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()
        if random.random() > 0.5:
            img  = np.flipud(img).copy()
            mask = np.flipud(mask).copy()
        angle = random.uniform(-15, 15)
        h, w  = img.shape[:2]
        M     = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img   = cv2.warpAffine(img,  M, (w, h),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT)
        mask  = cv2.warpAffine(mask, M, (w, h),
                                flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_REFLECT)
        if random.random() > 0.5:
            img = np.clip(img * random.uniform(0.85, 1.15), -3.5, 3.5)
        if random.random() > 0.5:
            img, mask = self._elastic(img, mask)
        return img, mask

    @staticmethod
    def _elastic(img: np.ndarray, mask: np.ndarray,
                 alpha: float = 30.0, sigma: float = 5.0):
        """Random elastic deformation via Gaussian-smoothed displacement fields."""
        h, w = img.shape[:2]
        dx   = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1) * alpha,
                                 (0, 0), sigma)
        dy   = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1) * alpha,
                                 (0, 0), sigma)
        gx, gy = np.meshgrid(np.arange(w), np.arange(h))
        img    = cv2.remap(img,  (gx + dx).astype(np.float32),
                           (gy + dy).astype(np.float32),
                           cv2.INTER_LINEAR,  borderMode=cv2.BORDER_REFLECT)
        mask   = cv2.remap(mask, (gx + dx).astype(np.float32),
                           (gy + dy).astype(np.float32),
                           cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
        return img, mask

    # ── Public interface ───────────────────────────────────────────────────────

    def __getitem__(self, idx: int):
        img, mask, lbl = self._load(self.files[idx])
        img            = self._preprocess_img(img)
        mask           = cv2.resize(mask, (self.img_size, self.img_size),
                                    interpolation=cv2.INTER_NEAREST)
        mask           = (mask > 0.5).astype(np.float32)
        if self.augment:
            img, mask  = self._augment(img, mask)
        img_t  = torch.from_numpy(img.transpose(2, 0, 1))
        mask_t = torch.from_numpy(mask[None])
        return img_t, mask_t, lbl, self.files[idx]


# ── Dataloader builder ─────────────────────────────────────────────────────────

def build_dataloaders(cfg: dict):
    """
    Builds train / validation / test DataLoaders from the FigShare dataset.

    Split strategy (stratified per class, seeded):
        80% → training
        10% → validation  (early-stopping / LR scheduling only)
        10% → test        (strictly held-out; all reported metrics)

    The exact file-path partition is written to cfg["SPLIT_PATH"] on the
    first call so that the identical split can be reloaded without re-running
    this function.

    Parameters
    ----------
    cfg : dict
        Must contain: DATA_DIR, SPLIT_PATH, IMG_SIZE, TRAIN_FRAC, VAL_FRAC,
                      BATCH_SIZE, NUM_WORKERS.

    Returns
    -------
    train_dl, val_dl, test_dl : DataLoader
    """
    mats = glob.glob(
        os.path.join(cfg["DATA_DIR"], "**", "*.mat"), recursive=True)
    mats = [f for f in mats if os.path.getsize(f) < 10 * 1024 * 1024]
    if not mats:
        mats = [f for f in glob.glob(os.path.join(cfg["DATA_DIR"], "*.mat"))
                if os.path.getsize(f) < 10 * 1024 * 1024]

    print(f"Total .mat slices : {len(mats)}")

    # ── Read label for each file ───────────────────────────────────────────────
    labels = []
    for f in tqdm(mats, desc="Reading labels", leave=False):
        try:
            with h5py.File(f, "r") as hf:
                lbl = int(np.array(hf["cjdata"]["label"]).item())
        except Exception:
            try:
                lbl = int(sio.loadmat(f, simplify_cells=True)["cjdata"]["label"])
            except Exception:
                lbl = 0
        labels.append(lbl)

    # ── Stratified 80 / 10 / 10 split ─────────────────────────────────────────
    by_label = defaultdict(list)
    for f, l in zip(mats, labels):
        by_label[l].append(f)

    train_f, val_f, test_f = [], [], []
    for lbl, files in sorted(by_label.items()):
        random.shuffle(files)
        n      = len(files)
        n_val  = max(1, int(n * cfg["VAL_FRAC"]))
        n_test = max(1, int(n * (1.0 - cfg["TRAIN_FRAC"] - cfg["VAL_FRAC"])))
        n_test = min(n_test, n - n_val - 1)

        test_f  += files[:n_test]
        val_f   += files[n_test: n_test + n_val]
        train_f += files[n_test + n_val:]

        print(f"  Label {lbl}: {n} total → "
              f"{len(files) - n_test - n_val} train, "
              f"{n_val} val, {n_test} test")

    print(f"\nSplit summary — "
          f"Train: {len(train_f)}  |  Val: {len(val_f)}  |  Test: {len(test_f)}")

    # ── Persist split for reproducibility ─────────────────────────────────────
    if not os.path.exists(cfg["SPLIT_PATH"]):
        split_record = {"train": train_f, "val": val_f, "test": test_f}
        with open(cfg["SPLIT_PATH"], "w") as fp:
            json.dump(split_record, fp, indent=2)
        print(f"Split saved → {cfg['SPLIT_PATH']}")
    else:
        print(f"Split file exists, skipping save → {cfg['SPLIT_PATH']}")

    # ── Datasets ───────────────────────────────────────────────────────────────
    train_ds = FigShareDataset(train_f, cfg["IMG_SIZE"], augment=True)
    val_ds   = FigShareDataset(val_f,   cfg["IMG_SIZE"], augment=False)
    test_ds  = FigShareDataset(test_f,  cfg["IMG_SIZE"], augment=False)

    # ── DataLoaders ────────────────────────────────────────────────────────────
    train_dl = DataLoader(train_ds, batch_size=cfg["BATCH_SIZE"],
                          shuffle=True,  num_workers=cfg["NUM_WORKERS"],
                          pin_memory=True, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=cfg["BATCH_SIZE"],
                          shuffle=False, num_workers=cfg["NUM_WORKERS"],
                          pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=cfg["BATCH_SIZE"],
                          shuffle=False, num_workers=cfg["NUM_WORKERS"],
                          pin_memory=True)
    return train_dl, val_dl, test_dl
