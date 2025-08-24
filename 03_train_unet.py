# 03_train_unet.py
# Train a 2D U-Net on 2.5D (3-channel) patches created from your z-stacks.

# ---- Windows OpenMP guard (prevents libiomp conflict) ----
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path
import numpy as np
import tifffile as tiff
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import segmentation_models_pytorch as smp

# ----------------- Paths & basic config -----------------
ROOT      = Path(__file__).resolve().parent
PATCH_DIR = (ROOT / "../data/patches").resolve()   # where tiling wrote patches
WORK_DIR  = (ROOT / "../work").resolve()           # where checkpoints & extras go
WORK_DIR.mkdir(parents=True, exist_ok=True)

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
EPOCHS     = 30
LR         = 1e-3

# Normalization used during training (keep in sync with inference)
IMG_MEAN = 0.5
IMG_STD  = 0.5

# ----------------- Dataset -----------------
class SegDataset(Dataset):
    def __init__(self, split="train", augment=True, in_channels=3):
        base = Path(PATCH_DIR) / split
        img_dir = base / "images"
        msk_dir = base / "masks"

        self.img_paths = sorted(list(img_dir.glob("*.tif")) + list(img_dir.glob("*.tiff")))
        self.msk_paths = [(msk_dir / p.name) for p in self.img_paths]
        self.augment = augment
        self.in_channels = in_channels

        # Version-safe augmentation set
        self.tfs = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(scale=(0.85, 1.15), rotate=(-45, 45),
                     translate_percent=(0.0, 0.1), shear=(-10, 10), p=0.5),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.RandomBrightnessContrast(p=0.3),
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, i):
        ip = self.img_paths[i]
        mp = self.msk_paths[i]

        img = tiff.imread(str(ip))   # uint8 HxW or HxWx3
        msk = tiff.imread(str(mp))   # uint8 HxW (0 or 255)

        # ensure binary {0,1}
        msk = (msk > 127).astype(np.uint8)

        # channel handling (keep in sync with how patches were made)
        if img.ndim == 2 and self.in_channels == 1:
            pass  # keep 1ch
        elif img.ndim == 2 and self.in_channels == 3:
            img = np.stack([img, img, img], axis=-1)  # 1ch -> 3ch
        elif img.ndim == 3 and self.in_channels == 1:
            img = img.mean(axis=-1).astype(np.uint8)  # 3ch -> 1ch
        elif img.ndim == 3 and self.in_channels == 3:
            pass  # keep 3ch

        # paired augmentations
        if self.augment:
            data = self.tfs(image=img, mask=msk)
            img, msk = data["image"], data["mask"]

        # to float tensors with the same normalization used during training
        img = img.astype(np.float32) / 255.0
        img = (img - IMG_MEAN) / IMG_STD
        if img.ndim == 2:
            img = img[None, ...]                  # 1xHxW
        else:
            img = np.transpose(img, (2, 0, 1))    # CxHxW

        msk = msk.astype(np.float32)[None, ...]   # 1xHxW

        return torch.from_numpy(img), torch.from_numpy(msk)

# ----------------- Data -----------------
IN_CHANNELS = 3  # 3 for 2.5D patches; set 1 if you trained single-slice

train_ds = SegDataset("train", augment=True,  in_channels=IN_CHANNELS)
val_ds   = SegDataset("val",   augment=False, in_channels=IN_CHANNELS)

print("Using PATCH_DIR:", PATCH_DIR)
print("Expecting to read from:", (Path(PATCH_DIR) / "train/images").resolve())
print("train patches:", len(train_ds), "| val patches:", len(val_ds))
assert len(train_ds) > 0, "No training patches found. Check PATCH_DIR and filenames."
assert len(val_ds)   > 0, "No validation patches found. Check PATCH_DIR and filenames."

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

# ----------------- Model / Loss / Optim -----------------
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=IN_CHANNELS,
    classes=1
).to(DEVICE)

bce = nn.BCEWithLogitsLoss()

@torch.no_grad()
def dice_coeff(logits, y, eps=1e-7):
    p = torch.sigmoid(logits)
    p = (p > 0.5).float()
    inter = (p * y).sum(dim=(1, 2, 3))
    denom = p.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3)) + eps
    return (2 * inter / denom).mean()

opt = torch.optim.AdamW(model.parameters(), lr=LR)

# ----------------- Train -----------------
best_dice = 0.0
for epoch in range(1, EPOCHS + 1):
    # --- train ---
    model.train()
    tr_loss = 0.0
    for x, y in train_dl:
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = bce(logits, y)
        loss.backward()
        opt.step()
        tr_loss += loss.item() * x.size(0)
    tr_loss /= len(train_ds)

    # --- validate ---
    model.eval()
    va_loss = 0.0
    va_dice = 0.0
    with torch.no_grad():
        for x, y in val_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            va_loss += bce(logits, y).item() * x.size(0)
            va_dice += dice_coeff(logits, y).item() * x.size(0)
    va_loss /= len(val_ds)
    va_dice /= len(val_ds)

    print(f"Epoch {epoch:02d} | train_loss {tr_loss:.4f} | val_loss {va_loss:.4f} | val_dice {va_dice:.4f}")

    if va_dice > best_dice:
        best_dice = va_dice
        torch.save(model.state_dict(), WORK_DIR / "unet_resnet34_best.pt")
        print(f"  Saved best with dice {best_dice:.4f}")

# ----------------- Threshold sweep (on best checkpoint) -----------------
def sweep_threshold(model, val_dl, device, ths=np.linspace(0.3, 0.8, 11)):
    model.eval()
    best = (0.5, 0.0)
    with torch.no_grad():
        for th in ths:
            num = den = 0.0
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                p = (torch.sigmoid(model(x)) > th).float()
                inter = (p * y).sum(dim=(1, 2, 3))
                denom = p.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3)) + 1e-7
                num += (2 * inter).sum().item()
                den += denom.sum().item()
            dice = num / den
            if dice > best[1]:
                best = (th, dice)
    print("Best TH", best[0], "Dice", round(best[1], 4))
    return best

# reload the best weights we just saved
best_ckpt = WORK_DIR / "unet_resnet34_best.pt"
best_model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,   # loading trained weights -> no imagenet init
    in_channels=IN_CHANNELS,
    classes=1
).to(DEVICE)
best_model.load_state_dict(torch.load(str(best_ckpt), map_location=DEVICE))

best_th, best_d = sweep_threshold(best_model, val_dl, DEVICE)
print(f"[THRESH SWEEP] best_th={best_th:.3f} | val_dice={best_d:.4f}")

# save threshold for the predict script to read
with open(WORK_DIR / "best_threshold.txt", "w") as f:
    f.write(f"{best_th:.4f}\n")
print(f"Wrote threshold to {WORK_DIR / 'best_threshold.txt'}")
