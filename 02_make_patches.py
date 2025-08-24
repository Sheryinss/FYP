# 02_make_patches_stacks.py
import numpy as np, tifffile as tiff
from pathlib import Path
from math import ceil

# ---- CONFIG ----
SCRIPT_DIR = Path(__file__).resolve().parent
IMG_DIR = (SCRIPT_DIR / "../data/train/images").resolve()
MSK_DIR = (SCRIPT_DIR / "../data/train/masks").resolve()

OUT_TRAIN_IMG = (SCRIPT_DIR / "../data/patches/train/images").resolve()
OUT_TRAIN_MSK = (SCRIPT_DIR / "../data/patches/train/masks").resolve()
OUT_VAL_IMG   = (SCRIPT_DIR / "../data/patches/val/images").resolve()
OUT_VAL_MSK   = (SCRIPT_DIR / "../data/patches/val/masks").resolve()
for p in [OUT_TRAIN_IMG, OUT_TRAIN_MSK, OUT_VAL_IMG, OUT_VAL_MSK]:
    p.mkdir(parents=True, exist_ok=True)

PATCH   = 256        # smaller to guarantee patches with H=700
OVERLAP = 128
VAL_STEMS = set(["frame003"])   # hold out one whole stack for validation
CONTEXT = 1                     # 0=single slice (1ch), 1=2.5D [z-1,z,z+1] (3ch)
MIN_MASK_MEAN = 0.0             # keep all patches

def list_tiffs(p: Path):
    return sorted([f for f in p.glob("*") if f.suffix.lower() in (".tif", ".tiff")])

def norm_slice(x):
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, (1, 99))
    if hi <= lo:
        mx = x.max() if x.max() > 0 else 1.0
        y = x / mx
    else:
        y = (x - lo) / (hi - lo)
    y = np.clip(y, 0, 1)
    return (y * 255).astype(np.uint8)

def binarize(m): return (m > 127).astype(np.uint8)

def make_context(stack, z):
    if CONTEXT == 0:
        return norm_slice(stack[z])                           # HxW
    Z = stack.shape[0]
    zs = [max(0, min(Z-1, z-1)), z, min(Z-1, z+1)]
    chans = [norm_slice(stack[zz]) for zz in zs]
    return np.stack(chans, axis=-1)                           # HxWx3

def save_patch(stem, split, z, yi, xi, img_p, msk_p):
    if split == "train":
        ip = OUT_TRAIN_IMG / f"{stem}_z{z:03d}_{yi:04d}_{xi:04d}.tif"
        mp = OUT_TRAIN_MSK / f"{stem}_z{z:03d}_{yi:04d}_{xi:04d}.tif"
    else:
        ip = OUT_VAL_IMG / f"{stem}_z{z:03d}_{yi:04d}_{xi:04d}.tif"
        mp = OUT_VAL_MSK / f"{stem}_z{z:03d}_{yi:04d}_{xi:04d}.tif"
    tiff.imwrite(str(ip), img_p.astype(np.uint8))
    tiff.imwrite(str(mp), (msk_p * 255).astype(np.uint8))

# ---- Pair stacks ----
imgs = list_tiffs(IMG_DIR)
pairs = []
for ip in imgs:
    mp = (MSK_DIR / ip.name)
    if mp.exists(): pairs.append((ip, mp))
print(f"Found {len(pairs)} stacks")

for ip, mp in pairs:
    stem, split = ip.stem, ("val" if ip.stem in VAL_STEMS else "train")
    img = tiff.imread(str(ip))   # (Z,H,W)
    msk = tiff.imread(str(mp))   # (Z,H,W)
    assert img.ndim == 3 and msk.ndim == 3
    Z, H, W = msk.shape
    assert img.shape[:3] == msk.shape[:3]

    stride = PATCH - OVERLAP
    ny = ceil((H - OVERLAP) / stride)
    nx = ceil((W - OVERLAP) / stride)

    saved = discarded = 0
    for z in range(Z):
        x2d = make_context(img, z)     # HxW or HxWx3
        y2d = binarize(msk[z])         # HxW

        for yi in range(ny):
            for xi in range(nx):
                y0 = max(0, min(yi*stride, H - PATCH)); y1 = y0 + PATCH if H >= PATCH else H
                x0 = max(0, min(xi*stride, W - PATCH)); x1 = x0 + PATCH if W >= PATCH else W

                img_p = x2d[y0:y1, x0:x1]
                msk_p = y2d[y0:y1, x0:x1]

                if MIN_MASK_MEAN > 0 and msk_p.mean() < MIN_MASK_MEAN:
                    discarded += 1; continue

                save_patch(stem, split, z, yi, xi, img_p, msk_p)
                saved += 1

    print(f"{stem} -> Z={Z}, saved={saved}, discarded={discarded}, split={split}")

print("Done. Patches in data/patches/{train,val}/{images,masks}")

