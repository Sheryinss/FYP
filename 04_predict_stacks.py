# 04_predict_stacks.py
# Predict binary masks for 3D z-stacks using a 2D U-Net (2.5D context).

import os
import argparse
from pathlib import Path
import numpy as np
import tifffile as tiff
import torch
import segmentation_models_pytorch as smp

# ---- Model/data settings (match your training) ----
CONTEXT = 1          # 0 = single slice (1ch), 1 = 2.5D [z-1,z,z+1] (3ch)
IN_CHANNELS = 3      # 1 if CONTEXT=0, 3 if CONTEXT=1
ENCODER_NAME = "resnet34"
THRESH_DEFAULT = 0.3

# ---- Paths (your absolute work dir with the checkpoint) ----
WORK_DIR = Path(r"C:\Users\sherl\anaconda3\envs\fyp\work")  # <-- your path
CKPT = WORK_DIR / "unet_resnet34_best.pt"

# OpenMP workaround on Windows (safe for training/inference)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCRIPT_DIR = Path(__file__).resolve().parent

def parse_args():
    p = argparse.ArgumentParser(description="Predict masks for 3D TIFF stacks.")
    p.add_argument("--thresh", type=float, default=THRESH_DEFAULT, help="probability threshold")
    p.add_argument("--in_dir", type=str, default=str((SCRIPT_DIR / "../data/train/images").resolve()),
                   help="folder containing input .tif/.tiff stacks")
    p.add_argument("--out_subdir", type=str, default="preds_stacks",
                   help="subfolder under WORK_DIR to write predictions")
    p.add_argument("--morph", action="store_true", help="apply light morphology (open+close)")
    return p.parse_args()

def norm_slice(x):
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, (1, 99))
    if hi <= lo:
        mx = x.max() if x.max() > 0 else 1.0
        y = x / mx
    else:
        y = (x - lo) / (hi - lo)
    return np.clip(y, 0, 1)

def make_context(img, z):
    if CONTEXT == 0:
        return norm_slice(img[z])[None, ...]  # 1xHxW
    Z = img.shape[0]
    zs = [max(0, min(Z-1, z-1)), z, min(Z-1, z+1)]
    chans = [norm_slice(img[zz]) for zz in zs]
    return np.stack(chans, axis=0)           # 3xHxW

def pad_to_divisible(x, d=32):
    C, H, W = x.shape
    H2 = (H + d - 1)//d * d
    W2 = (W + d - 1)//d * d
    if H2 == H and W2 == W:
        return x, (0, 0)
    xpad = np.pad(x, ((0,0),(0, H2-H),(0, W2-W)), mode="reflect")
    return xpad, (H2-H, W2-W)

def maybe_morph(binary_uint8):
    try:
        import cv2
    except Exception:
        return binary_uint8
    k = np.ones((3,3), np.uint8)
    out = cv2.morphologyEx(binary_uint8, cv2.MORPH_OPEN, k)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k)
    return out

def main():
    args = parse_args()
    in_dir = Path(args.in_dir).resolve()
    out_dir = (WORK_DIR / args.out_subdir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not CKPT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT}")

    # Build model & load weights
    model = smp.Unet(encoder_name=ENCODER_NAME, encoder_weights=None,
                     in_channels=IN_CHANNELS, classes=1).to(DEVICE)
    try:
        sd = torch.load(str(CKPT), map_location=DEVICE, weights_only=True)  # PyTorch ≥2.4
    except TypeError:
        sd = torch.load(str(CKPT), map_location=DEVICE)
    model.load_state_dict(sd)
    model.eval()

    files = sorted(list(in_dir.glob("*.tif")) + list(in_dir.glob("*.tiff")))
    if not files:
        print(f"No .tif/.tiff found in: {in_dir}")
        return

    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CKPT}")
    print(f"Reading from: {in_dir}")
    print(f"Writing to  : {out_dir}")
    print(f"THRESH={args.thresh} | CONTEXT={CONTEXT} | IN_CHANNELS={IN_CHANNELS}")

    for ip in files:
        img = tiff.imread(str(ip))  # (Z,H,W) or (H,W)
        if img.ndim == 2:
            img = img[None, ...]
        assert img.ndim == 3, f"Expected 3D stack; got {img.shape} for {ip.name}"
        Z, H, W = img.shape
        preds = np.zeros((Z, H, W), dtype=np.uint8)

        for z in range(Z):
            x = make_context(img, z)                    # CxHxW
            x, pads = pad_to_divisible(x, 32)
            # 🔧 match training normalization:
            x = (x - 0.5) / 0.5          # <— add this
            x = torch.from_numpy(x[None, ...].astype(np.float32)).to(DEVICE)  # 1xCxHxW
            with torch.no_grad():
                prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
            if any(pads):
                ph, pw = pads
                prob = prob[:H, :W]
            pred = (prob > args.thresh).astype(np.uint8) * 255
            if args.morph:
                pred = maybe_morph(pred)
            preds[z] = pred

        out_path = out_dir / f"{ip.stem}_pred{ip.suffix}"
        tiff.imwrite(str(out_path), preds)
        print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
