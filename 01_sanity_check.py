import numpy as np
import tifffile as tiff
from pathlib import Path

# Resolve relative to THIS file (so CWD doesn't matter)
SCRIPT_DIR = Path(__file__).resolve().parent
IMG_DIR = (SCRIPT_DIR / "../data/train/images").resolve()
MSK_DIR = (SCRIPT_DIR / "../data/train/masks").resolve()

def list_tiffs(p: Path):
    return sorted([f for f in p.glob("*") if f.suffix.lower() in (".tif", ".tiff")])

print("Script dir:", SCRIPT_DIR)
print("IMG_DIR:", IMG_DIR, "exists:", IMG_DIR.is_dir())
print("MSK_DIR:", MSK_DIR, "exists:", MSK_DIR.is_dir())

imgs = list_tiffs(IMG_DIR)
msks = list_tiffs(MSK_DIR)
print(f"Found {len(imgs)} image files and {len(msks)} mask files")

# Build a lookup for masks allowing common suffixes
suffixes = ["", "_mask", "-mask", "_label", "-label", "_seg", "-seg", "_gt", "-gt"]
msk_lookup = {}
for m in msks:
    stem = m.stem.lower()
    msk_lookup.setdefault(stem, m)
    for suf in suffixes[1:]:
        if stem.endswith(suf):
            msk_lookup.setdefault(stem[: -len(suf)], m)

pairs = []
for ip in imgs:
    stem = ip.stem.lower()
    mp = None
    for suf in suffixes:
        mp = msk_lookup.get(stem + suf)
        if mp:
            break
    if not mp:
        print(f"[MISS] No mask for image: {ip.name}")
        continue

    img = tiff.imread(str(ip))
    msk = tiff.imread(str(mp))

    print(f"\n== {ip.name}  |  mask: {mp.name}")
    print(" image:", img.shape, img.dtype, " | mask:", msk.shape, msk.dtype)
    if img.shape[:2] != msk.shape[:2]:
        print(" [WARN] (H,W) mismatch — resize/crop one to match before training.")
    uniq = np.unique(msk)
    print(" mask unique values (first 10):", uniq[:10])
    pairs.append((ip, mp))

print(f"\nFound {len(pairs)} image–mask pairs.")
if len(imgs) and not pairs:
    print("Tip: rename masks to exactly match image basenames "
          + "(e.g., cell001.tif <-> cell001.tif), or keep a suffix like _mask.tif.")
