# 05_eval_dice.py — compute Dice for one stack (e.g., frame003)

import argparse, csv
from pathlib import Path
import numpy as np
import tifffile as tiff

def find_default_root(script_dir: Path) -> Path:
    # Try script_dir, then parent, then parent.parent — pick the first containing data/train/masks
    candidates = [script_dir, script_dir.parent, script_dir.parent.parent]
    for c in candidates:
        if (c / "data" / "train" / "masks").exists():
            return c
    return script_dir  # fallback; user should pass --root

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stack", default="frame003", help="basename without extension")
    p.add_argument("--root", type=str, default=None,
                   help="project root containing data/ and work/ (e.g., C:\\Users\\sherl\\anaconda3\\envs\\fyp)")
    p.add_argument("--pred_subdir", default="preds_stacks", help="subfolder under work/ with predictions")
    args = p.parse_args()

    script_dir = Path(__file__).resolve().parent
    root = Path(args.root) if args.root else find_default_root(script_dir)

    mask_dir = root / "data" / "train" / "masks"
    pred_dir = root / "work" / args.pred_subdir
    out_dir  = root / "work"
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_path = mask_dir / f"{args.stack}.tif"
    pd_path = pred_dir / f"{args.stack}_pred.tif"

    assert gt_path.exists(), f"Missing GT: {gt_path}"
    assert pd_path.exists(), f"Missing pred: {pd_path}"

    gt = tiff.imread(str(gt_path)) > 127
    pd = tiff.imread(str(pd_path)) > 127
    assert gt.shape == pd.shape, f"Shape mismatch: gt {gt.shape} vs pred {pd.shape}"

    def dice(g, p):
        inter = np.logical_and(g, p).sum()
        denom = g.sum() + p.sum()
        return (2.0 * inter) / (denom + 1e-7)

    Z = gt.shape[0]
    per_slice = [dice(gt[z], pd[z]) for z in range(Z)]
    mean, std = float(np.mean(per_slice)), float(np.std(per_slice))
    print(f"{args.stack}: Mean Dice = {mean:.4f} (±{std:.4f}) | slices={Z} | "
          f"min={min(per_slice):.4f} | max={max(per_slice):.4f}")

    csv_path = out_dir / f"{args.stack}_dice_per_slice.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["slice", "dice"])
        for i, d in enumerate(per_slice):
            w.writerow([i, d])
    print(f"Wrote per-slice Dice to {csv_path}")

if __name__ == "__main__":
    main()
