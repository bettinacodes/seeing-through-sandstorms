"""Draw converted YOLO labels on random images of a split, for checking by eye.

Usage (from the repo root):
  python src/check_labels.py data/processed/dawn_sand test
  python src/check_labels.py data/processed/bdd_clear train --n 20
Writes results/figures/label_check_<dataset>_<split>.jpg
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from common import ROOT, load_classes

COLORS = [(0, 200, 255), (255, 120, 0), (0, 0, 255), (0, 255, 0), (255, 0, 255), (255, 255, 0)]  # BGR


def draw(img_path, label_path, names, tile_w=480):
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    for line in label_path.read_text().split("\n"):
        if not line.strip():
            continue
        c, cx, cy, bw, bh = line.split()
        c, cx, cy, bw, bh = int(c), float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
        p1, p2 = (int(cx - bw / 2), int(cy - bh / 2)), (int(cx + bw / 2), int(cy + bh / 2))
        thick = max(1, w // 400)
        cv2.rectangle(img, p1, p2, COLORS[c], thick)
        cv2.putText(img, names[c], (p1[0], max(12, p1[1] - 3)), 0, w / 1600, COLORS[c], thick)
    img = cv2.resize(img, (tile_w, round(tile_w * 9 / 16)))  # uniform 16:9 tiles
    cv2.putText(img, img_path.name, (4, 16), 0, 0.45, (255, 255, 255), 1)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path)
    ap.add_argument("split")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    names, _, _ = load_classes()
    imgs = sorted((args.dataset / "images" / args.split).glob("*.jpg"))
    picks = random.Random(args.seed).sample(imgs, min(args.n, len(imgs)))
    tiles = [draw(p, args.dataset / "labels" / args.split / f"{p.stem}.txt", names) for p in picks]
    cols = 4
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % cols)
    grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    out = ROOT / "results" / "figures" / f"label_check_{args.dataset.name}_{args.split}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(out)


if __name__ == "__main__":
    main()
