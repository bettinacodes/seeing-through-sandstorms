"""Shared helpers for data preparation: class mapping, YOLO labels, file linking.

No torch import here: make_dusty.py worker processes use these helpers, and loading torch
in every worker cost ~1 GB of RAM each.
"""
import os
import shutil
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
CLASSES_YAML = ROOT / "configs" / "classes.yaml"


def load_classes():
    """Return (names, name_to_id, cfg) from configs/classes.yaml."""
    cfg = yaml.safe_load(CLASSES_YAML.read_text())
    names = [cfg["names"][i] for i in sorted(cfg["names"])]
    return names, {n: i for i, n in enumerate(names)}, cfg


def label_path(img_path):
    """labels/<split>/<stem>.txt next to images/<split>/<stem>.jpg."""
    return Path(str(img_path.parent).replace("images", "labels", 1)) / f"{img_path.stem}.txt"


def read_labels(img_path, w, h):
    """YOLO txt -> (boxes [N, 4] absolute xyxy float32, classes [N] int64)."""
    rows = [list(map(float, line.split())) for line in label_path(img_path).read_text().split("\n")
            if line.strip()]
    if not rows:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.int64)
    a = np.array(rows, np.float64)
    cx, cy, bw, bh = a[:, 1] * w, a[:, 2] * h, a[:, 3] * w, a[:, 4] * h
    boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
    return boxes.astype(np.float32), a[:, 0].astype(np.int64)


def clip_box(x1, y1, x2, y2, w, h):
    """Clip an xyxy box to the image; return None if it degenerates (< 1 px)."""
    x1, x2 = sorted((max(0.0, min(x1, w)), max(0.0, min(x2, w))))
    y1, y2 = sorted((max(0.0, min(y1, h)), max(0.0, min(y2, h))))
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return x1, y1, x2, y2


def yolo_line(cls_id, x1, y1, x2, y2, w, h):
    return (f"{cls_id} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
            f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")


def link_or_copy(src, dst):
    """Hard-link src to dst (no extra disk space); fall back to copying."""
    dst = Path(dst)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def reset_split_dirs(out_root, splits):
    """Create empty images/<split> and labels/<split> dirs (removing old ones)."""
    for kind in ("images", "labels"):
        for s in splits:
            d = Path(out_root) / kind / s
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)
    for cache in Path(out_root).glob("labels/*.cache"):  # stale label caches left by earlier tools
        cache.unlink()
