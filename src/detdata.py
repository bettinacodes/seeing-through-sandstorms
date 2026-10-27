"""Model-agnostic detection dataset over the YOLO-format splits in data/processed/.

Images are returned at their original resolution; resizing is done inside each model
(models.py), with the same rule for Faster R-CNN and RT-DETR, so predictions come back
in original image coordinates for both.

A dataset is a list of "data_yaml:split" specs ("@N" = seeded random N images of the
split), so mixes (fix C: DAWN few-shot + some BDD clear) need no extra data directory:
    DetDataset(["configs/data_dawn_sand.yaml:train", "configs/data_bdd_clear.yaml:train@500"])
"""
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from common import ROOT, label_path, read_labels  # noqa: F401 (re-exported)


def split_images(data_yaml, split):
    """(dataset name, sorted image paths) for one split of a configs/data_*.yaml file."""
    cfg = yaml.safe_load(Path(data_yaml).read_text())
    img_dir = ROOT / cfg["path"] / cfg[split]
    return Path(cfg["path"]).name, sorted(img_dir.glob("*.jpg"))


def spec_images(spec, seed=0):
    """Images of a "data_yaml:split" spec; "data_yaml:split@N" keeps a seeded random N of them."""
    data_yaml, split = spec.rsplit(":", 1)
    split, _, n = split.partition("@")
    images = split_images(ROOT / data_yaml, split)[1]
    if n:
        images = sorted(random.Random(seed).sample(images, int(n)))
    return images


def to_tensor(img_bgr):
    """BGR uint8 HWC -> RGB float CHW in [0, 1]."""
    return torch.from_numpy(np.ascontiguousarray(img_bgr[:, :, ::-1])).permute(2, 0, 1).float().div_(255)


class DetDataset(torch.utils.data.Dataset):
    """Items: (image RGB float CHW [0,1], {"boxes": xyxy [N,4], "labels": shared class id [N]}, path).

    train=True adds a random horizontal flip (scale jitter happens in the model).
    fraction < 1 keeps a seeded random subset (smoke tests).
    """

    def __init__(self, specs, train=False, fraction=1.0, seed=0):
        self.images = []
        for spec in specs:
            self.images += spec_images(spec, seed)
        if fraction < 1:
            k = max(1, round(fraction * len(self.images)))
            self.images = sorted(random.Random(seed).sample(self.images, k))
        self.train = train

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        p = self.images[i]
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        boxes, classes = read_labels(p, w, h)
        if self.train and random.random() < 0.5:
            img = img[:, ::-1]
            boxes = boxes.copy()
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        return to_tensor(img), {"boxes": torch.from_numpy(boxes), "labels": torch.from_numpy(classes)}, str(p)


def collate(batch):
    """Lists, not stacked tensors: images in a batch can differ in size."""
    return tuple(map(list, zip(*batch)))
