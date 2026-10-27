"""Monocular relative depth for the dust generator (Depth Anything V2-Small, Hugging Face).

dust.py needs scene depth. A flat-ground-plane proxy (one depth per image row) hazes the
part of a nearby truck or building above the horizon row like sky; fixing that with the
labelled boxes put visible rectangles into the dust, i.e. it leaked the labels into the
image. A monocular depth network gives every pixel a depth without looking at labels.

Output: per image, the network's relative inverse depth (disparity), normalised per image
to [0, 1] between its 2nd and 98th percentiles (far -> 0, near -> 1), saved as a 16-bit PNG
at 644x364 in data/processed/depth/<dataset>/<image stem>.png (git-ignored). dust.py turns
it into depth as near_depth / disparity.

Usage (from the repo root):
  python src/depth.py configs/data_bdd_clear.yaml:train configs/data_bdd_clear.yaml:val
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from common import ROOT

# torch is imported only where the network is used: make_dusty.py worker processes import
# this module for load_disparity, and torch costs ~1 GB of RAM per process.

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
SIZE = (644, 364)  # (w, h): 16:9, multiples of the ViT patch size 14
OUT = ROOT / "data" / "processed" / "depth"


def depth_path(img_path):
    """data/processed/depth/<dataset>/<stem>.png for data/processed/<dataset>/images/<split>/<stem>.jpg."""
    return OUT / Path(img_path).parents[2].name / f"{Path(img_path).stem}.png"


def load_disparity(img_path):
    """Normalised disparity in [0, 1] (float32, 644x364), or None if not computed."""
    d = cv2.imread(str(depth_path(img_path)), cv2.IMREAD_UNCHANGED)
    return None if d is None else d.astype(np.float32) / 65535


class DepthModel:
    def __init__(self, device="cuda"):
        import torch
        from transformers import AutoModelForDepthEstimation
        self.model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID).to(device).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        self.device = device

    def disparity(self, img_bgr):
        import torch
        img = cv2.cvtColor(cv2.resize(img_bgr, SIZE, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(img).to(self.device).permute(2, 0, 1)[None].float() / 255
        with torch.no_grad():
            d = self.model(pixel_values=(x - self.mean) / self.std).predicted_depth[0].cpu().numpy()
        lo, hi = np.percentile(d, [2, 98])
        return np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("specs", nargs="+", help="data_yaml:split specs")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    from tqdm import tqdm

    from detdata import spec_images
    model = DepthModel()
    for spec in args.specs:
        images = spec_images(spec)
        for p in tqdm(images, desc=spec):
            out = depth_path(p)
            if out.exists() and not args.overwrite:
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            d = model.disparity(cv2.imread(str(p)))
            cv2.imwrite(str(out), (d * 65535).round().astype(np.uint16))


if __name__ == "__main__":
    main()
