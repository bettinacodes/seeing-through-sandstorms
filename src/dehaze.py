"""Fix B: dehazing / colour-correction pre-processing at inference time (no training).

Every method maps a BGR uint8 image to a BGR uint8 image of the same size, so evaluate.py
can run it in front of any detector (--preprocess <method>; its time is reported as
extra_preprocess_ms):

  cc    gray-world colour correction + CLAHE. Sand dust absorbs blue, so storm images have a
        strong orange cast; gray-world scales each channel to the common mean, then CLAHE on
        the L channel (LAB) restores local contrast without shifting colour again.
  dcp   dark channel prior (He et al., CVPR 2009) with a guided-filter refined transmission
        (He et al., ECCV 2010). Inverts the same scattering model dust.py uses,
        I = J*t + A*(1-t). Run on the raw image: DCP estimates the (orange) airlight A itself.

Parameters are the papers' defaults, not tuned on any test set: the DAWN test split and the
paired dusty val sets are never used to choose them (only --grid looks at images, and it
uses DAWN few-shot train and BDD dusty *train*).

Usage:
  python src/evaluate.py --weights runs/FRCNN-B1/weights/best.pt --name FRCNN-B-cc --preprocess cc
  python src/dehaze.py --grid      # results/figures/dehaze_examples.jpg
"""
import argparse

import cv2
import numpy as np

from common import ROOT


def gray_world_clahe(img, clip=2.0, tiles=8):
    f = img.astype(np.float32)
    means = f.reshape(-1, 3).mean(0)
    f *= means.mean() / np.maximum(means, 1e-3)
    lab = cv2.cvtColor(np.clip(f, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def dark_channel(f, patch):
    return cv2.erode(f.min(2), cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch)))


def guided_filter(guide, src, r, eps):
    box = lambda x: cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1))
    m_i, m_p = box(guide), box(src)
    a = (box(guide * src) - m_i * m_p) / (box(guide * guide) - m_i * m_i + eps)
    b = m_p - a * m_i
    return box(a) * guide + box(b)


def dark_channel_prior(img, patch=15, omega=0.95, t0=0.1, top=0.001, r=60, eps=1e-3):
    f = img.astype(np.float32) / 255.0
    dark = dark_channel(f, patch)
    # Airlight: mean colour of the brightest pixels among the 0.1% haziest (highest dark channel).
    flat, dflat = f.reshape(-1, 3), dark.ravel()
    idx = np.argpartition(dflat, -max(1, int(top * dflat.size)))[-max(1, int(top * dflat.size)):]
    A = np.maximum(flat[idx].mean(0), 1e-3)
    t = 1.0 - omega * dark_channel(f / A, patch)
    t = guided_filter(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), t, r, eps)
    J = (f - A) / np.maximum(t, t0)[..., None] + A
    return (np.clip(J, 0, 1) * 255).astype(np.uint8)


METHODS = {"cc": gray_world_clahe, "dcp": dark_channel_prior}


def _grid(n_dawn=4, n_bdd=2, seed=0, out="results/figures/dehaze_examples.jpg", width=480):
    """Rows: DAWN few-shot train and BDD dusty train images; columns: raw + each method."""
    rng = np.random.default_rng(seed)
    pick = lambda d, n: [d / p for p in rng.choice(sorted(x.name for x in d.glob("*.jpg")), n, replace=False)]
    paths = (pick(ROOT / "data/processed/dawn_sand/images/fewshot_train", n_dawn)
             + pick(ROOT / "data/processed/bdd_dusty/images/train", n_bdd))
    height, rows = round(width * 9 / 16), []
    for p in paths:
        img = cv2.imread(str(p))
        s = min(width / img.shape[1], height / img.shape[0])
        img = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)))
        cells = []
        for cell in [img] + [fn(img) for fn in METHODS.values()]:  # letterbox, keep aspect ratio
            canvas = np.zeros((height, width, 3), np.uint8)
            y, x = (height - cell.shape[0]) // 2, (width - cell.shape[1]) // 2
            canvas[y:y + cell.shape[0], x:x + cell.shape[1]] = cell
            cells.append(canvas)
        rows.append(np.hstack(cells))
    header = np.full((32, width * (1 + len(METHODS)), 3), 255, np.uint8)
    for i, name in enumerate(["raw", *METHODS]):
        cv2.putText(header, name, (i * width + 8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    path = ROOT / out
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.vstack([header, *rows]), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"-> {path.relative_to(ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", action="store_true", help="write results/figures/dehaze_examples.jpg")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.grid:
        _grid(seed=args.seed)
