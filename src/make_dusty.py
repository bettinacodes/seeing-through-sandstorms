"""Build the synthetic-dust datasets from the BDD clear subset (uses dust.py).

data/processed/bdd_dusty/
  images/train        fix A training set: the 6000 BDD clear train images, each given
                      synthetic dust with probability 0.5 (random severity), seed 1000
  images/val_light    the 1000 BDD clear val images, all with light dust,  seed 2000
  images/val_medium   ... medium dust
  images/val_heavy    ... heavy dust
  labels/val_*        unchanged copies of the BDD labels (dust does not move objects)
  labels/train        dusty images keep only the objects the dust leaves visible
                      (transmission >= dust.VISIBLE_T), as an annotator of a real storm photo
                      would; clear images keep all labels
  dust_params.json    the dust parameters of every image (reproducibility)
  visibility.json     per image, the transmission of every labelled object (label-file order):
                      the fraction of its contrast the dust leaves. evaluate.py treats objects
                      below dust.VISIBLE_T (0.3) as ignore regions for the *_visible scores,
                      because nobody could have labelled them in a real storm photo.

Dust v3 (Oct 15): scene depth comes from the cached monocular depth maps (src/depth.py, run
it first). The labels are only read for visibility (visibility.json, train label filtering);
they never shape the dust. Severities calibrated against DAWN's intensity thirds (Oct 16).

The val_* sets are paired with bdd_clear val (same images, same labels), so the drop
from clear -> light -> medium -> heavy isolates the effect of dust. They are for
evaluation only: fix A still selects its checkpoint on the clear val set, like B1.

Usage (from the repo root):
  python src/make_dusty.py [--procs 2]
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

import dust
from common import ROOT, link_or_copy, read_labels, reset_split_dirs
from depth import load_disparity

SRC = ROOT / "data" / "processed" / "bdd_clear"
OUT = ROOT / "data" / "processed" / "bdd_dusty"
TRAIN_SEED, EVAL_SEED = 1000, 2000
P_DUST = 0.5
JPEG_QUALITY = 95


def plan_jobs():
    """[(src image, out split, params or None)] - all randomness drawn here, in a fixed order."""
    jobs = []
    rng = np.random.default_rng(TRAIN_SEED)
    for p in sorted((SRC / "images" / "train").glob("*.jpg")):
        params = dust.sample_params(rng) if rng.random() < P_DUST else None
        jobs.append((p, "train", params))
    rng = np.random.default_rng(EVAL_SEED)
    val = sorted((SRC / "images" / "val").glob("*.jpg"))
    for sev in dust.SEVERITIES:
        jobs += [(p, f"val_{sev}", dust.sample_params(rng, sev)) for p in val]
    return jobs


def run_job(job):
    """Write one image + label; return (split, name, per-object transmission or None)."""
    src, split, params = job
    src_split = "train" if split == "train" else "val"
    dst = OUT / "images" / split / src.name
    vis = None
    if params is None:
        link_or_copy(src, dst)
    else:
        img = cv2.imread(str(src))
        disp = load_disparity(src)
        assert disp is not None, f"no depth map for {src.name}: run src/depth.py first"
        cv2.imwrite(str(dst), dust.add_dust(img, params, disp), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        boxes = read_labels(src, img.shape[1], img.shape[0])[0]
        vis = [round(t, 4) for t in dust.object_visibility(img.shape, params, boxes, disp)]
    label = f"{src.stem}.txt"
    if split == "train" and vis is not None:
        # Train only on what can be seen: hidden objects would teach the detector to guess.
        lines = [l for l in (SRC / "labels" / src_split / label).read_text().split("\n") if l.strip()]
        kept = [l for l, t in zip(lines, vis) if t >= dust.VISIBLE_T]
        dst_label = OUT / "labels" / split / label
        dst_label.unlink(missing_ok=True)  # never write through a hard link into bdd_clear
        dst_label.write_text("".join(l + "\n" for l in kept))
    else:
        link_or_copy(SRC / "labels" / src_split / label, OUT / "labels" / split / label)
    return split, src.name, vis


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--procs", type=int, default=2, help="worker processes (keep low while training)")
    args = ap.parse_args()

    jobs = plan_jobs()
    splits = ["train"] + [f"val_{s}" for s in dust.SEVERITIES]
    reset_split_dirs(OUT, splits)
    visibility = {s: {} for s in splits}
    with Pool(args.procs) as pool:
        for split, name, vis in tqdm(pool.imap_unordered(run_job, jobs, chunksize=8), total=len(jobs), desc="dusting"):
            if vis is not None:
                visibility[split][name] = vis
    (OUT / "visibility.json").write_text(json.dumps(visibility))

    params = {s: {} for s in splits}
    for src, split, p in jobs:
        params[split][src.name] = p
    (OUT / "dust_params.json").write_text(json.dumps(params))

    summary = {}
    for s in splits:
        ps = [p for p in params[s].values() if p is not None]
        summary[s] = {"images": len(params[s]), "dusty": len(ps),
                      **{sev: sum(p["severity"] == sev for p in ps) for sev in dust.SEVERITIES},
                      "beta_median": float(np.median([p["beta"] for p in ps])),
                      "objects_hidden": round(float(np.mean([t < dust.VISIBLE_T for v in visibility[s].values() for t in v])), 3)}
        print(f"{s:<11} {summary[s]}")
    (ROOT / "results" / "data_stats_bdd_dusty.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
