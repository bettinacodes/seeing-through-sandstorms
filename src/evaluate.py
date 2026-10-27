"""Evaluate a detector on YOLO-format test sets with pycocotools + bootstrap CIs.

For each dataset split:
  1. predict with score threshold 0.001, up to 300 boxes per image (optionally after a
     pre-processing step),
  2. write ground truth and predictions as COCO JSON,
  3. score with pycocotools: mAP50, mAP50-95, AP small/medium/large, per-class AP,
  4. bootstrap 95% CIs for mAP50 and mAP50-95 by resampling test images (1000x).

Synthetic-dust sets also get a "<set>_visible" score: objects the dust has physically hidden
(transmission < dust.VISIBLE_T, from the dataset's visibility.json) become ignore regions
(COCO iscrowd), neither misses nor false positives, as a human could not have labelled them.

--weights is a checkpoint from train.py, or coco:frcnn / coco:rtdetr for the zero-shot
baseline B0 (COCO head, the 6 shared classes mapped by name, see models.py).

Usage (from the repo root):
  python src/evaluate.py --weights coco:frcnn --name FRCNN-B0
  python src/evaluate.py --weights runs/RTDETR-B1/weights/best.pt --name RTDETR-B1 --sets configs/data_dawn_sand.yaml:test
Writes results/metrics/<name>__<dataset>_<split>.json, updates results/summary.csv,
and keeps raw predictions in results/predictions/<name>/ (git-ignored).
"""
import argparse
import contextlib
import io
import json
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from common import ROOT, load_classes
from dehaze import METHODS as DEHAZE_METHODS
from detdata import label_path, split_images, to_tensor
from dust import VISIBLE_T

# Clear val, its paired synthetic-dust copies (dose-response), and the real sandstorm test set.
DEFAULT_SETS = ["configs/data_bdd_clear.yaml:val", "configs/data_bdd_dusty.yaml:val_light",
                "configs/data_bdd_dusty.yaml:val_medium", "configs/data_bdd_dusty.yaml:val_heavy",
                "configs/data_dawn_sand.yaml:test"]
N_BOOT, BOOT_SEED = 1000, 0
PREPROCESSORS = {"none": None, **DEHAZE_METHODS}  # fix B: dehaze.py


def coco_ground_truth(images, names, visibility=None):
    """COCO GT dict from YOLO labels (labels/<split>/ next to images/<split>/).

    visibility: {image name: [transmission per label line]}; objects below VISIBLE_T are
    marked iscrowd, which COCOeval treats as ignore regions.
    """
    gt = {"images": [], "annotations": [],
          "categories": [{"id": i + 1, "name": n} for i, n in enumerate(names)]}
    for img_id, p in enumerate(images):
        w, h = Image.open(p).size
        gt["images"].append({"id": img_id, "file_name": p.name, "width": w, "height": h})
        lines = [line for line in label_path(p).read_text().split("\n") if line.strip()]
        vis = visibility[p.name] if visibility is not None else [1.0] * len(lines)
        assert len(vis) == len(lines), f"{p.name}: visibility does not match the label file"
        for line, t in zip(lines, vis):
            c, cx, cy, bw, bh = map(float, line.split())
            bw, bh = bw * w, bh * h
            gt["annotations"].append({
                "id": len(gt["annotations"]) + 1, "image_id": img_id, "category_id": int(c) + 1,
                "bbox": [cx * w - bw / 2, cy * h - bh / 2, bw, bh], "area": bw * bh,
                "iscrowd": int(t < VISIBLE_T)})
    return gt


def load_detector(weights, device):
    from models import Detector, build
    det = build(weights.split(":", 1)[1]) if weights.startswith("coco:") else Detector.load(weights)
    return det.to(device).eval()


def predict(det, images, preprocess=None, batch=1):
    """COCO-format detections (shared class id + 1) plus mean timings (ms/img)."""
    import cv2
    device = next(det.parameters()).device
    dets, t_pre, t_inf = [], 0.0, 0.0
    for start in range(0, len(images), batch):
        chunk = []
        for p in images[start:start + batch]:
            img = cv2.imread(str(p))
            if preprocess is not None:
                t0 = time.perf_counter()
                img = preprocess(img)
                t_pre += time.perf_counter() - t0
            chunk.append(to_tensor(img).to(device))
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        results = det.predict(chunk)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_inf += time.perf_counter() - t0
        for img_id, r in enumerate(results, start):
            for (x1, y1, x2, y2), c, s in zip(r["boxes"].tolist(), r["labels"].tolist(), r["scores"].tolist()):
                dets.append({"image_id": img_id, "category_id": c + 1,
                             "bbox": [x1, y1, x2 - x1, y2 - y1], "score": s})
    return dets, {"inference_ms": 1000 * t_inf / len(images),
                  "extra_preprocess_ms": 1000 * t_pre / len(images)}


def coco_eval(gt, dets):
    coco_gt = COCO()
    coco_gt.dataset = gt
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(dets) if dets else COCO()
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return ev


class Bootstrap:
    """Re-accumulates COCO AP (area 'all', maxDets 100) with per-image weights.

    Resampling images with replacement == weighting each image by how often it was
    drawn, so COCOeval.evaluate() (the expensive matching) runs only once.
    """

    def __init__(self, ev):
        p = ev.params
        n_area, n_img = len(p.areaRng), len(p.imgIds)
        self.rec_thrs, self.n_iou, self.n_img = p.recThrs, len(p.iouThrs), n_img
        self.classes = []
        for k in range(len(p.catIds)):
            scores, tp, fp, img_idx, n_gt = [], [], [], [], np.zeros(n_img)
            for i in range(n_img):
                e = ev.evalImgs[k * n_area * n_img + i]  # area index 0 = 'all'
                if e is None:
                    continue
                n_gt[i] = np.count_nonzero(~np.asarray(e["gtIgnore"], dtype=bool))
                if len(e["dtScores"]) == 0:
                    continue
                m, ign = e["dtMatches"][:, :p.maxDets[-1]], e["dtIgnore"][:, :p.maxDets[-1]].astype(bool)
                scores.append(np.asarray(e["dtScores"][:p.maxDets[-1]]))
                tp.append((m > 0) & ~ign)
                fp.append((m == 0) & ~ign)
                img_idx.append(np.full(len(scores[-1]), i))
            if scores:
                order = np.argsort(-np.concatenate(scores), kind="mergesort")  # same tie-break as COCOeval
                self.classes.append((np.concatenate(tp, 1)[:, order].astype(float),
                                     np.concatenate(fp, 1)[:, order].astype(float),
                                     np.concatenate(img_idx)[order], n_gt))
            else:
                self.classes.append((None, None, None, n_gt))

    def ap(self, weights):
        """AP[t, k] for image weights; NaN for classes without ground truth."""
        out = np.full((self.n_iou, len(self.classes)), np.nan)
        for k, (tp, fp, img_idx, n_gt) in enumerate(self.classes):
            npig = float(weights @ n_gt)
            if npig == 0:
                continue
            if tp is None:
                out[:, k] = 0.0
                continue
            w = weights[img_idx]
            tps, fps = np.cumsum(tp * w, 1), np.cumsum(fp * w, 1)
            rc = tps / npig
            pr = tps / np.maximum(tps + fps, np.spacing(1))
            pr = np.maximum.accumulate(pr[:, ::-1], axis=1)[:, ::-1]  # precision envelope
            for t in range(tp.shape[0]):
                inds = np.searchsorted(rc[t], self.rec_thrs, side="left")
                q = np.where(inds < len(pr[t]), pr[t][np.minimum(inds, len(pr[t]) - 1)], 0.0)
                out[t, k] = q.mean()
        return out

    def confidence_intervals(self, n_boot=N_BOOT, seed=BOOT_SEED):
        rng = np.random.default_rng(seed)
        samples = []
        for _ in range(n_boot):
            w = np.bincount(rng.integers(0, self.n_img, self.n_img), minlength=self.n_img).astype(float)
            ap = self.ap(w)
            samples.append((np.nanmean(ap[0]), np.nanmean(ap)))
        s = np.array(samples)
        lo, hi = np.percentile(s, [2.5, 97.5], axis=0)
        return {"mAP50": [lo[0], hi[0]], "mAP50_95": [lo[1], hi[1]]}


def summarize(ev, names):
    s = ev.stats  # [AP, AP50, AP75, APs, APm, APl, ...]
    prec = ev.eval["precision"][:, :, :, 0, -1]  # [T, R, K], area all, maxDets 100

    def ap_of(x):
        x = x[x > -1]
        return float(x.mean()) if x.size else None

    n_gt = {n: 0 for n in names}
    for a in ev.cocoGt.dataset["annotations"]:
        if not a["iscrowd"]:
            n_gt[names[a["category_id"] - 1]] += 1
    per_class = {n: {"AP50": ap_of(prec[0, :, k]), "AP50_95": ap_of(prec[:, :, k]), "n_gt": n_gt[n]}
                 for k, n in enumerate(names)}
    return {"mAP50": s[1], "mAP50_95": s[0], "mAP75": s[2],
            "AP_small": s[3], "AP_medium": s[4], "AP_large": s[5], "per_class": per_class}


def update_summary(row):
    path = ROOT / "results" / "summary.csv"
    df = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if not df.empty:
        df = df[~((df["name"] == row["name"]) & (df["dataset"] == row["dataset"]))]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True).sort_values(["name", "dataset"])
    df.to_csv(path, index=False, float_format="%.4f")


def score(args, det, names, tag, gt, dets, timing):
    """Score one test set: metrics + bootstrap CIs -> results/metrics, summary.csv, predictions."""
    pred_dir = ROOT / "results" / "predictions" / args.name
    pred_dir.mkdir(parents=True, exist_ok=True)
    (pred_dir / f"{tag}_gt.json").write_text(json.dumps(gt))
    (pred_dir / f"{tag}.json").write_text(json.dumps(dets))

    ev = coco_eval(gt, dets)
    metrics = summarize(ev, names)
    boot = Bootstrap(ev)
    ap_all = boot.ap(np.ones(len(gt["images"])))  # self-check against pycocotools
    assert abs(np.nanmean(ap_all) - metrics["mAP50_95"]) < 1e-6, (np.nanmean(ap_all), metrics["mAP50_95"])
    assert abs(np.nanmean(ap_all[0]) - metrics["mAP50"]) < 1e-6
    t0 = time.time()
    metrics["ci95"] = boot.confidence_intervals(args.n_boot)
    print(f"  bootstrap ({args.n_boot}x): {time.time() - t0:.0f}s")

    result = {"name": args.name, "arch": det.arch, "params_M": det.n_params() / 1e6,
              "weights": str(args.weights), "preprocess": args.preprocess,
              "dataset": tag, "n_images": len(gt["images"]),
              "n_gt": sum(not a["iscrowd"] for a in gt["annotations"]),
              "n_boot": args.n_boot, "timing": timing, **metrics}
    out = ROOT / "results" / "metrics" / f"{args.name}__{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))

    ci = metrics["ci95"]
    print(f"  mAP50    {metrics['mAP50']:.3f}  [{ci['mAP50'][0]:.3f}, {ci['mAP50'][1]:.3f}]")
    print(f"  mAP50-95 {metrics['mAP50_95']:.3f}  [{ci['mAP50_95'][0]:.3f}, {ci['mAP50_95'][1]:.3f}]")
    print(f"  AP small/medium/large {metrics['AP_small']:.3f} / {metrics['AP_medium']:.3f} / {metrics['AP_large']:.3f}")
    print("  per-class AP50: " + ", ".join(
        f"{n} {v['AP50']:.3f} (n={v['n_gt']})" if v["AP50"] is not None else f"{n} - (n=0)"
        for n, v in metrics["per_class"].items()))
    print(f"  speed ms/img: {', '.join(f'{k} {v:.1f}' for k, v in timing.items())}")
    update_summary({"name": args.name, "arch": det.arch, "dataset": tag, "preprocess": args.preprocess,
                    "mAP50": metrics["mAP50"], "mAP50_lo": ci["mAP50"][0], "mAP50_hi": ci["mAP50"][1],
                    "mAP50_95": metrics["mAP50_95"], "mAP50_95_lo": ci["mAP50_95"][0],
                    "mAP50_95_hi": ci["mAP50_95"][1], "AP_small": metrics["AP_small"],
                    "AP_medium": metrics["AP_medium"], "AP_large": metrics["AP_large"],
                    "infer_ms": timing["inference_ms"], "extra_ms": timing["extra_preprocess_ms"],
                    "weights": str(args.weights)})
    print(f"  -> {out.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="runs/<name>/weights/best.pt, or coco:frcnn / coco:rtdetr")
    ap.add_argument("--name", required=True, help="method ID for the results table, e.g. FRCNN-B1, RTDETR-A")
    ap.add_argument("--sets", nargs="+", default=DEFAULT_SETS, help="data_yaml:split pairs")
    ap.add_argument("--preprocess", default="none", choices=sorted(PREPROCESSORS))
    ap.add_argument("--batch", type=int, default=1,
                    help="1 = per-frame latency; Faster R-CNN at batch 4 spills past 4 GB and runs 3.5x slower")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()

    names = load_classes()[0]
    det = load_detector(args.weights, "cuda" if torch.cuda.is_available() else "cpu")

    for spec in args.sets:
        data_yaml, split = spec.rsplit(":", 1)
        dataset, images = split_images(ROOT / data_yaml, split)
        tag = f"{dataset}_{split}"
        print(f"\n[{args.name}] {tag}: {len(images)} images")

        gt = coco_ground_truth(images, names)
        dets, timing = predict(det, images, PREPROCESSORS[args.preprocess], args.batch)
        score(args, det, names, tag, gt, dets, timing)
        vis_file = images[0].parent.parent.parent / "visibility.json"
        if images and vis_file.exists() and split in (vis_all := json.loads(vis_file.read_text())):
            gt_vis = coco_ground_truth(images, names, vis_all[split])
            hidden = sum(a["iscrowd"] for a in gt_vis["annotations"])
            print(f"\n[{args.name}] {tag}_visible: {hidden} of {len(gt_vis['annotations'])} objects hidden by dust -> ignored")
            score(args, det, names, f"{tag}_visible", gt_vis, dets, timing)


if __name__ == "__main__":
    main()
