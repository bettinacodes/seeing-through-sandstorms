"""Paired comparison of two evaluated runs on the same test set: is the difference real?

Both runs are re-scored from their saved predictions (results/predictions/<name>/), then
the *same* 1000 bootstrap resamples of test images are applied to both, giving a 95% CI
for the difference in mAP50 and mAP50-95 (run A - run B). If the CI excludes 0 the
difference is significant at the 5% level. Pairing removes the shared image-difficulty
variance, so this is much tighter than comparing two separate CIs.

Usage (from the repo root, after evaluate.py has run both):
  python src/compare.py RTDETR-B1 FRCNN-B1                     # every test set both have
  python src/compare.py RTDETR-A RTDETR-B1 --sets dawn_sand_test
Appends to results/comparisons.csv.
"""
import argparse
import json

import numpy as np
import pandas as pd

from common import ROOT, load_classes
from evaluate import BOOT_SEED, N_BOOT, Bootstrap, coco_eval, summarize

PRED = ROOT / "results" / "predictions"


def load_run(name, tag):
    gt = json.loads((PRED / name / f"{tag}_gt.json").read_text())
    dets = json.loads((PRED / name / f"{tag}.json").read_text())
    return gt, dets


def paired_difference(ev_a, ev_b, n_boot=N_BOOT, seed=BOOT_SEED):
    """(point difference, 95% CI) for mAP50 and mAP50-95, A - B, on shared resamples."""
    boot_a, boot_b = Bootstrap(ev_a), Bootstrap(ev_b)
    n = boot_a.n_img
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        a, b = boot_a.ap(w), boot_b.ap(w)
        diffs.append((np.nanmean(a[0]) - np.nanmean(b[0]), np.nanmean(a) - np.nanmean(b)))
    d = np.array(diffs)
    lo, hi = np.percentile(d, [2.5, 97.5], axis=0)
    one = np.ones(n)
    a, b = boot_a.ap(one), boot_b.ap(one)
    point = (np.nanmean(a[0]) - np.nanmean(b[0]), np.nanmean(a) - np.nanmean(b))
    return {"mAP50": (point[0], lo[0], hi[0]), "mAP50_95": (point[1], lo[1], hi[1])}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--sets", nargs="+", help="dataset tags, e.g. dawn_sand_test (default: all shared)")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()

    tags = args.sets or sorted(
        {p.stem for p in (PRED / args.run_a).glob("*.json") if not p.stem.endswith("_gt")}
        & {p.stem for p in (PRED / args.run_b).glob("*.json") if not p.stem.endswith("_gt")})
    if not tags:
        raise SystemExit(f"no test set evaluated for both {args.run_a} and {args.run_b}")
    names = load_classes()[0]
    rows = []
    for tag in tags:
        gt_a, dets_a = load_run(args.run_a, tag)
        gt_b, dets_b = load_run(args.run_b, tag)
        assert gt_a == gt_b, f"{tag}: the two runs were scored on different ground truth"
        ev_a, ev_b = coco_eval(gt_a, dets_a), coco_eval(gt_b, dets_b)
        diff = paired_difference(ev_a, ev_b, args.n_boot)
        m_a, m_b = summarize(ev_a, names), summarize(ev_b, names)
        print(f"\n{args.run_a} vs {args.run_b} on {tag} ({len(gt_a['images'])} images)")
        row = {"run_a": args.run_a, "run_b": args.run_b, "dataset": tag, "n_boot": args.n_boot}
        for k, (d, lo, hi) in diff.items():
            sig = lo > 0 or hi < 0
            print(f"  {k:9s} {m_a[k]:.3f} vs {m_b[k]:.3f}: diff {d:+.3f} [{lo:+.3f}, {hi:+.3f}]"
                  f"{'  significant' if sig else ''}")
            row.update({f"{k}_a": m_a[k], f"{k}_b": m_b[k], f"{k}_diff": d,
                        f"{k}_diff_lo": lo, f"{k}_diff_hi": hi, f"{k}_significant": sig})
        rows.append(row)

    path = ROOT / "results" / "comparisons.csv"
    df = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if not df.empty:
        keys = {(r["run_a"], r["run_b"], r["dataset"]) for r in rows}
        df = df[[k not in keys for k in zip(df["run_a"], df["run_b"], df["dataset"])]]
    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True).sort_values(["dataset", "run_a", "run_b"])
    df.to_csv(path, index=False, float_format="%.4f")
    print(f"\n-> {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
