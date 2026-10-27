"""Results tables, figures and the bias breakdowns, built from results/ (no re-prediction).

Reads results/summary.csv, results/comparisons.csv, results/metrics/*.json and the saved
predictions in results/predictions/<name>/ (written by evaluate.py), and writes:

  results/tables/main_map50_95.md     every method x test set, mAP50-95 [95% CI]
  results/tables/main_map50.md        the same for mAP50
  results/tables/fix_vs_b1.md         paired differences of every fix vs B1, and RT-DETR vs Faster R-CNN
  results/tables/size_buckets.md      AP small / medium / large on BDD clear val vs DAWN (sandstorm drop
                                      within object size, plan item 3) plus the size mix of each set
  results/tables/per_class_dawn.md    per-class AP50 on DAWN test, with object counts
  results/tables/storm_thirds.md      DAWN test split into thirds of storm intensity (dust.storm_thirds,
                                      ranked within the test set), mAP50-95 [95% CI] per third
  results/figures/dose_response.png   clear -> light -> medium -> heavy synthetic dust, all objects and visible-only
  results/figures/dawn_methods.png    DAWN mAP50-95 with CIs for every method
  results/figures/storm_thirds.png    DAWN mAP50-95 per storm-intensity third
  results/figures/size_buckets.png    AP per object size, BDD clear val vs DAWN

--speed (needs the GPU) times both B1 models and both pre-processing methods on the same 100
DAWN test images in one session -> results/tables/speed.md. evaluate.py's per-run timings
come from different sessions and vary by ~30% on this laptop, so only this table compares speed.

Usage (from the repo root):
  python src/results.py            # tables + figures
  python src/results.py --speed    # also the speed benchmark
"""
import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ROOT, load_classes
from evaluate import Bootstrap, coco_eval

ARCHS = [("FRCNN", "Faster R-CNN"), ("RTDETR", "RT-DETR")]
# Fixed method order and colours (categorical slots of the dataviz reference palette; B0 is a
# neutral dashed reference). Colour follows the method in every figure.
METHODS = [("B0", "B0 zero-shot COCO", "#8c8b86"), ("B1", "B1 clear BDD", "#2a78d6"),
           ("A", "A synthetic dust", "#eb6834"), ("C", "C few-shot DAWN", "#1baf7a"),
           ("B-cc", "B colour corr.", "#eda100"), ("B-dcp", "B dark channel", "#e87ba4")]
SETS = [("bdd_clear_val", "Clear"), ("bdd_dusty_val_light", "Light"), ("bdd_dusty_val_medium", "Medium"),
        ("bdd_dusty_val_heavy", "Heavy"), ("bdd_dusty_val_light_visible", "Light vis."),
        ("bdd_dusty_val_medium_visible", "Medium vis."), ("bdd_dusty_val_heavy_visible", "Heavy vis."),
        ("dawn_sand_test", "DAWN sand")]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
TABLES, FIGURES = ROOT / "results" / "tables", ROOT / "results" / "figures"


def run(arch, method):
    return f"{arch}-{method}"


def style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"-> results/figures/{name}")


def write(name, text):
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / name).write_text(text, encoding="utf-8")
    print(f"-> results/tables/{name}")


# ---------------------------------------------------------------- tables

def main_tables(summary):
    for metric, fname in (("mAP50_95", "main_map50_95.md"), ("mAP50", "main_map50.md")):
        lines = [f"# {metric.replace('_', '-')} [95% bootstrap CI] per method and test set\n",
                 "Dusty sets are BDD clear val with synthetic dust (same images and labels); *vis.* = objects "
                 "the dust hides (transmission < 0.3) are ignored. DAWN sand = held-out real sandstorm test set "
                 "(222 images).\n",
                 "| Method | " + " | ".join(label for _, label in SETS) + " |",
                 "|---|" + "---|" * len(SETS)]
        for arch, _ in ARCHS:
            for m, label, _ in METHODS:
                cells = []
                for tag, _ in SETS:
                    r = summary[(summary["name"] == run(arch, m)) & (summary["dataset"] == tag)]
                    cells.append("-" if r.empty else
                                 f"{r[metric].iloc[0]:.3f} [{r[metric + '_lo'].iloc[0]:.3f}, {r[metric + '_hi'].iloc[0]:.3f}]")
                lines.append(f"| {run(arch, m)} | " + " | ".join(cells) + " |")
        write(fname, "\n".join(lines) + "\n")


def fix_table(comp):
    def cell(a, b, tag, metric="mAP50_95"):
        r = comp[(comp["run_a"] == a) & (comp["run_b"] == b) & (comp["dataset"] == tag)]
        if r.empty:
            return "-"
        r = r.iloc[0]
        s = f"{r[metric + '_diff']:+.3f} [{r[metric + '_diff_lo']:+.3f}, {r[metric + '_diff_hi']:+.3f}]"
        return f"**{s}**" if str(r[metric + "_significant"]) == "True" else s

    lines = ["# Paired differences, mAP50-95 (A - B) [95% CI]; bold = CI excludes 0\n",
             "Same 1000 bootstrap resamples of test images applied to both runs (compare.py).\n",
             "| Comparison | " + " | ".join(label for _, label in SETS) + " |",
             "|---|" + "---|" * len(SETS)]
    for arch, _ in ARCHS:
        for m in ("A", "B-cc", "B-dcp", "C"):
            lines.append(f"| {run(arch, m)} - {run(arch, 'B1')} | "
                         + " | ".join(cell(run(arch, m), run(arch, "B1"), t) for t, _ in SETS) + " |")
        lines.append(f"| {run(arch, 'B1')} - {run(arch, 'B0')} | "
                     + " | ".join(cell(run(arch, "B1"), run(arch, "B0"), t) for t, _ in SETS) + " |")
    for m in ("B0", "B1", "A", "C"):
        lines.append(f"| RTDETR-{m} - FRCNN-{m} | "
                     + " | ".join(cell(f"RTDETR-{m}", f"FRCNN-{m}", t) for t, _ in SETS) + " |")
    write("fix_vs_b1.md", "\n".join(lines) + "\n")


def metrics_json(name, tag):
    return json.loads((ROOT / "results" / "metrics" / f"{name}__{tag}.json").read_text())


def size_mix(name, tag):
    """Share of (non-ignored) GT boxes per COCO size bucket, from the saved ground truth."""
    gt = json.loads((ROOT / "results" / "predictions" / name / f"{tag}_gt.json").read_text())
    a = np.array([g["area"] for g in gt["annotations"] if not g["iscrowd"]])
    return [(a < 32 ** 2).mean(), ((a >= 32 ** 2) & (a < 96 ** 2)).mean(), (a >= 96 ** 2).mean()]


def size_table():
    lines = ["# AP50-95 per object size: clear BDD val vs real sandstorm (DAWN)\n",
             "COCO buckets: small < 32^2 px, medium < 96^2 px, large above. The raw BDD-vs-DAWN gap is confounded "
             "by object size and image domain; comparing within a size bucket removes the size part.\n",
             "| Method | Small BDD | Small DAWN | Medium BDD | Medium DAWN | Large BDD | Large DAWN |",
             "|---|---|---|---|---|---|---|"]
    for arch, _ in ARCHS:
        for m, _, _ in METHODS:
            b, d = metrics_json(run(arch, m), "bdd_clear_val"), metrics_json(run(arch, m), "dawn_sand_test")
            lines.append(f"| {run(arch, m)} | " + " | ".join(
                f"{b[k]:.3f} | {d[k]:.3f}" for k in ("AP_small", "AP_medium", "AP_large")) + " |")
    mb, md = size_mix("FRCNN-B1", "bdd_clear_val"), size_mix("FRCNN-B1", "dawn_sand_test")
    lines += ["", "Size mix of the ground truth (share of boxes small / medium / large):", "",
              f"- BDD clear val: {mb[0]:.0%} / {mb[1]:.0%} / {mb[2]:.0%}",
              f"- DAWN sand test: {md[0]:.0%} / {md[1]:.0%} / {md[2]:.0%}"]
    write("size_buckets.md", "\n".join(lines) + "\n")


def class_table(names):
    lines = ["# Per-class AP50 on DAWN sand test\n",
             "Classes with few test objects (bicycle, bus) have unreliable AP; counts in the header.\n"]
    n = metrics_json("FRCNN-B1", "dawn_sand_test")["per_class"]
    lines += ["| Method | " + " | ".join(f"{c} (n={n[c]['n_gt']})" for c in names) + " |",
              "|---|" + "---|" * len(names)]
    for arch, _ in ARCHS:
        for m, _, _ in METHODS:
            pc = metrics_json(run(arch, m), "dawn_sand_test")["per_class"]
            lines.append(f"| {run(arch, m)} | " + " | ".join(
                "-" if pc[c]["AP50"] is None else f"{pc[c]['AP50']:.3f}" for c in names) + " |")
    write("per_class_dawn.md", "\n".join(lines) + "\n")


# ---------------------------------------------------------------- storm-intensity thirds

def subset_score(gt, dets, keep):
    """mAP50-95 and 95% CI on the images in `keep` (ids), from saved GT and predictions."""
    keep = set(keep)
    sub = {"images": [i for i in gt["images"] if i["id"] in keep],
           "annotations": [a for a in gt["annotations"] if a["image_id"] in keep],
           "categories": gt["categories"]}
    ev = coco_eval(sub, [d for d in dets if d["image_id"] in keep])
    ci = Bootstrap(ev).confidence_intervals()["mAP50_95"]
    return ev.stats[0], ci[0], ci[1]


def storm_thirds_analysis():
    import dust
    from detdata import split_images
    _, images = split_images(ROOT / "configs" / "data_dawn_sand.yaml", "test")
    third = dust.storm_thirds(images)
    by_name = {p.name: third[p] for p in images}
    rows = []
    for arch, _ in ARCHS:
        for m, _, _ in METHODS:
            d = ROOT / "results" / "predictions" / run(arch, m)
            gt = json.loads((d / "dawn_sand_test_gt.json").read_text())
            dets = json.loads((d / "dawn_sand_test.json").read_text())
            for sev in dust.SEVERITIES:
                keep = [i["id"] for i in gt["images"] if by_name[i["file_name"]] == sev]
                ap, lo, hi = subset_score(gt, dets, keep)
                rows.append({"name": run(arch, m), "arch": arch, "method": m, "third": sev,
                             "n_images": len(keep), "mAP50_95": ap, "lo": lo, "hi": hi})
    df = pd.DataFrame(rows)
    n = df[df["name"] == "FRCNN-B1"].set_index("third")["n_images"]
    lines = ["# DAWN sand test by storm intensity: mAP50-95 [95% CI]\n",
             "Thirds ranked within the test set by background structure left by the dust "
             "(dust.storm_thirds, the same measure used to calibrate the synthetic severities on few-shot train).\n",
             "| Method | " + " | ".join(f"{s} (n={n[s]})" for s in dust.SEVERITIES) + " |", "|---|---|---|---|"]
    for name, g in df.groupby("name", sort=False):
        g = g.set_index("third")
        lines.append(f"| {name} | " + " | ".join(
            f"{g.loc[s, 'mAP50_95']:.3f} [{g.loc[s, 'lo']:.3f}, {g.loc[s, 'hi']:.3f}]" for s in dust.SEVERITIES) + " |")
    write("storm_thirds.md", "\n".join(lines) + "\n")
    return df


# ---------------------------------------------------------------- figures

def dose_response_figure(summary):
    get = lambda name, tag: summary[(summary["name"] == name) & (summary["dataset"] == tag)]["mAP50_95"].iloc[0]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.6), sharey=True)
    x = np.arange(4)
    for col, (arch, title) in enumerate(ARCHS):
        for row, suffix in enumerate(("", "_visible")):
            ax = axes[row, col]
            tags = ["bdd_clear_val"] + [f"bdd_dusty_val_{s}{suffix}" for s in ("light", "medium", "heavy")]
            for m, label, color in METHODS:
                y = [get(run(arch, m), t) for t in tags]
                ax.plot(x, y, color=color, linewidth=2, linestyle="--" if m == "B0" else "-",
                        marker="o", markersize=4, label=label)
            ax.set_xticks(x, ["Clear", "Light", "Medium", "Heavy"])
            ax.set_title(f"{title}: {'all objects' if not suffix else 'visible objects only'}",
                         fontsize=10, color=INK, loc="left")
            style(ax)
            if col == 0:
                ax.set_ylabel("mAP50-95", color=MUTED, fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Detection accuracy as synthetic dust gets denser (BDD val, same 1000 images)",
                 x=0.01, y=1.07, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    save(fig, "dose_response.png")


def dawn_figure(summary):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharex=True)
    for ax, (arch, title) in zip(axes, ARCHS):
        for i, (m, label, color) in enumerate(METHODS):
            r = summary[(summary["name"] == run(arch, m)) & (summary["dataset"] == "dawn_sand_test")].iloc[0]
            y = len(METHODS) - 1 - i
            ax.plot([r["mAP50_95_lo"], r["mAP50_95_hi"]], [y, y], color=color, linewidth=2)
            ax.plot(r["mAP50_95"], y, "o", color=color, markersize=8, markeredgecolor="white", markeredgewidth=1.5)
            ax.text(r["mAP50_95_hi"] + 0.006, y, f"{r['mAP50_95']:.3f}", va="center", fontsize=8, color=MUTED)
        b1 = summary[(summary["name"] == run(arch, "B1")) & (summary["dataset"] == "dawn_sand_test")]["mAP50_95"].iloc[0]
        ax.axvline(b1, color=MUTED, linewidth=0.8, linestyle=":")
        ax.set_yticks(range(len(METHODS)), [label for _, label, _ in METHODS][::-1], fontsize=8.5)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("mAP50-95 on DAWN sand test [95% CI]", color=MUTED, fontsize=9)
        style(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.grid(axis="y", visible=False)
    axes[1].tick_params(labelleft=False)
    fig.suptitle("Real sandstorm test set (222 images); dotted line = B1", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    save(fig, "dawn_methods.png")


def thirds_figure(df):
    import dust
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    x = np.arange(3)
    for ax, (arch, title) in zip(axes, ARCHS):
        for m, label, color in METHODS:
            g = df[df["name"] == run(arch, m)].set_index("third").loc[list(dust.SEVERITIES)]
            ax.plot(x, g["mAP50_95"], color=color, linewidth=2, marker="o", markersize=4,
                    linestyle="--" if m == "B0" else "-", label=label)
        ax.set_xticks(x, ["Lightest third", "Middle third", "Most intense third"])
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        style(ax)
    axes[0].set_ylabel("mAP50-95", color=MUTED, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("DAWN sand test by storm intensity (~74 images per third; CIs in storm_thirds.md)",
                 x=0.01, y=1.1, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    save(fig, "storm_thirds.png")


def size_figure():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    x, w = np.arange(3), 0.36
    for ax, (arch, title) in zip(axes, ARCHS):
        b, d = metrics_json(run(arch, "B1"), "bdd_clear_val"), metrics_json(run(arch, "B1"), "dawn_sand_test")
        keys = ("AP_small", "AP_medium", "AP_large")
        # Colours not used by any method, so they are not read as B1 / A.
        for off, src, label, color in ((-w / 2, b, "Clear BDD val", "#b9b8b3"), (w / 2, d, "DAWN sandstorm", "#6250d6")):
            ys = [src[k] for k in keys]
            ax.bar(x + off, ys, w - 0.03, color=color, label=label)
            for xi, y in zip(x + off, ys):
                ax.text(xi, y + 0.01, f"{y:.2f}", ha="center", fontsize=8, color=MUTED)
        ax.set_xticks(x, ["Small", "Medium", "Large"])
        ax.set_title(f"{title} B1", fontsize=10, color=INK, loc="left")
        style(ax)
    axes[0].set_ylabel("AP50-95", color=MUTED, fontsize=9)
    axes[0].legend(frameon=False, fontsize=8.5)
    fig.suptitle("Clear vs sandstorm within each object-size bucket", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    save(fig, "size_buckets.png")


# ---------------------------------------------------------------- speed

def speed(n=100):
    import cv2
    import torch
    from dehaze import METHODS as DEHAZE
    from detdata import split_images, to_tensor
    from evaluate import load_detector
    _, images = split_images(ROOT / "configs" / "data_dawn_sand.yaml", "test")
    imgs = [cv2.imread(str(p)) for p in images[:n]]
    rows = []
    for arch, title in ARCHS:
        det = load_detector(f"runs/{arch}-B1/weights/best.pt", "cuda")
        for img in imgs[:5]:  # warm-up
            det.predict([to_tensor(img).cuda()])
        ts = []
        for img in imgs:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            det.predict([to_tensor(img).cuda()])
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        rows.append((title, det.n_params() / 1e6, 1000 * np.median(ts), 1000 * np.mean(ts)))
        del det
        torch.cuda.empty_cache()
    pre = {k: 1000 * np.mean([(lambda t0: (fn(im), time.perf_counter() - t0)[1])(time.perf_counter()) for im in imgs])
           for k, fn in DEHAZE.items()}
    lines = [f"# Inference speed, one session, {n} DAWN test images, batch 1, GTX 1650 (FP32)\n",
             "Detector time includes resizing inside the model (short side 640); pre-processing runs on the CPU.\n",
             "| Detector | Params (M) | Median ms/img | Mean ms/img | FPS (median) |", "|---|---|---|---|---|"]
    lines += [f"| {t} | {p:.1f} | {med:.1f} | {mean:.1f} | {1000 / med:.1f} |" for t, p, med, mean in rows]
    lines += ["", "| Pre-processing (CPU) | Mean ms/img |", "|---|---|"]
    lines += [f"| {k} | {v:.1f} |" for k, v in pre.items()]
    write("speed.md", "\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--speed", action="store_true", help="also run the GPU speed benchmark")
    args = ap.parse_args()
    summary = pd.read_csv(ROOT / "results" / "summary.csv")
    comp = pd.read_csv(ROOT / "results" / "comparisons.csv")
    names = load_classes()[0]
    main_tables(summary)
    fix_table(comp)
    size_table()
    class_table(names)
    dose_response_figure(summary)
    dawn_figure(summary)
    size_figure()
    thirds_figure(storm_thirds_analysis())
    if args.speed:
        speed()


if __name__ == "__main__":
    main()
