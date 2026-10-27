"""Failure-case and qualitative figures from the saved predictions (no re-prediction).

Detections are shown at score >= 0.5 and matched to ground truth greedily by score (IoU >= 0.5,
same class), like a deployed detector with one operating threshold:
  green   correct detection (true positive)
  red     false detection (false positive)
  yellow  missed object (false negative)
  grey    object hidden by synthetic dust (transmission < 0.3; ignored in the visible-only score)

Writes:
  results/figures/gallery_dawn.jpg         DAWN test images where the models miss the most objects,
                                           side by side for FRCNN-B0, FRCNN-B1, RTDETR-B1, RTDETR-C
  results/figures/gallery_dust_levels.jpg  one BDD val image at clear / light / medium / heavy synthetic
                                           dust, RTDETR-B1 vs RTDETR-A
  results/tables/failure_counts.md         TP / FP / FN at score 0.5 on DAWN test per run, and which
                                           object sizes the misses are

Usage (from the repo root):
  python src/gallery.py
"""
import argparse
import json

import cv2
import numpy as np

from common import ROOT, load_classes

THR, IOU = 0.5, 0.5
TP, FP, FN, HIDDEN = (60, 170, 0), (40, 40, 220), (0, 200, 255), (150, 150, 150)  # BGR
IMAGE_DIRS = {"dawn_sand_test": "dawn_sand/images/test", "bdd_clear_val": "bdd_clear/images/val",
              **{f"bdd_dusty_val_{s}": f"bdd_dusty/images/val_{s}" for s in ("light", "medium", "heavy")}}
DAWN_RUNS = ["FRCNN-B0", "FRCNN-B1", "RTDETR-B1", "RTDETR-C"]
RUNS = ["FRCNN-B0", "FRCNN-B1", "FRCNN-A", "FRCNN-C", "FRCNN-B-cc", "FRCNN-B-dcp",
        "RTDETR-B0", "RTDETR-B1", "RTDETR-A", "RTDETR-C", "RTDETR-B-cc", "RTDETR-B-dcp"]


def load(name, tag):
    d = ROOT / "results" / "predictions" / name
    gt = json.loads((d / f"{tag}_gt.json").read_text())
    pred_tag = tag.removesuffix("_visible")
    return gt, json.loads((d / f"{pred_tag}.json").read_text())


def iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    return inter / max(a[2] * a[3] + b[2] * b[3] - inter, 1e-9)


def match(gts, dets):
    """-> (tp dets, fp dets, missed gts, hidden gts) for one image at THR."""
    dets = sorted((d for d in dets if d["score"] >= THR), key=lambda d: -d["score"])
    used, tp, fp = set(), [], []
    for d in dets:
        best, best_iou = None, IOU
        for k, g in enumerate(gts):
            if k not in used and g["category_id"] == d["category_id"] and iou(d["bbox"], g["bbox"]) >= best_iou:
                best, best_iou = k, iou(d["bbox"], g["bbox"])
        if best is None:
            # a detection on a hidden (ignored) object is neither right nor wrong
            if not any(g["iscrowd"] and iou(d["bbox"], g["bbox"]) >= IOU for g in gts):
                fp.append(d)
        else:
            used.add(best)
            (tp if not gts[best]["iscrowd"] else []).append(d)
    missed = [g for k, g in enumerate(gts) if k not in used and not g["iscrowd"]]
    hidden = [g for g in gts if g["iscrowd"]]
    return tp, fp, missed, hidden


def per_image(gt, dets):
    by_img = {i["id"]: ([], []) for i in gt["images"]}
    for a in gt["annotations"]:
        by_img[a["image_id"]][0].append(a)
    for d in dets:
        by_img[d["image_id"]][1].append(d)
    return by_img


def draw(img, tp, fp, missed, hidden, names, scale):
    def box(b, color, width, label=None):
        x, y, w, h = (np.array(b) * scale).round().astype(int)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, width, cv2.LINE_AA)
        if label:
            cv2.putText(img, label, (x + 2, max(y - 3, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
    for g in hidden:
        box(g["bbox"], HIDDEN, 1)
    for g in missed:
        box(g["bbox"], FN, 2)
    for d in fp:
        box(d["bbox"], FP, 2, f"{names[d['category_id'] - 1]} {d['score']:.2f}")
    for d in tp:
        box(d["bbox"], TP, 2)
    return img


def tile(path, entry, names, width):
    img = cv2.imread(str(path))
    scale = width / img.shape[1]
    img = cv2.resize(img, (width, round(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    return draw(img, *entry, names, scale)


def caption(img, text, height=26):
    bar = np.full((height, img.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (6, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def legend(width):
    bar = np.full((30, width, 3), 255, np.uint8)
    x = 8
    for color, text in ((TP, "correct"), (FP, "false detection"), (FN, "missed object"), (HIDDEN, "hidden by dust")):
        cv2.rectangle(bar, (x, 9), (x + 18, 21), color, 2)
        cv2.putText(bar, text, (x + 24, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
        x += 40 + 9 * len(text)
    cv2.putText(bar, f"score >= {THR}", (x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1, cv2.LINE_AA)
    return bar


def pad_rows(rows):
    h = max(r.shape[0] for r in rows)
    return [np.vstack([r, np.full((h - r.shape[0], r.shape[1], 3), 255, np.uint8)]) for r in rows]


def save(img, name):
    path = ROOT / "results" / "figures" / name
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"-> results/figures/{name}")


def gallery_dawn(names, n=5, width=360):
    data = {r: per_image(*load(r, "dawn_sand_test")) for r in DAWN_RUNS}
    gt = load(DAWN_RUNS[0], "dawn_sand_test")[0]
    files = {i["id"]: i["file_name"] for i in gt["images"]}
    matched = {r: {i: match(*data[r][i]) for i in files} for r in DAWN_RUNS}
    # images with the most misses summed over the runs, at least 3 labelled objects
    score = {i: sum(len(matched[r][i][2]) for r in DAWN_RUNS) for i in files if len(data[DAWN_RUNS[0]][i][0]) >= 3}
    # DAWN duplicate groups stay together inside the test split: skip near-copies (incl. mirrored)
    def thumb(i):
        g = cv2.imread(str(ROOT / "data" / "processed" / IMAGE_DIRS["dawn_sand_test"] / files[i]), cv2.IMREAD_GRAYSCALE)
        t = cv2.resize(g, (24, 24), interpolation=cv2.INTER_AREA).astype(np.float32)
        return (t - t.mean()) / (t.std() + 1e-6)
    pick, thumbs = [], []
    for i in sorted(score, key=lambda i: -score[i]):
        t = thumb(i)
        if all(max(np.mean(t * u), np.mean(t[:, ::-1] * u)) < 0.8 for u in thumbs):  # correlation
            pick.append(i)
            thumbs.append(t)
        if len(pick) == n:
            break
    rows = []
    for i in pick:
        cells = []
        for r in DAWN_RUNS:
            tp, fp, missed, _ = matched[r][i]
            t = tile(ROOT / "data" / "processed" / IMAGE_DIRS["dawn_sand_test"] / files[i], matched[r][i], names, width)
            cells.append(caption(t, f"{r}: {len(tp)} found, {len(missed)} missed, {len(fp)} false"))
        rows.append(np.hstack(pad_rows(cells)))
    grid = np.vstack(rows)
    save(np.vstack([legend(grid.shape[1]), grid]), "gallery_dawn.jpg")


def gallery_dust_levels(names, runs=("RTDETR-B1", "RTDETR-A"), width=420):
    tags = ["bdd_clear_val"] + [f"bdd_dusty_val_{s}_visible" for s in ("light", "medium", "heavy")]
    gt0 = load(runs[0], "bdd_clear_val")[0]
    files = {i["id"]: i["file_name"] for i in gt0["images"]}
    data = {(r, t): per_image(*load(r, t)) for r in runs for t in tags}
    # a busy but readable scene: 8-20 objects, at least 3 of them medium/large (area >= 32^2)
    def ok(i):
        g = data[(runs[0], tags[0])][i][0]
        return 8 <= len(g) <= 20 and sum(a["area"] >= 32 ** 2 for a in g) >= 3
    cands = [i for i in sorted(files) if ok(i)]
    # the scene where dust-trained A recovers the most objects that B1 misses at medium dust
    gain = lambda i: (len(match(*data[(runs[0], tags[2])][i])[2]) - len(match(*data[(runs[1], tags[2])][i])[2]))
    i = max(cands, key=gain)
    rows = []
    for t, label in zip(tags, ["clear", "light dust", "medium dust", "heavy dust"]):
        cells = []
        for r in runs:
            m = match(*data[(r, t)][i])
            d = "bdd_clear_val" if t == "bdd_clear_val" else t.removesuffix("_visible")
            tl = tile(ROOT / "data" / "processed" / IMAGE_DIRS[d] / files[i], m, names, width)
            cells.append(caption(tl, f"{r}, {label}: {len(m[0])} found, {len(m[2])} missed, "
                                     f"{len(m[3])} hidden, {len(m[1])} false"))
        rows.append(np.hstack(cells))
    grid = np.vstack(rows)
    save(np.vstack([legend(grid.shape[1]), grid]), "gallery_dust_levels.jpg")


def failure_counts():
    lines = [f"# Errors at score >= {THR} on DAWN sand test (IoU >= {IOU}, same class)\n",
             "A deployed detector uses one score threshold; these counts show what it would do at 0.5. "
             "Missed objects are broken down by COCO size (small < 32^2 px, medium < 96^2 px).\n",
             "| Run | Correct | False detections | Missed | Missed small / medium / large | Recall | Precision |",
             "|---|---|---|---|---|---|---|"]
    for r in RUNS:
        gt, dets = load(r, "dawn_sand_test")
        n_tp = n_fp = 0
        missed = []
        for gts, ds in per_image(gt, dets).values():
            tp, fp, m, _ = match(gts, ds)
            n_tp, n_fp = n_tp + len(tp), n_fp + len(fp)
            missed += m
        a = np.array([g["area"] for g in missed])
        sizes = f"{(a < 32 ** 2).sum()} / {((a >= 32 ** 2) & (a < 96 ** 2)).sum()} / {(a >= 96 ** 2).sum()}"
        lines.append(f"| {r} | {n_tp} | {n_fp} | {len(missed)} | {sizes} | {n_tp / (n_tp + len(missed)):.2f} "
                     f"| {n_tp / max(n_tp + n_fp, 1):.2f} |")
    path = ROOT / "results" / "tables" / "failure_counts.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("-> results/tables/failure_counts.md")


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    names = load_classes()[0]
    gallery_dawn(names)
    gallery_dust_levels(names)
    failure_counts()
