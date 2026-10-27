"""DAWN sandstorm subset: VOC XML -> YOLO txt, plus a frozen, seeded split.

Split (79 / 22 / 222 images once near-duplicate groups are kept whole):
  fewshot_train  80 images  - fix C fine-tuning
  fewshot_val    20 images  - fix C validation
  test           the rest   - held-out test set for every method, never tuned on

The split is stratified by source prefix (dusttornado / sand_storm / sand_storm_g2)
and keeps near-duplicate images in the same split, so the test
set cannot leak into few-shot training. The split is written once to
configs/splits/dawn_sand.json and reused on every later run (frozen). Pass
--resplit only if you really mean to change the test set.

Usage (from the repo root):
  python src/prepare_dawn.py
"""
import argparse
import json
import random
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import cv2
import numpy as np

from common import ROOT, clip_box, link_or_copy, load_classes, reset_split_dirs, yolo_line

RAW = ROOT / "data" / "raw" / "dawn" / "Sand"
OUT = ROOT / "data" / "processed" / "dawn_sand"
SPLIT_FILE = ROOT / "configs" / "splits" / "dawn_sand.json"
SEED = 42
N_FEWSHOT, N_FEWSHOT_VAL = 100, 20
# DAWN re-uses the same photos under different names, sometimes recoloured, cropped
# or mirrored. Global hashes / thumbnail correlation chain unrelated hazy images
# together, so duplicates are verified with ORB keypoints + RANSAC homography.
# Calibrated by eye on 2026-10-06: true duplicates >= 36 inliers (mirrored copy),
# different scenes mostly <= 7; one false match at exactly 30 (sand_storm-212 /
# sand_storm_g2_-063). A false match only merges two groups into the same split, so
# it can never leak test images - erring low is the safe direction.
DUP_MIN_INLIERS = 30


def source_of(name):
    """dusttornado-001.jpg -> dusttornado; sand_storm_g2_-005.jpg -> sand_storm_g2."""
    return re.sub(r"_?-\d+$", "", Path(name).stem).rstrip("_")


def orb_features(path, orb, clahe):
    """ORB keypoints/descriptors for the image and its mirror, at 640 px width."""
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    g = clahe.apply(cv2.resize(g, (640, round(640 * g.shape[0] / g.shape[1]))))
    return [orb.detectAndCompute(img, None) for img in (g, cv2.flip(g, 1))]


def ransac_inliers(fa, fb, matcher):
    (ka, da), (kb, db) = fa, fb
    if da is None or db is None or len(da) < 2 or len(db) < 2:
        return 0
    good = [p[0] for p in matcher.knnMatch(da, db, k=2) if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < 8:
        return 0
    _, mask = cv2.findHomography(np.float32([ka[m.queryIdx].pt for m in good]),
                                 np.float32([kb[m.trainIdx].pt for m in good]), cv2.RANSAC, 5.0)
    return 0 if mask is None else int(mask.sum())


def duplicate_pairs(images):
    """All (i, j, inliers) pairs with >= DUP_MIN_INLIERS, comparing plain and mirrored."""
    orb, clahe, matcher = cv2.ORB_create(1500), cv2.createCLAHE(3.0, (8, 8)), cv2.BFMatcher(cv2.NORM_HAMMING)
    feats = [orb_features(p, orb, clahe) for p in images]
    pairs = []
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            n = max(ransac_inliers(feats[i][0], feats[j][k], matcher) for k in (0, 1))
            if n >= DUP_MIN_INLIERS:
                pairs.append((i, j, n))
    return pairs


def near_duplicate_groups(images, pairs):
    """Union-find over duplicate pairs."""
    parent = list(range(len(images)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j, _ in pairs:
        parent[find(i)] = find(j)
    groups = defaultdict(list)
    for i, p in enumerate(images):
        groups[find(i)].append(p.name)
    return sorted((sorted(g) for g in groups.values()), key=lambda g: g[0])


def make_split(groups):
    rng = random.Random(SEED)
    n_total = sum(len(g) for g in groups)
    by_source = defaultdict(list)
    for g in groups:
        by_source[source_of(g[0])].append(g)

    fewshot = []
    for src in sorted(by_source):
        src_groups = by_source[src]
        quota = round(N_FEWSHOT * sum(len(g) for g in src_groups) / n_total)
        rng.shuffle(src_groups)
        taken = 0
        for g in src_groups:
            if taken >= quota:
                break
            fewshot.append(g)
            taken += len(g)
    fewshot_names = {n for g in fewshot for n in g}

    rng.shuffle(fewshot)
    val, n_val = [], 0
    for g in fewshot:
        if n_val >= N_FEWSHOT_VAL:
            break
        val.append(g)
        n_val += len(g)
    val_names = {n for g in val for n in g}

    all_names = sorted(n for g in groups for n in g)
    return {
        "fewshot_train": sorted(fewshot_names - val_names),
        "fewshot_val": sorted(val_names),
        "test": sorted(n for n in all_names if n not in fewshot_names),
    }


def convert(xml_path, img_path, name_to_id, dawn_map, stats):
    """Parse one VOC file; return YOLO label lines."""
    root = ET.parse(xml_path).getroot()
    h, w = cv2.imread(str(img_path)).shape[:2]
    xw, xh = int(root.findtext("size/width", 0)), int(root.findtext("size/height", 0))
    if (xw, xh) != (w, h):
        stats["size_mismatch"].append(f"{img_path.name}: xml {xw}x{xh} vs image {w}x{h}")
    lines = []
    for obj in root.iter("object"):
        name = obj.findtext("name").strip().lower()
        if name not in dawn_map:
            stats["dropped_labels"][name] += 1
            continue
        bb = obj.find("bndbox")
        box = clip_box(*(float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")), w, h)
        if box is None:
            stats["degenerate_boxes"] += 1
            continue
        lines.append(yolo_line(name_to_id[dawn_map[name]], *box, w, h))
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resplit", action="store_true", help="discard the frozen split and make a new one")
    args = ap.parse_args()

    names, name_to_id, cfg = load_classes()
    images = sorted(RAW.glob("*.jpg"))
    xml_dir = RAW / "Sand_PASCAL_VOC"
    missing = [p.name for p in images if not (xml_dir / f"{p.stem}.xml").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} images have no XML, e.g. {missing[:5]}")
    print(f"{len(images)} sandstorm images in {RAW}")

    if SPLIT_FILE.exists() and not args.resplit:
        manifest = json.loads(SPLIT_FILE.read_text())
        print(f"Using frozen split {SPLIT_FILE.relative_to(ROOT)} (created {manifest['created']})")
        on_disk = {p.name for p in images}
        in_split = {n for s in manifest["splits"].values() for n in s}
        if on_disk != in_split:
            raise SystemExit("Images on disk differ from the frozen split; investigate before --resplit.")
    else:
        print("Finding near-duplicates (ORB + RANSAC over all pairs, ~15 min)...")
        pairs = duplicate_pairs(images)
        groups = near_duplicate_groups(images, pairs)
        dups = [g for g in groups if len(g) > 1]
        print(f"Near-duplicate groups (>= {DUP_MIN_INLIERS} inliers): {len(dups)} "
              f"covering {sum(len(g) for g in dups)} images")
        for i, j, n in sorted(pairs, key=lambda p: p[2])[:10]:
            print(f"  weakest: {n:4d} inliers {images[i].name} {images[j].name}")
        manifest = {
            "created": date.today().isoformat(),
            "seed": SEED,
            "dup_min_inliers": DUP_MIN_INLIERS,
            "duplicate_pairs": [[images[i].name, images[j].name, n] for i, j, n in pairs],
            "near_duplicate_groups": dups,
            "splits": make_split(groups),
        }
        SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
        SPLIT_FILE.write_text(json.dumps(manifest, indent=1))
        print(f"Wrote new split to {SPLIT_FILE.relative_to(ROOT)}")

    splits = manifest["splits"]
    assert not set(splits["test"]) & (set(splits["fewshot_train"]) | set(splits["fewshot_val"]))

    reset_split_dirs(OUT, splits)
    stats = {"size_mismatch": [], "dropped_labels": Counter(), "degenerate_boxes": 0}
    summary = {}
    for split, files in splits.items():
        cls_counts, n_empty = Counter(), 0
        for fname in files:
            img = RAW / fname
            lines = convert(xml_dir / f"{img.stem}.xml", img, name_to_id, cfg["dawn"], stats)
            link_or_copy(img, OUT / "images" / split / fname)
            (OUT / "labels" / split / f"{img.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            cls_counts.update(names[int(l.split()[0])] for l in lines)
            n_empty += not lines
        summary[split] = {"images": len(files), "empty_images": n_empty, "boxes": sum(cls_counts.values()),
                          **{n: cls_counts[n] for n in names}}

    print(f"\n{'split':<14}{'imgs':>5}{'empty':>6}{'boxes':>6}" + "".join(f"{n[:5]:>7}" for n in names))
    for split, s in summary.items():
        print(f"{split:<14}{s['images']:>5}{s['empty_images']:>6}{s['boxes']:>6}"
              + "".join(f"{s[n]:>7}" for n in names))
    print(f"\nDropped labels: {dict(stats['dropped_labels']) or 'none'}; "
          f"degenerate boxes: {stats['degenerate_boxes']}; XML/image size mismatches: {len(stats['size_mismatch'])}")
    for m in stats["size_mismatch"][:10]:
        print("  ", m)

    stats_file = ROOT / "results" / "data_stats_dawn_sand.json"
    stats_file.write_text(json.dumps({"splits": summary, "dropped_labels": stats["dropped_labels"],
                                      "degenerate_boxes": stats["degenerate_boxes"],
                                      "size_mismatch": stats["size_mismatch"]}, indent=1))
    print(f"Stats -> {stats_file.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
