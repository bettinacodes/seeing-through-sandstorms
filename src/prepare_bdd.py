"""BDD100K clear + daytime subset: BDD JSON -> YOLO txt with the shared 6 classes.

  train  6000 images sampled from the official BDD100K train split
  val    1000 images sampled from the official BDD100K val split

Only images with weather == "clear", timeofday == "daytime" and at least one box of
the shared classes are eligible; images that are actually dark (mis-tagged night
scenes) are skipped while sampling. The 1.45 GB train label file is streamed (ijson)
and the filtered result cached in data/processed/bdd_cache/, so later runs are fast.
The sampled file lists are frozen in configs/splits/bdd_clear.json and reused;
pass --resample to draw a new sample.

Usage (from the repo root):
  python src/prepare_bdd.py
"""
import argparse
import json
import random
from collections import Counter
from datetime import date

import cv2
import ijson
from tqdm import tqdm

from common import ROOT, clip_box, link_or_copy, load_classes, reset_split_dirs, yolo_line

RAW = ROOT / "data" / "raw" / "bdd100k"
OUT = ROOT / "data" / "processed" / "bdd_clear"
CACHE = ROOT / "data" / "processed" / "bdd_cache"
SPLIT_FILE = ROOT / "configs" / "splits" / "bdd_clear.json"
SEED = 42
N_IMAGES = {"train": 6000, "val": 1000}
IMG_W, IMG_H = 1280, 720  # all BDD100K images (checked on 2026-10-06)
# ~1% of "daytime" images are actually night. Mean gray level (0-255) below this is
# rejected; checked by eye on 2026-10-06: < 45 is night, 50-58 dusk (kept), > 60 day.
MIN_BRIGHTNESS = 45


def is_dark(path):
    return cv2.resize(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE), (160, 90)).mean() < MIN_BRIGHTNESS


def filtered_labels(split, bdd_map):
    """{image name: [[class name, x1, y1, x2, y2], ...]} for clear-daytime images, cached."""
    cache = CACHE / f"{split}_clear_daytime.json"
    if cache.exists():
        return json.loads(cache.read_text())
    src = RAW / "labels" / f"bdd100k_labels_images_{split}.json"
    out, weather_time = {}, Counter()
    with open(src, "rb") as f:
        for item in tqdm(ijson.items(f, "item", use_float=True), desc=f"streaming {src.name}", unit=" img"):
            attrs = item.get("attributes", {})
            weather_time[(attrs.get("weather"), attrs.get("timeofday"))] += 1
            if attrs.get("weather") != "clear" or attrs.get("timeofday") != "daytime":
                continue
            out[item["name"]] = [[bdd_map[l["category"]], *(l["box2d"][k] for k in ("x1", "y1", "x2", "y2"))]
                                 for l in item.get("labels") or [] if l["category"] in bdd_map and "box2d" in l]
    CACHE.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out))
    print(f"{split}: {sum(weather_time.values())} images, {len(out)} clear+daytime -> {cache.relative_to(ROOT)}")
    return out


def image_index(split):
    """Image name -> path. The train images may sit in sub-folders (trainA, testB, ...)."""
    return {p.name: p for p in (RAW / "images" / "100k" / split).rglob("*.jpg")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resample", action="store_true", help="discard the frozen sample and draw a new one")
    args = ap.parse_args()

    names, name_to_id, cfg = load_classes()
    labels = {s: filtered_labels(s, cfg["bdd100k"]) for s in N_IMAGES}
    index = {s: image_index(s) for s in N_IMAGES}

    if SPLIT_FILE.exists() and not args.resample:
        manifest = json.loads(SPLIT_FILE.read_text())
        print(f"Using frozen sample {SPLIT_FILE.relative_to(ROOT)} (created {manifest['created']})")
    else:
        rng = random.Random(SEED)
        splits, pools, rejected = {}, {}, {}
        for s, n in N_IMAGES.items():
            pool = sorted(k for k, v in labels[s].items() if v and k in index[s])
            pools[s] = {"clear_daytime": len(labels[s]), "with_objects_on_disk": len(pool)}
            rng.shuffle(pool)
            splits[s], rejected[s] = [], []
            for name in pool:  # draw in shuffled order, skipping night images mis-tagged as daytime
                if len(splits[s]) == n:
                    break
                (rejected[s] if is_dark(index[s][name]) else splits[s]).append(name)
            splits[s].sort()
            print(f"{s}: rejected {len(rejected[s])} dark images while drawing {n}")
        manifest = {"created": date.today().isoformat(), "seed": SEED, "min_brightness": MIN_BRIGHTNESS,
                    "pools": pools, "rejected_dark": rejected, "splits": splits}
        SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
        SPLIT_FILE.write_text(json.dumps(manifest, indent=1))
        print(f"Wrote new sample to {SPLIT_FILE.relative_to(ROOT)}; pools: {pools}")

    reset_split_dirs(OUT, manifest["splits"])
    summary, n_degenerate = {}, 0
    for split, files in manifest["splits"].items():
        cls_counts, box_area = Counter(), Counter()
        for fname in tqdm(files, desc=f"writing {split}"):
            lines = []
            for cls, *xyxy in labels[split][fname]:
                box = clip_box(*xyxy, IMG_W, IMG_H)
                if box is None:
                    n_degenerate += 1
                    continue
                lines.append(yolo_line(name_to_id[cls], *box, IMG_W, IMG_H))
                cls_counts[cls] += 1
                area = (box[2] - box[0]) * (box[3] - box[1])  # COCO size buckets
                box_area["small" if area < 32 ** 2 else "medium" if area < 96 ** 2 else "large"] += 1
            link_or_copy(index[split][fname], OUT / "images" / split / fname)
            (OUT / "labels" / split / fname.replace(".jpg", ".txt")).write_text("\n".join(lines) + "\n")
        summary[split] = {"images": len(files), "boxes": sum(cls_counts.values()),
                          **{n: cls_counts[n] for n in names}, **box_area}

    print(f"\n{'split':<7}{'imgs':>6}{'boxes':>7}" + "".join(f"{n[:5]:>7}" for n in names)
          + f"{'small':>7}{'med':>7}{'large':>7}")
    for split, s in summary.items():
        print(f"{split:<7}{s['images']:>6}{s['boxes']:>7}" + "".join(f"{s[n]:>7}" for n in names)
              + f"{s['small']:>7}{s['medium']:>7}{s['large']:>7}")
    print(f"Degenerate boxes dropped: {n_degenerate}")

    stats_file = ROOT / "results" / "data_stats_bdd_clear.json"
    stats_file.write_text(json.dumps({"splits": summary, "pools": manifest.get("pools"),
                                      "degenerate_boxes": n_degenerate}, indent=1))
    print(f"Stats -> {stats_file.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
