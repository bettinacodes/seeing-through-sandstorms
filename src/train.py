"""Fine-tune Faster R-CNN or RT-DETR (models.py) with fixed per-architecture recipes.

Every run of one architecture uses the same recipe and seed, so runs differ only in
starting weights, data and the few arguments below. Both architectures get an effective
batch of 2 and about the same wall-clock budget (~6 h overnight on the GTX 1650):

  frcnn   SGD momentum 0.9, wd 1e-4, lr 0.0025 (torchvision reference: 0.02 at batch 16,
          scaled linearly to 2). Batch 1 x 2 accumulation steps: batch 2 reserves 4.3 GB and
          spills into system RAM (2.2x slower per image); the backbone BatchNorm is frozen,
          so this equals batch 2. 4 epochs (1.44 h/epoch on 6000 images).
  rtdetr  AdamW, lr 1e-4 (backbone 1e-5), wd 1e-4, grad-norm clip 0.1 (RT-DETR paper).
          Batch 2. 8 epochs (0.76 h/epoch).
  both    500-step linear warm-up then cosine decay to 5%; FP32 (no AMP: fp16 is no faster
          on the GTX 1650); horizontal flip + short-side jitter 576-704 (models.py).

After every epoch the model is scored (mAP50-95, pycocotools) on --val; the best epoch is
kept as best.pt. Weights end up in runs/<name>/weights/{best,last}.pt (git-ignored).

Usage (from the repo root, laptop plugged in):
  python src/train.py --arch frcnn --name FRCNN-B1 --data configs/data_bdd_clear.yaml:train
  python src/train.py --arch rtdetr --name smoke --data configs/data_bdd_clear.yaml:train --epochs 1 --fraction 0.02
  python src/train.py --arch frcnn --name FRCNN-C --init runs/FRCNN-B1/weights/best.pt \\
      --data configs/data_dawn_sand.yaml:train configs/data_bdd_clear.yaml:train@500 \\
      --val configs/data_dawn_sand.yaml:val --epochs 30 --lr0 0.00025
"""
import argparse
import json
import math
import platform
import random
import subprocess
import time

import numpy as np
import torch

from common import ROOT, load_classes
from detdata import DetDataset, collate, spec_images
from evaluate import coco_eval, coco_ground_truth, predict
from models import ARCHS, Detector, build

RECIPES = {
    "frcnn": dict(optimizer="sgd", lr0=0.0025, backbone_lr_mult=1.0, weight_decay=1e-4,
                  batch=1, accumulate=2, epochs=4, clip=None),
    "rtdetr": dict(optimizer="adamw", lr0=1e-4, backbone_lr_mult=0.1, weight_decay=1e-4,
                   batch=2, accumulate=1, epochs=8, clip=0.1),
}
WARMUP_STEPS, FINAL_LR_FRAC, SEED = 500, 0.05, 0


def on_battery():
    """True if a Windows laptop is running on battery (Win32_Battery status 1 = discharging)."""
    if platform.system() != "Windows":
        return False
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "(Get-CimInstance Win32_Battery).BatteryStatus"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        return out == "1"
    except (OSError, subprocess.SubprocessError):
        return False


def make_optimizer(det, recipe, lr0):
    backbone, rest = [], []
    for n, p in det.named_parameters():
        if p.requires_grad:
            (backbone if ".backbone." in f".{n}" and "fpn" not in n else rest).append(p)
    groups = [{"params": rest, "lr": lr0},
              {"params": backbone, "lr": lr0 * recipe["backbone_lr_mult"]}]
    for g in groups:
        g["initial_lr"] = g["lr"]
    if recipe["optimizer"] == "sgd":
        return torch.optim.SGD(groups, momentum=0.9, weight_decay=recipe["weight_decay"])
    return torch.optim.AdamW(groups, weight_decay=recipe["weight_decay"])


def lr_factor(step, total):
    if step < WARMUP_STEPS:
        return (step + 1) / WARMUP_STEPS
    t = (step - WARMUP_STEPS) / max(1, total - WARMUP_STEPS)
    return FINAL_LR_FRAC + (1 - FINAL_LR_FRAC) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))


def validate(det, images, gt, batch):
    det.eval()
    dets, _ = predict(det, images, batch=batch)
    det.train()
    s = coco_eval(gt, dets).stats
    return {"mAP50": float(s[1]), "mAP50_95": float(s[0])}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", required=True, choices=ARCHS)
    ap.add_argument("--name", required=True, help="run name / method ID (runs/<name>)")
    ap.add_argument("--data", required=True, nargs="+", help="training data_yaml:split[@N] specs")
    ap.add_argument("--val", default="configs/data_bdd_clear.yaml:val", help="checkpoint-selection set")
    ap.add_argument("--init", help="start from this checkpoint (fix C) instead of the COCO weights")
    ap.add_argument("--epochs", type=int, help="default: the architecture's recipe")
    ap.add_argument("--lr0", type=float, help="default: the architecture's recipe")
    ap.add_argument("--fraction", type=float, default=1.0, help="fraction of the training set (smoke tests)")
    ap.add_argument("--val-fraction", type=float, default=1.0, help="fraction of the val set (smoke tests)")
    ap.add_argument("--patience", type=int, default=8, help="stop after this many epochs without a val gain")
    # On Windows each DataLoader worker re-imports torch (~3 GB committed memory each); with 4
    # workers the 16 GB laptop reached its commit limit (Oct 15). Decoding 2 images per ~2 s
    # step needs only one worker.
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--exist-ok", action="store_true", help="overwrite runs/<name>")
    ap.add_argument("--allow-battery", action="store_true", help="train even when not plugged in")
    args = ap.parse_args()

    if on_battery() and not args.allow_battery:
        raise SystemExit("On battery: plug the laptop in (or pass --allow-battery).")
    out = ROOT / "runs" / args.name
    if out.exists() and not args.exist_ok:
        raise SystemExit(f"{out} exists (pass --exist-ok to overwrite)")
    (out / "weights").mkdir(parents=True, exist_ok=True)

    recipe = dict(RECIPES[args.arch])
    recipe.update({k: v for k, v in (("epochs", args.epochs), ("lr0", args.lr0)) if v is not None})
    (out / "args.json").write_text(json.dumps({**vars(args), "recipe": recipe, "seed": SEED,
                                               "warmup_steps": WARMUP_STEPS}, indent=1))
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = "cuda"

    names = load_classes()[0]
    if args.init:
        det = Detector.load(args.init)
        assert det.arch == args.arch, f"{args.init} is a {det.arch} checkpoint"
    else:
        det = build(args.arch, num_classes=len(names))
    det.to(device).train()

    train_set = DetDataset(args.data, train=True, fraction=args.fraction, seed=SEED)
    loader = torch.utils.data.DataLoader(
        train_set, batch_size=recipe["batch"], shuffle=True, collate_fn=collate,
        num_workers=args.workers, persistent_workers=args.workers > 0,
        generator=torch.Generator().manual_seed(SEED))
    val_images = spec_images(args.val)
    if args.val_fraction < 1:
        val_images = sorted(random.Random(SEED).sample(val_images, max(1, round(args.val_fraction * len(val_images)))))
    val_gt = coco_ground_truth(val_images, names)

    opt = make_optimizer(det, recipe, recipe["lr0"])
    steps_per_epoch = len(loader) // recipe["accumulate"]
    total_steps = steps_per_epoch * recipe["epochs"]
    print(f"[{args.name}] {args.arch}, {len(train_set)} train / {len(val_images)} val images, "
          f"{recipe['epochs']} epochs x {steps_per_epoch} steps (batch {recipe['batch']} x {recipe['accumulate']})")

    log, best, bad_epochs, step = [], -1.0, 0, 0
    for epoch in range(1, recipe["epochs"] + 1):
        t0, losses, skipped = time.time(), [], 0
        torch.cuda.reset_peak_memory_stats()
        opt.zero_grad()
        for i, (images, targets, _) in enumerate(loader, 1):
            loss = det.loss([im.to(device) for im in images],
                            [{k: v.to(device) for k, v in t.items()} for t in targets])
            if not torch.isfinite(loss):
                skipped += 1
                if skipped > 20:
                    raise SystemExit("more than 20 non-finite losses in one epoch; stopping")
                opt.zero_grad()
                continue
            (loss / recipe["accumulate"]).backward()
            losses.append(loss.item())
            if i % recipe["accumulate"]:
                continue
            if recipe["clip"]:
                torch.nn.utils.clip_grad_norm_(det.parameters(), recipe["clip"])
            for g in opt.param_groups:
                g["lr"] = g["initial_lr"] * lr_factor(step, total_steps)
            opt.step()
            opt.zero_grad()
            step += 1
            if step % 100 == 0:
                rate = (time.time() - t0) / (i / recipe["accumulate"])
                eta = rate * (steps_per_epoch - i // recipe["accumulate"]) / 60
                print(f"  epoch {epoch} step {step % steps_per_epoch or steps_per_epoch}/{steps_per_epoch} "
                      f"loss {np.mean(losses[-100:]):.3f} lr {opt.param_groups[0]['lr']:.2e} "
                      f"eta {eta:.0f} min", flush=True)

        train_min = (time.time() - t0) / 60
        peak_gb = torch.cuda.max_memory_reserved() / 2**30  # > ~3.9 GB means spilling into system RAM
        torch.cuda.empty_cache()
        val = validate(det, val_images, val_gt, batch=1)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **val, "skipped_steps": skipped,
               "train_min": train_min, "val_min": (time.time() - t0) / 60 - train_min,
               "peak_mem_gb": peak_gb}
        log.append(row)
        (out / "results.json").write_text(json.dumps(log, indent=1))
        det.save(out / "weights" / "last.pt")
        if val["mAP50_95"] > best:
            best, bad_epochs = val["mAP50_95"], 0
            det.save(out / "weights" / "best.pt")
        else:
            bad_epochs += 1
        print(f"  epoch {epoch}: loss {row['train_loss']:.3f}, val mAP50 {val['mAP50']:.3f} "
              f"mAP50-95 {val['mAP50_95']:.3f} (best {best:.3f}), {train_min:.0f}+{row['val_min']:.0f} min, "
              f"{row['peak_mem_gb']:.2f} GB", flush=True)
        if bad_epochs >= args.patience:
            print(f"  no improvement for {args.patience} epochs; stopping")
            break
    print(f"done: runs/{args.name}/weights/best.pt (val mAP50-95 {best:.3f})")


if __name__ == "__main__":
    main()
