"""Download the six trained models from the GitHub release into runs/<name>/weights/best.pt.

Evaluation, galleries and the demo then run without retraining. B0 needs no download: it uses
the COCO weights, which torchvision and Hugging Face fetch on first use. Every file is checked
against the release's SHA256SUMS.txt; files that are already present and match are skipped.

Usage (from the repo root):
  python src/download_weights.py                 # all six (~760 MB)
  python src/download_weights.py RTDETR-B1 FRCNN-B1
"""
import argparse
import hashlib
import shutil
import urllib.request

from common import ROOT

RELEASE = "https://github.com/bettinacodes/seeing-through-sandstorms/releases/download/weights-v1"
RUNS = ["FRCNN-B1", "FRCNN-A", "FRCNN-C", "RTDETR-B1", "RTDETR-A", "RTDETR-C"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", choices=RUNS, metavar="RUN", help=f"default: all ({' '.join(RUNS)})")
    args = ap.parse_args()
    runs = args.runs or RUNS

    with urllib.request.urlopen(f"{RELEASE}/SHA256SUMS.txt") as r:
        sums = {name.strip().lstrip("*")[:-3]: digest for digest, name in
                (line.split(maxsplit=1) for line in r.read().decode().splitlines() if line.strip())}

    for run in runs:
        dst = ROOT / "runs" / run / "weights" / "best.pt"
        if dst.exists() and sha256(dst) == sums[run]:
            print(f"{run}: already present")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".part")
        print(f"{run}: downloading ...", flush=True)
        with urllib.request.urlopen(f"{RELEASE}/{run}.pt") as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
        if sha256(tmp) != sums[run]:
            tmp.unlink()
            raise SystemExit(f"{run}: checksum mismatch, download again")
        tmp.replace(dst)
        print(f"{run}: -> {dst.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
