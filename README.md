# Seeing Through Sandstorms

Robust object detection for autonomous driving in desert sandstorm conditions: a comparison of a
CNN detector (**Faster R-CNN R50-FPN v2**) and a transformer detector (**RT-DETR-R18**).
Both detectors are fine-tuned on clear BDD100K images and tested on physics-based synthetic dust (light,
medium, heavy) and on real sandstorm photos from DAWN. Three fixes are compared: training with synthetic
dust (A), colour correction or dehazing before detection (B), and few-shot fine-tuning on 79 real storm
images (C). Tested on one laptop GPU (GTX 1650, 4 GB).

## Setup (Windows, Python 3.12)
1. Update the NVIDIA driver (tested: 617.14). Train plugged in, Windows power mode set to Best performance.
2. Create the environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
   pip install -r requirements.txt
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```
   Tested with torch 2.14.1+cu126, torchvision 0.29.1, transformers 5.19.0. Pretrained weights are
   downloaded automatically on first use: the COCO detectors (torchvision cache, Hugging Face hub
   `PekingU/rtdetr_r18vd`) and the Depth Anything V2 depth model (Hugging Face hub
   `depth-anything/Depth-Anything-V2-Small-hf`), a ViT that supplies the scene depth for the synthetic dust.

## Trained weights (skip training)
The six trained models are attached to the GitHub release `weights-v1` (~760 MB in total). This
puts each one at `runs/<name>/weights/best.pt`, where every script below expects it:
```powershell
python src/download_weights.py                  # all six; or name some, e.g. RTDETR-B1 FRCNN-B1
python app/demo.py                              # runs straight away on your own images or videos
```
With the weights downloaded, skip the Training section. Re-running the evaluation still needs the
datasets (next section), because the images cannot be redistributed with the code.

## Data
Place raw downloads under `data/raw/` (git-ignored):
- `data/raw/bdd100k/`: 100K images + detection labels (registration at bdd-data.berkeley.edu)
- `data/raw/dawn/`: DAWN dataset (Mendeley Data)

Build the datasets (from the repo root). Splits are frozen in `configs/splits/` and re-used,
so these commands reproduce the exact same data:
```powershell
python src/prepare_dawn.py   # DAWN sandstorm -> data/processed/dawn_sand (fewshot_train/fewshot_val/test)
python src/prepare_bdd.py    # BDD100K clear+daytime -> data/processed/bdd_clear (train 6000 / val 1000)
python src/depth.py configs/data_bdd_clear.yaml:train configs/data_bdd_clear.yaml:val   # monocular depth for the dust (~17 min)
python src/make_dusty.py     # synthetic dust -> data/processed/bdd_dusty (fix A train + paired dusty val sets)
python src/check_labels.py data/processed/dawn_sand test   # label overlay -> results/figures/
```
Dataset configs: `configs/data_bdd_clear.yaml`, `configs/data_bdd_dusty.yaml`, `configs/data_dawn_sand.yaml`
(relative paths, so run from the repo root). Scripts refer to a split as `data_yaml:split`. The DAWN
`test` split is the held-out sandstorm test set: never train or tune on it.

## Training
```powershell
python src/train.py --arch frcnn  --name FRCNN-B1  --data configs/data_bdd_clear.yaml:train   # ~6 h
python src/train.py --arch rtdetr --name RTDETR-B1 --data configs/data_bdd_clear.yaml:train   # ~6 h
python src/train.py --arch frcnn  --name FRCNN-A   --data configs/data_bdd_dusty.yaml:train   # fix A
python src/train.py --arch rtdetr --name RTDETR-A  --data configs/data_bdd_dusty.yaml:train   # fix A
python src/train.py --arch frcnn --name FRCNN-C --init runs/FRCNN-B1/weights/best.pt `
    --data configs/data_dawn_sand.yaml:train configs/data_bdd_clear.yaml:train@500 `
    --val configs/data_dawn_sand.yaml:val --epochs 30 --lr0 0.00025                          # fix C (~1 h)
python src/train.py --arch rtdetr --name RTDETR-C --init runs/RTDETR-B1/weights/best.pt `
    --data configs/data_dawn_sand.yaml:train configs/data_bdd_clear.yaml:train@500 `
    --val configs/data_dawn_sand.yaml:val --epochs 30 --lr0 0.00001                          # fix C (~40 min)
```
Per-architecture recipes (optimiser, learning rate, batch, epochs) are fixed in `src/train.py`.
The best epoch on clear BDD val is kept as `runs/<name>/weights/best.pt`; per-epoch metrics go to
`runs/<name>/results.json`.

## Evaluation
All 12 runs (both architectures x B0, B1, A, B-cc, B-dcp, C), then the 14 paired comparisons that
`results.py` turns into tables:
```powershell
# zero-shot COCO baseline (B0)
python src/evaluate.py --weights coco:frcnn  --name FRCNN-B0
python src/evaluate.py --weights coco:rtdetr --name RTDETR-B0
# trained models (B1, A, C)
foreach ($r in "FRCNN-B1", "FRCNN-A", "FRCNN-C", "RTDETR-B1", "RTDETR-A", "RTDETR-C") {
    python src/evaluate.py --weights runs/$r/weights/best.pt --name $r
}
# fix B: B1 with pre-processing in front
foreach ($a in "FRCNN", "RTDETR") { foreach ($m in "cc", "dcp") {
    python src/evaluate.py --weights runs/$a-B1/weights/best.pt --name $a-B-$m --preprocess $m
} }
# paired differences + 95% CIs -> results/comparisons.csv
foreach ($a in "FRCNN", "RTDETR") {
    foreach ($f in "A", "B-cc", "B-dcp", "C") { python src/compare.py $a-$f $a-B1 }
    python src/compare.py $a-B1 $a-B0
}
foreach ($v in "B0", "B1", "A", "C") { python src/compare.py RTDETR-$v FRCNN-$v }
```
Fix B pre-processing (`src/dehaze.py`): `cc` = gray-world colour correction + CLAHE, `dcp` = dark
channel prior; `python src/dehaze.py --grid` writes `results/figures/dehaze_examples.jpg`.
`evaluate.py` scores BDD clear val, the paired dusty val sets (light/medium/heavy) and the DAWN
sandstorm test set with pycocotools (mAP50, mAP50-95, AP small/medium/large, per-class AP), and adds
1000x bootstrap 95% CIs over test images. The dusty sets are also scored as `<set>_visible`, where
objects the dust has hidden (transmission < 0.3) are ignored. Per-run metrics go to `results/metrics/`; one row per
(method, test set) goes to `results/summary.csv`. `compare.py` applies the same bootstrap resamples
to two runs and reports a CI of their difference in `results/comparisons.csv`.

## Results, figures and demo
```powershell
python src/results.py --speed   # tables -> results/tables/, figures -> results/figures/ (--speed needs the GPU)
python src/gallery.py           # failure-case galleries + results/tables/failure_counts.md
python app/demo.py              # Gradio demo at http://127.0.0.1:7860 (image or short video)
```
- `results/tables/main_map50_95.md`, `main_map50.md`: every method x test set with 95% CIs.
- `results/tables/fix_vs_b1.md`: paired difference of every fix vs B1, and RT-DETR vs Faster R-CNN.
- `results/tables/storm_thirds.md`, `size_buckets.md`, `per_class_dawn.md`, `failure_counts.md`: DAWN results
  by storm intensity, object size and class, and the errors at score 0.5.
- `results/tables/speed.md`: both detectors and both pre-processing methods timed in one session.
- Figures: `dose_response.png` (clear -> heavy synthetic dust), `dawn_methods.png`, `storm_thirds.png`,
  `size_buckets.png`, `gallery_dawn.jpg`, `gallery_dust_levels.jpg`. Only the charts (`.png`) are in the
  repository; figures built from dataset photos (`.jpg`) show faces and number plates and cannot be
  redistributed, so the scripts above regenerate them locally.

The demo runs any trained model (or the zero-shot COCO baseline) on an uploaded image or video,
optionally after adding synthetic dust and/or a fix B pre-processing step. Videos are written as
WebM (VP8), which needs no ffmpeg install.

## Layout
```
configs/   class mapping and dataset configs
data/      raw/ and processed/ (git-ignored)
src/       data preparation, dust synthesis, dehazing, models, training, evaluation, comparison, tables, galleries
app/       Gradio demo
results/   metrics, tables and figures
```
