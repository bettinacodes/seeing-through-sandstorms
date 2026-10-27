"""Gradio demo: run either detector on an image or a short video, optionally with synthetic dust
added first (src/dust.py) and/or a fix B pre-processing step (src/dehaze.py).

Run from the repo root (needs the trained weights in runs/<name>/weights/best.pt; B0 uses the
COCO weights):
  python app/demo.py            # opens http://127.0.0.1:7860
"""
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import cv2
import gradio as gr
import numpy as np
import torch

import dust
from common import load_classes
from dehaze import METHODS as DEHAZE
from detdata import to_tensor
from evaluate import load_detector

NAMES = load_classes()[0]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LABELS = {"B0": "zero-shot COCO", "B1": "clear BDD", "A": "+ synthetic dust training", "C": "+ few-shot DAWN"}
MODELS = {f"{arch}-{m} ({LABELS[m]})": (f"coco:{arch.lower()}" if m == "B0" else f"runs/{arch}-{m}/weights/best.pt")
          for arch in ("FRCNN", "RTDETR") for m in LABELS
          if m == "B0" or (ROOT / "runs" / f"{arch}-{m}" / "weights" / "best.pt").exists()}
COLORS = [(214, 120, 42), (52, 104, 235), (122, 175, 27), (0, 161, 237), (164, 123, 232), (0, 131, 0)]  # BGR per class
MAX_FRAMES, VIDEO_WIDTH = 300, 960
_cache = {}


def detector(choice):
    if choice not in _cache:
        _cache.clear()  # one model on the 4 GB GPU at a time
        torch.cuda.empty_cache()
        _cache[choice] = load_detector(str(ROOT / MODELS[choice]) if not MODELS[choice].startswith("coco:")
                                       else MODELS[choice], DEVICE)
    return _cache[choice]


def prepare(img_bgr, severity, preprocess, seed=0):
    if severity != "none":
        img_bgr = dust.add_dust(img_bgr, dust.sample_params(np.random.default_rng(seed), severity))
    if preprocess != "none":
        img_bgr = DEHAZE[preprocess](img_bgr)
    return img_bgr


@torch.no_grad()
def detect(det, img_bgr, thr):
    r = {k: v.cpu().numpy() for k, v in det.predict([to_tensor(img_bgr).to(DEVICE)])[0].items()}
    keep = r["scores"] >= thr
    return r["boxes"][keep], r["labels"][keep], r["scores"][keep]


def draw(img, boxes, labels, scores):
    img = img.copy()
    t = max(1, round(img.shape[1] / 640))
    for (x1, y1, x2, y2), c, s in zip(boxes.round().astype(int), labels, scores):
        cv2.rectangle(img, (x1, y1), (x2, y2), COLORS[c], 2 * t, cv2.LINE_AA)
        text = f"{NAMES[c]} {s:.2f}"
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45 * t, t)
        cv2.rectangle(img, (x1, max(y1 - h - 6, 0)), (x1 + w + 4, max(y1, h + 6)), COLORS[c], -1)
        cv2.putText(img, text, (x1 + 2, max(y1 - 4, h + 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * t, (255, 255, 255), t, cv2.LINE_AA)
    return img


def summary(labels):
    counts = {n: int((labels == i).sum()) for i, n in enumerate(NAMES) if (labels == i).any()}
    return ", ".join(f"{v} {k}" for k, v in counts.items()) or "nothing detected"


def run_image(image_rgb, model, severity, preprocess, thr):
    if image_rgb is None:
        raise gr.Error("Upload an image first.")
    det = detector(model)
    img = prepare(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), severity, preprocess)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    boxes, labels, scores = detect(det, img, thr)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    ms = 1000 * (time.perf_counter() - t0)
    out = cv2.cvtColor(draw(img, boxes, labels, scores), cv2.COLOR_BGR2RGB)
    return out, f"{summary(labels)}  |  detector {ms:.0f} ms on {DEVICE}"


def run_video(path, model, severity, preprocess, thr, progress=gr.Progress()):
    if not path:
        raise gr.Error("Upload a video first.")
    det = detector(model)
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    n = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or MAX_FRAMES, MAX_FRAMES)
    out_path = Path(tempfile.mkdtemp()) / "detections.webm"  # VP8: plays in the browser, no ffmpeg needed
    writer, frames, total, t_det = None, 0, 0, 0.0
    for _ in progress.tqdm(range(n), desc="frames"):
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[1] > VIDEO_WIDTH:
            frame = cv2.resize(frame, (VIDEO_WIDTH, round(frame.shape[0] * VIDEO_WIDTH / frame.shape[1])))
        frame = prepare(frame, severity, preprocess, seed=0)  # same dust parameters on every frame
        t0 = time.perf_counter()
        boxes, labels, scores = detect(det, frame, thr)
        t_det += time.perf_counter() - t0
        total += len(labels)
        if writer is None:
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"VP80"), fps, frame.shape[1::-1])
        writer.write(draw(frame, boxes, labels, scores))
        frames += 1
    cap.release()
    if writer is None:
        raise gr.Error("Could not read any frames from this video.")
    writer.release()
    return str(out_path), (f"{frames} frames (max {MAX_FRAMES}), {total / frames:.1f} detections/frame, "
                           f"detector {1000 * t_det / frames:.0f} ms/frame on {DEVICE}")


def controls():
    model = gr.Dropdown(list(MODELS), value=next(iter(MODELS)), label="Model")
    severity = gr.Radio(["none", *dust.SEVERITIES], value="none", label="Add synthetic dust")
    preprocess = gr.Radio(["none", *DEHAZE], value="none", label="Pre-processing (fix B: cc = colour correction, dcp = dark channel prior)")
    thr = gr.Slider(0.05, 0.95, value=0.5, step=0.05, label="Score threshold")
    return [model, severity, preprocess, thr]


with gr.Blocks(title="Seeing Through Sandstorms") as demo:
    gr.Markdown("# Seeing Through Sandstorms\nFaster R-CNN (CNN) vs RT-DETR (transformer) on sandstorm images. "
                "Classes: " + ", ".join(NAMES) + ". Synthetic dust uses a flat-ground depth proxy here "
                "(the evaluation sets used monocular depth).")
    with gr.Tab("Image"):
        with gr.Row():
            with gr.Column():
                image_in = gr.Image(type="numpy", label="Input image")
                image_ctrl = controls()
                image_btn = gr.Button("Detect", variant="primary")
            with gr.Column():
                image_out = gr.Image(label="Detections")
                image_info = gr.Textbox(label="Result", interactive=False)
        image_btn.click(run_image, [image_in, *image_ctrl], [image_out, image_info])
    with gr.Tab("Video"):
        with gr.Row():
            with gr.Column():
                video_in = gr.Video(label=f"Input video (first {MAX_FRAMES} frames are processed)")
                video_ctrl = controls()
                video_btn = gr.Button("Detect", variant="primary")
            with gr.Column():
                video_out = gr.Video(label="Detections")
                video_info = gr.Textbox(label="Result", interactive=False)
        video_btn.click(run_video, [video_in, *video_ctrl], [video_out, video_info])


if __name__ == "__main__":
    demo.launch()
