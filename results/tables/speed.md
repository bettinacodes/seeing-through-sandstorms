# Inference speed, one session, 100 DAWN test images, batch 1, GTX 1650 (FP32)

Detector time includes resizing inside the model (short side 640); pre-processing runs on the CPU.

| Detector | Params (M) | Median ms/img | Mean ms/img | FPS (median) |
|---|---|---|---|---|
| Faster R-CNN | 43.3 | 298.8 | 291.8 | 3.3 |
| RT-DETR | 20.1 | 80.7 | 79.1 | 12.4 |

| Pre-processing (CPU) | Mean ms/img |
|---|---|
| cc | 32.3 |
| dcp | 136.0 |
