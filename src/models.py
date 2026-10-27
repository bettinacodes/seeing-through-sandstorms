"""The two compared detectors behind one interface.

  frcnn   Faster R-CNN ResNet-50-FPN v2 (torchvision): two-stage CNN, 43.7M params, COCO 46.7 AP
  rtdetr  RT-DETR-R18 (Hugging Face, PekingU/rtdetr_r18vd): transformer detector with a
          ResNet-18 backbone, 20.2M params, COCO 46.5 AP

Both models see the same input: images are resized inside the model so the short side is
640 (long side capped at 1152; BDD 1280x720 -> 640x1138), with the short side drawn from
TRAIN_SIZES during training. Faster R-CNN does this in its own GeneralizedRCNNTransform;
the RT-DETR adapter copies the same rule. Predictions come back in original image pixels.

Labels are always the shared class ids of configs/classes.yaml (0..5). COCO-pretrained
weights (zero-shot baseline B0) keep their COCO head and are mapped to the shared classes
by name; detections of other COCO classes are dropped.

  det = build("frcnn")                       # COCO head, zero-shot
  det = build("rtdetr", num_classes=6)       # new 6-class head, for fine-tuning
  loss = det.loss(images, targets)           # train mode
  preds = det.predict(images)                # [{"boxes", "scores", "labels"}]
  det.save("runs/x/weights/best.pt"); det = Detector.load(...)
"""
import torch
import torch.nn.functional as F

from common import load_classes

ARCHS = ("frcnn", "rtdetr")
RTDETR_ID = "PekingU/rtdetr_r18vd"
TRAIN_SIZES, EVAL_SIZE, MAX_SIZE = (576, 608, 640, 672, 704), 640, 1152
MAX_DET = 300
COCO_ALIASES = {"motorbike": "motorcycle"}  # HF RT-DETR uses VOC-style names for some COCO classes


def resize(img, short, max_size=MAX_SIZE):
    """torchvision GeneralizedRCNNTransform's resize rule, for one CHW image."""
    h, w = img.shape[-2:]
    scale = min(short / min(h, w), max_size / max(h, w))
    out = F.interpolate(img[None], scale_factor=scale, mode="bilinear",
                        recompute_scale_factor=True, align_corners=False)[0]
    return out, (out.shape[-1] / w, out.shape[-2] / h)


def name_mapping(model_names):
    """{model label -> shared class id} for the model classes that are in the shared set."""
    shared = load_classes()[1]
    return {i: shared[COCO_ALIASES.get(n, n)] for i, n in model_names.items()
            if COCO_ALIASES.get(n, n) in shared}


class Detector(torch.nn.Module):
    arch = None

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes  # None = COCO head (zero-shot)

    def save(self, path):
        torch.save({"arch": self.arch, "num_classes": self.num_classes,
                    "state_dict": self.state_dict()}, path)

    @staticmethod
    def load(path, device="cpu"):
        ckpt = torch.load(path, map_location=device, weights_only=True)
        # Rebuilt from the (cached) COCO weights, so frozen layers and FrozenBatchNorm match training.
        det = build(ckpt["arch"], ckpt["num_classes"])
        det.load_state_dict(ckpt["state_dict"])
        return det

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class FasterRCNN(Detector):
    arch = "frcnn"

    def __init__(self, num_classes=None):
        super().__init__(num_classes)
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        # trainable_backbone_layers=3 (torchvision's default): conv1 and layer1 stay frozen.
        self.model = fasterrcnn_resnet50_fpn_v2(
            weights=weights, trainable_backbone_layers=3, min_size=EVAL_SIZE, max_size=MAX_SIZE,
            box_score_thresh=0.001, box_detections_per_img=MAX_DET)
        if num_classes is None:
            cats = weights.meta["categories"]
            self.to_shared = name_mapping(dict(enumerate(cats)))
            # Other COCO classes are dropped after NMS, so keep more boxes before filtering.
            self.model.roi_heads.detections_per_img = 1000
        else:
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
            self.to_shared = {i + 1: i for i in range(num_classes)}  # label 0 = background

    def train(self, mode=True):
        super().train(mode)
        self.model.transform.min_size = TRAIN_SIZES if mode else (EVAL_SIZE,)
        return self

    def loss(self, images, targets):
        targets = [{"boxes": t["boxes"], "labels": t["labels"] + 1} for t in targets]
        return sum(self.model(images, targets).values())

    @torch.no_grad()
    def predict(self, images):
        out = []
        for r in self.model(images):
            keep = [i for i, c in enumerate(r["labels"].tolist()) if c in self.to_shared][:MAX_DET]
            out.append({"boxes": r["boxes"][keep], "scores": r["scores"][keep],
                        "labels": torch.tensor([self.to_shared[c] for c in r["labels"][keep].tolist()],
                                               dtype=torch.long)})
        return out


class RTDETR(Detector):
    arch = "rtdetr"

    def __init__(self, num_classes=None):
        super().__init__(num_classes)
        from transformers import RTDetrForObjectDetection
        from transformers.utils import logging
        logging.set_verbosity_error()  # the expected "newly initialised head" warning
        kw = {}
        if num_classes is not None:
            names = load_classes()[0]
            kw = dict(num_labels=num_classes, id2label=dict(enumerate(names)),
                      label2id={n: i for i, n in enumerate(names)})
        self.model = RTDetrForObjectDetection.from_pretrained(RTDETR_ID, ignore_mismatched_sizes=True, **kw)
        if num_classes is None:
            self.to_shared = name_mapping(self.model.config.id2label)
        else:
            self.to_shared = {i: i for i in range(num_classes)}

    def batch(self, images, targets=None):
        """Resize (same rule as Faster R-CNN) and zero-pad to a common size divisible by 32.

        The checkpoint's image processor only rescales to [0, 1] (do_normalize=False), which
        the dataset already does. Boxes are normalised to the padded tensor, which is the frame
        the model predicts in.
        """
        sizes = [TRAIN_SIZES[torch.randint(len(TRAIN_SIZES), (1,)).item()] if self.training else EVAL_SIZE
                 for _ in images]
        resized = [resize(img, s) for img, s in zip(images, sizes)]
        H = -(-max(r.shape[-2] for r, _ in resized) // 32) * 32
        W = -(-max(r.shape[-1] for r, _ in resized) // 32) * 32
        x = images[0].new_zeros(len(images), 3, H, W)
        for i, (r, _) in enumerate(resized):
            x[i, :, :r.shape[-2], :r.shape[-1]] = r
        scales = [s for _, s in resized]
        if targets is None:
            return x, scales
        labels = []
        for t, (sx, sy) in zip(targets, scales):
            b = t["boxes"] * torch.tensor([sx, sy, sx, sy], device=x.device) / torch.tensor([W, H, W, H], device=x.device)
            cxcywh = torch.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2,
                                  b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], 1)
            labels.append({"class_labels": t["labels"], "boxes": cxcywh})
        return x, labels

    def loss(self, images, targets):
        x, labels = self.batch(images, targets)
        return self.model(pixel_values=x, labels=labels).loss

    @torch.no_grad()
    def predict(self, images):
        """Top-300 (query, class) pairs by sigmoid score, as in RT-DETR's own post-processing."""
        x, scales = self.batch(images)
        out = self.model(pixel_values=x)
        model_ids = sorted(self.to_shared)
        probs = out.logits.sigmoid()[:, :, model_ids]  # [B, Q, kept classes]
        H, W = x.shape[-2:]
        preds = []
        for p, bx, (sx, sy) in zip(probs, out.pred_boxes, scales):
            scores, idx = p.flatten().topk(min(MAX_DET, p.numel()))
            q, c = idx // p.shape[1], idx % p.shape[1]
            cx, cy, bw, bh = bx[q].unbind(1)
            boxes = torch.stack([(cx - bw / 2) * W / sx, (cy - bh / 2) * H / sy,
                                 (cx + bw / 2) * W / sx, (cy + bh / 2) * H / sy], 1)
            keep = scores > 0.001
            labels = torch.tensor([self.to_shared[model_ids[k]] for k in c.tolist()], dtype=torch.long)
            preds.append({"boxes": boxes[keep], "scores": scores[keep], "labels": labels[keep.cpu()]})
        return preds


def build(arch, num_classes=None):
    """COCO-pretrained detector. num_classes=None keeps the COCO head (zero-shot);
    an int replaces the classification head with a new one of that size."""
    return {"frcnn": FasterRCNN, "rtdetr": RTDETR}[arch](num_classes)
