"""Synthetic sandstorm generator (reusable deliverable).

Physics: the atmospheric scattering model used for haze and fog,

    I(x) = J(x) * t(x) + A * (1 - t(x)),    t(x) = exp(-beta * d(x))

  J     clear image             A     airlight: sand-coloured, bright yellow-orange
  t     transmission            beta  dust density (light / medium / heavy)
  d     relative scene depth    - from monocular depth (Depth Anything V2, src/depth.py):
                                  d = near_depth / disparity, capped at 1 (far field, sky);
                                  without a depth map: flat-ground-plane proxy d ~ 1 / (y - y_horizon)

On top of the scattering model, real sandstorm photos show:
  - a colour cast on near objects too (dust absorbs blue): J is tinted towards A,
  - patchy, non-uniform dust: beta is modulated by smooth low-frequency noise,
  - loss of fine detail with distance: small-angle forward scattering by sand grains blurs
    the far field; a spatially varying blur with sigma proportional to (1 - t) keeps near
    objects sharp,
  - fine particle noise: sensor-like Gaussian noise plus sparse bright specks,
  - blowing sand near the ground: long horizontal streaks (in half of the images).

Version 3 (Oct 15): an ablation with zero-shot RT-DETR on 300 BDD val images showed that v2's
uniform blur caused most of the detection loss (light dust: clear 0.471 -> 0.184 mAP50; without
blur 0.341; haze alone 0.386), and that the per-row depth hazed nearby vehicles above the horizon
row like sky. v3 uses monocular depth and a depth-dependent blur, and is calibrated on
per-object statistics against DAWN (--box-stats), not only whole-image statistics. (A first
attempt gave each labelled box the depth of its ground-contact row; that drew the box outlines
into the dust, leaking the labels, and was dropped.)

Severity calibration (Oct 16): each severity is calibrated against its own third of DAWN's
storm-intensity range (background structure and visible objects; see BETA_RANGE), so the
levels span light haze to the opaque storms of DAWN's most intense third.

Usage as a library:
    import numpy as np, dust, depth
    rng = np.random.default_rng(0)
    params = dust.sample_params(rng, severity="medium")   # a plain dict, JSON-serialisable
    disp = depth.load_disparity(img_path)                 # or depth.DepthModel().disparity(img_bgr); None = ground plane
    dusty = dust.add_dust(img_bgr, params, disp)          # deterministic given params and disp
    t = dust.object_visibility(img_bgr.shape, params, boxes, disp)  # per-object transmission (contrast left)

Command line (tuning against real DAWN few-shot *train* images only, never the test set):
    python src/dust.py --grid        # side-by-side grid -> results/figures/dust_vs_real.jpg
    python src/dust.py --stats       # contrast / haze / colour / sharpness -> results/dust_stats.json + figure
    python src/dust.py --box-stats   # per severity vs its DAWN intensity third -> results/dust_box_stats.json
"""
import argparse

import cv2
import numpy as np

SEVERITIES = ("light", "medium", "heavy")
# Severity calibration (Oct 16; --box-stats): the 79 DAWN few-shot train images are ranked by
# storm intensity (background_stats: structure left in the upper half of the image, which avoids
# the dashcam hood) and split into thirds. Medium is matched to DAWN's middle third and heavy to
# its most intense third, on both the background (local contrast 0.027 vs 0.024; 0.013 vs 0.007)
# and the visible objects (contrast 0.092 vs 0.086, 0.087 vs 0.080; sharpness -0.94 vs -0.95,
# -0.87 vs -0.81). Light is a mild haze, slightly lighter than DAWN's lightest third (objects
# 0.120 / -0.65 vs 0.112 / -0.57), so the three levels span light haze to opaque storm.
# v3's single calibration (beta 0.24-1.40, far field capped at depth 1) matched the objects
# on average, but its heavy level had the background of DAWN's *lightest* third: distant
# buildings and sky stayed visible. Denser dust hides more of BDD's many small, distant objects
# (light 17%, medium 72%, heavy 88%); visibility-aware scoring evaluates what is left.
BETA_RANGE = {"light": (0.30, 0.60), "medium": (1.00, 1.60), "heavy": (2.00, 3.00)}
# Depth cap for the far field and sky, per severity: denser storms hide the far field.
FAR_DEPTH = {"light": 2.5, "medium": 2.0, "heavy": 3.0}
# Max blur sigma (px at 640 px width, at zero transmission; scaled by 1 - t).
BLUR_RANGE = {"light": (0.3, 1.8), "medium": (0.8, 3.0), "heavy": (1.2, 4.0)}
# Airlight colour, measured on the 79 DAWN few-shot *train* images (dark-channel-prior
# estimate: median colour of the brightest 0.1% dark-channel pixels, clipped pixels
# excluded). Real dust colour lies on one axis from saturated orange to grey-beige:
#   B/R in [0.27, 1.03] (5th-95th pct),  G/R = 0.47 * B/R + 0.55  (corr 0.96),  R in [174, 248]
# Sampling R, G, B independently (the original plan) gives unrealistic green/pink casts.
AIRLIGHT_BR = (0.3, 1.0)
AIRLIGHT_R = (175, 245)
AIRLIGHT_GR_FIT = (0.47, 0.55)


def sample_params(rng, severity=None):
    """Random dust parameters. severity: 'light' | 'medium' | 'heavy' | None (random)."""
    severity = severity or SEVERITIES[rng.integers(len(SEVERITIES))]
    level = SEVERITIES.index(severity)  # 0, 1, 2
    br = rng.uniform(*AIRLIGHT_BR)
    gr = AIRLIGHT_GR_FIT[0] * br + AIRLIGHT_GR_FIT[1] + rng.normal(0, 0.02)
    r = rng.uniform(*AIRLIGHT_R)
    return {
        "severity": severity,
        "beta": float(rng.uniform(*BETA_RANGE[severity])),
        "far_depth": FAR_DEPTH[severity],
        "airlight_rgb": [float(r), float(min(255, r * gr)), float(min(255, r * br))],
        "horizon": float(rng.uniform(0.35, 0.55)),         # horizon row, fraction of height (drift; ground-plane fallback)
        "near_depth": float(rng.uniform(0.05, 0.15)),      # depth of the nearest scene point (far field = 1)
        "patchiness": float(rng.uniform(0.1, 0.4)),         # amplitude of beta modulation
        # blowing sand near the ground: off in half of the images, stronger when denser
        "drift": float(rng.uniform(0.1, 0.25) * (1 + level) / 2) if rng.random() < 0.5 else 0.0,
        "tint": float(rng.uniform(0.15, 0.35) + 0.1 * level),  # colour cast on the clear image
        # px at 640 px width, at zero transmission; scaled by (1 - t), so near objects stay sharp
        "blur_sigma": float(rng.uniform(*BLUR_RANGE[severity])),
        "noise_sigma": float(rng.uniform(2, 5) + 1.5 * level),     # gray levels
        "speck_density": float(rng.uniform(0, 0.0015) * (1 + level)),  # fraction of pixels
        "seed": int(rng.integers(2 ** 31)),                 # for the noise fields
    }


def depth_proxy(h, w, p):
    """Relative depth in [near_depth, 1] from a flat-ground-plane camera model.

    For a camera above a flat road, the distance to the ground seen at image row y is
    proportional to 1 / (y - y_horizon). Scaled so the bottom row has depth near_depth;
    capped at 1 near and above the horizon (sky, far buildings). Compared with a linear
    ramp, near objects stay clear and the dust thickens quickly towards the horizon.
    """
    y = np.arange(h, dtype=np.float32)[:, None] / max(h - 1, 1)
    hz = p["horizon"]
    k = p["near_depth"] * (1 - hz)
    d = np.minimum(1.0, k / np.maximum(y - hz, 1e-6))
    return np.broadcast_to(d, (h, w)).astype(np.float32)


def ground_drift(h, w, p, rng, scale):
    """Horizontal streaks of blowing sand, strongest at the bottom of the image, in [0, 1]."""
    n = rng.random((max(2, h // 4), max(2, w // 4))).astype(np.float32)
    n = cv2.resize(n, (w, h), interpolation=cv2.INTER_LINEAR)
    n = cv2.GaussianBlur(n, (0, 0), sigmaX=30 * scale, sigmaY=1.5 * scale)  # long, thin streaks
    n = np.clip((n - n.mean()) / (n.std() + 1e-6) * 0.5 + 0.5, 0, 1) ** 2
    y = np.arange(h, dtype=np.float32)[:, None] / max(h - 1, 1)
    weight = np.clip((y - p["horizon"]) / (1 - p["horizon"]), 0, 1) ** 0.7
    return n * weight


def smooth_noise(h, w, rng, cells=6):
    """Smooth random field in [-1, 1] (bicubic-upsampled coarse noise)."""
    coarse = rng.uniform(-1, 1, (cells, max(2, round(cells * w / h)))).astype(np.float32)
    field = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
    return np.clip(field, -1, 1)


def scene_depth(h, w, p, disparity=None):
    """Relative depth in [near_depth, 1].

    disparity: normalised monocular disparity (far 0 .. near 1, any resolution; src/depth.py).
    Depth is inverse disparity, scaled so the nearest point has near_depth and capped at 1.
    Without it, the flat-ground-plane proxy (one depth per image row) is used.
    """
    if disparity is None:
        return depth_proxy(h, w, p)
    disp = cv2.resize(disparity, (w, h), interpolation=cv2.INTER_LINEAR)
    far = p.get("far_depth", 1.0)  # v3 parameter files have no far_depth: it was fixed at 1
    return np.minimum(far, p["near_depth"] / np.maximum(disp, 1e-3)).astype(np.float32)


def _transmission(h, w, p, disparity, rng):
    """t = exp(-beta * d), with beta modulated by smooth noise (patchy dust). Uses rng first."""
    beta = p["beta"] * (1 + p["patchiness"] * smooth_noise(h, w, rng))
    return np.exp(-beta * scene_depth(h, w, p, disparity))


def object_visibility(shape, params, boxes, disparity=None):
    """Median transmission inside each box: the fraction of the object's contrast left.

    Uses the same dust field as add_dust(img, params, disparity) for the same arguments.
    The labels are only read here, to report visibility; they never shape the dust.
    """
    h, w = shape[:2]
    t = _transmission(h, w, params, disparity, np.random.default_rng(params["seed"]))
    out = []
    for x1, y1, x2, y2 in np.asarray(boxes, np.float64).reshape(-1, 4):
        y1i, x1i = int(np.clip(y1, 0, h - 1)), int(np.clip(x1, 0, w - 1))
        out.append(float(np.median(t[y1i:max(y1i + 1, int(np.ceil(y2))), x1i:max(x1i + 1, int(np.ceil(x2)))])))
    return out


def depth_blur(img, sigma, n_levels=4):
    """Spatially varying Gaussian blur: per-pixel sigma, interpolated between n_levels blurs.

    A true blur, not a blend of sharp and blurred copies: blending keeps the sharp edges at
    full strength times the weight, so object sharpness barely drops (measured with --box-stats).
    """
    top = float(sigma.max())
    if top < 0.3:
        return img
    levels = np.linspace(0, top, n_levels)
    stack = [img] + [cv2.GaussianBlur(img, (0, 0), s) for s in levels[1:]]
    pos = sigma / top * (n_levels - 1)
    i0 = np.minimum(pos.astype(np.int32), n_levels - 2)
    f = (pos - i0)[..., None]
    out = np.zeros_like(img)
    for k in range(n_levels - 1):
        m = (i0 == k)[..., None]
        out += m * (stack[k] * (1 - f) + stack[k + 1] * f)
    return out


def add_dust(img_bgr, params, disparity=None):
    """Apply synthetic sandstorm to a uint8 BGR image; deterministic given params and disparity.

    disparity: normalised monocular disparity from src/depth.py (None: ground-plane depth).
    """
    p = params
    rng = np.random.default_rng(p["seed"])
    h, w = img_bgr.shape[:2]
    scale = w / 640.0
    J = img_bgr.astype(np.float32) / 255.0
    A = np.array(p["airlight_rgb"][::-1], np.float32) / 255.0  # RGB -> BGR

    # Colour cast on the clear image (dust absorbs blue even close to the camera).
    J = J * (1 - p["tint"]) + J * (A / A.max()) * p["tint"]

    # Patchy dust: modulate beta with smooth noise.
    t = _transmission(h, w, p, disparity, rng)[..., None]
    I = J * t + A * (1 - t)

    # Blowing sand close to the ground.
    if p.get("drift", 0) > 0:
        a = (p["drift"] * ground_drift(h, w, p, rng, scale))[..., None]
        I = I * (1 - a) + A * a

    # Loss of fine detail, growing with optical depth: near objects stay sharp.
    if p["blur_sigma"] > 0:
        I = depth_blur(I, p["blur_sigma"] * scale * (1 - t[..., 0]))

    # Fine particle noise: Gaussian noise plus sparse bright specks.
    I = I + rng.normal(0, p["noise_sigma"] / 255.0, I.shape).astype(np.float32)
    n_specks = int(p["speck_density"] * h * w)
    if n_specks:
        ys, xs = rng.integers(0, h, n_specks), rng.integers(0, w, n_specks)
        I[ys, xs] = I[ys, xs] * 0.3 + A * 0.7 + 0.1
    return (np.clip(I, 0, 1) * 255).round().astype(np.uint8)


def _grid(n_rows=4, seed=0, out="results/figures/dust_vs_real.jpg"):
    """Rows: clear BDD | synthetic light | medium | heavy || real DAWN light | medium | intense third.

    Real images are DAWN few-shot *train* images only, one from each storm-intensity third.
    """
    import random

    from common import ROOT
    from depth import load_disparity
    rng = np.random.default_rng(seed)
    pick = random.Random(seed)
    clear = pick.sample(sorted((ROOT / "data/processed/bdd_clear/images/train").glob("*.jpg")), n_rows)
    thirds = storm_thirds(sorted((ROOT / "data/processed/dawn_sand/images/fewshot_train").glob("*.jpg")))
    real = {s: pick.sample(sorted(p for p, t in thirds.items() if t == s), n_rows) for s in SEVERITIES}
    tw, th = 384, 216

    def tile(img, label):
        img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
        cv2.putText(img, label, (5, 18), 0, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return img

    rows = []
    for i, c in enumerate(clear):
        img = cv2.imread(str(c))
        disp = load_disparity(c)
        tiles = [tile(img.copy(), "clear (BDD)")]
        for sev in SEVERITIES:
            p = sample_params(rng, sev)
            tiles.append(tile(add_dust(img, p, disp), f"{sev} b={p['beta']:.1f}"))
        sep = np.full((th, 8, 3), 255, np.uint8)
        tiles += [sep] + [tile(cv2.imread(str(real[s][i])), f"DAWN {s} third") for s in SEVERITIES]
        rows.append(np.hstack(tiles))
    out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(out)


def image_stats(img_bgr):
    """Global statistics that separate clear from dusty images (computed at 640 px width)."""
    h, w = img_bgr.shape[:2]
    img = cv2.resize(img_bgr, (640, round(640 * h / w)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    dark = cv2.erode(img.min(2), np.ones((15, 15), np.uint8)).astype(np.float32)
    b, _, r = img.reshape(-1, 3).mean(0)
    return {
        "contrast": float(gray.std() / 255),                       # RMS contrast
        "dark_channel": float(dark.mean() / 255),                  # haze density (He et al.)
        "blue_red": float(b / max(r, 1)),                          # colour cast
        "sharpness": float(np.log10(cv2.Laplacian(gray, cv2.CV_32F).var() + 1)),  # fine detail
    }


def _stats(n_clear=150, seed=1):
    """Compare clear BDD, the same images with synthetic dust, and real DAWN few-shot train."""
    import json
    import random

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from common import ROOT
    from depth import load_disparity
    rng = np.random.default_rng(seed)
    clear = random.Random(seed).sample(sorted((ROOT / "data/processed/bdd_clear/images/train").glob("*.jpg")), n_clear)
    real = sorted((ROOT / "data/processed/dawn_sand/images/fewshot_train").glob("*.jpg"))
    groups = {"clear (BDD)": [], "synthetic dust": [], "real sandstorm (DAWN)": []}
    for p in clear:
        img = cv2.imread(str(p))
        groups["clear (BDD)"].append(image_stats(img))
        groups["synthetic dust"].append(image_stats(add_dust(img, sample_params(rng), load_disparity(p))))
    groups["real sandstorm (DAWN)"] = [image_stats(cv2.imread(str(p))) for p in real]

    metrics = {"contrast": "RMS contrast", "dark_channel": "Dark-channel mean (haze)",
               "blue_red": "Blue / red ratio (colour cast)", "sharpness": "log10 Laplacian variance (detail)"}
    summary = {g: {m: dict(zip(("p25", "median", "p75"), np.percentile([s[m] for s in v], [25, 50, 75]).round(3).tolist()))
                   for m in metrics} | {"n": len(v)} for g, v in groups.items()}
    (ROOT / "results" / "dust_stats.json").write_text(json.dumps(summary, indent=1))

    # Categorical slots 1-3 (validated: CVD dE 9.2, normal-vision dE 27.6); legend + JSON table as relief.
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                         "xtick.color": "#52514e", "ytick.color": "#52514e", "text.color": "#0b0b0b"})
    fig, axes = plt.subplots(1, 4, figsize=(13, 3), facecolor="#fcfcfb")
    for ax, (m, label) in zip(axes, metrics.items()):
        ax.set_facecolor("#fcfcfb")
        vals = {g: [s[m] for s in v] for g, v in groups.items()}
        allv = np.concatenate(list(vals.values()))
        bins = np.linspace(*np.percentile(allv, [0.5, 99.5]), 25)  # clip rare outliers
        for c, (g, v) in zip(colors, vals.items()):
            ax.hist(v, bins=bins, density=True, histtype="step", linewidth=2, color=c, label=g)
        ax.set_title(label, fontsize=9, loc="left")
        ax.set_yticks([])
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(f"Synthetic vs real sandstorm image statistics (clear: BDD n={n_clear}; synthetic: the same "
                 f"images + dust; real: DAWN few-shot train n={len(real)})", x=0.01, y=0.98, ha="left", fontsize=10)
    fig.legend(handles, labels, frameon=False, loc="upper left", bbox_to_anchor=(0.005, 0.9), ncol=3, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.8))
    out = ROOT / "results" / "figures" / "dust_stats.png"
    fig.savefig(out, dpi=130, facecolor=fig.get_facecolor())
    print(json.dumps(summary, indent=1))
    print(out)


VISIBLE_T = 0.3  # objects with less contrast left than this count as hidden by the dust
BOX_H_RANGE = (32, 128)  # object height in px at 640 px image width, for the per-object statistics


def box_stats(img_bgr, boxes):
    """Per-object contrast and scale- and contrast-normalised sharpness, for objects in BOX_H_RANGE.

    Whole-image sharpness is confounded by object size and by web-photo compression (DAWN);
    comparing objects of similar size, each resized to 64 px height, removes most of that.
    """
    h, w = img_bgr.shape[:2]
    s = 640 / w
    img = cv2.resize(img_bgr, (640, round(h * s)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    out = []
    for x1, y1, x2, y2 in np.asarray(boxes, np.float64).reshape(-1, 4) * s:
        bh, bw = y2 - y1, x2 - x1
        if not BOX_H_RANGE[0] <= bh <= BOX_H_RANGE[1] or bw < 8:
            out.append(None)
            continue
        crop = gray[int(y1):int(np.ceil(y2)), int(x1):int(np.ceil(x2))]
        crop = cv2.resize(crop, (max(8, round(64 * bw / bh)), 64), interpolation=cv2.INTER_LINEAR)
        std = crop.std()
        # Laplacian variance grows with contrast^2; dividing by the crop's own std separates
        # edge sharpness from the contrast loss that "contrast" already measures.
        out.append({"contrast": float(std / 255),
                    "sharpness": float(np.log10(cv2.Laplacian(crop / max(std, 1.0), cv2.CV_32F).var() + 1e-6))})
    return out


def background_stats(img_bgr):
    """Storm intensity from the upper half of the image (background; avoids the dashcam hood).

    local_contrast: mean local std / mean brightness at 160 px width, where sensor noise and
    JPEG artefacts average out: how much scene structure the dust leaves (opaque storm -> 0).
    """
    h, w = img_bgr.shape[:2]
    img = cv2.resize(img_bgr, (640, round(640 * h / w)), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(img[: img.shape[0] // 2], cv2.COLOR_BGR2GRAY).astype(np.float32)
    gs = cv2.resize(g, (160, max(2, round(g.shape[0] / 4))), interpolation=cv2.INTER_AREA)
    m = cv2.blur(gs, (5, 5))
    local_std = np.sqrt(np.maximum(cv2.blur(gs * gs, (5, 5)) - m * m, 0))
    return {"local_contrast": float(local_std.mean() / max(g.mean(), 1)), "rms_contrast": float(g.std() / 255)}


def storm_thirds(paths):
    """{path: 'light' | 'medium' | 'heavy'}: real images split into thirds of storm intensity.

    Intensity = mean rank of low background local contrast and low background RMS contrast.
    """
    from scipy.stats import rankdata
    st = [background_stats(cv2.imread(str(p))) for p in paths]
    score = (rankdata([-s["local_contrast"] for s in st]) + rankdata([-s["rms_contrast"] for s in st])) / 2
    order = np.argsort(score, kind="stable")
    return {paths[i]: SEVERITIES[min(2, k * 3 // len(paths))] for k, i in enumerate(order)}


def _box_stats(n_clear=300, seed=1):
    """Per severity: synthetic dust on BDD vs the matching third of DAWN few-shot train.

    Background structure (background_stats) plus per-object contrast and sharpness. Synthetic
    objects hidden by the dust (transmission < VISIBLE_T) are left out of the object statistics,
    because DAWN annotators could only label objects they could see.
    """
    import json
    import random

    from common import ROOT, read_labels
    from depth import load_disparity
    rng = np.random.default_rng(seed)
    clear = random.Random(seed).sample(sorted((ROOT / "data/processed/bdd_clear/images/train").glob("*.jpg")), n_clear)
    real = sorted((ROOT / "data/processed/dawn_sand/images/fewshot_train").glob("*.jpg"))

    def new():
        return {"background": [], "objects": []}
    groups = {"clear (BDD)": new(), **{f"synthetic {s}": new() for s in SEVERITIES},
              **{f"DAWN {s} third": new() for s in SEVERITIES}, "DAWN all": new()}
    hidden = {s: [0, 0] for s in SEVERITIES}
    for p in clear:
        img = cv2.imread(str(p))
        boxes = read_labels(p, img.shape[1], img.shape[0])[0]
        disp = load_disparity(p)
        groups["clear (BDD)"]["background"].append(background_stats(img))
        groups["clear (BDD)"]["objects"] += box_stats(img, boxes)
        for sev in SEVERITIES:
            prm = sample_params(rng, sev)
            vis = object_visibility(img.shape, prm, boxes, disp)
            dusty = add_dust(img, prm, disp)
            hidden[sev][0] += sum(t < VISIBLE_T for t in vis)
            hidden[sev][1] += len(vis)
            groups[f"synthetic {sev}"]["background"].append(background_stats(dusty))
            groups[f"synthetic {sev}"]["objects"] += [st for st, t in zip(box_stats(dusty, boxes), vis) if t >= VISIBLE_T]
    thirds = storm_thirds(real)
    for p in real:
        img = cv2.imread(str(p))
        bg, objs = background_stats(img), box_stats(img, read_labels(p, img.shape[1], img.shape[0])[0])
        for g in (f"DAWN {thirds[p]} third", "DAWN all"):
            groups[g]["background"].append(bg)
            groups[g]["objects"] += objs

    def q(v):
        return dict(zip(("p25", "median", "p75"), np.percentile(v, [25, 50, 75]).round(3).tolist()))
    summary = {}
    for g, d in groups.items():
        objs = [x for x in d["objects"] if x is not None]
        summary[g] = {**{f"background_{m}": q([b[m] for b in d["background"]]) for m in ("local_contrast", "rms_contrast")},
                      **{f"object_{m}": q([x[m] for x in objs]) for m in ("contrast", "sharpness")},
                      "n_images": len(d["background"]), "n_objects": len(objs)}
    summary["hidden_fraction"] = {s: round(a / max(b, 1), 3) for s, (a, b) in hidden.items()}
    summary["dawn_thirds"] = {p.stem: t for p, t in thirds.items()}
    summary["settings"] = {"visible_t": VISIBLE_T, "box_h_range_px_at_640w": BOX_H_RANGE, "beta_range": BETA_RANGE,
                           "far_depth": FAR_DEPTH, "blur_range": BLUR_RANGE}
    (ROOT / "results" / "dust_box_stats.json").write_text(json.dumps(summary, indent=1))
    print(f"{'':22s} {'bg local contrast':>18s} {'bg RMS':>7s} {'obj contrast':>13s} {'obj sharpness':>14s}")
    for g in groups:
        s = summary[g]
        print(f"{g:22s} {s['background_local_contrast']['median']:18.3f} {s['background_rms_contrast']['median']:7.3f} "
              f"{s['object_contrast']['median']:13.3f} {s['object_sharpness']['median']:14.2f}  "
              f"(n={s['n_images']} images, {s['n_objects']} objects)")
    print("hidden fraction (all object sizes):", summary["hidden_fraction"])
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", action="store_true", help="write results/figures/dust_vs_real.jpg")
    ap.add_argument("--stats", action="store_true", help="synthetic vs real image statistics -> results/")
    ap.add_argument("--box-stats", action="store_true", help="per severity vs its DAWN intensity third -> results/")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.grid:
        _grid(seed=args.seed)
    if args.stats:
        _stats()
    if args.box_stats:
        _box_stats()
