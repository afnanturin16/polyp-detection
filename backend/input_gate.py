"""
input_gate.py
=============
Input gate: refuses images that are clearly not H&E-stained tissue, before any model runs.

check_he_tissue(pil_image) -> {"verdict": "pass" | "warn" | "reject", "reasons": [str], "scores": {...}}

The gate only READS a downscaled copy of the image; it never alters what the models receive.

Optical density is computed here, standalone (OD = -log10((I+1)/256)).  It deliberately does NOT reuse
stain_norm_training.py (that file reproduces the training-time log10/exp quirk and must stay byte-identical).
Only the *directions* of the stain vectors in stain_reference.npz are used, and directions do not depend on
the log base.

Scores (all computed on a copy whose longer side is <= 512 px):
  tissue_fraction   share of pixels with OD norm > OD_TISSUE
  residual          mean norm of OD not explained by the two dominant stain directions / mean OD norm
  angle_h, angle_e  angle (deg) between each estimated stain direction (Macenko-style, from the SVD plane) and the
                    matching reference stain direction (H = the vector with the larger red OD)
  plane_angle       largest principal angle (deg) between the estimated 2-D stain plane and the reference plane
  hue_share         share of saturated pixels (S > 0.06) whose hue is in the pink-purple band (>= 250 or <= 25 deg),
                    same idea as the client-side check in app.js
  sat_fraction      share of pixels that are saturated (S > 0.06)
  min_side          shorter side of the ORIGINAL image in px (real patches are 1812 px; tiny images carry no usable tissue)
  border_band       widest band of exactly uniform rows/columns along any image edge, as a fraction of that side,
                    measured on the ORIGINAL pixels (downscaling would average scanner noise away).  Margins from a
                    screenshot or a crop with padding are exactly uniform; scanner background carries noise.
                    WARN-ONLY: it never rejects.
  dark_fraction     share of near-black pixels (max channel < 40).  Real patches contain essentially none (max 0.2%
                    over 666 patches); photos on black backgrounds, many screenshots and scans do.  This sixth score
                    was added after calibration showed the five requested ones left a few silent passes.

Verdict: reject only when a score is far outside anything seen in real patches; warn when it is just outside the
real range; otherwise pass.  THRESHOLDS were set from the distributions of 666 real UniToPatho patches (test + train,
all six classes) and checked against a non-tissue set; they are not guesses.
"""

from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
MAX_SIDE = 512
OD_TISSUE = 0.15
MIN_SIDE_REJECT = 64
UNIFORM_STD = 1.0        # a row/column counts as uniform if every channel's std along it is <= this
BORDER_SCAN = 0.25       # only the outer 25% of each side is scanned for a uniform band
SAT_MIN = 0.06

_ref = np.load(HERE / "stain_reference.npz")["stain_matrix"].astype(np.float64)   # (3, 2): columns H, E
_ref = _ref / np.linalg.norm(_ref, axis=0, keepdims=True)
_REF_H, _REF_E = _ref[:, 0], _ref[:, 1]
_REF_Q, _ = np.linalg.qr(_ref)                                                    # orthonormal basis of the reference plane


def _angle(a, b):
    return float(np.degrees(np.arccos(np.clip(abs(float(a @ b)) / (np.linalg.norm(a) * np.linalg.norm(b)), 0, 1))))


def _leading_uniform(lines: np.ndarray) -> int:
    """lines: (n, length, 3) rows (or columns) ordered from the edge inwards; count of leading uniform lines."""
    uniform = lines.std(axis=1).max(axis=1) <= UNIFORM_STD
    return int(np.argmin(uniform)) if not uniform.all() else len(uniform)


def border_band(rgb_u8: np.ndarray) -> float:
    """Widest exactly-uniform band along any of the four edges, as a fraction of the side it is measured on."""
    h, w = rgb_u8.shape[:2]
    nh, nw = max(1, int(h * BORDER_SCAN)), max(1, int(w * BORDER_SCAN))
    a = rgb_u8.astype(np.float32, copy=False)
    top = _leading_uniform(a[:nh]) / h
    bottom = _leading_uniform(a[::-1][:nh]) / h
    left = _leading_uniform(a[:, :nw].transpose(1, 0, 2)) / w
    right = _leading_uniform(a[:, ::-1][:, :nw].transpose(1, 0, 2)) / w
    return float(max(top, bottom, left, right))


def compute_scores(pil_image: Image.Image) -> dict:
    im = pil_image.convert("RGB")
    w, h = im.size
    min_side = min(w, h)
    band = border_band(np.asarray(im)) if min_side >= MIN_SIDE_REJECT else None
    if min_side < MIN_SIDE_REJECT:      # statistics on a handful of pixels are meaningless
        return {"tissue_fraction": None, "sat_fraction": None, "hue_share": None, "dark_fraction": None,
                "residual": None, "angle_h": None, "angle_e": None, "plane_angle": None,
                "min_side": min_side, "width": w, "height": h, "border_band": None}
    if max(w, h) > MAX_SIDE:
        s = MAX_SIDE / max(w, h)
        im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)

    rgb = np.asarray(im, dtype=np.float64).reshape(-1, 3)
    od = -np.log10((rgb + 1.0) / 256.0)
    norm = np.linalg.norm(od, axis=1)
    tissue = norm > OD_TISSUE

    # hue band (HSV via PIL: H, S, V are 0-255)
    hsv = np.asarray(im.convert("HSV"), dtype=np.float64).reshape(-1, 3)
    hue, sat = hsv[:, 0] * 360.0 / 255.0, hsv[:, 1] / 255.0
    saturated = sat > SAT_MIN
    sat_fraction = float(saturated.mean())
    hue_share = float(((hue >= 250) | (hue <= 25))[saturated].mean()) if saturated.any() else 0.0

    dark_fraction = float((rgb.max(axis=1) < 40).mean())

    scores = {"min_side": min_side, "width": w, "height": h, "border_band": band, "tissue_fraction": float(tissue.mean()), "sat_fraction": sat_fraction, "hue_share": hue_share,
              "dark_fraction": dark_fraction, "residual": None, "angle_h": None, "angle_e": None, "plane_angle": None}
    if tissue.sum() < 200:
        return scores

    x = od[tissue]
    _, _, vt = np.linalg.svd(x, full_matrices=False)       # uncentred: stain directions pass through the origin
    v = vt[:2].copy()                                       # 2 x 3, orthonormal rows
    if v[0].sum() < 0:                                      # first axis points into the positive OD octant
        v[0] = -v[0]
    proj = x @ v.T                                          # n x 2
    scores["residual"] = float(np.linalg.norm(x - proj @ v, axis=1).mean() / norm[tissue].mean())

    # Macenko-style extremes: angles within the plane, 1st / 99th percentile
    phi = np.arctan2(proj[:, 1], proj[:, 0])
    lo, hi = np.percentile(phi, [1, 99])
    cand = [np.cos(a) * v[0] + np.sin(a) * v[1] for a in (lo, hi)]
    cand = [c if c.sum() >= 0 else -c for c in cand]
    cand = [c / np.linalg.norm(c) for c in cand]
    hv, ev = (cand[0], cand[1]) if cand[0][0] > cand[1][0] else (cand[1], cand[0])    # H has the larger red OD
    scores["angle_h"], scores["angle_e"] = _angle(hv, _REF_H), _angle(ev, _REF_E)

    # principal angle between estimated plane and reference plane
    sv = np.linalg.svd(v @ _REF_Q, compute_uv=False)
    scores["plane_angle"] = float(np.degrees(np.arccos(np.clip(sv.min(), 0, 1))))
    return scores


# (reject_above / reject_below, warn_above / warn_below) per score
THRESHOLDS = {
    "min_side":        {"reject_below": 64, "warn_below": 224},       # real patches: 1812 px; other-lab tiles are often 224+
    "tissue_fraction": {"reject_below": 0.15, "warn_below": 0.25},    # real patches: min 0.35
    "hue_share":       {"reject_below": 0.80, "warn_below": 0.98},    # real patches: min 0.984
    "residual":        {"reject_above": 0.10, "warn_above": 0.075},   # real patches: max 0.066
    # Stain angles are WARN-ONLY: a different lab's stain legitimately shifts them, so they never reject on their own.
    "angle_h":         {"warn_above": 20.0},                          # real patches: max 16.6 deg
    "angle_e":         {"warn_above": 28.0},                          # real patches: max 24.5 deg
    "dark_fraction":   {"reject_above": 0.10, "warn_above": 0.02},    # real patches: max 0.002
    "border_band":     {"warn_at_least": 0.03},                       # warn-only; see calibration below
}

_TEXT = {
    "tissue_fraction": "very little stained tissue (mostly blank or bright background)",
    "hue_share": "the colours are not in the pink-purple range of H&E staining",
    "residual": "the colours cannot be explained by the two H&E stains (hematoxylin and eosin)",
    "angle_h": "the purple (hematoxylin) colour direction differs from H&E staining",
    "angle_e": "the pink (eosin) colour direction differs from H&E staining",
    "dark_fraction": "large near-black regions, which real tissue patches do not have",
    "border_band": "This image has uniform borders (maybe a screenshot). Crop to the tissue patch for a reliable result.",
}


def _grade(scores: dict):
    reject, warn = [], []
    for key, t in THRESHOLDS.items():
        v = scores.get(key)
        if v is None:
            continue
        if ("reject_below" in t and v < t["reject_below"]) or ("reject_above" in t and v > t["reject_above"]):
            reject.append(key)
        elif (("warn_below" in t and v < t["warn_below"]) or ("warn_above" in t and v > t["warn_above"])
              or ("warn_at_least" in t and v >= t["warn_at_least"])):
            warn.append(key)
    if "min_side" in reject:
        return ["min_side"], []
    if scores.get("residual") is None and "tissue_fraction" not in reject:
        reject.append("tissue_fraction")
    return reject, warn


def check_he_tissue(pil_image: Image.Image) -> dict:
    """Gate verdict for one image.  Reads a downscaled copy only; never modifies the image."""
    try:
        scores = compute_scores(pil_image)
        reject, warn = _grade(scores)
    except Exception as e:      # never let the gate itself take the app down
        return {"verdict": "warn", "reasons": [f"The input check could not be completed ({type(e).__name__})."],
                "scores": {}}
    verdict = "reject" if reject else "warn" if warn else "pass"
    reasons = []
    for k in (reject or warn):
        if k == "min_side":
            reasons.append(f"The image is very small ({scores['width']}×{scores['height']} px); the models were "
                           f"trained on patches of 1812×1812 px")
        elif k == "border_band":
            reasons.append(_TEXT[k])
        else:
            reasons.append(_TEXT[k][0].upper() + _TEXT[k][1:])
    return {"verdict": verdict, "reasons": reasons,
            "scores": {k: (None if v is None else round(float(v), 4)) for k, v in scores.items()}}
