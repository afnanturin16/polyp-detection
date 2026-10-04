"""
inference.py
============
Real model inference for the web app (extracted from the Gradio demo, demo/app.py, which was
verified against the recorded test-set predictions).  main.py and the frontend read the dict shape documented below.

Return shape of run_cascade() / run_caslite():
    pred_class    str    one of CLASSES; the original cascade routing (type head first; only ADENOMA reads shape+grade)
    confidence    float  routing confidence: max P(type) for HP/NORM, else P(ADENOMA)*max P(shape)*max P(grade)
    probs         dict   DERIVED secondary value over CLASSES (4 dp): P(HP), P(NORM), and
                         P(ADENOMA)*P(shape)*P(grade) for each adenoma class.  Its argmax is NOT used for pred_class.
    stages        list   3 dicts {name, ran, pred, conf}: Type / Architecture / Grade
    latency_ms    float  classifier time (stain-normalise + resize + forward passes); excludes overlays
    gland_mask    PIL    GlaS U-Net gland overlay (RAW pixels)
    attention_map PIL|None  Grad-CAM on the cascade type stage; None for CasLite
    banner        str|None  high-grade limitation notice, only when pred_class is TA.HG / TVA.HG

Classes, fixed order (the app's class order is never changed; each model's own label order is mapped BY NAME):
    0 NORM   1 HP   2 TA.LG   3 TA.HG   4 TVA.LG   5 TVA.HG

All checkpoints are loaded ONCE at import.  Preprocessing mirrors training exactly: Macenko stain normalisation at
native resolution (training-time normaliser, verbatim) -> Resize(stage size) -> ImageNet normalise.  The U-Net gets
RAW pixels + ImageNet normalisation only.  Nothing under 'UniToPatho 800/' is written.
"""

import base64
import io
import sys
import threading
import time
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from stain_norm_training import MacenkoNormalizer   # verbatim copy of UniToPatho 800/src/preprocessing/stain_norm.py
from polypnet_model import PolypNet, TYPE_CLASSES as LITE_TYPES, SHAPE_CLASSES as LITE_SHAPES, GRADE_CLASSES as LITE_GRADES

CLASSES = ["NORM", "HP", "TA.LG", "TA.HG", "TVA.LG", "TVA.HG"]

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODELS = HERE.parent / 'models'      # all checkpoints live inside webapp_v2 and load by relative path
CKPT = {
    'type':  MODELS / 'best_type.pt',           # cascade stage 1: HP / NORM / ADENOMA  (512 px)
    'shape': MODELS / 'best_ta_tva.pt',         # cascade stage 2: TA / TVA             (512 px)
    'grade': MODELS / 'best_grade.pt',          # cascade stage 3: HG / LG              (896 px)
    'lite':  MODELS / 'best_polypnet.pt',       # CasLite / PolypNet                    (896 px)
    'unet':  MODELS / 'glas_unet_r34_best.pt',  # GlaS gland-segmentation U-Net
}
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
EXPECTED_SIZE = {'type': 512, 'shape': 512, 'grade': 896, 'lite': 896}
SEG_MAX_SIDE = 2304

macenko = MacenkoNormalizer.load(HERE / 'stain_reference.npz')
_normalize = A.Compose([A.Normalize(mean=MEAN, std=STD)])

# Grad-CAM hooks the shared type model, so one request runs at a time.
_LOCK = threading.Lock()


def decode_image(b64_string: str) -> Image.Image:
    """Base64 data-URL (or raw base64) -> PIL Image. Used by main.py."""
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    raw = base64.b64decode(b64_string)
    img = Image.open(io.BytesIO(raw))
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        # Transparent pixels would otherwise become black; show them on white, as a browser does.
        # Fully opaque images (all real patches) come out pixel-identical.
        rgba = img.convert("RGBA")
        img = Image.alpha_composite(Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba)
    return img.convert("RGB")


def encode_image(img: Image.Image) -> str:
    """PIL Image -> base64 data URL, for sending overlays back to the browser."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------- preprocessing (same as the Gradio demo)

def stain_normalize(img_np):
    """Training-time StainNormalize at native resolution; too little tissue -> image left unchanged."""
    try:
        return macenko.transform(img_np), True
    except ValueError:
        return img_np, False


def classifier_tensor(stain_img_np, size):
    resized = A.Compose([A.Resize(size, size)])(image=stain_img_np)['image']
    x = _normalize(image=resized)['image']
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).unsqueeze(0).to(DEVICE)


def raw_tensor(img_np):
    x = (img_np.astype(np.float32) / 255.0 - np.array(MEAN, np.float32)) / np.array(STD, np.float32)
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).unsqueeze(0).to(DEVICE)


# ---------------------------------------------------------------- model loading (once, at import)

def _ckpt(path):
    return torch.load(path, map_location=DEVICE, weights_only=True)


def load_resnet(key):
    ck = _ckpt(CKPT[key])
    assert ck['backbone'] == 'resnet18', ck['backbone']
    size = int(ck['image_size'])
    assert size == EXPECTED_SIZE[key], f"{key}: checkpoint image_size {size} != expected {EXPECTED_SIZE[key]}"
    m = tvm.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, len(ck['classes']))
    m.load_state_dict(ck['model_state'])
    for p in m.parameters():
        p.requires_grad_(False)
    return m.eval().to(DEVICE), list(ck['classes']), size


print('loading models...')
M_TYPE, TYPE_NAMES, SIZE_TYPE = load_resnet('type')       # checkpoint label order, mapped by name below
M_SHAPE, SHAPE_NAMES, SIZE_SHAPE = load_resnet('shape')
M_GRADE, GRADE_NAMES, SIZE_GRADE = load_resnet('grade')

_ck = _ckpt(CKPT['lite'])
assert int(_ck['image_size']) == EXPECTED_SIZE['lite']
SIZE_LITE = int(_ck['image_size'])
M_LITE = PolypNet(pretrained=False)
M_LITE.load_state_dict(_ck['model_state'])
M_LITE = M_LITE.eval().to(DEVICE)

import segmentation_models_pytorch as smp
M_UNET = smp.Unet('resnet34', encoder_weights=None, in_channels=3, classes=1, activation=None)
M_UNET.load_state_dict(_ckpt(CKPT['unet'])['model_state'])
M_UNET = M_UNET.eval().to(DEVICE)
print('done. device:', DEVICE, '| sizes: type', SIZE_TYPE, 'shape', SIZE_SHAPE, 'grade', SIZE_GRADE, 'lite', SIZE_LITE)


# ---------------------------------------------------------------- Grad-CAM / segmentation overlays

def gradcam(model, x, target_layer, out_size):
    feats, grads = {}, {}

    def fwd(_, __, out):
        feats['v'] = out
        out.register_hook(lambda g: grads.__setitem__('v', g))

    h = target_layer.register_forward_hook(fwd)
    with torch.enable_grad():
        xx = x.clone().requires_grad_(True)
        out = model(xx)
        cls = int(out.argmax(1))
        model.zero_grad(set_to_none=True)
        out[0, cls].backward()
    h.remove()

    w = grads['v'].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((w * feats['v']).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=(out_size, out_size), mode='bilinear', align_corners=False)
    cam = cam[0, 0].detach().cpu().numpy()
    return (cam - cam.min()) / (float(cam.max() - cam.min()) + 1e-8)


def overlay(img: Image.Image, heat: np.ndarray, color=(220, 60, 40), size=None):
    size = size or heat.shape[0]
    base = np.array(img.convert('RGB').resize((size, size))).astype(float)
    tint = np.zeros_like(base)
    tint[..., 0], tint[..., 1], tint[..., 2] = color
    a = np.clip(heat, 0, 1)[..., None] * 0.45
    return Image.fromarray((base * (1 - a) + tint * a).astype(np.uint8))


@torch.no_grad()
def seg_prob(img_np):
    h, w = img_np.shape[:2]
    work = img_np
    if max(h, w) > SEG_MAX_SIDE:
        s = SEG_MAX_SIDE / max(h, w)
        work = cv2.resize(img_np, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
    x = raw_tensor(work)
    ph, pw = (-x.shape[2]) % 32, (-x.shape[3]) % 32
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode='reflect')
    p = torch.sigmoid(M_UNET(x))[0, 0, :work.shape[0], :work.shape[1]].float().cpu().numpy()
    if work.shape[:2] != (h, w):
        p = cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR)
    return p


def seg_overlay(img: Image.Image, prob: np.ndarray, max_side=1024):
    rgb = img.convert('RGB')
    if max(rgb.size) > max_side:
        s = max_side / max(rgb.size)
        rgb = rgb.resize((int(rgb.width * s), int(rgb.height * s)), Image.BILINEAR)
    m = cv2.resize((prob > 0.5).astype(np.uint8), rgb.size, interpolation=cv2.INTER_NEAREST) > 0
    base = np.array(rgb).astype(np.float32)
    base[m] = 0.62 * base[m] + 0.38 * np.array((30, 158, 117), np.float32)
    out = base.astype(np.uint8)
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cs, -1, (255, 235, 60), 2)
    return Image.fromarray(out)


# ---------------------------------------------------------------- routing + response assembly

def _route(p_type, type_names, p_shape, shape_names, p_grade, grade_names):
    """Original Gradio/cascade routing: read type head; only ADENOMA reads shape + grade. Names, not indices."""
    tname = type_names[int(p_type.argmax())]
    if tname != 'ADENOMA':
        return tname, float(p_type.max())
    shape = shape_names[int(p_shape.argmax())]
    grade = grade_names[int(p_grade.argmax())]
    conf = float(p_type[type_names.index('ADENOMA')] * p_shape.max() * p_grade.max())
    return f'{shape}.{grade}', conf


def _derive_probs(p_type, type_names, p_shape, shape_names, p_grade, grade_names):
    """Secondary value only: joint probability per app class, looked up BY NAME."""
    pt = {n: float(p) for n, p in zip(type_names, p_type)}
    ps = {n: float(p) for n, p in zip(shape_names, p_shape)}
    pg = {n: float(p) for n, p in zip(grade_names, p_grade)}
    probs = {"HP": pt["HP"], "NORM": pt["NORM"]}
    for shape in ("TA", "TVA"):
        for grade in ("LG", "HG"):
            probs[f"{shape}.{grade}"] = pt["ADENOMA"] * ps[shape] * pg[grade]
    return {c: round(probs[c], 4) for c in CLASSES}


def _stages(p_type, type_names, p_shape, shape_names, p_grade, grade_names):
    is_adenoma = type_names[int(p_type.argmax())] == 'ADENOMA'
    return [
        {"name": "Type", "ran": True, "pred": type_names[int(p_type.argmax())], "conf": round(float(p_type.max()), 3)},
        {"name": "Architecture", "ran": is_adenoma,
         "pred": shape_names[int(p_shape.argmax())] if is_adenoma else None,
         "conf": round(float(p_shape.max()), 3) if is_adenoma else None},
        {"name": "Grade", "ran": is_adenoma,
         "pred": grade_names[int(p_grade.argmax())] if is_adenoma else None,
         "conf": round(float(p_grade.max()), 3) if is_adenoma else None},
    ]


# Approved banner texts, verbatim. **bold** / *italic* markers are rendered by app.js.
BANNER_CASCADE = (
    "**Known limitation: high-grade dysplasia.** Distinguishing high-grade from low-grade dysplasia is this system's "
    "weakest task. In the full cascade, per-class F1 on the six-class test set was **0.142** for TA.HG and **0.480** "
    "for TVA.HG, against **0.899** for HP; the grading stage on its own reached **0.675** balanced accuracy. Of the 572 "
    "high-grade test patches, the cascade misclassified 370, and 251 of those (68%) were called low-grade of the same "
    "architecture (for example, 211 of 451 TVA.HG patches were called TVA.LG). Treat any high-grade prediction as "
    "unreliable. Research use only, not a diagnostic tool. *(2,399 test patches; single training run.)*"
)
BANNER_CASLITE = (
    "**Known limitation: high-grade dysplasia.** Distinguishing high-grade from low-grade dysplasia is this system's "
    "weakest task. In the lightweight CasLite model, per-class F1 on the six-class test set was **0.012** for TA.HG "
    "and **0.094** for TVA.HG, against **0.926** for HP. Of the 572 high-grade test patches, CasLite misclassified "
    "548, and 427 of those (78%) were called low-grade of the same architecture (for example, 364 of 451 TVA.HG "
    "patches were called TVA.LG). Treat any high-grade prediction as unreliable. Research use only, not a diagnostic "
    "tool. *(2,399 test patches; single training run.)*"
)


def _run(image: Image.Image, lite: bool) -> dict:
    arr = np.array(image.convert('RGB'))
    with _LOCK:
        t0 = time.time()
        stain_arr, _ = stain_normalize(arr)           # ONCE at native resolution, then resize per stage
        with torch.no_grad():
            if lite:
                t_out, s_out, g_out = M_LITE(classifier_tensor(stain_arr, SIZE_LITE))
                names = (LITE_TYPES, LITE_SHAPES, LITE_GRADES)
                x_cam = None
            else:
                x_type = classifier_tensor(stain_arr, SIZE_TYPE)
                x_shape = x_type if SIZE_SHAPE == SIZE_TYPE else classifier_tensor(stain_arr, SIZE_SHAPE)
                x_grade = classifier_tensor(stain_arr, SIZE_GRADE)
                t_out, s_out, g_out = M_TYPE(x_type), M_SHAPE(x_shape), M_GRADE(x_grade)
                names = (TYPE_NAMES, SHAPE_NAMES, GRADE_NAMES)
                x_cam = x_type
            pt, ps, pg = (torch.softmax(o, 1)[0] for o in (t_out, s_out, g_out))
        args = (pt, names[0], ps, names[1], pg, names[2])
        pred, conf = _route(*args)
        latency = round((time.time() - t0) * 1000, 1)

        gland = seg_overlay(image, seg_prob(arr))
        attn = None
        if x_cam is not None:
            attn = overlay(image, gradcam(M_TYPE, x_cam, M_TYPE.layer4[-1], SIZE_TYPE),
                           color=(239, 159, 39), size=SIZE_TYPE)

    banner = (BANNER_CASLITE if lite else BANNER_CASCADE) if pred in ("TA.HG", "TVA.HG") else None
    return {
        "pred_class": pred,
        "confidence": conf,
        "probs": _derive_probs(*args),
        "stages": _stages(*args),
        "latency_ms": latency,
        "gland_mask": gland,
        "attention_map": attn,
        "banner": banner,
    }


def run_cascade(image: Image.Image) -> dict:
    """Real 3-stage cascade (ResNet18 x3, 33.5M params) with Grad-CAM on the type stage."""
    return _run(image, lite=False)


def run_caslite(image: Image.Image) -> dict:
    """Real CasLite / PolypNet (1.53M params). No Grad-CAM: attention_map is None."""
    return _run(image, lite=True)


# Six example patches shown in the gallery — one per class.
# REPLACE with your real Section 5.10.4 verification examples.
EXAMPLES = [
    {"file": "hp.png", "label": "HP"},
    {"file": "norm.png", "label": "NORM"},
    {"file": "ta_lg.png", "label": "TA.LG"},
    {"file": "tva_lg.png", "label": "TVA.LG"},
    {"file": "tva_hg.png", "label": "TVA.HG"},
    {"file": "ta_hg.png", "label": "TA.HG"},
]
