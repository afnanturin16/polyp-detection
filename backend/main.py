"""
main.py — FastAPI backend.

Serves:
  GET  /                     the frontend (static files)
  GET  /api/examples         list of example patches + their image data
  POST /api/classify         {model: "cascade"|"caslite", image: dataURL} -> result
  POST /api/compare          {image: dataURL} -> {cascade: result, caslite: result}
  POST /api/batch            {model, images: [{name, image: dataURL}]} (max 10) -> {results: [...]}

Run:
    pip install fastapi uvicorn python-multipart pillow
    python main.py
Then open http://127.0.0.1:8000
"""

import pathlib

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from inference import CLASSES, EXAMPLES, decode_image, encode_image, run_cascade, run_caslite
from input_gate import check_he_tissue

BASE_DIR = pathlib.Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"
EXAMPLES_DIR = FRONTEND_DIR / "examples"

app = FastAPI(title="Colorectal Polyp Classifier API")

# Same-origin in normal use (frontend is served by this same app), but CORS
# is left open so you can also run the frontend from a separate dev server
# (e.g. VS Code Live Server) while testing.
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


class ClassifyRequest(BaseModel):
    model: str  # "cascade" | "caslite"
    image: str  # base64 data URL


class CompareRequest(BaseModel):
    image: str


def _serialize(result: dict) -> dict:
    """Swap any PIL.Image values for base64 strings so the result is JSON-safe."""
    out = dict(result)
    for key in ("gland_mask", "attention_map"):
        if out.get(key) is not None:
            out[key] = encode_image(out[key])
    return out


UNREADABLE = "The file could not be read as an image."


def _unreadable() -> JSONResponse:
    return JSONResponse(status_code=422, content={"error": "unreadable_image", "reasons": [UNREADABLE]})


def _rejection(gate: dict) -> JSONResponse:
    return JSONResponse(status_code=422, content={"error": "not_he_tissue", "reasons": gate["reasons"]})


@app.get("/api/classes")
def get_classes():
    return {"classes": CLASSES}


@app.get("/api/examples")
def get_examples():
    """Returns example metadata; the browser loads the actual image files
    directly from /examples/<file> (served as static files below)."""
    return {"examples": EXAMPLES}


@app.post("/api/classify")
def classify(req: ClassifyRequest):
    try:
        image = decode_image(req.image)
    except Exception:
        return _unreadable()
    gate = check_he_tissue(image)           # reads a downscaled copy; the models get the untouched image
    if gate["verdict"] == "reject":
        return _rejection(gate)
    fn = run_cascade if req.model == "cascade" else run_caslite
    result = _serialize(fn(image))
    if gate["verdict"] == "warn":
        result["gate_warning"] = gate["reasons"]
    return result


@app.post("/api/compare")
def compare(req: CompareRequest):
    try:
        image = decode_image(req.image)
    except Exception:
        return _unreadable()
    gate = check_he_tissue(image)
    if gate["verdict"] == "reject":
        return _rejection(gate)
    cascade_result = _serialize(run_cascade(image))
    caslite_result = _serialize(run_caslite(image))
    out = {"cascade": cascade_result, "caslite": caslite_result}
    if gate["verdict"] == "warn":
        out["gate_warning"] = gate["reasons"]
    return out


MAX_BATCH = 10
THUMB_PX = 96


class BatchItem(BaseModel):
    name: str
    image: str  # base64 data URL


class BatchRequest(BaseModel):
    model: str  # "cascade" | "caslite"
    images: list[BatchItem]


@app.post("/api/batch")
def batch(req: BatchRequest):
    """Runs the same run_cascade()/run_caslite() over each image, one at a time. Overlays are not returned."""
    if not 1 <= len(req.images) <= MAX_BATCH:
        raise HTTPException(status_code=400, detail=f"Send between 1 and {MAX_BATCH} images per batch.")
    fn = run_cascade if req.model == "cascade" else run_caslite
    results = []
    for item in req.images:
        try:
            try:
                image = decode_image(item.image)
            except Exception:
                results.append({"name": item.name, "thumbnail": None, "pred_class": None, "confidence": None,
                                "latency_ms": None, "banner": None, "error": UNREADABLE})
                continue
            gate = check_he_tissue(image)
            if gate["verdict"] == "reject":     # this file's error row; the rest of the batch still runs
                results.append({"name": item.name, "thumbnail": None, "pred_class": None, "confidence": None,
                                "latency_ms": None, "banner": None,
                                "error": "Not an H&E-stained tissue patch, so it was not analysed. "
                                         + "; ".join(gate["reasons"]) + "."})
                continue
            r = fn(image)
            thumb = image.copy()
            thumb.thumbnail((THUMB_PX, THUMB_PX))
            results.append({
                "name": item.name, "thumbnail": encode_image(thumb),
                "pred_class": r["pred_class"], "confidence": r["confidence"],
                "latency_ms": r["latency_ms"], "banner": r["banner"], "error": None,
                **({"gate_warning": gate["reasons"]} if gate["verdict"] == "warn" else {}),
            })
        except Exception as e:  # one unreadable file must not sink the whole batch
            results.append({"name": item.name, "thumbnail": None, "pred_class": None, "confidence": None,
                            "latency_ms": None, "banner": None, "error": f"Could not process this file ({type(e).__name__})."})
    return {"model": req.model, "results": results}


# Static files: the frontend itself, and example images.
app.mount("/examples", StaticFiles(directory=str(EXAMPLES_DIR)), name="examples")
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
