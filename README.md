# Polyp Detection

Web app for six-class colorectal polyp histopathology (HP, NORM, TA.LG, TA.HG, TVA.LG, TVA.HG): a three-stage
cascade and the lightweight CasLite model, plus gland segmentation, Grad-CAM, batch upload and a one-patch report.
**Research use only, not a diagnostic tool.**

Everything the app needs is inside this folder: the checkpoints in `models/` (stored with Git LFS), the stain
reference and helper modules in `backend/`, and the frontend in `frontend/`. It runs on CPU; no GPU is needed.

## Run it in GitHub Codespaces

1. Push this folder to a **private** GitHub repository. Git LFS must be installed locally first
   (`git lfs install`); `.gitattributes` already tracks the `*.pt` checkpoints.
   Check that the checkpoints went through LFS with `git lfs ls-files` (five `.pt` files).
2. On GitHub open the repository, then **Code -> Codespaces -> Create codespace on main**.
3. Wait for the build. The dev container installs Python 3.12 and runs `pip install -r requirements.txt`
   (CPU-only PyTorch, a few minutes the first time).
4. In the Codespace terminal start the server:

   ```bash
   bash start_server.sh
   ```

   The first start takes a little while because all five models load once at startup.
5. Open the **Ports** tab, find port **8000** and click the globe icon (or use the pop-up that appears).
   Keep the port visibility **Private** (the default).
6. Upload an H&E-stained colorectal tissue patch on the **Classify** tab.

## Check that the install reproduces the recorded predictions

```bash
python verification/run_fidelity.py
```

It runs six vetted test patches through both models (12 predictions) and compares them with
`verification/reference_predictions_test.csv`. The class must match exactly and the confidence must be within
2e-3. It prints `ALL 12 MATCH: True` and exits with code 0 when everything matches.

## Notes

- Inference runs one image at a time on CPU, so expect a few seconds per patch on a small Codespace machine.
  Batch upload is limited to 10 images.
- Codespaces stop after a period of inactivity. Start the server again with `bash start_server.sh`.
- Run on your own machine instead: create a Python 3.12 virtual environment, `pip install -r requirements.txt`,
  then `bash start_server.sh` (or `cd backend && uvicorn main:app --host 127.0.0.1 --port 8000`).

## Layout

```
backend/        FastAPI app (main.py), inference.py, input_gate.py, stain normaliser, PolypNet definition
frontend/       static web UI served by the backend
models/         checkpoints (Git LFS)
verification/   reference predictions, six vetted patches, run_fidelity.py
.devcontainer/  Codespaces configuration
start_server.sh starts the server on port 8000
```
