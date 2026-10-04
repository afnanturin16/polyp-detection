#!/usr/bin/env python3
"""
run_fidelity.py - checks that this install reproduces the recorded predictions.

Runs the six vetted test patches (verification/examples/*.png) through run_cascade() and run_caslite(), i.e. 12
predictions, and compares each with verification/reference_predictions_test.csv (per-patch predictions on the
official 2,399-patch UniToPatho test set).  A prediction passes when the class matches exactly and the confidence is
within 2e-3.

    python verification/run_fidelity.py

Exit code 0 = all 12 match, 1 = at least one mismatch.  Works from any directory (paths are relative to this file).
"""
import csv
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "backend"))

from PIL import Image  # noqa: E402

import inference  # noqa: E402  (loads all checkpoints once)

TOL = 2e-3
manifest = json.loads((HERE / "examples_manifest.json").read_text(encoding="utf-8"))
with open(HERE / "reference_predictions_test.csv", encoding="utf-8", newline="") as fh:
    ref = {int(r["idx"]): r for r in csv.DictReader(fh)}

print(f"device: {inference.DEVICE}")
print(f"{'patch':<8}{'model':<9}{'pred':<8}{'reference':<10}{'conf':<9}{'ref conf':<9}{'diff':<9}{'sec':<6}ok")
all_ok = True
worst = 0.0
for ex in manifest:
    row = ref[int(ex["test_row_idx"])]
    img = Image.open(HERE / ex["file"]).convert("RGB")
    for name, fn, key in (("cascade", inference.run_cascade, "cascade"), ("caslite", inference.run_caslite, "lite")):
        t0 = time.time()
        res = fn(img)
        sec = time.time() - t0
        exp, exp_conf = row[f"{key}_pred"], float(row[f"{key}_conf"])
        diff = abs(res["confidence"] - exp_conf)
        ok = res["pred_class"] == exp and diff < TOL
        all_ok &= ok
        worst = max(worst, diff)
        print(f"{ex['true_class']:<8}{name:<9}{res['pred_class']:<8}{exp:<10}{res['confidence']:<9.4f}{exp_conf:<9.4f}{diff:<9.1e}{sec:<6.1f}{ok}")

print(f"\nALL 12 MATCH: {all_ok}   (largest confidence difference {worst:.1e}, tolerance {TOL:.0e})")
sys.exit(0 if all_ok else 1)
