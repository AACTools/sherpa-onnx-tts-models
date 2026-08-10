#!/usr/bin/env python3
"""Pick the smallest downloadable model of each ``model_type``.

Used by the verify-models workflow to choose one representative per family
to actually synthesise — smallest first keeps CI minutes and disk usage
down. Writes a JSON array of ``{model_type, id, url, sha256, filesize_mb}``
to stdout so the workflow can matrix over it.

Models without a usable URL (MMS resolve dirs are validated by hitting
``<url>/model.onnx``) or with ``filesize_mb == 0`` are skipped for the
download path — MMS is verified separately by pointing sherpa-onnx at the
resolve directory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models.json"


def main() -> int:
    models = json.loads(MODELS.read_text(encoding="utf-8"))
    # Group by model_type, keep the smallest by filesize_mb (>0 so we skip
    # MMS resolve dirs which report 0.0).
    best: dict[str, dict] = {}
    for mid, entry in models.items():
        if entry.get("deprecated"):
            continue
        mt = entry["model_type"]
        size = entry.get("filesize_mb", 0.0) or 0.0
        if size <= 0:
            continue
        if mt not in best or size < best[mt]["filesize_mb"]:
            best[mt] = {
                "model_type": mt,
                "id": mid,
                "url": entry["url"],
                "sha256": entry.get("sha256", ""),
                "filesize_mb": size,
            }

    # Always include the canonical supertonic entry even though it's flagged
    # deprecated — it's the only supertonic model and worth verifying.
    if "supertonic-3-multilingual" in models and "supertonic" not in best:
        e = models["supertonic-3-multilingual"]
        best["supertonic"] = {
            "model_type": "supertonic",
            "id": "supertonic-3-multilingual",
            "url": e["url"],
            "sha256": e.get("sha256", ""),
            "filesize_mb": e["filesize_mb"],
        }

    # Also grab one MMS model (smallest is unknown — filesize is 0 — so pick
    # a stable, known-good one and let the verifier treat it specially).
    mms = next((m for m in models.values() if m["model_type"] == "mms"), None)
    if mms and "mms" not in best:
        best["mms"] = {
            "model_type": "mms",
            "id": mms["id"],
            "url": mms["url"],
            "sha256": "",
            "filesize_mb": 0.0,
        }

    out = sorted(best.values(), key=lambda m: m["model_type"])
    json.dump(out, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
