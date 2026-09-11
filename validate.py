#!/usr/bin/env python3
"""Validate models.json against schema.json and run sanity checks.

Three layers of checking:

  1. **Schema** — every entry conforms to ``schema.json`` (jsonschema).
  2. **Uniqueness** — ids and (model_type, url) pairs are unique.
  3. **Plausibility** — sample_rate, num_speakers, and sha256 (when present)
     look sane; non-deprecated models point at a reachable URL when
     ``--online`` is passed.

Exits non-zero if anything fails, so this is CI-safe.

Usage::

    python validate.py             # offline checks only (fast)
    python validate.py --online    # also HEAD-check a sample of URLs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jsonschema

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models.json"
SCHEMA = HERE / "schema.json"

SAMPLE_SIZE = 25  # cap online URL checks to keep the run quick


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--online", action="store_true", help="HEAD-check a sample of URLs")
    parser.add_argument("--schema-only", action="store_true", help="Skip uniqueness/plausibility checks")
    args = parser.parse_args()

    if not MODELS.exists():
        print(f"error: {MODELS.name} not found — run generate.py first", file=sys.stderr)
        return 1

    models = json.loads(MODELS.read_text(encoding="utf-8"))
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    print(f"Loaded {len(models)} models")

    errors: list[str] = []

    # 1. Schema conformance.
    for model_id, entry in models.items():
        try:
            jsonschema.validate(entry, schema)
        except jsonschema.ValidationError as exc:
            errors.append(f"[schema] {model_id}: {exc.message}")

    if args.schema_only:
        return _report(errors)

    # 2. Uniqueness — id is the JSON key, but catch dup URLs too.
    seen_urls: dict[str, str] = {}
    for model_id, entry in models.items():
        url = entry.get("url", "")
        if url and url in seen_urls:
            errors.append(
                f"[dup-url] {model_id} and {seen_urls[url]} share URL {url}"
            )
        seen_urls[url] = model_id

    # 3. Plausibility.
    for model_id, entry in models.items():
        sr = entry.get("sample_rate", 0)
        if sr and sr not in (8000, 16000, 22050, 24000, 44100, 48000):
            errors.append(f"[sample_rate] {model_id}: unusual rate {sr}")
        if entry.get("num_speakers", 1) < 1:
            errors.append(f"[num_speakers] {model_id}: must be >= 1")
        sha = entry.get("sha256", "")
        if sha and len(sha) != 64:
            errors.append(f"[sha256] {model_id}: wrong length ({len(sha)})")
        # License must be present and non-empty after enrichment.
        lic = entry.get("license", "")
        if not lic:
            errors.append(f"[license] {model_id}: missing license field")

    # 4. Optional online URL reachability (sampled).
    if args.online:
        import requests  # local import keeps offline runs dep-free

        checked = 0
        for model_id, entry in models.items():
            if checked >= SAMPLE_SIZE or entry.get("deprecated"):
                continue
            url = entry.get("url", "")
            if not url:
                continue
            target = f"{url}/model.onnx" if model_id.startswith("mms_") else url
            try:
                r = requests.head(target, timeout=10, allow_redirects=True)
                if r.status_code not in (200, 302):
                    errors.append(f"[url] {model_id}: HTTP {r.status_code}")
            except requests.RequestException as exc:
                errors.append(f"[url] {model_id}: {exc}")
            checked += 1
        print(f"Online-checked {checked} URLs")

    return _report(errors)


def _report(errors: list[str]) -> int:
    if errors:
        print(f"\n{len(errors)} problem(s):", file=sys.stderr)
        for e in errors[:50]:
            print(f"  - {e}", file=sys.stderr)
        if len(errors) > 50:
            print(f"  ... and {len(errors) - 50} more", file=sys.stderr)
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
