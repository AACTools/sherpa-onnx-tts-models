#!/usr/bin/env python3
"""JSON hygiene linter for models.json.

Distinct from ``validate.py`` (which checks *content* against the JSON
Schema): this checks *formatting* so the committed file is canonical and
diffs stay minimal. A non-canonical file fails with a concrete diff, which
makes "did you forget to commit the regenerated file?" obvious in CI.

Checks:
  1. The file parses as JSON (and is valid UTF-8).
  2. Top-level keys are sorted alphabetically.
  3. Indentation is exactly 2 spaces.
  4. No trailing whitespace on any line.
  5. File ends with a single trailing newline.
  6. Re-serialising canonically produces a byte-identical file.

Run locally::

    python scripts/lint_json.py            # check
    python scripts/lint_json.py --rewrite  # fix in place

Exits non-zero if anything is off, so this is CI-safe with no rewrites.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_TARGET = ROOT / "models.json"


def canonicalise(raw: str) -> str:
    """Re-serialise ``raw`` the way ``generate.py`` does.

    Top-level model ids are sorted alphabetically (for stable diffs); the
    field order *within* each entry is preserved so descriptions stay
    readable. This must stay in sync with the final ``json.dumps`` call in
    ``generate.py``.
    """
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object at the top level, got {type(data).__name__}")
    sorted_data = dict(sorted(data.items()))
    return json.dumps(sorted_data, indent=2, ensure_ascii=False) + "\n"


def line_level_checks(raw: str) -> list[str]:
    """Catch issues a byte comparison would conflate (helps the error message)."""
    problems: list[str] = []
    for i, line in enumerate(raw.splitlines(), 1):
        if line.endswith(" "):
            problems.append(f"line {i}: trailing whitespace")
        # Indentation must be multiples of 2 spaces and no tabs.
        stripped = line.lstrip(" ")
        indent_len = len(line) - len(stripped)
        if "\t" in line[:indent_len + 1]:
            problems.append(f"line {i}: tab in indentation (use spaces)")
        elif indent_len % 2 != 0:
            problems.append(f"line {i}: odd indentation ({indent_len} spaces, expected multiple of 2)")
    if not raw.endswith("\n"):
        problems.append("file must end with a trailing newline")
    if raw.endswith("\n\n"):
        problems.append("file ends with multiple blank lines")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", nargs="?", default=str(DEFAULT_TARGET), help="JSON file to lint")
    parser.add_argument("--rewrite", action="store_true", help="Rewrite the file in canonical form instead of checking")
    args = parser.parse_args()

    path = Path(args.target)
    if not path.exists():
        print(f"error: {path} does not exist", file=sys.stderr)
        return 1

    raw = path.read_text(encoding="utf-8")

    # 1. Parse (surfaces syntax errors with a useful location).
    try:
        canonical = canonicalise(raw)
    except json.JSONDecodeError as exc:
        print(f"error: {path.name} is not valid JSON: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"error: {path.name}: {exc}", file=sys.stderr)
        return 1

    if args.rewrite:
        path.write_text(canonical, encoding="utf-8")
        print(f"Rewrote {path.name} in canonical form ({len(canonical)} bytes).")
        return 0

    # 2. Line-level checks first (clearer messages than a blob diff).
    problems = line_level_checks(raw)
    # 3. Byte-identical to canonical re-serialisation (catches key order,
    #    escaping, and indent drift in one shot).
    if raw != canonical:
        problems.append("file does not match canonical form (sorted keys, 2-space indent)")

    # Confirm top-level keys really are sorted (canonicalise sorts, but be
    # explicit so the message names the problem).
    keys = list(json.loads(raw).keys())
    if keys != sorted(keys):
        problems.append("top-level keys are not sorted alphabetically")

    if problems:
        print(f"{len(problems)} problem(s) in {path.name}:", file=sys.stderr)
        for p in problems[:20]:
            print(f"  - {p}", file=sys.stderr)
        print("\nFix with: python scripts/lint_json.py --rewrite", file=sys.stderr)
        return 1

    print(f"{path.name}: canonical ({len(raw)} bytes, {len(json.loads(raw))} entries).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
