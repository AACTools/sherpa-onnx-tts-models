#!/usr/bin/env python3
"""(Re-)fetch sha256 digests for registry entries that can have one.

Durable version of the one-off #5 campaign: PR #5 committed digests
directly into models.json, but ``generate.py --force`` re-derives every
entry from upstream and the preserve-existing pass is skipped under
--force — a regeneration silently wiped ~1,226 digests (#7 regression,
caught by the wrapper's availability audit: every MMS voice flipped to
blocked:no-sha256).

Sources, all WITHOUT downloading model bytes:
  - Hugging Face repos: the paths-info API returns the LFS sha256 per
    file (MMS fleet = one call; ours.py artifacts = one call per repo)
  - GitHub release archives: the release API's asset ``digest`` field
    (sha256:…) when present, else streamed through hashlib

Also teaches generate.py-independent safety: digests are MERGED into
whatever models.json carries, and validate.py enforces a coverage floor
so a future regeneration cannot silently drop them again.

Usage: python scripts/fetch_sha256.py [--write]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent.parent
MODELS = HERE / "models.json"

HF = "https://huggingface.co"
GH = "https://api.github.com/repos/k2-fsa/sherpa-onnx/releases"


def hf_paths_info(repo: str, paths: list[str]) -> dict[str, str]:
    """{path: sha256} for one HF repo — a single API call."""
    url = f"{HF}/api/models/{repo}/paths-info/main"
    body = json.dumps({"paths": paths}).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        entries = json.load(r)
    out = {}
    for e in entries:
        oid = e.get("lfs", {}).get("oid")
        if oid:
            out[e["path"]] = oid
    return out


def gh_asset_digests() -> dict[str, str]:
    """{asset download url: sha256} across the sherpa-onnx releases."""
    out: dict[str, str] = {}
    for page in range(1, 12):
        with urllib.request.urlopen(f"{GH}?per_page=100&page={page}", timeout=60) as r:
            releases = json.load(r)
        if not releases:
            break
        for rel in releases:
            for asset in rel.get("assets", []):
                dig = asset.get("digest") or ""
                if dig.startswith("sha256:"):
                    out[asset["browser_download_url"]] = dig.removeprefix("sha256:")
    return out


def stream_sha256(url: str) -> str:
    h = hashlib.sha256()
    req = urllib.request.Request(url, headers={"User-Agent": "sherpa-onnx-models/1"})
    with urllib.request.urlopen(req, timeout=600) as r:
        for chunk in iter(lambda: r.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="update models.json in place")
    args = ap.parse_args()

    models = json.loads(MODELS.read_text(encoding="utf-8"))

    # ── group the digest-less entries by source ──
    hf_want: dict[str, set[str]] = {}     # repo -> {path}
    hf_entry: dict[tuple[str, str], list[str]] = {}  # (repo, path) -> [ids]
    gh_entries: list[str] = []
    for mid, m in models.items():
        if m.get("sha256"):
            continue
        url = m.get("url", "")
        if "/resolve/" in url and "huggingface.co" in url:
            prefix, path = url.split("/resolve/main/", 1)
            repo = prefix.removeprefix("https://huggingface.co/")
            if path.endswith(".onnx"):
                key = (repo, path)
                hf_want.setdefault(repo, set()).add(path)
                hf_entry.setdefault(key, []).append(mid)
            else:
                # HF directory url (MMS): the digest pins model.onnx
                key = (repo, f"{path}/model.onnx")
                hf_want.setdefault(repo, set()).add(key[1])
                hf_entry.setdefault(key, []).append(mid)
        elif "github.com" in url and "/releases/download/" in url:
            gh_entries.append(mid)

    added, failed = 0, []
    # ── HF: one paths-info call per repo ──
    for repo, paths in hf_want.items():
        print(f"HF {repo}: {len(paths)} paths…")
        try:
            info = hf_paths_info(repo, sorted(paths))
        except Exception as e:  # noqa: BLE001
            print(f"  !! paths-info failed: {e}", file=sys.stderr)
            info = {}
        for (r, path), ids in hf_entry.items():
            if r != repo:
                continue
            oid = info.get(path)
            if not oid:
                failed.extend(ids)
                continue
            for mid in ids:
                models[mid]["sha256"] = oid
                added += 1

    # ── GitHub archives: API digest, else stream ──
    if gh_entries:
        print(f"GitHub archives: {len(gh_entries)} entries…")
        digests = gh_asset_digests()
        for mid in gh_entries:
            url = models[mid]["url"]
            dig = digests.get(url)
            if not dig:
                try:
                    dig = stream_sha256(url)
                except Exception as e:  # noqa: BLE001
                    print(f"  !! {mid}: {e}", file=sys.stderr)
                    failed.append(mid)
                    continue
            models[mid]["sha256"] = dig
            added += 1

    print(f"added {added} digests; {len(failed)} failed")
    for mid in failed[:20]:
        print(f"  failed: {mid}")
    if args.write and added:
        MODELS.write_text(
            json.dumps(models, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"wrote {MODELS.name}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
