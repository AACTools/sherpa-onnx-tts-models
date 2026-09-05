#!/usr/bin/env python3
"""Generate the enriched sherpa-onnx TTS model registry.

This is the canonical builder for ``models.json``. It pulls every TTS model
asset from the ``k2-fsa/sherpa-onnx`` ``tts-models`` GitHub release plus the
Meta MMS multilingual set from Hugging Face, and writes a single enriched
registry keyed by model id.

Compared to the original ``createindex.py`` in tts-wrapper, this:

  * recognises the newer model types (``kitten``, ``supertonic``) that the
    legacy filename parser skipped;
  * captures the SHA-256 ``digest`` GitHub now attaches to every release
    asset, so downloads can be verified;
  * enriches each entry with ``license``, ``license_url``, ``description``,
    ``source_url``, ``voice_names``, ``min_sherpa_onnx_version``,
    ``deprecated``, and ``tags`` (see ``overrides.py`` for the curated bits).

The whole refresh is two HTTP fetches (one GitHub release + one HF JSON), so
a full run takes a few seconds — no model files are downloaded.

Usage::

    python generate.py                 # incremental (keeps manual edits)
    python generate.py --force         # rebuild from scratch
    python generate.py --no-validate   # skip URL reachability checks

The script never deletes a field an editor added manually when run without
``--force``: it merges new/changed entries onto the existing file. Use
``--force`` for a clean rebuild.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import langcodes
import requests

import overrides

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("sherpa-onnx-tts-models")

REPO = "k2-fsa/sherpa-onnx"
RELEASE_TAG = "tts-models"
HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "models.json"

# A whole filename segment that is exactly a 2-letter language code,
# optionally with a region ("en", "en_US", "zh_en"), used as a last-resort
# language guess for VITS-family models. Matching whole segments (never
# substrings — "-in" inside "inflect" is not Indonesian) and only the
# *filename* (the "fs" of "k2-fsa" in the URL once produced a bogus ``fs``).
LANG_SEGMENT_FALLBACK = re.compile(r"^[a-z]{2}(_[a-zA-Z]{2})?$")
ISO_PATTERN = re.compile(r"^[a-zA-Z0-9]{1,8}$")

# The set of model types sherpa-onnx's ``OfflineTtsModelConfig`` understands.
# Anything else in the release is skipped (ASR models, vocoders, espeak data…).
KNOWN_MODEL_TYPES = {"vits", "matcha", "kokoro", "kitten", "supertonic", "zipvoice", "pocket"}

# Trailing precision/quantization tags used by piper/coqui/kitten/zipvoice
# releases. The untagged archive of a voice is the full-precision (fp32) one.
QUANT_TAGS = {"int8", "fp16", "fp32"}


# ---------------------------------------------------------------------------
# Language helpers (ported from tts-wrapper/createindex.py)
# ---------------------------------------------------------------------------

def language_data(lang_code: str, region: str) -> dict[str, str]:
    """Resolve a 2-/3-letter code to ``{lang_code, language_name, country}``.

    Uses ``langcodes`` for the human name and a best-effort region. Bad /
    non-ASCII codes come back as ``"unknown"`` so they're visible rather than
    silently dropped.
    """
    if (
        not lang_code
        or lang_code == "unknown"
        or not lang_code.isascii()
        or not ISO_PATTERN.match(lang_code)
    ):
        return {"lang_code": "unknown", "language_name": "Unknown", "country": region or "Unknown"}

    try:
        info = langcodes.get(lang_code)
        country = region if region and region != "Unknown" else (info.maximize().region or "Unknown")
        return {
            "lang_code": lang_code,
            "language_name": info.language_name() or "Unknown",
            "country": country,
        }
    except LookupError:
        return {"lang_code": lang_code, "language_name": "Unknown", "country": region or "Unknown"}


def extract_languages(url: str, model_type: str, developer: str) -> list[tuple[str, str]]:
    """Pull ``(lang_code, region)`` pairs out of the asset filename/URL.

    Encodes the per-developer naming conventions sherpa-onnx's release uses:
    Piper and Mimic3 embed ``ll_RR``, Coqui/Kokoro use a bare ``ll``, etc.
    Falls back to the generic dash-code regex, then ``("unknown", ...)``.
    """
    filename = url.rsplit("/", 1)[-1]

    if developer == "piper":
        m = re.search(r"vits-piper-(\w+)_(\w+)-", filename)
        if m:
            return [(m.group(1).lower(), m.group(2))]

    if developer == "mimic3":
        m = re.search(r"vits-mimic3-(\w+)_(\w+)-", filename) or re.search(
            r"vits-mimic3-([a-z]{2})-", filename.lower()
        )
        if m:
            code = m.group(1).lower()
            region = m.group(2) if m.lastindex == 2 else "Unknown"
            return [(code, region)]

    if developer == "coqui":
        m = re.search(r"vits-coqui-(\w+)-", filename)
        if m:
            return [(m.group(1).lower(), "Unknown")]

    if developer == "kokoro":
        if "multi-lang" in filename:
            return [("zh", "CN"), ("en", "US")]
        # Find the first segment that looks like a language code (``en`` /
        # ``zh_en``), skipping variant tags like ``int8``. Matches the logic
        # in ``_kokoro_variant`` so the id and language list agree.
        kokoro_stem = re.sub(r"\.(tar\.bz2|tar\.gz|zip)$", "", filename)
        for seg in kokoro_stem.split("-")[1:]:
            if _LANG_SEGMENT.match(seg):
                region = seg.split("_")[1] if "_" in seg else "Unknown"
                return [(seg.split("_")[0], region)]

    # Generic fallback — whole filename segments only, never the URL.
    # Strip the archive extension first so a trailing "en.tar.bz2" is "en".
    stem = re.sub(r"\.(tar\.bz2|tar\.gz|zip)$", "", filename)
    for seg in stem.split("-"):
        m = LANG_SEGMENT_FALLBACK.match(seg)
        if m:
            code = seg[:2]
            region = seg[3:].upper() if len(seg) > 2 else ""
            return [(code, region or "Unknown")]
    return [("unknown", "Unknown")]


# ---------------------------------------------------------------------------
# Asset parsing
# ---------------------------------------------------------------------------

def parse_asset(asset: dict) -> dict | None:
    """Turn a GitHub release asset into a registry entry, or ``None`` to skip.

    Returns ``None`` for anything that isn't a TTS model archive (.exe,
    checksum files, espeak data, ASR models, vocoders distributed separately,
    superseded re-uploads).
    """
    filename = asset["name"]
    # Skip obvious non-model artefacts in the release.
    if (
        filename.endswith((".exe", ".txt", ".png", ".wav"))
        or filename.startswith("espeak-")
        or "-mms-" in filename
    ):
        return None

    # kokoro multi-lang v1_0 (53 speakers) was superseded by v1_1 (103
    # speakers) on the same day; only register the v1_1 uploads.
    if re.fullmatch(r"kokoro-(int8-)?multi-lang-v1_0\.(tar\.bz2|tar\.gz|zip)", filename):
        return None

    url = asset["browser_download_url"]
    stem = re.sub(r"\.(tar\.bz2|tar\.gz|zip)$", "", filename)
    parts = stem.split("-")

    # Newer "sherpa-onnx-<type>-…" packaged models (supertonic handled by the
    # caller; zipvoice/pocket/vits-zh-ll here).
    if stem.startswith("sherpa-onnx-"):
        sub = parts[2] if len(parts) > 2 else ""
        if sub == "zipvoice":
            return build_zipvoice_entry(asset)
        if sub == "pocket":
            return build_pocket_entry(asset)
        if sub == "vits" and stem == "sherpa-onnx-vits-zh-ll":
            return build_zh_ll_entry(asset)
        if sub not in ("supertonic", "kitten"):
            return None

    # Trailing quantization tag: ``vits-piper-en_US-amy-low-int8`` → the int8
    # build of the ``amy`` voice. Strip it so name/quality parsing is stable,
    # then re-attach it to the id and the ``quantization`` field — otherwise
    # all three variants of a voice collapse onto the same id and only the
    # last one (in GitHub API order) survives.
    quant = "fp32"
    if len(parts) >= 5 and parts[-1] in QUANT_TAGS:
        quant = parts[-1]
        parts = parts[:-1]

    model_type, developer, name, quality = _classify(parts, stem)
    if model_type is None:
        return None
    name = overrides.fix_name(filename, name)

    lang_pairs = extract_languages(url, model_type, developer)
    lang_pairs = overrides.fix_language(filename, lang_pairs)
    langs = [language_data(code, region) for code, region in lang_pairs]

    sample_rate = _sample_rate(developer, model_type)

    model_id = _model_id(developer, lang_pairs, name, quality, quant)

    entry: dict[str, Any] = {
        "id": model_id,
        "model_type": model_type,
        "developer": developer,
        "name": name,
        "language": langs,
        "quality": quality,
        "quantization": quant,
        "sample_rate": sample_rate,
        "num_speakers": 1,
        "url": url,
        "compression": True,
        "filesize_mb": round(asset["size"] / (1024 * 1024), 2),
    }

    # Curated per-filename facts (speaker counts, sample rates, voice names)
    # that can't be derived from the filename at all.
    overrides.apply_filename_meta(filename, entry)

    # SHA-256 straight from the GitHub API ``digest`` field (``sha256:…``).
    digest = asset.get("digest") or ""
    if digest.startswith("sha256:"):
        entry["sha256"] = digest.split(":", 1)[1]

    _enrich(entry)
    return entry


def _classify(parts: list[str], stem: str) -> tuple[str | None, str, str, str]:
    """Determine ``(model_type, developer, name, quality)`` from filename parts.

    sherpa-onnx release names follow a few conventions:
      ``vits-piper-en_GB-amy-low``      → vits, piper, amy, low
      ``matcha-icefall-en_US-ljspeech`` → matcha, icefall, ljspeech, unknown
      ``kokoro-en-v0_19``               → kokoro, kokoro, en, v0_19
      ``sherpa-onnx-supertonic-3-tts-int8-2026-05-11``  → supertonic, supertone, …
      ``sherpa-onnx-kitten-…``          → kitten, kitten, …
    """
    # Newer "sherpa-onnx-<type>-…" packaged models.
    if stem.startswith("sherpa-onnx-"):
        sub = parts[2] if len(parts) > 2 else ""
        if sub == "supertonic":
            return "supertonic", "supertone", "supertonic-3-int8", "int8"
        if sub == "kitten":
            return "kitten", "kitten", parts[-2] if len(parts) > 3 else "kitten", "unknown"
        return None, "", "", ""

    model_type = parts[0] if parts and parts[0] in KNOWN_MODEL_TYPES else None
    if model_type is None:
        return None, "", "", ""

    if model_type == "matcha" and len(parts) > 1 and parts[1] == "tts":
        # ``matcha-tts-fa_en-musa`` — the second segment is the literal "tts"
        # from the project name "Matcha-TTS", not a developer slug. Collapse
        # it so the developer/license resolve to the Matcha-TTS project.
        developer = "matcha"
        name = parts[3] if len(parts) > 3 else "unknown"
        quality = parts[4] if len(parts) > 4 else "unknown"
        quality = re.sub(r".*?_", "", quality)
        return model_type, developer, name, quality

    if model_type == "kokoro":
        # Kokoro filenames pack a variant tag (int8 / multi-lang) and a
        # language together: ``kokoro-en-v0_19``, ``kokoro-int8-en-v0_19``,
        # ``kokoro-int8-multi-lang-v1_1``. We can't naively take parts[1] as
        # the name because "int8" would be misread as the language. Instead,
        # pull out the language segment and treat the rest as the variant.
        developer = "kokoro"
        name = _kokoro_variant(parts[1:], stem)
        quality = "unknown"
        return model_type, developer, name, quality

    developer = parts[1] if len(parts) > 1 else "unknown"
    name = parts[3] if len(parts) > 3 else "unknown"
    quality = parts[4] if len(parts) > 4 else "unknown"
    # Strip ``*_`` prefixes sherpa-onnx sometimes appends to quality.
    quality = re.sub(r".*?_", "", quality)
    return model_type, developer, name, quality


# A 2-letter language code optionally followed by ``_RR`` region, e.g. ``en`` / ``zh_en``.
_LANG_SEGMENT = re.compile(r"^[a-z]{2}(_[a-z]{2})?$")


def _kokoro_variant(segments: list[str], filename: str) -> str:
    """Variant tag for a Kokoro model, with the language segment removed.

    ``kokoro-en-v0_19``           → segments [en, v0_19]      → "v0_19"
    ``kokoro-int8-en-v0_19``      → [int8, en, v0_19]          → "int8-v0_19"
    ``kokoro-int8-multi-lang-v1_1`` → [int8, multi, lang, v1_1] → "int8"
    ``kokoro-multi-lang-v1_1``    → [multi, lang, v1_1]        → "multi-lang"

    Drops the redundant ``multi-lang``/``lang``/version tail because
    multilingual-ness is already encoded in the language list and the version
    is noise for the id.
    """
    if "multi-lang" in filename:
        # Multilingual: keep meaningful variant tags like "int8", drop the rest.
        kept = [s for s in segments if s not in {"multi", "lang", "multi-lang"} and not s.startswith("v")]
        return "-".join(kept) if kept else "multi-lang"
    kept = []
    for s in segments:
        if _LANG_SEGMENT.match(s):
            continue  # the language
        kept.append(s)
    return "-".join(kept) if kept else "unknown"


def _sample_rate(developer: str, model_type: str) -> int:
    if developer == "piper":
        return 22050
    if model_type == "kokoro":
        return 24000
    if model_type == "supertonic":
        return 24000
    if model_type == "zipvoice":
        return 24000
    if model_type == "pocket":
        return 24000
    return 16000


def _model_id(
    developer: str,
    lang_pairs: list[tuple[str, str]],
    name: str,
    quality: str,
    quant: str = "fp32",
) -> str:
    """Build the registry id.

    Piper and Mimic3 voices are regional (``en_US`` vs ``en_GB``) and
    different regions can ship the same voice name (e.g. both ``en_GB-miro``
    and ``en_US-miro`` exist), so their id keeps the region. The
    quantization is appended when it isn't the default fp32 so the three
    builds of a voice stay distinct.
    """
    if developer in ("piper", "mimic3"):
        tokens = [f"{code}_{region}" if region and region != "Unknown" else code
                  for code, region in lang_pairs]
    else:
        tokens = [code for code, _ in lang_pairs]
    base = f"{developer}-{'_'.join(tokens)}-{name}"
    if quality and quality != "unknown":
        base = f"{base}-{quality}"
    if quant in ("int8", "fp16"):
        base = f"{base}-{quant}"
    return base


def _pretty_developer(name: str) -> str:
    """Capitalise a developer/model_type for descriptions.

    Acronyms (MMS) stay upper-case; everything else gets title-cased so we
    get ``Piper`` / ``Supertone`` / ``Kokoro`` rather than the raw lowercase
    slug from the filename.
    """
    upper = {"mms"}
    return name.upper() if name in upper else name.capitalize()


def _enrich(entry: dict) -> None:
    """Add the curated / derived metadata fields in place."""
    model_id = entry["id"]
    model_type = entry["model_type"]
    developer = entry["developer"]

    # --- description: one human-readable line ------------------------------
    # Avoid "Mms mms TTS" / "Kokoro kokoro TTS" when developer == model_type
    # by collapsing the duplicate into a single token.
    langs = entry["language"]
    lang_part = langs[0]["language_name"] if langs else "unknown"
    if len(langs) > 1:
        lang_part = f"{len(langs)} languages"
    speaker_part = f", {entry['num_speakers']} voices" if entry["num_speakers"] > 1 else ""
    quality = entry.get("quality", "unknown")
    quant = entry.get("quantization", "fp32")
    if quality != "unknown" and quant in ("int8", "fp16"):
        quality_part = f" ({quality}, {quant})"
    elif quality != "unknown":
        quality_part = f" ({quality})"
    elif quant in ("int8", "fp16"):
        quality_part = f" ({quant})"
    else:
        quality_part = ""
    if developer == model_type:
        source = _pretty_developer(developer)
    else:
        source = f"{_pretty_developer(developer)} {model_type}"
    entry["description"] = f"{source} TTS — {lang_part}{speaker_part}{quality_part}"

    # --- license -----------------------------------------------------------
    entry.update(overrides.resolve_license(developer, model_type))

    # --- min sherpa-onnx version ------------------------------------------
    entry["min_sherpa_onnx_version"] = overrides.resolve_min_version(model_type)

    # --- floravox support ---------------------------------------------------
    # Which engines can drive this family. floravox (the permissive,
    # SSML-native engine with measured word timings) drives the
    # VITS/Matcha/Kokoro graphs; everything else needs sherpa-onnx.
    # Values: "floravox" | "sherpa-onnx" | "both" (reserved).
    entry["engines"] = (
        "floravox" if model_type in ("vits", "mms", "matcha", "kokoro") else "sherpa-onnx"
    )

    # --- source_url: best guess -------------------------------------------
    entry.setdefault(
        "source_url",
        f"https://github.com/{REPO}/releases/tag/{RELEASE_TAG}",
    )

    # --- tags --------------------------------------------------------------
    tags = {model_type, "on-device", "sherpa-onnx"}
    if entry["num_speakers"] > 1:
        tags.add("multi-speaker")
    if len(langs) > 1:
        tags.add("multilingual")
    if entry.get("quantization") == "int8":
        tags.add("int8")
    entry["tags"] = sorted(tags)

    # --- curated overrides (voice_names, deprecated, source_url, …) -------
    overrides.apply_overrides(model_id, entry)


# ---------------------------------------------------------------------------
# Supertonic special case: one multilingual entry, 31 languages
# ---------------------------------------------------------------------------

SUPERTONIC_LANGS = [
    ("en", "English"), ("ko", "Korean"), ("ja", "Japanese"), ("ar", "Arabic"),
    ("bg", "Bulgarian"), ("cs", "Czech"), ("da", "Danish"), ("de", "German"),
    ("el", "Greek"), ("es", "Spanish"), ("et", "Estonian"), ("fi", "Finnish"),
    ("fr", "French"), ("hi", "Hindi"), ("hr", "Croatian"), ("hu", "Hungarian"),
    ("id", "Indonesian"), ("it", "Italian"), ("lt", "Lithuanian"), ("lv", "Latvian"),
    ("nl", "Dutch"), ("pl", "Polish"), ("pt", "Portuguese"), ("ro", "Romanian"),
    ("ru", "Russian"), ("sk", "Slovak"), ("sl", "Slovenian"), ("sv", "Swedish"),
    ("tr", "Turkish"), ("uk", "Ukrainian"), ("vi", "Vietnamese"),
]


def build_supertonic_entry(asset: dict) -> dict:
    """The sherpa-onnx Supertonic bundle is one multilingual model — expand
    its language list to all 31 supported codes rather than the single
    ``unknown`` the filename parser would infer."""
    entry = parse_asset(asset)
    if entry is None:
        return None
    entry["language"] = [language_data(code, "") for code, _ in SUPERTONIC_LANGS]
    entry["num_speakers"] = 10
    entry["id"] = "supertonic-3-multilingual"
    # Description/name reflect the multilingual reality, not the filename.
    entry["name"] = "supertonic-3-int8"
    entry["description"] = (
        "Supertone Supertonic 3 multilingual flow-matching TTS — "
        "31 languages, 10 voices (M1–M5, F1–F5), int8"
    )
    # Re-apply overrides now that the id is final, then re-derive tags.
    overrides.apply_overrides(entry["id"], entry)
    entry["tags"] = sorted(
        {"supertonic", "on-device", "sherpa-onnx", "multilingual", "multi-speaker", "int8"}
    )
    return entry


# ---------------------------------------------------------------------------
# ZipVoice: zero-shot voice cloning (zh + en), packaged as
# ``sherpa-onnx-zipvoice[-distill][-int8|-fp32]-zh-en-emilia``
# ---------------------------------------------------------------------------

def build_zipvoice_entry(asset: dict) -> dict:
    distill = "-distill" in asset["name"]
    quant = "int8" if "-int8-" in asset["name"] else "fp32"

    # The Dec-2025 ``distill-fp32`` re-upload is a different build from the
    # Aug-2025 ``distill`` archive (both fp32), so keep the explicit tag in
    # the name to disambiguate the ids.
    name = "emilia-distill" if distill else "emilia"
    if distill and "-fp32-" in asset["name"]:
        name = "emilia-distill-fp32"

    entry = _build_packaged_entry(asset, "zipvoice", "zipvoice", name, quant,
                                  [("zh", "CN"), ("en", "US")])
    entry["description"] = (
        "ZipVoice zero-shot voice-cloning TTS (flow matching) — Chinese + "
        "English; needs reference audio and its transcript"
    )
    entry["tags"] = sorted({
        "zipvoice", "on-device", "sherpa-onnx", "multilingual",
        "zero-shot", "voice-cloning", *(["int8"] if quant == "int8" else []),
    })
    return entry


# ---------------------------------------------------------------------------
# PocketTTS (Kyutai): zero-shot voice cloning, English, reference audio only.
# Packaged as ``sherpa-onnx-pocket-tts[-int8]-2026-01-26``.
# ---------------------------------------------------------------------------

def build_pocket_entry(asset: dict) -> dict:
    quant = "int8" if "-int8-" in asset["name"] else "fp32"
    entry = _build_packaged_entry(asset, "pocket", "kyutai", "pocket-tts", quant,
                                  [("en", "US")])
    entry["description"] = (
        "Kyutai PocketTTS zero-shot voice-cloning TTS — English; needs a "
        "short reference audio clip (no transcript)"
    )
    entry["tags"] = sorted({
        "pocket", "on-device", "sherpa-onnx",
        "zero-shot", "voice-cloning", *(["int8"] if quant == "int8" else []),
    })
    return entry


# ---------------------------------------------------------------------------
# sherpa-onnx-vits-zh-ll: Chinese multi-speaker VITS (5 speakers), trained
# with Plachtaa/VITS-fast-fine-tuning, hosted by csukuangfj.
# ---------------------------------------------------------------------------

def build_zh_ll_entry(asset: dict) -> dict:
    entry = _build_packaged_entry(asset, "vits", "csukuangfj", "ll", "fp32",
                                  [("zh", "CN")])
    entry["num_speakers"] = 5
    entry["description"] = (
        "Multi-speaker Chinese VITS (VITS-fast-fine-tuning) — 5 voices"
    )
    entry["tags"] = sorted({"vits", "on-device", "sherpa-onnx", "multi-speaker"})
    return entry


def _build_packaged_entry(
    asset: dict,
    model_type: str,
    developer: str,
    name: str,
    quant: str,
    lang_pairs: list[tuple[str, str]],
) -> dict:
    """Shared skeleton for the ``sherpa-onnx-<type>-…`` packaged models."""
    url = asset["browser_download_url"]
    model_id = _model_id(developer, lang_pairs, name, "unknown", quant)
    entry: dict[str, Any] = {
        "id": model_id,
        "model_type": model_type,
        "developer": developer,
        "name": name,
        "language": [language_data(code, region) for code, region in lang_pairs],
        "quality": "unknown",
        "quantization": quant,
        "sample_rate": _sample_rate(developer, model_type),
        "num_speakers": 1,
        "url": url,
        "compression": True,
        "filesize_mb": round(asset["size"] / (1024 * 1024), 2),
    }
    digest = asset.get("digest") or ""
    if digest.startswith("sha256:"):
        entry["sha256"] = digest.split(":", 1)[1]
    _enrich(entry)
    return entry


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

def fetch_github_models() -> dict[str, dict]:
    """One API call → every asset on the ``tts-models`` release."""
    url = f"https://api.github.com/repos/{REPO}/releases/tags/{RELEASE_TAG}"
    logger.info("Fetching %s …", url)
    resp = requests.get(url, timeout=30, headers={"Accept": "application/vnd.github+json"})
    resp.raise_for_status()
    assets = resp.json().get("assets", [])
    logger.info("Release has %d assets", len(assets))

    models: dict[str, dict] = {}
    skipped = 0
    for asset in assets:
        fname = asset["name"]
        # Supertonic v3 needs special handling (one multilingual model).
        if "supertonic-3-tts-int8-2026-05-11" in fname:
            entry = build_supertonic_entry(asset)
            if entry:
                models[entry["id"]] = entry
            continue
        # Older Supertonic bundles (v1/v2, e.g. ``...-2026-03-06.tar.bz2``)
        # are superseded by the v3 multilingual release above and live on a
        # separate upstream branch — skip them rather than emit garbled ids.
        if "supertonic" in fname:
            skipped += 1
            continue
        entry = parse_asset(asset)
        if entry is None:
            skipped += 1
            continue
        models[entry["id"]] = entry
    logger.info("Parsed %d models from GitHub (skipped %d non-model assets)", len(models), skipped)
    return models


def fetch_mms_models() -> dict[str, dict]:
    """Meta MMS multilingual set — lives on Hugging Face, not the GH release."""
    src = "https://huggingface.co/willwade/mms-tts-multilingual-models-onnx/raw/main/languages-supported.json"
    logger.info("Fetching MMS languages from %s …", src)
    resp = requests.get(src, timeout=30)
    resp.raise_for_status()
    rows = resp.json()

    models: dict[str, dict] = {}
    skipped = 0
    for row in rows:
        iso = row["Iso Code"]
        # Some MMS languages are listed but have no ONNX upload yet — their
        # URL is empty or the literal "Not available". Skip them rather than
        # emit entries that can never be downloaded.
        raw_url = row.get("ONNX Model URL", "")
        if not raw_url or raw_url == "Not available":
            skipped += 1
            continue
        model_id = f"mms_{iso}"
        # Rewrite the HF API tree URL to a resolve URL so it's directly downloadable.
        raw_url = raw_url.replace("api/models/", "", 1).replace("/tree/", "/resolve/")
        langs = [language_data(iso, row.get("Country", "Unknown"))]
        entry = {
            "id": model_id,
            "model_type": "mms",
            "developer": "mms",
            "name": row.get("Language Name", iso),
            "language": langs,
            "quality": "unknown",
            "quantization": "fp32",
            "sample_rate": 16000,
            "num_speakers": 1,
            "url": raw_url,
            "compression": False,
            "filesize_mb": 0.0,
        }
        _enrich(entry)
        models[model_id] = entry
    logger.info("Parsed %d MMS models (skipped %d without an ONNX URL)", len(models), skipped)
    return models


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_urls(models: dict[str, dict], limit: int | None = None) -> None:
    """HEAD-check each model URL and flag unreachable ones in-place.

    Disabled by default (``--no-validate``); only useful for a periodic audit
    since sherpa-onnx release URLs are stable. MMS entries point at a
    *directory* so they're checked via ``<url>/model.onnx``.
    """
    checked = 0
    for model_id, entry in models.items():
        if limit is not None and checked >= limit:
            break
        url = entry.get("url", "")
        if not url:
            continue
        target = f"{url}/model.onnx" if model_id.startswith("mms_") else url
        try:
            r = requests.head(target, timeout=10, allow_redirects=True)
            entry["url_valid"] = r.status_code in (200, 302)
        except requests.RequestException:
            entry["url_valid"] = False
        checked += 1
    logger.info("Validated %d URLs", checked)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="Rebuild from scratch, ignoring existing models.json")
    parser.add_argument("--no-validate", action="store_true", help="Skip URL reachability checks")
    parser.add_argument("--limit-validate", type=int, default=None, help="Only validate the first N URLs")
    args = parser.parse_args()

    existing: dict[str, dict] = {}
    if OUTPUT.exists() and not args.force:
        try:
            existing = json.loads(OUTPUT.read_text(encoding="utf-8"))
            logger.info("Loaded %d existing models from %s", len(existing), OUTPUT.name)
        except json.JSONDecodeError:
            logger.warning("Existing %s is malformed — starting fresh", OUTPUT.name)

    models: dict[str, dict] = {}
    models.update(fetch_github_models())
    models.update(fetch_mms_models())

    # Preserve any manual edits on ids we didn't re-derive this run.
    if existing and not args.force:
        for k, v in existing.items():
            models.setdefault(k, v)

    # Always preserve durations_url and has_durations from existing file,
    # even with --force. These are added by the patch pipeline, not by
    # generate.py, and must survive regeneration.
    existing_meta: dict[str, dict] = {}
    if OUTPUT.exists():
        try:
            existing_meta = json.loads(OUTPUT.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    if existing_meta:
        for k, v in existing_meta.items():
            if k not in models:
                continue
            for field in ("durations_url", "has_durations"):
                if field in v:
                    models[k][field] = v[field]
            # Preserve the "patched" tag
            if "tags" in v and "patched" in v["tags"]:
                if "tags" not in models[k]:
                    models[k]["tags"] = []
                if "patched" not in models[k]["tags"]:
                    models[k]["tags"].append("patched")

    if not args.no_validate:
        validate_urls(models, args.limit_validate)

    # Sort by id so output is deterministic across runs (the GitHub API and
    # Hugging Face can return assets in different orders). This keeps diffs
    # minimal and lets CI assert that the committed models.json matches a
    # fresh regeneration byte-for-byte.
    models = dict(sorted(models.items()))

    OUTPUT.write_text(json.dumps(models, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("Wrote %d models to %s", len(models), OUTPUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
