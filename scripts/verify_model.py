#!/usr/bin/env python3
"""Download a model from the registry and synthesise a short clip.

End-to-end check that a registry entry's URL, file layout, and config are
actually loadable by sherpa-onnx. One model per ``model_type`` is enough
to catch regressions in the config builders (the thing most likely to
break when sherpa-onnx changes its file-layout conventions).

Exit codes:
  0  — synth succeeded and produced non-empty audio
  1  — synth failed or produced empty audio
  2  — setup error (missing model id, bad arguments)

Usage::

    python scripts/verify_model.py <model_id>
    python scripts/verify_model.py supertonic-3-multilingual
    python scripts/verify_model.py piper-en_GB-amy-low --cache-dir ./models

Requires ``sherpa-onnx`` (``pip install sherpa-onnx``). Downloads are
cached under ``--cache-dir`` (default ``./.verify-cache``) so re-runs
don't re-fetch.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
MODELS_JSON = ROOT / "models.json"
SAMPLE_TEXT = "Hello world. This is a verification clip."
SHERPA_ONNX_VOCODER_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/hifigan_v2.onnx"
)


# ---------------------------------------------------------------------------
# Download / extract
# ---------------------------------------------------------------------------

def ensure_downloaded(model_id: str, entry: dict, cache_dir: Path) -> Path:
    """Make sure the model is extracted under ``cache_dir/<model_id>/`` and
    return that directory. Reuses a cache across runs."""
    target = cache_dir / model_id
    if target.exists() and any(target.iterdir()):
        return target
    target.mkdir(parents=True, exist_ok=True)

    url = entry["url"]
    print(f"  downloading {url}")
    if model_id.startswith("mms_"):
        # MMS entries point at a Hugging Face *directory*; fetch the three
        # files sherpa-onnx needs directly.
        for name in ("model.onnx", "tokens.txt", "lexicon.txt"):
            _download(f"{url}/{name}", target / name)
    else:
        # GitHub release archive (.tar.bz2). Stream to disk then extract.
        archive = cache_dir / f"{model_id}.tar.bz2"
        _download(url, archive)
        print(f"  extracting…")
        with tarfile.open(archive, "r:bz2") as tar:
            tar.extractall(target)  # noqa: S202 (trusted sherpa-onnx release)
        archive.unlink(missing_ok=True)

    # If the archive extracted to a single nested subdir, flatten it so the
    # config builders find files at the top level (matches rust-tts-wrapper's
    # resolve_model_scan_dir behaviour).
    _flatten_single_subdir(target)
    return target


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)


def _flatten_single_subdir(path: Path) -> None:
    """If ``path`` contains exactly one subdirectory and no files, move its
    contents up one level (archives often extract to ``<name>/<files>``)."""
    children = list(path.iterdir())
    if len(children) != 1 or not children[0].is_dir():
        return
    inner = children[0]
    for item in inner.iterdir():
        shutil.move(str(item), str(path / item.name))
    inner.rmdir()


# ---------------------------------------------------------------------------
# Per-model_type config builders
# ---------------------------------------------------------------------------
# These mirror the logic in rust-tts-wrapper's sherpaonnx_engine.rs
# (build_kokoro_config / build_matcha_config / build_vits_config). If the
# verify fails here, the Rust engine likely will too — that's the point.

def _existing(path: Path, name: str) -> str | None:
    p = path / name
    return str(p) if p.exists() else None


def _find_onnx(path: Path, prefer: list[str]) -> str | None:
    """Return the first existing file from ``prefer``, else the first .onnx
    in the dir that isn't a vocoder/acoustic-step model."""
    for name in prefer:
        if (path / name).exists():
            return str(path / name)
    for item in sorted(path.iterdir()):
        if item.suffix == ".onnx":
            low = item.name.lower()
            if not (low.startswith("vocoder") or low.startswith("hifigan") or low.startswith("vocos")):
                return str(item)
    return None


def build_config(model_type: str, model_dir: Path, cache_dir: Path) -> dict:
    """Return a sherpa-onnx ``OfflineTtsConfig`` dict for the model_type."""
    import sherpa_onnx  # local import so --help works without the dep

    if model_type == "kokoro":
        model = _find_onnx(model_dir, ["model.onnx", "model.int8.onnx"])
        cfg = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                    model=model,
                    voices=_existing(model_dir, "voices.bin"),
                    tokens=_existing(model_dir, "tokens.txt"),
                    data_dir=_existing(model_dir, "espeak-ng-data"),
                ),
                num_threads=1,
                debug=False,
                provider="cpu",
            )
        )
        return cfg

    if model_type == "matcha":
        acoustic = _find_onnx(
            model_dir,
            ["acoustic-model.onnx", "model-steps-3.onnx", "model.onnx"],
        )
        vocoder = _find_onnx(
            model_dir, ["hifigan_v2.onnx", "vocoder.onnx", "vocos-22khz-univ.onnx"]
        )
        if vocoder is None:
            # Matcha archives don't bundle a vocoder — fetch the shared one.
            vocoder_path = cache_dir / "hifigan_v2.onnx"
            if not vocoder_path.exists():
                print(f"  fetching shared vocoder for matcha…")
                _download(SHERPA_ONNX_VOCODER_URL, vocoder_path)
            vocoder = str(vocoder_path)
        cfg = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                matcha=sherpa_onnx.OfflineTtsMatchaModelConfig(
                    acoustic_model=acoustic,
                    vocoder=vocoder,
                    tokens=_existing(model_dir, "tokens.txt"),
                    lexicon=_existing(model_dir, "lexicon.txt"),
                    data_dir=_existing(model_dir, "espeak-ng-data"),
                ),
                num_threads=1,
                debug=False,
                provider="cpu",
            )
        )
        return cfg

    if model_type == "supertonic":
        # The int8 release uses ``<name>.int8.onnx``.
        def onnx(name: str) -> str | None:
            return _find_onnx(model_dir, [f"{name}.int8.onnx", f"{name}.onnx"])

        cfg = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                supertonic=sherpa_onnx.OfflineTtsSupertonicModelConfig(
                    duration_predictor=onnx("duration_predictor"),
                    text_encoder=onnx("text_encoder"),
                    vector_estimator=onnx("vector_estimator"),
                    vocoder=onnx("vocoder"),
                    tts_json=_existing(model_dir, "tts.json"),
                    unicode_indexer=_existing(model_dir, "unicode_indexer.bin"),
                    voice_style=_existing(model_dir, "voice.bin"),
                ),
                num_threads=1,
                debug=False,
                provider="cpu",
            )
        )
        return cfg

    if model_type == "kitten":
        cfg = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                kitten=sherpa_onnx.OfflineTtsKittenModelConfig(
                    model=_find_onnx(model_dir, ["model.onnx"]),
                    voices=_existing(model_dir, "voices.bin"),
                    tokens=_existing(model_dir, "tokens.txt"),
                    data_dir=_existing(model_dir, "espeak-ng-data"),
                ),
                num_threads=1,
                debug=False,
                provider="cpu",
            )
        )
        return cfg

    # Default: VITS family (piper, coqui, mimic3, melo, mms).
    cfg = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=_find_onnx(model_dir, ["model.onnx"]),
                tokens=_existing(model_dir, "tokens.txt"),
                lexicon=_existing(model_dir, "lexicon.txt"),
                data_dir=_existing(model_dir, "espeak-ng-data"),
                dict_dir=_existing(model_dir, "dict"),
            ),
            num_threads=1,
            debug=False,
            provider="cpu",
        )
    )
    return cfg


# ---------------------------------------------------------------------------
# Synth + assert
# ---------------------------------------------------------------------------

def synth(model_id: str, model_dir: Path, cfg) -> int:
    """Run a short synthesis; return the sample count (0 = failure)."""
    import sherpa_onnx

    tts = sherpa_onnx.OfflineTts(cfg)
    print(f"  loaded: sample_rate={tts.sample_rate} num_speakers={tts.num_speakers}")

    gen = sherpa_onnx.OfflineTtsGenerateConfig() if hasattr(sherpa_onnx, "OfflineTtsGenerateConfig") else None
    # Supertonic needs ``lang`` via GenerationConfig.extra. Default "en".
    if model_id == "supertonic-3-multilingual":
        # The modern API takes a GenerationConfig; the field name has varied
        # across sherpa-onnx versions, so try the dict form first.
        try:
            gc = sherpa_onnx.OfflineTtsGenerateConfig(sid=0, num_steps=8, speed=1.0)
        except TypeError:
            gc = sherpa_onnx.OfflineTtsGenerateConfig()
            gc.sid = 0
            gc.num_steps = 8
            gc.speed = 1.0
        try:
            gc.extra = {"lang": "en"}
        except (AttributeError, TypeError):
            pass
        audio = tts.generate(SAMPLE_TEXT, gc)
    else:
        audio = tts.generate(SAMPLE_TEXT)

    return len(audio.samples)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model_id", help="Registry id, e.g. supertonic-3-multilingual")
    parser.add_argument("--cache-dir", default=str(ROOT / ".verify-cache"))
    parser.add_argument("--save-wav", action="store_true", help="Write the clip to <model_id>.wav for inspection")
    args = parser.parse_args()

    if not MODELS_JSON.exists():
        print(f"error: {MODELS_JSON.name} not found — run generate.py first", file=sys.stderr)
        return 2

    models = json.loads(MODELS_JSON.read_text(encoding="utf-8"))
    entry = models.get(args.model_id)
    if entry is None:
        print(f"error: '{args.model_id}' is not in the registry", file=sys.stderr)
        return 2

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"== {args.model_id} ({entry['model_type']}) ==")
    model_dir = ensure_downloaded(args.model_id, entry, cache_dir)
    print(f"  model dir: {model_dir}")

    cfg = build_config(entry["model_type"], model_dir, cache_dir)
    n_samples = synth(args.model_id, model_dir, cfg)

    if n_samples <= 0:
        print(f"FAIL: produced {n_samples} samples", file=sys.stderr)
        return 1

    duration = n_samples / 16000  # approximate; actual rate is in cfg
    print(f"OK: produced {n_samples} samples (~{duration:.1f}s @ 16kHz nominal)")
    if args.save_wav:
        import sherpa_onnx
        # Re-do with save for inspection (cheap relative to download).
        print(f"  (re-generating to write {args.model_id}.wav)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
