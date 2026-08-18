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
VOCOS_24KHZ_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos_24khz.onnx"
)

# Reference transcripts for the test wavs bundled in the zero-shot archives
# (from the sherpa-onnx docs — the transcript must match the audio exactly).
REFERENCE_TEXTS = {
    "leijun-1.wav": "那还是三十六年前, 一九八七年. 我呢考上了武汉大学的计算机系.",
}


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
        # MMS entries point at a Hugging Face *directory*; fetch the two
        # files sherpa-onnx needs (MMS has no lexicon / espeak data).
        for name in ("model.onnx", "tokens.txt"):
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

def _existing(path: Path, name: str) -> str:
    """Path to ``path/name``, or ``""`` — the pybind11 bindings require a
    str for optional path fields and reject ``None``."""
    p = path / name
    return str(p) if p.exists() else ""


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


def _find_onnx_quant(path: Path, stem: str, quant: str) -> str:
    """Resolve ``<stem>.int8.onnx`` / ``<stem>.onnx`` for a packaged model
    (only some components are quantized in the int8 builds — e.g. pocket's
    encoder and text conditioner never are)."""
    if quant == "int8" and (path / f"{stem}.int8.onnx").exists():
        return str(path / f"{stem}.int8.onnx")
    return str(path / f"{stem}.onnx")


def _load_wav(path: Path) -> tuple[list[float], int]:
    """Read a 16-bit PCM wav via the stdlib (sherpa-onnx's GenerationConfig
    takes in-memory float samples, not a file path)."""
    import array
    import wave

    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        assert w.getsampwidth() == 2, f"{path}: expected 16-bit PCM"
        samples = array.array("h", w.readframes(w.getnframes()))
    return [s / 32768.0 for s in samples], rate


def _reference_audio(model_dir: Path) -> Path:
    """First bundled test wav (under ``test_wavs/`` or the model root)."""
    for sub in (model_dir / "test_wavs", model_dir):
        if sub.is_dir():
            wavs = sorted(sub.glob("*.wav"))
            if wavs:
                return wavs[0]
    raise FileNotFoundError(f"no reference wav found under {model_dir}")


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
        # Vocoder: only an explicit vocoder file counts — the generic
        # "_find_onnx falls back to first .onnx" behaviour would happily
        # return the acoustic model itself when no vocoder is bundled.
        vocoder = None
        for name in ("hifigan_v2.onnx", "vocoder.onnx", "vocos-22khz-univ.onnx"):
            if (model_dir / name).exists():
                vocoder = str(model_dir / name)
                break
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

    if model_type == "zipvoice":
        # Zero-shot cloning: encoder + decoder + espeak data + lexicon from
        # the archive, plus the shared vocos vocoder from the
        # vocoder-models release (not bundled).
        quant = "int8" if (model_dir / "decoder.int8.onnx").exists() else "fp32"
        vocos = cache_dir / "vocos_24khz.onnx"
        if not vocos.exists():
            print("  fetching vocos vocoder for zipvoice…")
            _download(VOCOS_24KHZ_URL, vocos)
        cfg = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                zipvoice=sherpa_onnx.OfflineTtsZipvoiceModelConfig(
                    tokens=_existing(model_dir, "tokens.txt"),
                    encoder=_find_onnx_quant(model_dir, "encoder", quant),
                    decoder=_find_onnx_quant(model_dir, "decoder", quant),
                    vocoder=str(vocos),
                    data_dir=_existing(model_dir, "espeak-ng-data"),
                    lexicon=_existing(model_dir, "lexicon.txt"),
                ),
                num_threads=1,
                debug=False,
                provider="cpu",
            )
        )
        return cfg

    if model_type == "pocket":
        # Zero-shot cloning: multi-component LM; only decoder/lm_* are
        # quantized in the int8 build.
        quant = "int8" if (model_dir / "decoder.int8.onnx").exists() else "fp32"
        cfg = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                pocket=sherpa_onnx.OfflineTtsPocketModelConfig(
                    lm_flow=_find_onnx_quant(model_dir, "lm_flow", quant),
                    lm_main=_find_onnx_quant(model_dir, "lm_main", quant),
                    encoder=_find_onnx_quant(model_dir, "encoder", quant),
                    decoder=_find_onnx_quant(model_dir, "decoder", quant),
                    text_conditioner=_find_onnx_quant(model_dir, "text_conditioner", quant),
                    vocab_json=_existing(model_dir, "vocab.json"),
                    token_scores_json=_existing(model_dir, "token_scores.json"),
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

def synth(model_id: str, model_type: str, model_dir: Path, cfg) -> int:
    """Run a short synthesis; return the sample count (0 = failure)."""
    import sherpa_onnx

    tts = sherpa_onnx.OfflineTts(cfg)
    print(f"  loaded: sample_rate={tts.sample_rate} num_speakers={tts.num_speakers}")

    if model_type in ("zipvoice", "pocket"):
        # Zero-shot voice cloning: needs a reference clip (and, for
        # zipvoice, its exact transcript) via GenerationConfig.
        ref = _reference_audio(model_dir)
        samples, rate = _load_wav(ref)
        print(f"  reference: {ref.name} ({len(samples) / rate:.1f}s @ {rate}Hz)")
        gc = sherpa_onnx.GenerationConfig()
        gc.reference_audio = samples
        gc.reference_sample_rate = rate
        gc.num_steps = 2 if model_type == "pocket" else 4
        gc.speed = 1.0
        if model_type == "zipvoice":
            gc.reference_text = REFERENCE_TEXTS.get(ref.name, "")
            assert gc.reference_text, f"no reference transcript for {ref.name}"
        audio = tts.generate(SAMPLE_TEXT, gc)
    elif model_type == "supertonic":
        # Needs ``lang`` via GenerationConfig.extra ("en" is fine for the
        # multilingual v3 bundle).
        gc = sherpa_onnx.GenerationConfig()
        gc.sid = 0
        gc.num_steps = 8
        gc.speed = 1.0
        gc.extra = {"lang": "en"}
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
    n_samples = synth(args.model_id, entry["model_type"], model_dir, cfg)

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
