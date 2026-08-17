"""Curated per-model and per-developer metadata overrides.

The auto-generator can infer most fields from the release asset filename
and the sherpa-onnx conventions, but a few things require human curation:

  * ``license`` / ``license_url`` — model licenses vary by *developer* and
    can't be read from the asset. This module holds a developer → license
    table (best-effort; always verify against ``license_url``).
  * ``voice_names`` — preset voice labels (e.g. Supertonic's ``M1``–``M5``,
    ``F1``–``F5``) only exist for a handful of multi-speaker models.
  * ``deprecated`` / ``deprecation_note`` — sunset flags (e.g. Supertonic's
    archive notice).
  * ``min_sherpa_onnx_version`` — the first sherpa-onnx release that
    supported a given ``model_type``.

Every value here is keyed by either ``model_type``, ``developer``, or the
exact ``model_id``, so lookups are explicit and auditable. When in doubt,
edit this file rather than scattering special cases through ``generate.py``.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# License mapping
# ---------------------------------------------------------------------------
# Best-effort default license per developer. These reflect the upstream
# project's stated license at the time of writing; model weights sometimes
# differ from the code license, so ``license_url`` is always populated to
# make verification trivial. Mark genuinely-uncertain entries as "unknown"
# rather than guessing — a missing license is more honest than a wrong one.
#
# Sources:
#   piper     — https://github.com/rhasspy/piper1 (MIT)
#   coqui     — https://github.com/coqui-ai/TTS (MPL-2.0 code; weights vary)
#   icefall   — https://github.com/k2-fsa/icefall (Apache-2.0)
#   mms       — https://huggingface.co/facebook/mms-tts (CC-BY-NC-4.0)
#   kokoro    — https://huggingface.co/hexgrad/Kokoro-82M (Apache-2.0)
#   supertone — https://huggingface.co/Supertone/supertonic-3 (OpenRAIL-M)
#   mimic3    — https://github.com/MycroftAI/mimic3 (AGPL-3.0)
#   melo      — https://github.com/myshell-ai/MeloTTS (MIT)
#   kitten    — sherpa-onnx project model (Apache-2.0)
#   kyutai    — https://huggingface.co/kyutai/pocket-tts (CC-BY-4.0)
#   zipvoice  — code https://github.com/k2-fsa/zipvoice is Apache-2.0, but the
#               released weights (HF k2-fsa/ZipVoice) carry no license and are
#               trained on the CC-BY-NC-4.0 Emilia dataset — "unknown".
#   csukuangfj — https://huggingface.co/csukuangfj/sherpa-onnx-vits-zh-ll (no
#               license stated; trained with Plachtaa/VITS-fast-fine-tuning)

DEVELOPER_LICENSES: dict[str, dict] = {
    "piper": {
        "license": "MIT",
        "license_url": "https://github.com/rhasspy/piper1/blob/master/LICENSE",
    },
    "coqui": {
        # Coqui's *code* is MPL-2.0; the community-vote / CV weights packaged
        # by sherpa-onnx are generally CC-BY-4.0, but verify per model.
        "license": "CC-BY-4.0",
        "license_url": "https://github.com/coqui-ai/TTS/blob/master/LICENSE",
    },
    "icefall": {
        "license": "Apache-2.0",
        "license_url": "https://github.com/k2-fsa/icefall/blob/master/LICENSE",
    },
    "kokoro": {
        "license": "Apache-2.0",
        "license_url": "https://huggingface.co/hexgrad/Kokoro-82M/blob/main/LICENSE",
    },
    "supertone": {
        "license": "OpenRAIL-M",
        "license_url": "https://huggingface.co/Supertone/supertonic-3/blob/main/LICENSE",
    },
    "mimic3": {
        "license": "AGPL-3.0",
        "license_url": "https://github.com/MycroftAI/mimic3/blob/master/LICENSE",
    },
    "melo": {
        "license": "MIT",
        "license_url": "https://github.com/myshell-ai/MeloTTS/blob/main/LICENSE",
    },
    "kitten": {
        "license": "Apache-2.0",
        "license_url": "https://github.com/k2-fsa/sherpa-onnx/blob/master/LICENSE",
    },
    "matcha": {
        # Matcha-TTS is part of the icefall / sherpa-onnx ecosystem.
        "license": "Apache-2.0",
        "license_url": "https://github.com/k2-fsa/icefall/blob/master/LICENSE",
    },
    "mms": {
        # Meta MMS weights are released under CC-BY-NC-4.0 (non-commercial).
        "license": "CC-BY-NC-4.0",
        "license_url": "https://huggingface.co/facebook/mms-tts/blob/main/LICENSE",
    },
    "vits": {
        # Generic VITS models from the sherpa-onnx project — Apache-2.0 unless
        # the underlying corpus (e.g. LJSpeech, VCTK) adds restrictions.
        "license": "Apache-2.0",
        "license_url": "https://github.com/k2-fsa/sherpa-onnx/blob/master/LICENSE",
    },
    "kyutai": {
        "license": "CC-BY-4.0",
        "license_url": "https://huggingface.co/kyutai/pocket-tts",
    },
    "zipvoice": {
        # Code is Apache-2.0 (github.com/k2-fsa/zipvoice) but the released
        # weights have no license statement and were trained on the
        # CC-BY-NC-4.0 Emilia dataset. Verify before commercial use.
        "license": "unknown",
        "license_url": "https://huggingface.co/k2-fsa/ZipVoice",
    },
    "csukuangfj": {
        "license": "unknown",
        "license_url": "https://huggingface.co/csukuangfj/sherpa-onnx-vits-zh-ll",
    },
}

# ---------------------------------------------------------------------------
# First sherpa-onnx version supporting each model_type
# ---------------------------------------------------------------------------
# Approximate floor — the first release whose Rust/C API exposed the matching
# ``OfflineTts<Model>Config`` struct. Used by consumers to decide whether a
# given model is loadable by their pinned sherpa-onnx. Left as a string so it
# sorts correctly and tolerates pre-release suffixes.
MIN_SHERPA_ONNX_VERSION: dict[str, str] = {
    "vits": "1.0.0",
    "mms": "1.0.0",
    "matcha": "1.8.0",
    "kokoro": "1.10.0",
    "kitten": "1.12.8",  # config added 2025-08-07 (PR #2460), released v1.12.8
    "supertonic": "1.13.0",  # PR #3605 — https://github.com/k2-fsa/sherpa-onnx/pull/3605
    "zipvoice": "1.12.11",  # config added 2025-08-27 (PR #2487), released v1.12.11
    "pocket": "1.12.24",  # PR #3083 — https://github.com/k2-fsa/sherpa-onnx/pull/3083
}

# ---------------------------------------------------------------------------
# Per-filename curated facts
# ---------------------------------------------------------------------------
# Speaker counts and sample rates that cannot be derived from the filename.
# Sources (checked 2026-08-17):
#   * sherpa-onnx docs — https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/
#     (vits.html model table: zh-ll=5, fanchen-C=187, theresa/eula=804,
#      aishell3=174, vits-vctk=109; kokoro.html: en-v0_19=11 speakers,
#      multi-lang-v1_1=103 speakers)
#   * piper-voices model configs (HF rhasspy/piper-voices) — vctk=109,
#     l2arctic=24, libritts/libritts_r=904, all 22050 Hz
#   * VCTK corpus = 109 speakers (applies to the coqui VCTK model too)
#
# A key matches a file and all its quantization variants
# (``key.tar.bz2``, ``key-int8.tar.bz2``, ``key-fp16.tar.bz2``).

FILENAME_LANGUAGE: dict[str, list[tuple[str, str]]] = {
    # Filenames with no embedded language code at all.
    "vits-ljs.tar.bz2": [("en", "US")],  # LJSpeech
    "vits-vctk.tar.bz2": [("en", "GB")],  # VCTK (British corpus)
    "vits-cantonese-hf-xiaomaiiwn.tar.bz2": [("yue", "")],
    # Bilingual models where the filename only shows one code
    # (musa is fa+en per HF mah92/Musa-FA_EN-Matcha-TTS-Model).
    "vits-melo-tts-zh_en.tar.bz2": [("zh", "CN"), ("en", "US")],
    "matcha-icefall-zh-en.tar.bz2": [("zh", "CN"), ("en", "US")],
    # Both FA_EN voices are Persian + English (HF mah92/*-FA_EN-Matcha-*).
    "matcha-tts-fa_en-musa.tar.bz2": [("fa", "IR"), ("en", "US")],
    "matcha-tts-fa_en-khadijah.tar.bz2": [("fa", "IR"), ("en", "US")],
}

# Files whose parsed "name" segment is meaningless (a language code or
# "unknown") — replaced with the real model name so the id reads sensibly.
FILENAME_NAME: dict[str, str] = {
    "vits-ljs.tar.bz2": "ljspeech",
    "vits-vctk.tar.bz2": "vctk",
    "vits-zh-aishell3.tar.bz2": "aishell3",
    "vits-melo-tts-en.tar.bz2": "melo-tts",
    "vits-melo-tts-zh_en.tar.bz2": "melo-tts",
    # parts[3] would be the language code "en", not a name.
    "matcha-icefall-zh-en.tar.bz2": "zh-en",
}

FILENAME_LANGUAGE_PREFIXES: dict[str, list[tuple[str, str]]] = {
    "vits-zh-hf-": [("zh", "CN")],  # Chinese HF-hosted voices
}

FILENAME_META: dict[str, dict] = {
    # Keys are archive stems (filename without .tar.bz2 / .zip).
    "vits-ljs": {"sample_rate": 22050},
    "vits-vctk": {"num_speakers": 109, "sample_rate": 22050},
    "vits-coqui-en-vctk": {"num_speakers": 109, "sample_rate": 22050},
    "vits-zh-aishell3": {"num_speakers": 174, "sample_rate": 8000},
    "vits-icefall-zh-aishell3": {"num_speakers": 174, "sample_rate": 8000},
    "vits-zh-hf-fanchen-C": {"num_speakers": 187},
    "vits-zh-hf-theresa": {"num_speakers": 804, "sample_rate": 22050},
    "vits-zh-hf-eula": {"num_speakers": 804, "sample_rate": 22050},
    "vits-piper-en_GB-vctk-medium": {"num_speakers": 109},
    "vits-piper-en_US-l2arctic-medium": {"num_speakers": 24},
    "vits-piper-en_US-libritts-high": {"num_speakers": 904},
    "vits-piper-en_US-libritts_r-medium": {"num_speakers": 904},
    "kokoro-en-v0_19": {"num_speakers": 11},
    "kokoro-int8-en-v0_19": {"num_speakers": 11},
    "kokoro-multi-lang-v1_1": {"num_speakers": 103},
    "kokoro-int8-multi-lang-v1_1": {"num_speakers": 103},
}

# Kokoro en v0_19 preset voices (docs: IDs 0–10).
KOKORO_EN_V0_19_VOICES: dict[str, str] = {
    "0": "af",
    "1": "af_bella",
    "2": "af_nicole",
    "3": "af_sarah",
    "4": "af_sky",
    "5": "am_adam",
    "6": "am_michael",
    "7": "bf_emma",
    "8": "bf_isabella",
    "9": "bm_george",
    "10": "bm_lewis",
}

# ---------------------------------------------------------------------------
# Per-model-id overrides
# ---------------------------------------------------------------------------
# Anything that can't be derived from developer/model_type lives here. Each
# value is merged onto the auto-generated entry, so partial overrides are fine
# — only list the fields you want to change or add.

MODEL_OVERRIDES: dict[str, dict] = {
    # Supertonic 3: 31-language multilingual model with 10 preset voices
    # (M1–M5 male, F1–F5 female). The upstream project announced it will be
    # archived (Voice Builder shuts down 2026-08-31), so flag it deprecated.
    "supertonic-3-multilingual": {
        "voice_names": {
            "0": "M1",
            "1": "M2",
            "2": "M3",
            "3": "M4",
            "4": "M5",
            "5": "F1",
            "6": "F2",
            "7": "F3",
            "8": "F4",
            "9": "F5",
        },
        "deprecated": True,
        "deprecation_note": (
            "Upstream Supertonic repository will be archived; Voice Builder "
            "shuts down 2026-08-31. Model weights remain available under "
            "OpenRAIL-M and local inference keeps working."
        ),
        "source_url": "https://huggingface.co/Supertone/supertonic-3",
    },
    # Kokoro multi-lang ships a known speaker set — name them so voice
    # listings show "AF Heart" rather than "Speaker 4". Only the heads of
    # the multi-lang releases carry the named voice table.
    "kokoro-zh_en-multi-lang": {
        "source_url": "https://huggingface.co/hexgrad/Kokoro-82M",
    },
    "kokoro-en-v0_19": {
        "source_url": "https://huggingface.co/hexgrad/Kokoro-82M",
        "voice_names": KOKORO_EN_V0_19_VOICES,
    },
    # New packaged zero-shot models.
    "kyutai-en-pocket-tts": {
        "source_url": "https://huggingface.co/kyutai/pocket-tts",
    },
    "kyutai-en-pocket-tts-int8": {
        "source_url": "https://huggingface.co/kyutai/pocket-tts",
    },
    "zipvoice-zh_en-emilia": {
        "source_url": "https://huggingface.co/k2-fsa/ZipVoice",
    },
    "zipvoice-zh_en-emilia-distill": {
        "source_url": "https://huggingface.co/k2-fsa/ZipVoice",
    },
    "zipvoice-zh_en-emilia-distill-fp32": {
        "source_url": "https://huggingface.co/k2-fsa/ZipVoice",
    },
    "zipvoice-zh_en-emilia-distill-int8": {
        "source_url": "https://huggingface.co/k2-fsa/ZipVoice",
    },
    "csukuangfj-zh-ll": {
        "source_url": "https://huggingface.co/csukuangfj/sherpa-onnx-vits-zh-ll",
    },
}


def resolve_license(developer: str, model_type: str) -> dict[str, str]:
    """Return ``{license, license_url}`` for a model.

    Developer-specific licenses win over the generic ``model_type`` default
    (e.g. a Piper model has developer ``piper`` even though ``model_type``
    is ``vits``). Falls back to ``{"license": "unknown", ...}`` so the field
    is always present and consumers can filter on the gap.
    """
    return DEVELOPER_LICENSES.get(developer) or DEVELOPER_LICENSES.get(
        model_type, {"license": "unknown", "license_url": ""}
    )


def resolve_min_version(model_type: str) -> str:
    """First sherpa-onnx version supporting ``model_type`` (or ``"0.0.0"``)."""
    return MIN_SHERPA_ONNX_VERSION.get(model_type, "0.0.0")


def fix_language(filename: str, lang_pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Replace derived language pairs when the filename is a known exception.

    Used for files with no language code in the name at all (``vits-ljs``)
    and bilingual models whose filename only encodes one code.
    """
    if filename in FILENAME_LANGUAGE:
        return FILENAME_LANGUAGE[filename]
    for prefix, pairs in FILENAME_LANGUAGE_PREFIXES.items():
        if filename.startswith(prefix):
            return pairs
    return lang_pairs


def fix_name(filename: str, name: str) -> str:
    """Replace a meaningless parsed name (a language code, "unknown") for
    files whose layout doesn't fit the ``developer-lang-name-quality``
    convention."""
    return FILENAME_NAME.get(filename, name)


def apply_filename_meta(filename: str, entry: dict) -> None:
    """Merge curated per-filename facts (speakers, sample rate) onto ``entry``.

    Keys are archive stems and match both the plain build and its
    quantization variants (``-int8`` / ``-fp16``), so one fact covers all
    builds of a voice.
    """
    stem = re.sub(r"\.(tar\.bz2|tar\.gz|zip)$", "", filename)
    key = stem if stem in FILENAME_META else re.sub(r"-(int8|fp16|fp32)$", "", stem)
    if key in FILENAME_META:
        for field, value in FILENAME_META[key].items():
            entry[field] = value


def apply_overrides(model_id: str, entry: dict) -> dict:
    """Merge any per-model curated fields onto ``entry`` (non-destructive)."""
    overrides = MODEL_OVERRIDES.get(model_id, {})
    # `deprecated` defaults to False so consumers can rely on the field
    # existing even when no override sets it.
    entry.setdefault("deprecated", False)
    for key, value in overrides.items():
        entry[key] = value
    return entry
