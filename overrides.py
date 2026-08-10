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
    "kitten": "1.11.0",
    "supertonic": "1.13.0",  # PR #3605 — https://github.com/k2-fsa/sherpa-onnx/pull/3605
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
    "kokoro-en-en-19": {
        "source_url": "https://huggingface.co/hexgrad/Kokoro-82M",
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


def apply_overrides(model_id: str, entry: dict) -> dict:
    """Merge any per-model curated fields onto ``entry`` (non-destructive)."""
    overrides = MODEL_OVERRIDES.get(model_id, {})
    # `deprecated` defaults to False so consumers can rely on the field
    # existing even when no override sets it.
    entry.setdefault("deprecated", False)
    for key, value in overrides.items():
        entry[key] = value
    return entry
