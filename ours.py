"""Curated AACTools-trained artifacts appended to the registry.

Everything here is ours (trained/patched by AACTools) and lives on
Hugging Face. Entries are merged AFTER the upstream fetches so a
regeneration never loses them (they'd also survive via the
preserve-existing pass, but being explicit keeps the source auditable).

Sources of truth:
  - teachers/students: https://huggingface.co/willwade/piper-voices-patched
  - patched MMS set:   https://huggingface.co/willwade/mms-tts-multilingual-models-onnx
  - phoneme pocket:    https://huggingface.co/willwade/pocket-tts-en-phonemes-24l
"""

from __future__ import annotations

PP = "https://huggingface.co/willwade/piper-voices-patched/resolve/main"
MMS = "https://huggingface.co/willwade/mms-tts-multilingual-models-onnx"
PK = "https://huggingface.co/willwade/pocket-tts-en-phonemes-24l/resolve/main/onnx"

_PATCHED_TEACHERS = [
    # file stem (upstream voice name), lang code, language name, country
    ("en_GB-alan-medium", "en-GB", "English", "GB"),
    ("en_GB-alba-medium", "en-GB", "English", "GB"),
    ("en_GB-northern_english_male-medium", "en-GB", "English", "GB"),
    ("en_US-hfc_male-medium", "en-US", "English", "US"),
    ("en_US-kristin-medium", "en-US", "English", "US"),
    ("en_US-ryan-high", "en-US", "English", "US"),
    ("cy_GB-bu_tts-medium", "cy", "Welsh", "GB"),
    ("fa_IR-amir-medium", "fa", "Persian", "IR"),
]

_FLEET_STUDENTS = [
    ("alan", "en-GB", "English", "GB"),
    ("alba", "en-GB", "English", "GB"),
    ("northern_english_male", "en-GB", "English", "GB"),
    ("kristin", "en-US", "English", "US"),
    ("hfc_male", "en-US", "English", "US"),
    ("ryan", "en-US", "English", "US"),
]

_RESEARCH_STUDENTS = [
    # (stem, files-suffix, lang, description)
    ("engdict2", ("acoustic", "decoder"), "en", "fleet-best English student (dict pack, b4)"),
    ("eng-a96", ("acoustic", "decoder"), "en", "hidden-96 English student"),
    ("enginter", ("acoustic", "decoder", "duration"), "en", "interleaved-convention English student"),
    ("guj-student", ("acoustic", "decoder"), "gu", "Gujarati MMS-teacher student pilot"),
]


def our_models() -> dict[str, dict]:
    out: dict[str, dict] = {}

    # --- patched piper teachers (durations-output tap; floravox training source)
    for stem, lang, lname, country in _PATCHED_TEACHERS:
        out[f"willwade-patched-{stem}"] = {
            "id": f"willwade-patched-{stem}",
            "model_type": "vits",
            "developer": "willwade",
            "name": stem,
            "language": [{"lang_code": lang, "language_name": lname, "country": country}],
            "quality": "verified",
            "quantization": "fp32",
            "sample_rate": 22050,
            "num_speakers": 1,
            "url": f"{PP}/{stem}.onnx",
            "compression": False,
            "description": "Piper voice re-exported with the floravox durations-output tap (enables measured word timings); pairs with students/* in the same repo",
            "license": "MIT",
            "license_url": "https://github.com/rhasspy/piper",
            "engines": "sherpa-onnx,floravox",
            "source_url": "https://huggingface.co/willwade/piper-voices-patched",
            "tags": ["patched", "floravox", "teacher"],
            "deprecated": False,
        }

    # --- fleet students (floravox student format: acoustic+decoder+duration)
    for voice, lang, lname, country in _FLEET_STUDENTS:
        files = ", ".join(
            f"{PP}/students/fleet-{voice}-{part}.onnx"
            for part in ("acoustic", "decoder", "duration")
        )
        out[f"willwade-student-fleet-{voice}"] = {
            "id": f"willwade-student-fleet-{voice}",
            "model_type": "floravox-student",
            "developer": "willwade",
            "name": f"fleet-{voice}",
            "language": [{"lang_code": lang, "language_name": lname, "country": country}],
            "quality": "verified",
            "quantization": "fp32",
            "sample_rate": 22050,
            "num_speakers": 1,
            "url": f"{PP}/students/fleet-{voice}-acoustic.onnx",
            "compression": False,
            "description": f"floravox student voice '{voice}': acoustic+decoder+duration ONNX triple distilled from the patched piper teacher; v2 warm-started decoders, ASR-verbatim, ear-approved",
            "license": "MIT",
            "license_url": "https://github.com/rhasspy/piper",
            "engines": "floravox",
            "source_url": "https://huggingface.co/willwade/piper-voices-patched",
            "tags": ["floravox", "student", "word-timings"],
            "deprecated": False,
        }

    # --- research/pilot students
    for stem, parts, lang, desc in _RESEARCH_STUDENTS:
        out[f"willwade-student-{stem}"] = {
            "id": f"willwade-student-{stem}",
            "model_type": "floravox-student",
            "developer": "willwade",
            "name": stem,
            "language": [{"lang_code": lang, "language_name": "English" if lang == "en" else "Gujarati", "country": "GB" if lang == "en" else "IN"}],
            "quality": "experimental",
            "quantization": "fp32",
            "sample_rate": 22050,
            "num_speakers": 1,
            "url": f"{PP}/students/{stem}-{parts[0]}.onnx",
            "compression": False,
            "description": f"floravox student ({desc})",
            "license": "MIT",
            "engines": "floravox",
            "source_url": "https://huggingface.co/willwade/piper-voices-patched",
            "tags": ["floravox", "student", "research"],
            "deprecated": False,
        }

    # --- patched MMS umbrella (1,138 voices with the durations tap)
    out["willwade-mms-patched"] = {
        "id": "willwade-mms-patched",
        "model_type": "mms",
        "developer": "willwade",
        "name": "mms-tts-multilingual-models-onnx",
        "language": [{"lang_code": "mul", "language_name": "Multilingual", "country": ""}],
        "quality": "verified",
        "quantization": "fp32",
        "sample_rate": 16000,
        "num_speakers": 1138,
        "url": MMS,
        "compression": False,
        "description": "Meta MMS TTS voices re-exported with the floravox durations-output tap (per-voice dirs: model.onnx/tokens.txt/sample.wav, iso-639-3 keys matching the registry's mms_* ids); the coverage tier (~1,100 languages)",
        "license": "CC-BY-4.0",
        "license_url": "https://huggingface.co/willwade/mms-tts-multilingual-models-onnx",
        "engines": "sherpa-onnx,floravox",
        "source_url": MMS,
        "tags": ["patched", "floravox", "mms", "coverage-tier"],
        "deprecated": False,
    }

    # --- phoneme pocket-tts (IPA phoneme input + attention timings)
    out["willwade-pocket-phoneme-en-24l"] = {
        "id": "willwade-pocket-phoneme-en-24l",
        "model_type": "pocket",
        "developer": "willwade",
        "name": "pocket-tts-en-phonemes-24l",
        "language": [{"lang_code": "en", "language_name": "English", "country": "US"}],
        "quality": "verified",
        "quantization": "fp32",
        "sample_rate": 24000,
        "num_speakers": 1,
        "url": f"{PK}/flow_lm_main_tapped.onnx",
        "compression": False,
        "filesize_mb": 1210,
        "description": "Kyutai pocket-tts 24L fine-tuned on 380h LibriTTS-R to read IPA phonemes natively: zero-shot voice cloning from any donor wav, attention-measured word boundaries (en-US phoneme inventory; en-GB subset). ONNX bundle in onnx/, safetensors at repo root. Driven by rust-tts-wrapper PocketTtsEngine (pocket-timing feature)",
        "license": "CC-BY-4.0",
        "license_url": "https://huggingface.co/willwade/pocket-tts-en-phonemes-24l",
        "engines": "rust-tts-wrapper",
        "source_url": "https://huggingface.co/willwade/pocket-tts-en-phonemes-24l",
        "tags": ["pocket", "phonemes", "voice-cloning", "word-timings", "floravox"],
        "deprecated": False,
    }

    return out
