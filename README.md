# sherpa-onnx-tts-models

A canonical, enriched registry of [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
text-to-speech models — the metadata layer that lets a TTS client list models,
show licenses, verify downloads, and pick a voice without hard-coding anything.

This is the shared successor to the `merged_models.json` files previously
vendored inside [rust-tts-wrapper](https://github.com/AACTools/rust-tts-wrapper)
and [tts-wrapper](https://github.com/willwade/tts-wrapper). One source of truth
for every wrapper to consume.

## What's in the box

| File | Purpose |
|------|---------|
| `models.json` | The registry — one entry per model, keyed by stable id. |
| `schema.json` | JSON Schema (draft 2020-12) documenting every field. |
| `generate.py` | Builds `models.json` from the sherpa-onnx GitHub release + the Meta MMS Hugging Face set. |
| `overrides.py` | Curated per-developer / per-model metadata (licenses, voice names, deprecation flags). |
| `validate.py` | Validates `models.json` against the schema + uniqueness + plausibility checks. |

A full refresh is two HTTP fetches (one GitHub release + one Hugging Face JSON)
and takes a few seconds — **no model files are downloaded**.

## Why a separate registry?

The sherpa-onnx project ships ~1,400 TTS models across a GitHub release and a
Hugging Face repo. Each model is just a tarball; the *metadata* that makes it
usable — which language, how many voices, what license, how big, which
sherpa-onnx version supports it, whether it's deprecated — is scattered across
filenames, READMEs, and model cards. This registry gathers it in one
machine-readable place so every TTS wrapper doesn't reinvent the same parsing.

## Schema

Every entry in `models.json` conforms to `schema.json`. The fields:

### Core (always present)
| Field | Type | Notes |
|-------|------|-------|
| `id` | string | Stable unique id, convention `<developer>-<lang>-<name>[-<quality>]`. Used as the registry key. |
| `model_type` | enum | `vits` \| `matcha` \| `kokoro` \| `kitten` \| `supertonic` \| `mms` — the sherpa-onnx `OfflineTtsModelConfig` branch. |
| `developer` | string | Upstream project (piper, coqui, icefall, mms, kokoro, supertone, mimic3, melo, kitten). |
| `name` | string | Variant name parsed from the release filename. |
| `language` | array | `{lang_code, language_name, country}` — one per supported language; multilingual models list all. |
| `quality` | string | Size/variant tag (low / medium / high / int8 / v0_19 / unknown). |
| `sample_rate` | int | Output audio rate in Hz (16000 / 22050 / 24000). |
| `num_speakers` | int | Number of preset voices addressable by speaker id. |
| `url` | string | Download URL for the archive (or HF resolve directory for MMS). |
| `compression` | bool | Whether `url` is a `.tar.bz2` / `.tar.gz` / `.zip`. |
| `filesize_mb` | number | Approximate download size in MiB. |

### Enriched (the reason this registry exists)
| Field | Type | Notes |
|-------|------|-------|
| `description` | string | One-line human-readable summary, auto-derived. |
| `license` | string | Best-effort license id (SPDX-style: `MIT`, `Apache-2.0`, `CC-BY-NC-4.0`, `OpenRAIL-M`, …). **`unknown` means verify manually — never assume.** |
| `license_url` | string | Link to the authoritative license text. |
| `sha256` | string | SHA-256 of the archive, from the GitHub release `digest` field. Absent for MMS. |
| `source_url` | string | Model card / upstream page for attribution. |
| `min_sherpa_onnx_version` | string | First sherpa-onnx release supporting this `model_type`. |
| `deprecated` | bool | `true` when the upstream is archived / no longer recommended. |
| `deprecation_note` | string | Why it's deprecated, and any migration path. |
| `voice_names` | object | Optional `{sid: name}` mapping (e.g. Supertonic `M1`–`M5`, `F1`–`F5`). |
| `tags` | array | Search facets: `on-device`, `multilingual`, `multi-speaker`, `int8`, … |
| `url_valid` | bool | Set by `--validate`; absent until the URL has been HEAD-checked. |

Consumers should treat any field beyond `id` / `model_type` / `url` as optional
and fall back gracefully — the schema is additive, and a field being absent
means "not known", not "false".

## Usage

### Regenerate the registry
```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python generate.py            # incremental (preserves manual edits)
.venv/bin/python generate.py --force    # full rebuild from scratch
```

### Validate
```bash
.venv/bin/python validate.py            # offline: schema + uniqueness + plausibility
.venv/bin/python validate.py --online   # also HEAD-check a sample of URLs
```

### Consume (Python)
```python
import json, urllib.request
reg = json.load(open("models.json"))
supertonic = reg["supertonic-3-multilingual"]
print(supertonic["license"])            # OpenRAIL-M
print(supertonic["voice_names"])        # {"0": "M1", "1": "M2", ...}
```

### Consume (Rust — rust-tts-wrapper)
rust-tts-wrapper already embeds a copy of this registry via
`include_str!("merged_models.json")`. To sync it with this canonical source,
copy `models.json` over `src/merged_models.json`:

```bash
cp models.json path/to/rust-tts-wrapper/src/merged_models.json
```

The enriched fields (`license`, `sha256`, `voice_names`, …) are ignored by the
current `parse_model` until you opt into them, so the copy is safe. (See the
rust-tts-wrapper README for the `SherpaModelInfo` struct.)

## Curating metadata

Auto-generation can't know a model's license or preset voice names. Those live
in [`overrides.py`](./overrides.py):

- **`DEVELOPER_LICENSES`** — default `{license, license_url}` per developer.
  Best-effort; `license_url` always points at the authoritative source so
  consumers can verify.
- **`MIN_SHERPA_ONNX_VERSION`** — first sherpa-onnx version per `model_type`.
- **`MODEL_OVERRIDES`** — per-model-id fields that can't be derived
  (`voice_names`, `deprecated`, `deprecation_note`, a specific `source_url`).

When in doubt, **edit `overrides.py`** rather than scattering special cases
through `generate.py` or hand-editing `models.json` (hand edits are lost on the
next `--force` rebuild).

## License caveats

The `license` field is **best-effort**. Model weights sometimes carry a
different license than the upstream codebase, and sherpa-onnx repackages models
from many sources. The registry populates `license_url` on every entry
specifically so you can verify before commercial use. Entries marked
`"unknown"` are genuinely unverified — treat them as "all rights reserved"
until you've checked the source.

Notable examples:
- **MMS** (`mms_*`) — `CC-BY-NC-4.0`, **non-commercial only**.
- **Supertonic** (`supertonic-*`) — `OpenRAIL-M` (use-based restrictions), and
  the upstream project is being archived (see `deprecation_note`).
- **Piper** (`piper-*`) — `MIT`.
- **Coqui** (`coqui-*`) — `CC-BY-4.0` (code is MPL-2.0); verify per model.

## License

The registry data and the scripts in this repository are released under
**Apache-2.0**. The *model weights* they describe are each governed by their
own license (see the `license` / `license_url` fields) — this repository
grants no rights over the models themselves, only over the metadata.
