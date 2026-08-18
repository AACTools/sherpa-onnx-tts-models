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

The sherpa-onnx project ships ~1,750 TTS models across a GitHub release and a
Hugging Face repo. Each model is just a tarball; the *metadata* that makes it
usable — which language, how many voices, what license, how big, which
sherpa-onnx version supports it, whether it's deprecated — is scattered across
filenames, READMEs, and model cards. This registry gathers it in one
machine-readable place so every TTS wrapper doesn't reinvent the same parsing.

## Coverage

Every TTS archive on the `tts-models` release is registered (622 of the 642
assets; the other 20 are deliberate skips: checksums, sample wavs, espeak
data, the Windows binary, the 8 `vits-mms-*` bundles duplicated by the HF MMS
set, the superseded kokoro multi-lang v1.0 and Supertonic v2 re-uploads).
Speaker counts, sample rates, and licenses are verified against the
sherpa-onnx docs, the piper-voices configs, and the upstream model cards —
anything unverifiable is left `unknown` rather than guessed.

## Schema

Every entry in `models.json` conforms to `schema.json`. The fields:

### Core (always present)
| Field | Type | Notes |
|-------|------|-------|
| `id` | string | Stable unique id, convention `<developer>-<lang>-<name>[-<quality>][-<quantization>]`. Piper/Mimic3 keep the region (`en_US`) since voice names repeat across regions. Used as the registry key. |
| `model_type` | enum | `vits` \| `matcha` \| `kokoro` \| `kitten` \| `supertonic` \| `zipvoice` \| `pocket` \| `mms` — the sherpa-onnx `OfflineTtsModelConfig` branch. |
| `developer` | string | Upstream project (piper, coqui, icefall, mms, kokoro, supertone, mimic3, melo, kitten, kyutai, zipvoice, …). |
| `name` | string | Variant name parsed from the release filename. |
| `language` | array | `{lang_code, language_name, country}` — one per supported language; multilingual models list all. |
| `quality` | string | Size/variant tag (low / medium / high / x_low / v0_19 / unknown). |
| `quantization` | enum | `fp32` (default) \| `fp16` \| `int8` — numeric precision of the ONNX build. Most piper voices ship all three; ids get an `-int8`/`-fp16` suffix so each build is addressable. |
| `sample_rate` | int | Output audio rate in Hz (16000 / 22050 / 24000; one legacy model at 8000). |
| `num_speakers` | int | Number of preset voices addressable by speaker id. Zero-shot cloning models (zipvoice, pocket) are 1 — they clone any voice from reference audio. |
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

## Building & validating locally

```bash
make help                       # list all targets
python -m venv .venv && .venv/bin/pip install -r requirements.txt

.venv/bin/python generate.py            # incremental (preserves manual edits)
.venv/bin/python generate.py --force    # full rebuild from scratch

.venv/bin/python scripts/lint_json.py   # canonical-format check
.venv/bin/python validate.py            # schema + uniqueness + plausibility
.venv/bin/python validate.py --online   # + HEAD-check a sample of URLs
```

CI (`.github/workflows/ci.yml`) runs lint + validate + a reproducibility
check on every PR. Model verification
(`.github/workflows/verify-models.yml`) actually synthesises audio for the
smallest model of each type.

## Releases

Tags cut a **versioned, checksummed** release that consumers can pin. Push a
tag (`vYYYY-MM-DD`, optionally `.1`/`.2` for same-day re-rolls) and the
[`release.yml`](./.github/workflows/release.yml) workflow rebuilds
`models.json` from the committed `generate.py`/`overrides.py` at that tag,
then uploads:

| Asset | Purpose |
|-------|---------|
| `models.json` | The registry — pin a tag and fetch this. |
| `models.json.sha256` | SHA-256 of `models.json` for verified downloads. |
| `schema.json` | The JSON Schema, so consumers can validate post-sync. |
| `REGISTRY-VERSION.txt` | Tag, entry count, and checksum for provenance. |

```bash
git tag v2026-08-10
git push origin v2026-08-10
```

Consumers should **never** track `main` — always pin a tag so a build is
reproducible.

## Consuming

### rust-tts-wrapper (the canonical ingestion pattern)

rust-tts-wrapper vendors a copy of the registry via
`include_str!("merged_models.json")` and ships a sync script that pulls a
specific tagged release, verifies its checksum, and drops it into `src/`:

```bash
# from the rust-tts-wrapper checkout
./scripts/sync-registry.sh v2026-08-10
```

That script writes `src/merged_models.json` **and** `src/registry-version.txt`
(provenance: tag + sha256). Commit both — the build then needs no network.
See [rust-tts-wrapper's sync-registry.sh](https://github.com/AACTools/rust-tts-wrapper/blob/main/scripts/sync-registry.sh).

The same pattern ports directly to `js-tts-wrapper` and (python) `tts-wrapper`:
fetch the tagged `models.json`, verify the sha256, replace the vendored copy,
bump a version constant.

### Generic (any language)
```python
import json, urllib.request, hashlib
TAG = "v2026-08-10"
base = f"https://github.com/AACTools/sherpa-onnx-tts-models/releases/download/{TAG}"
data = urllib.request.urlopen(f"{base}/models.json").read()
expected = urllib.request.urlopen(f"{base}/models.json.sha256").read().strip().decode()
assert hashlib.sha256(data).hexdigest() == expected, "checksum mismatch"
registry = json.loads(data)
```

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
- **PocketTTS** (`kyutai-*`) — `CC-BY-4.0`.
- **ZipVoice** (`zipvoice-*`) — `unknown`: the [code](https://github.com/k2-fsa/zipvoice)
  is Apache-2.0, but the released weights carry no license statement and were
  trained on the CC-BY-NC-4.0 Emilia dataset — treat as non-commercial until
  verified.
- **vits-zh-ll** (`csukuangfj-zh-ll`) — `unknown`: no license stated upstream.

## License

The registry data and the scripts in this repository are released under
**Apache-2.0**. The *model weights* they describe are each governed by their
own license (see the `license` / `license_url` fields) — this repository
grants no rights over the models themselves, only over the metadata.
