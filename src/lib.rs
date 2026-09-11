//! # sherpa-onnx-models
//!
//! The [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) TTS model
//! registry as typed data, published from the canonical repository
//! ([AACTools/sherpa-onnx-tts-models](https://github.com/AACTools/sherpa-onnx-tts-models)).
//! Every model there is here — vits/mms/matcha/kokoro voices, plus the
//! audio-LM families — with languages, licenses, sample rates, sizes,
//! and the `engines` routing field (`"floravox"` or `"sherpa-onnx"`).
//!
//! The registry is embedded at compile time: no network, no runtime
//! fetches, `~0` resident cost until first access. Registry updates are
//! crate versions — bump the dependency to refresh.
//!
//! ```
//! use sherpa_onnx_models::{get, models};
//!
//! assert!(models().len() > 1700);
//! let info = get("piper-en_US-lessac-high").expect("registered");
//! assert_eq!(info.model_type, "vits");
//! assert_eq!(info.engines, "floravox");
//! assert!(info.language.iter().any(|l| l.lang_code == "en"));
//! ```

use std::collections::HashMap;
use std::sync::OnceLock;

/// The raw registry JSON, exactly as published by the canonical repo.
pub static MODELS_JSON: &str = include_str!("../models.json");

/// One language entry of a model.
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct ModelLanguage {
    /// BCP-47-ish language code (`"en"`, `"pt-BR"`, ...).
    pub lang_code: String,
    /// Human-readable language name.
    pub language_name: String,
    /// Country/region code (may be empty).
    #[serde(default)]
    pub country: String,
}

/// One registry entry (one model).
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct ModelInfo {
    /// Registry id (`"vits-piper-en_US-lessac-low"`).
    pub id: String,
    /// Graph family: `vits`, `mms`, `matcha`, `kokoro`, `kitten`,
    /// `pocket`, `supertonic`, `zipvoice`.
    pub model_type: String,
    /// Which TTS engines can drive this family:
    /// `"floravox"` (piper/MMS VITS, Matcha, Kokoro — SSML, measured
    /// word timings) or `"sherpa-onnx"` (everything else).
    #[serde(default = "default_engines")]
    pub engines: String,
    /// Upstream developer (`"piper"`, `"icefall"`, ...).
    pub developer: String,
    /// Human-readable model name.
    pub name: String,
    /// Languages supported.
    pub language: Vec<ModelLanguage>,
    /// Quality/variant tier (`"low"`, `"medium"`, `"high"`, ...).
    #[serde(default)]
    pub quality: String,
    /// Quantization (`"fp32"`, `"int8"`, `"fp16"`).
    #[serde(default)]
    pub quantization: String,
    /// Output sample rate in Hz.
    pub sample_rate: u32,
    /// Speaker count.
    pub num_speakers: u32,
    /// Download URL of the model archive.
    pub url: String,
    /// Whether the archive is compressed.
    pub compression: bool,
    /// Archive size in MiB.
    #[serde(default)]
    pub filesize_mb: f64,
    /// Model license name.
    pub license: String,
    /// License text URL.
    #[serde(default)]
    pub license_url: String,
    /// First sherpa-onnx release supporting this model type.
    #[serde(default)]
    pub min_sherpa_onnx_version: String,
    /// Where this entry was sourced from.
    #[serde(default)]
    pub source_url: String,
    /// Free-form tags.
    #[serde(default)]
    pub tags: Vec<String>,
    /// One-line description.
    #[serde(default)]
    pub description: String,
    /// SHA-256 of the archive, when known.
    #[serde(default)]
    pub sha256: Option<String>,
    /// Deprecated flag.
    #[serde(default)]
    pub deprecated: bool,
    /// Why it was deprecated, when it was.
    #[serde(default)]
    pub deprecation_note: Option<String>,
    /// Preset voice labels for multi-speaker models (e.g. Supertonic's
    /// `M1`..`F5`).
    #[serde(default)]
    pub voice_names: Option<std::collections::BTreeMap<String, String>>,
}

fn default_engines() -> String {
    "sherpa-onnx".into()
}

/// The parsed registry, keyed by model id. Parsed once on first access.
///
/// # Panics
///
/// Never in practice: the embedded `models.json` is schema-validated in
/// this repository's CI before any release is cut.
#[must_use]
pub fn models() -> &'static HashMap<String, ModelInfo> {
    static MODELS: OnceLock<HashMap<String, ModelInfo>> = OnceLock::new();
    MODELS.get_or_init(|| serde_json::from_str(MODELS_JSON).expect("embedded models.json is valid"))
}

/// Look up one model by id.
#[must_use]
pub fn get(id: &str) -> Option<&'static ModelInfo> {
    models().get(id)
}

/// Number of registered models.
#[must_use]
pub fn len() -> usize {
    models().len()
}

/// True when the registry is empty (never, in practice).
#[must_use]
pub fn is_empty() -> bool {
    models().is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_parses_and_is_large() {
        assert!(models().len() > 1700, "registry shrank: {}", len());
    }

    #[test]
    fn engines_field_is_populated() {
        let floravox = models()
            .values()
            .filter(|m| m.engines == "floravox")
            .count();
        let sherpa = models()
            .values()
            .filter(|m| m.engines == "sherpa-onnx")
            .count();
        assert!(floravox > 1500, "floravox-drivable: {floravox}");
        assert!(sherpa > 0);
        assert_eq!(floravox + sherpa, len(), "engines must cover every model");
    }

    #[test]
    fn spot_checks() {
        let m = get("piper-en_US-lessac-high").unwrap();
        assert_eq!(m.model_type, "vits");
        assert_eq!(m.engines, "floravox");
        assert!(m.language.iter().any(|l| l.lang_code == "en"));
        assert_eq!(m.sample_rate, 22_050);

        let k = models()
            .values()
            .find(|m| m.model_type == "kokoro")
            .expect("a kokoro model");
        assert_eq!(k.engines, "floravox");
    }
}
