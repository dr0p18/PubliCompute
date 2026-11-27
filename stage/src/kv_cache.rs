//! Requested and effective attention cache settings, validated without the native engine.
//!
//! The coordinator declares one profile per chain and the agent passes it down as command line options.
//! A quantized value cache requires FlashAttention to be requested explicitly and enabled in the loaded context.

use std::fmt;

/// Storage type of the K or V cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheType {
    F16,
    Q8_0,
    Q4_0,
}

impl CacheType {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "f16" => Ok(Self::F16),
            "q8_0" => Ok(Self::Q8_0),
            "q4_0" => Ok(Self::Q4_0),
            other => Err(format!("unsupported cache type \"{other}\" (f16 | q8_0 | q4_0)")),
        }
    }

    pub fn quantized(self) -> bool {
        self != Self::F16
    }
}

impl fmt::Display for CacheType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::F16 => "f16",
            Self::Q8_0 => "q8_0",
            Self::Q4_0 => "q4_0",
        })
    }
}

/// The requested FlashAttention mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FlashAttn {
    Auto,
    On,
    Off,
}

impl FlashAttn {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "auto" => Ok(Self::Auto),
            "on" => Ok(Self::On),
            "off" => Ok(Self::Off),
            other => Err(format!("unsupported flash attention mode \"{other}\" (auto | on | off)")),
        }
    }
}

impl fmt::Display for FlashAttn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Auto => "auto",
            Self::On => "on",
            Self::Off => "off",
        })
    }
}

/// Cache settings requested for every main context and draft context.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CacheConfig {
    pub k: CacheType,
    pub v: CacheType,
    pub flash_attn: FlashAttn,
}

/// The allocated key and value types, when the context has an attention cache, and the resolved mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EffectiveCacheConfig {
    pub types: Option<(CacheType, CacheType)>,
    pub flash_attn: bool,
}

impl EffectiveCacheConfig {
    pub fn describe(&self) -> String {
        let flash_attn = if self.flash_attn { FlashAttn::On } else { FlashAttn::Off };
        match self.types {
            Some((k, v)) => format!(
                "attention_cache=present cache_type_k_effective={k} cache_type_v_effective={v} flash_attn_effective={flash_attn}"
            ),
            None => format!(
                "attention_cache=none cache_type_k_effective=none cache_type_v_effective=none flash_attn_effective={flash_attn}"
            ),
        }
    }
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self { k: CacheType::F16, v: CacheType::F16, flash_attn: FlashAttn::Auto }
    }
}

impl CacheConfig {
    /// Parse the three option values and reject a combination the engine cannot honor.
    pub fn parse(k: Option<&str>, v: Option<&str>, flash_attn: Option<&str>) -> Result<Self, String> {
        let d = Self::default();
        let c = Self {
            k: k.map_or(Ok(d.k), CacheType::parse).map_err(|e| format!("--cache-type-k: {e}"))?,
            v: v.map_or(Ok(d.v), CacheType::parse).map_err(|e| format!("--cache-type-v: {e}"))?,
            flash_attn: flash_attn.map_or(Ok(d.flash_attn), FlashAttn::parse).map_err(|e| format!("--flash-attn: {e}"))?,
        };
        c.requested_flash_attn()?;
        Ok(c)
    }

    /// Return the requested mode after checking that a quantized value cache explicitly enables FlashAttention.
    pub fn requested_flash_attn(&self) -> Result<FlashAttn, String> {
        match (self.v.quantized(), self.flash_attn) {
            (true, FlashAttn::On) => Ok(FlashAttn::On),
            (true, other) => Err(format!("--cache-type-v {} requires --flash-attn on, not {other}", self.v)),
            (false, fa) => Ok(fa),
        }
    }

    pub fn validate_effective(&self, effective: EffectiveCacheConfig) -> Result<(), String> {
        let Some((k, v)) = effective.types else {
            return Ok(());
        };
        if (k, v) != (self.k, self.v) {
            return Err(format!(
                "allocated K/V cache types {k}/{v} differ from requested {}/{}",
                self.k, self.v
            ));
        }
        if v.quantized() && !effective.flash_attn {
            return Err(format!("allocated V cache type {v} requires effective FlashAttention to be on"));
        }
        match (self.flash_attn, effective.flash_attn) {
            (FlashAttn::On, false) => Err("requested FlashAttention on, but effective mode is off".into()),
            (FlashAttn::Off, true) => Err("requested FlashAttention off, but effective mode is on".into()),
            _ => Ok(()),
        }
    }

    /// Format the requested cache types and FlashAttention mode for the stage log.
    pub fn describe_requested(&self) -> String {
        format!(
            "cache_type_k_requested={} cache_type_v_requested={} flash_attn_requested={}",
            self.k, self.v, self.flash_attn
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_types_reject_quantized_v_without_flash_attention() {
        for v in ["q8_0", "q4_0"] {
            for fa in ["off", "auto"] {
                let err = CacheConfig::parse(Some("f16"), Some(v), Some(fa)).unwrap_err();
                assert!(err.contains("requires --flash-attn on"), "{v} with {fa}: {err}");
            }
        }
        // Quantized K alone does not need FlashAttention.
        assert!(CacheConfig::parse(Some("q8_0"), Some("f16"), Some("off")).is_ok());
        // Mixed types keep K and V apart.
        let mixed = CacheConfig::parse(Some("q8_0"), Some("q4_0"), Some("on")).unwrap();
        assert_eq!((mixed.k, mixed.v), (CacheType::Q8_0, CacheType::Q4_0));
    }

    #[test]
    fn quantized_v_requires_explicit_on() {
        let c = CacheConfig::parse(Some("f16"), Some("q8_0"), Some("on")).unwrap();
        assert_eq!(c.requested_flash_attn(), Ok(FlashAttn::On));
        assert_eq!(
            c.describe_requested(),
            "cache_type_k_requested=f16 cache_type_v_requested=q8_0 flash_attn_requested=on"
        );
        assert!(CacheConfig::parse(Some("f16"), Some("q8_0"), Some("auto")).is_err());
        let f16 = CacheConfig::parse(None, None, Some("auto")).unwrap();
        assert_eq!(f16.requested_flash_attn(), Ok(FlashAttn::Auto));
        assert_eq!(
            f16.describe_requested(),
            "cache_type_k_requested=f16 cache_type_v_requested=f16 flash_attn_requested=auto"
        );
    }

    #[test]
    fn cache_types_reject_unknown_values() {
        assert!(CacheConfig::parse(Some("q5_0"), None, None).unwrap_err().starts_with("--cache-type-k"));
        assert!(CacheConfig::parse(None, Some("bf16"), None).unwrap_err().starts_with("--cache-type-v"));
        assert!(CacheConfig::parse(None, None, Some("yes")).unwrap_err().starts_with("--flash-attn"));
    }

    #[test]
    fn no_options_use_default_profile() {
        assert_eq!(CacheConfig::parse(None, None, None).unwrap(), CacheConfig::default());
    }

    #[test]
    fn effective_cache_checks_key_and_value_types() {
        let requested = CacheConfig::parse(Some("q8_0"), Some("q4_0"), Some("on")).unwrap();
        for types in [(CacheType::F16, CacheType::Q4_0), (CacheType::Q8_0, CacheType::F16)] {
            let effective = EffectiveCacheConfig { types: Some(types), flash_attn: true };
            assert!(requested.validate_effective(effective).unwrap_err().contains("differ from requested"));
        }
    }

    #[test]
    fn effective_quantized_value_cache_requires_flash_attention() {
        let requested = CacheConfig::parse(Some("q8_0"), Some("q8_0"), Some("on")).unwrap();
        let types = Some((CacheType::Q8_0, CacheType::Q8_0));
        let error = requested.validate_effective(EffectiveCacheConfig { types, flash_attn: false }).unwrap_err();
        assert!(error.contains("requires effective FlashAttention to be on"));
        assert!(requested.validate_effective(EffectiveCacheConfig { types, flash_attn: true }).is_ok());
    }

    #[test]
    fn effective_flash_attention_can_resolve_either_way_for_f16() {
        let requested = CacheConfig::default();
        for (flash_attn, mode) in [(true, "on"), (false, "off")] {
            let effective = EffectiveCacheConfig { types: Some((CacheType::F16, CacheType::F16)), flash_attn };
            assert!(requested.validate_effective(effective).is_ok());
            assert!(effective.describe().ends_with(&format!("flash_attn_effective={mode}")));
        }
    }

    #[test]
    fn effective_flash_attention_must_match_explicit_mode_for_f16() {
        let types = Some((CacheType::F16, CacheType::F16));
        for (requested, enabled, mismatch) in [
            ("on", true, "requested FlashAttention on, but effective mode is off"),
            ("off", false, "requested FlashAttention off, but effective mode is on"),
        ] {
            let requested = CacheConfig::parse(None, None, Some(requested)).unwrap();
            assert!(requested.validate_effective(EffectiveCacheConfig { types, flash_attn: enabled }).is_ok());
            assert_eq!(
                requested.validate_effective(EffectiveCacheConfig { types, flash_attn: !enabled }).unwrap_err(),
                mismatch
            );
        }
    }

    #[test]
    fn effective_cache_can_be_absent_on_a_recurrent_stage() {
        let requested = CacheConfig::parse(Some("q8_0"), Some("q8_0"), Some("on")).unwrap();
        let effective = EffectiveCacheConfig { types: None, flash_attn: false };
        assert!(requested.validate_effective(effective).is_ok());
        assert!(effective.describe().contains("attention_cache=none cache_type_k_effective=none cache_type_v_effective=none"));
    }
}
