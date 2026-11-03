//! Internal quantization types (Release 1 - no legacy compatibility)
//!
//! These types are used internally for quantization operations.
//! The proto QuantizationConfig is simplified for user-facing API.

pub use proximadb_quantization_model::*;

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "experimental-turboquant")]
    use proximadb_quantization_types::CalibrationMode;

    #[cfg(feature = "experimental-turboquant")]
    #[test]
    fn test_turboquant_variant_2bit_bytes_per_vector() {
        // d=1536, bit_width=2 → ceil(1536*2/8) = 384 bytes + 4 scale = 388
        let q = UnifiedQuantizationLevel::turboquant(2, 0xdeadbeef);
        assert_eq!(q.bits_per_element(), 2);
        assert_eq!(q.bytes_per_vector(1536), 388);
        // Compression ratio matches the headline 16x at 2-bit minus the scale
        // overhead (~6144 / 388 ≈ 15.8).
        let r = q.compression_ratio(1536);
        assert!(r > 15.0 && r < 16.0, "ratio = {}", r);
    }

    #[cfg(feature = "experimental-turboquant")]
    #[test]
    fn test_turboquant_variant_4bit_bytes_per_vector() {
        // d=1536, bit_width=4 → ceil(1536*4/8) = 768 bytes + 4 scale = 772
        let q = UnifiedQuantizationLevel::turboquant(4, 0xdeadbeef);
        assert_eq!(q.bits_per_element(), 4);
        assert_eq!(q.bytes_per_vector(1536), 772);
    }

    #[cfg(feature = "experimental-turboquant")]
    #[test]
    fn test_turboquant_identity_constructor() {
        let q = UnifiedQuantizationLevel::turboquant_identity(4, 42);
        match &q.level_type {
            Some(QuantizationScheme::TurboQuant(tq)) => {
                assert_eq!(tq.bit_width, 4);
                assert_eq!(tq.rotation_seed, 42);
                assert_eq!(tq.calibration_mode, CalibrationMode::Identity);
            }
            _ => panic!("expected TurboQuant variant"),
        }
    }

    #[cfg(feature = "experimental-turboquant")]
    #[test]
    fn test_turboquant_serde_round_trip() {
        let q = TurboQuantization::tq_plus(2, 0xcafe_babe_dead_beef);
        let s = serde_json::to_string(&q).unwrap();
        let back: TurboQuantization = serde_json::from_str(&s).unwrap();
        assert_eq!(q, back);
    }

    // ------------------------------------------------------------------
    // QuantizationMethod trait coverage
    // (Phase A — Quantization Trait Convergence Plan)
    // ------------------------------------------------------------------

    #[test]
    fn quantization_method_on_no_quantization() {
        let m = NoQuantization {};
        assert_eq!(m.quantization_type(), QuantizationType::None);
        assert_eq!(m.bit_width(), 32);
        assert_eq!(m.lifecycle(), QuantizationLifecycle::Identity);
        assert!(m.durable_state().is_none());
        assert!(!m.supports_candidate_mask());
        assert_eq!(m.metric_label(), "none");
    }

    #[test]
    fn quantization_method_on_binary_quantization() {
        let m = BinaryQuantization {
            threshold: None,
            sign_based: false,
        };
        assert_eq!(m.quantization_type(), QuantizationType::Binary);
        assert_eq!(m.bit_width(), 1);
        assert_eq!(m.lifecycle(), QuantizationLifecycle::WriteTime);
        assert!(m.durable_state().is_none());
        assert!(!m.supports_candidate_mask());
        assert_eq!(m.metric_label(), "binary");
    }

    #[test]
    fn quantization_method_on_scalar_quantization() {
        let m = ScalarQuantization {
            bits: 8,
            scale: 1.0,
            offset: 0.0,
            clamp_values: false,
        };
        assert_eq!(m.quantization_type(), QuantizationType::Scalar);
        assert_eq!(m.bit_width(), 8);
        assert_eq!(m.lifecycle(), QuantizationLifecycle::WriteTime);
        assert!(m.durable_state().is_none());
        assert!(!m.supports_candidate_mask());
        assert_eq!(m.metric_label(), "scalar");
    }

    #[test]
    fn quantization_method_on_scalar_clamps_bits_into_u8_range() {
        // Pathological i32 values shouldn't cause wrap-around when packed
        // into the u8 trait return — the clamp guards routing/metrics from
        // a config-typo-induced denial of service.
        let m = ScalarQuantization {
            bits: 999,
            scale: 1.0,
            offset: 0.0,
            clamp_values: false,
        };
        assert_eq!(m.bit_width(), 32);
        let neg = ScalarQuantization {
            bits: -7,
            scale: 1.0,
            offset: 0.0,
            clamp_values: false,
        };
        assert_eq!(neg.bit_width(), 1);
    }

    #[test]
    fn quantization_method_on_product_quantization_durable_state_carries_codebook() {
        let m = ProductQuantization {
            bits_per_code: 4,
            num_subvectors: 8,
            codebook_id: Some("cb-xyz".to_string()),
            adaptive_subvectors: false,
        };
        assert_eq!(m.quantization_type(), QuantizationType::Product);
        assert_eq!(m.bit_width(), 4);
        assert_eq!(m.lifecycle(), QuantizationLifecycle::WriteTime);
        assert!(!m.supports_candidate_mask());
        assert_eq!(m.metric_label(), "product");
        let ds = m.durable_state().expect("PQ has durable state");
        assert_eq!(ds.seed_or_codebook_id.as_deref(), Some("cb-xyz"));
        assert!(ds.calibration.is_none());
        assert_eq!(ds.encoded_epoch, 0);
    }

    #[test]
    fn quantization_method_on_uniform_and_custom_route_through_scalar() {
        // Uniform and Custom both surface as Scalar at the router level —
        // their per-variant semantics live in their own encode/score paths.
        let u = UniformQuantization {
            bits: 8,
            scale: None,
            offset: None,
        };
        assert_eq!(u.quantization_type(), QuantizationType::Scalar);
        assert_eq!(u.lifecycle(), QuantizationLifecycle::WriteTime);
        assert_eq!(u.metric_label(), "uniform");

        let c = CustomQuantization {
            type_id: "custom-magic".to_string(),
            bits_per_element: 6,
            config: Default::default(),
        };
        assert_eq!(c.quantization_type(), QuantizationType::Scalar);
        assert_eq!(c.bit_width(), 6);
        assert_eq!(c.metric_label(), "custom");
        let cds = c.durable_state().expect("custom carries its type_id");
        assert_eq!(cds.seed_or_codebook_id.as_deref(), Some("custom-magic"));
    }

    #[cfg(feature = "experimental-turboquant")]
    #[test]
    fn quantization_method_on_turboquantization() {
        let m = TurboQuantization::tq_plus(4, 0xdead_beef_cafe_babe);
        assert_eq!(m.quantization_type(), QuantizationType::TurboQuant);
        assert_eq!(m.bit_width(), 4);
        assert_eq!(m.lifecycle(), QuantizationLifecycle::ReadTime);
        assert!(m.supports_candidate_mask());
        assert_eq!(m.metric_label(), "turboquant");
        let ds = m.durable_state().expect("TurboQuant has durable state");
        // rotation_seed is tunneled as a hex string for cross-protocol
        // stability. Pinning the format here protects xCatalog round-trip.
        assert_eq!(
            ds.seed_or_codebook_id.as_deref(),
            Some("0xdeadbeefcafebabe")
        );
        assert_eq!(ds.calibration.as_deref(), Some("tq_plus"));
        assert_eq!(ds.encoded_epoch, 0);
    }

    #[cfg(feature = "experimental-turboquant")]
    #[test]
    fn quantization_method_on_turboquantization_identity_tunnels_correct_label() {
        let m = TurboQuantization::identity(2, 0x1);
        let ds = m.durable_state().expect("identity carries durable state");
        assert_eq!(ds.calibration.as_deref(), Some("identity"));
        assert_eq!(m.bit_width(), 2);
    }

    #[test]
    fn quantization_method_blanket_on_quantization_level_delegates_correctly() {
        // QuantizationScheme is the enum every router holds — verify each
        // arm delegates to its inner variant.
        let none = QuantizationScheme::None(NoQuantization {});
        assert_eq!(none.quantization_type(), QuantizationType::None);
        assert_eq!(none.lifecycle(), QuantizationLifecycle::Identity);

        let bin = QuantizationScheme::Binary(BinaryQuantization {
            threshold: None,
            sign_based: false,
        });
        assert_eq!(bin.quantization_type(), QuantizationType::Binary);
        assert_eq!(bin.bit_width(), 1);

        let pq = QuantizationScheme::Pq(ProductQuantization {
            bits_per_code: 8,
            num_subvectors: 16,
            codebook_id: Some("cb".to_string()),
            adaptive_subvectors: false,
        });
        assert_eq!(pq.quantization_type(), QuantizationType::Product);
        assert!(pq.durable_state().is_some());
    }

    #[test]
    fn quantization_method_blanket_on_unified_wrapper_handles_none_level_type() {
        // UnifiedQuantizationLevel { level_type: None } means "no
        // quantization configured" — the wrapper must route identical to
        // the explicit NoQuantization arm so downstream code can't tell the
        // two apart. This is what kills the "unwrap-or-default 32" bugs we
        // would have seen scattered across call sites.
        let empty = UnifiedQuantizationLevel { level_type: None };
        assert_eq!(empty.quantization_type(), QuantizationType::None);
        assert_eq!(empty.bit_width(), 32);
        assert_eq!(empty.lifecycle(), QuantizationLifecycle::Identity);
        assert!(empty.durable_state().is_none());
        assert!(!empty.supports_candidate_mask());
        assert_eq!(empty.metric_label(), "none");
    }

    #[test]
    fn quantization_method_blanket_on_unified_wrapper_delegates_to_inner() {
        // When a level is present, the wrapper just forwards.
        let q = UnifiedQuantizationLevel::int8();
        assert_eq!(q.quantization_type(), QuantizationType::Scalar);
        assert_eq!(q.bit_width(), 8);
        assert_eq!(q.lifecycle(), QuantizationLifecycle::WriteTime);
    }

    #[cfg(feature = "experimental-turboquant")]
    #[test]
    fn quantization_method_blanket_on_unified_wrapper_routes_turboquant_correctly() {
        let q = UnifiedQuantizationLevel::turboquant(4, 0xabcd);
        assert_eq!(q.quantization_type(), QuantizationType::TurboQuant);
        assert_eq!(q.lifecycle(), QuantizationLifecycle::ReadTime);
        assert!(q.supports_candidate_mask());
        assert_eq!(q.metric_label(), "turboquant");
    }
}
