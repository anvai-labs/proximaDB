// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for ProximaCodec
//!
//! Tests the complete encode-decode pipeline through WireFormatManager

use crate::baseline::{BaselineDecoder, BaselineEncoder};
use crate::traits::{RawDecoder, RawEncoder};
use crate::types::{ProximaScheme, TypeId};
use crate::wire_format::WireFormatManager;

/// Test complete roundtrip: encode → wire format → decode
#[test]
fn test_complete_roundtrip_f32_delta() {
    let encoder = BaselineEncoder;
    let decoder = BaselineDecoder;
    let wire_manager = WireFormatManager::new();

    let values = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let scheme = ProximaScheme::Delta { base: 0 };

    // Encode raw data
    let raw_encoded = encoder.encode_f32(&values, &scheme).unwrap();

    // Create wire format header
    let header = wire_manager.write_header(&scheme, values.len(), TypeId::F32);

    // Combine header + raw data
    let mut with_header = header;
    with_header.extend_from_slice(&raw_encoded);

    // Read wire format header
    let parsed_header = wire_manager.read_header(&with_header).unwrap();

    assert_eq!(parsed_header.type_id, TypeId::F32);
    assert_eq!(parsed_header.count, values.len());
    match parsed_header.scheme {
        ProximaScheme::Delta { .. } => {} // Success
        _ => panic!("Expected Delta scheme"),
    }

    // Extract raw data (skip header bytes)
    let raw_data = &with_header[parsed_header.data_offset..];

    // Decode raw data
    let decoded = decoder
        .decode_f32(raw_data, &parsed_header.scheme, parsed_header.count)
        .unwrap();

    assert_eq!(decoded, values);
}

#[test]
fn test_complete_roundtrip_i64_delta() {
    let encoder = BaselineEncoder;
    let decoder = BaselineDecoder;
    let wire_manager = WireFormatManager::new();

    let values = vec![100i64, 200, 300, 400, 500];
    let scheme = ProximaScheme::Delta { base: 0 };

    // Encode raw data
    let raw_encoded = encoder.encode_i64(&values, &scheme).unwrap();

    // Create wire format header
    let header = wire_manager.write_header(&scheme, values.len(), TypeId::I64);

    // Combine header + raw data
    let mut with_header = header;
    with_header.extend_from_slice(&raw_encoded);

    // Read wire format header
    let parsed_header = wire_manager.read_header(&with_header).unwrap();

    assert_eq!(parsed_header.type_id, TypeId::I64);
    assert_eq!(parsed_header.count, values.len());
    match parsed_header.scheme {
        ProximaScheme::Delta { .. } => {} // Success
        _ => panic!("Expected Delta scheme"),
    }

    // Extract raw data (skip header bytes)
    let raw_data = &with_header[parsed_header.data_offset..];

    // Decode raw data
    let decoded = decoder
        .decode_i64(raw_data, &parsed_header.scheme, parsed_header.count)
        .unwrap();

    assert_eq!(decoded, values);
}

#[test]
fn test_complete_roundtrip_i32_delta() {
    let encoder = BaselineEncoder;
    let decoder = BaselineDecoder;
    let wire_manager = WireFormatManager::new();

    let values = vec![10i32, 20, 30, 40, 50];
    let scheme = ProximaScheme::Delta { base: 0 };

    // Encode raw data
    let raw_encoded = encoder.encode_i32(&values, &scheme).unwrap();

    // Create wire format header
    let header = wire_manager.write_header(&scheme, values.len(), TypeId::I32);

    // Combine header + raw data
    let mut with_header = header;
    with_header.extend_from_slice(&raw_encoded);

    // Read wire format header
    let parsed_header = wire_manager.read_header(&with_header).unwrap();

    assert_eq!(parsed_header.type_id, TypeId::I32);
    assert_eq!(parsed_header.count, values.len());
    match parsed_header.scheme {
        ProximaScheme::Delta { .. } => {} // Success
        _ => panic!("Expected Delta scheme"),
    }

    // Extract raw data (skip header bytes)
    let raw_data = &with_header[parsed_header.data_offset..];

    // Decode raw data
    let decoded = decoder
        .decode_i32(raw_data, &parsed_header.scheme, parsed_header.count)
        .unwrap();

    assert_eq!(decoded, values);
}

/// Test sequential data compression
#[test]
fn test_sequential_data_compression() {
    let encoder = BaselineEncoder;
    let decoder = BaselineDecoder;

    let values: Vec<i64> = (0..1000).collect();
    let scheme = ProximaScheme::Delta { base: 0 };

    let encoded = encoder.encode_i64(&values, &scheme).unwrap();
    let decoded = decoder.decode_i64(&encoded, &scheme, values.len()).unwrap();

    assert_eq!(decoded, values);

    // Verify compression
    let original_bytes = values.len() * 8;
    assert!(
        encoded.len() < original_bytes,
        "Should compress sequential data: {} bytes vs {} bytes",
        encoded.len(),
        original_bytes
    );
}

/// Test large floating-point vectors
#[test]
fn test_large_f32_vector() {
    let encoder = BaselineEncoder;
    let decoder = BaselineDecoder;

    // Simulate embedding vectors
    let values: Vec<f32> = (0..1536).map(|i| (i as f32) * 0.001).collect();
    let scheme = ProximaScheme::Delta { base: 0 };

    let encoded = encoder.encode_f32(&values, &scheme).unwrap();
    let decoded = decoder.decode_f32(&encoded, &scheme, values.len()).unwrap();

    assert_eq!(decoded.len(), values.len());
    for (original, decoded_val) in values.iter().zip(decoded.iter()) {
        assert_eq!(original, decoded_val);
    }
}

/// Test edge case: empty vector
#[test]
fn test_empty_vector() {
    let encoder = BaselineEncoder;
    let decoder = BaselineDecoder;

    let values: Vec<i64> = vec![];
    let scheme = ProximaScheme::Delta { base: 0 };

    let encoded = encoder.encode_i64(&values, &scheme).unwrap();
    assert!(encoded.is_empty());

    let decoded = decoder.decode_i64(&encoded, &scheme, 0).unwrap();
    assert!(decoded.is_empty());
}

/// Test edge case: single value
#[test]
fn test_single_value() {
    let encoder = BaselineEncoder;
    let decoder = BaselineDecoder;

    let values = vec![42i64];
    let scheme = ProximaScheme::Delta { base: 0 };

    let encoded = encoder.encode_i64(&values, &scheme).unwrap();
    let decoded = decoder.decode_i64(&encoded, &scheme, 1).unwrap();

    assert_eq!(decoded, values);
}

/// Test negative deltas
#[test]
fn test_negative_deltas() {
    let encoder = BaselineEncoder;
    let decoder = BaselineDecoder;

    let values = vec![1000i64, 900, 800, 700, 600];
    let scheme = ProximaScheme::Delta { base: 0 };

    let encoded = encoder.encode_i64(&values, &scheme).unwrap();
    let decoded = decoder.decode_i64(&encoded, &scheme, values.len()).unwrap();

    assert_eq!(decoded, values);
}

/// Test mixed positive and negative deltas
#[test]
fn test_mixed_deltas() {
    let encoder = BaselineEncoder;
    let decoder = BaselineDecoder;

    let values = vec![100i64, 200, 150, 300, 250];
    let scheme = ProximaScheme::Delta { base: 0 };

    let encoded = encoder.encode_i64(&values, &scheme).unwrap();
    let decoded = decoder.decode_i64(&encoded, &scheme, values.len()).unwrap();

    assert_eq!(decoded, values);
}

/// Test with different base values
#[test]
fn test_different_base_values() {
    let encoder = BaselineEncoder;
    let decoder = BaselineDecoder;

    let values = vec![1000i64, 1100, 1200, 1300, 1400];

    for base in [0, 500, 1000, 1500] {
        let scheme = ProximaScheme::Delta { base };

        let encoded = encoder.encode_i64(&values, &scheme).unwrap();
        let decoded = decoder.decode_i64(&encoded, &scheme, values.len()).unwrap();

        assert_eq!(decoded, values, "Failed with base={}", base);
    }
}

/// Test f32 special values
#[test]
fn test_f32_special_values() {
    let encoder = BaselineEncoder;
    let decoder = BaselineDecoder;

    let values = vec![0.0f32, -0.0, 1.0, -1.0, 100.5];
    let scheme = ProximaScheme::Delta { base: 0 };

    let encoded = encoder.encode_f32(&values, &scheme).unwrap();
    let decoded = decoder.decode_f32(&encoded, &scheme, values.len()).unwrap();

    assert_eq!(decoded.len(), values.len());
    for (original, decoded_val) in values.iter().zip(decoded.iter()) {
        // Use exact bit comparison for f32
        assert_eq!(original.to_bits(), decoded_val.to_bits());
    }
}

// ============================================================================
// PForDelta SIMD/baseline wire divergence pins (TD-CODEC-1)
//
// The SIMD and baseline PForDelta encoders write DIFFERENT wire formats:
//   - baseline (canonical): patches as pos:4B + i64 value:8B; regular values
//     bit-packed from the i64 domain (`functions/pfor_delta.rs`)
//   - simd: patches as pos:4B + i32 value:4B; regular values bit-packed from
//     the i32 domain (`simd/simd.rs`), and `simd_pfor_delta_decode_f32`
//     delegates header+patch parsing to the BASELINE parser.
//
// Consequences pinned below: the bytes differ, the baseline wire round-trips
// (it is the canonical format), and simd-encode → simd-decode is NOT a
// faithful round-trip once any patch exists. Today no persisted path routes
// PForDelta through the SIMD encoder (every PAX writer overrides the scheme
// first — see TD-CODEC-1 for the reachability analysis), so this is a
// tripwire, not a data-loss repro. If a change makes these tests fail, the
// wire format moved: update TD-CODEC-1 and add mixed-read-safe handling.
// ============================================================================

/// Input that forces PForDelta patches on both implementations. The
/// encoders work on BIT-CAST deltas (f32::to_bits as int, minus base), not
/// arithmetic ones: the 32 alternating 0.0/1.0 values give deltas of
/// {0, 0x3F800000 ≈ 1.07e9}, so the p90-driven majority width lands at 30
/// bits on both sides, while the two 1e30 outliers (0x7149F2C9 ≈ 1.9e9)
/// exceed 2^30 − 1 and must be patched (baseline: 12B/patch; simd: 8B/patch
/// — 161B vs 153B total). Keep this mechanism in mind when editing the
/// fixture: an outlier whose BIT PATTERN is small (e.g. 1e-30) would NOT
/// patch. The SIMD encoder also IGNORES its majority_bits argument
/// (underscore param) and picks its own width.
fn pfor_patch_forcing_values() -> Vec<f32> {
    let mut values = Vec::with_capacity(66);
    for i in 0..32 {
        values.push(i as f32 % 2.0); // deltas of ±1
    }
    values.push(1e30); // outlier delta far beyond any majority width
    values.push(1e30);
    values
}

/// Baseline PForDelta must round-trip exactly, patches included — it is the
/// canonical wire format that persisted bytes must keep matching.
#[test]
fn test_pfor_delta_baseline_round_trips_with_patches() {
    let values = pfor_patch_forcing_values();
    let encoded = crate::baseline::functions::pfor_delta::encode_f32(&values, 0).expect("encode");
    let decoded =
        crate::baseline::functions::pfor_delta::decode_f32(&encoded, values.len()).expect("decode");

    assert_eq!(decoded.len(), values.len());
    for (original, decoded_val) in values.iter().zip(decoded.iter()) {
        assert_eq!(original.to_bits(), decoded_val.to_bits());
    }
}

/// The two PForDelta encoders must not silently converge: their outputs
/// differ (12B i64 patches vs 8B i32 patches among other layout differences),
/// and any change to that fact is a wire-format change requiring TD-CODEC-1
/// to be revisited.
#[test]
fn test_pfor_delta_simd_baseline_wire_divergence_is_pinned() {
    let values = pfor_patch_forcing_values();

    let baseline_bytes =
        crate::baseline::functions::pfor_delta::encode_f32(&values, 0).expect("baseline encode");
    let simd_bytes = crate::simd::simd_pfor_delta_encode_f32(&values, 8, 0).expect("simd encode");

    assert_ne!(
        baseline_bytes.len(),
        simd_bytes.len(),
        "PForDelta wires converged (baseline {}B vs simd {}B) — wire format changed, \
         revisit TD-CODEC-1",
        baseline_bytes.len(),
        simd_bytes.len()
    );
    assert_ne!(baseline_bytes, simd_bytes);
}

/// simd-encode → simd-decode is NOT a faithful round-trip today when patches
/// exist (the simd decoder parses the baseline patch layout). Pinned so that
/// a fix to either side surfaces here instead of silently changing bytes.
#[test]
fn test_pfor_delta_simd_self_round_trip_divergence_is_pinned() {
    let values = pfor_patch_forcing_values();
    let simd_bytes = crate::simd::simd_pfor_delta_encode_f32(&values, 8, 0).expect("simd encode");
    let decoded = crate::simd::simd_pfor_delta_decode_f32(&simd_bytes, 8, 0, values.len());

    let faithful = decoded
        .map(|v| {
            v.len() == values.len()
                && v.iter()
                    .zip(values.iter())
                    .all(|(d, o)| d.to_bits() == o.to_bits())
        })
        .unwrap_or(false);

    assert!(
        !faithful,
        "simd PForDelta became self-consistent — wire format changed, revisit TD-CODEC-1"
    );
}
