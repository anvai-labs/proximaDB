//! Canonical-WAL wire-format compatibility (ADR-097 data format 2).
//!
//! Entries are `rmp_serde::to_vec_named` maps, so a field may be ADDED to
//! `CanonicalWalEntry` or `ProximaRecord` if it is `#[serde(default)]` (or an
//! `Option`): old entries decode with the default, and older binaries ignore
//! the new key. Renaming or removing a field, or changing a variant's shape,
//! makes every existing WAL unreadable — the frozen bytes below catch that.
//!
//! Never regenerate `GOLDEN` to make a failure go away. A new data format adds
//! a new table beside this one (`cargo test -p proximadb-storage-common --test
//! wal_format_compat print_golden -- --ignored --nocapture`).
//!
//!   cargo test -p proximadb-storage-common --test wal_format_compat

use proximadb_records::{ProximaRecord, ProximaTreeNode, ProximaValue};
use proximadb_storage_common::{CanonicalOperation, CanonicalWalEntry};
use serde::Serialize;

fn record(oid: &str, tenant: &str, branch: Option<&str>) -> ProximaRecord {
    let mut record = ProximaRecord {
        oid: oid.to_string(),
        tenant_id: tenant.to_string(),
        record_version: 3,
        created_at_ns: 1_700_000_000_000_000_000,
        updated_at_ns: 1_700_000_000_000_000_001,
        branch_id: branch.map(str::to_string),
        ..ProximaRecord::default()
    };
    // One prop: `props` is a HashMap, so more than one key has no fixed order.
    record.props.insert(
        "price".to_string(),
        ProximaTreeNode::Value(ProximaValue::Int64(42)),
    );
    record
}

/// The entries `GOLDEN` was written from: one of each operation the record
/// tier and the catalog stores replay.
fn samples() -> Vec<CanonicalWalEntry> {
    let entry = |sequence_number, operation, tenant_id: Option<&str>| CanonicalWalEntry {
        sequence_number,
        timestamp_ms: 1_760_000_000_000,
        operation,
        tenant_id: tenant_id.map(str::to_string),
    };
    vec![
        entry(
            7,
            CanonicalOperation::RecordUpsert {
                collection_id: "orders".to_string(),
                record: Box::new(record("o1", "t1", None)),
                projections: Vec::new(),
            },
            Some("t1"),
        ),
        entry(
            8,
            CanonicalOperation::RecordUpsert {
                collection_id: "orders".to_string(),
                record: Box::new(record("o2", "", Some("feature"))),
                projections: Vec::new(),
            },
            None,
        ),
        entry(
            9,
            CanonicalOperation::RecordDelete {
                collection_id: "orders".to_string(),
                oid: "o1".to_string(),
                projections: Vec::new(),
            },
            Some("t1"),
        ),
        entry(
            10,
            CanonicalOperation::RecordPartitionDrop {
                collection_id: "orders".to_string(),
            },
            None,
        ),
        entry(
            11,
            CanonicalOperation::CatalogMutation {
                key: "fn:double".to_string(),
                bytes: vec![1, 2, 3],
            },
            None,
        ),
    ]
}

/// The fields replay depends on, rendered stably (independent of fields a
/// later version adds with defaults).
fn summary(entry: &CanonicalWalEntry) -> String {
    let op = match &entry.operation {
        CanonicalOperation::RecordUpsert {
            collection_id,
            record,
            projections,
        } => format!(
            "upsert {collection_id} oid={} tenant={:?} version={} created={} updated={} \
             branch={:?} props={:?} projections={}",
            record.oid,
            record.tenant_id,
            record.record_version,
            record.created_at_ns,
            record.updated_at_ns,
            record.branch_id,
            record.props,
            projections.len()
        ),
        CanonicalOperation::RecordDelete {
            collection_id,
            oid,
            projections,
        } => format!(
            "delete {collection_id} oid={oid} projections={}",
            projections.len()
        ),
        CanonicalOperation::RecordPartitionDrop { collection_id } => {
            format!("drop {collection_id}")
        }
        CanonicalOperation::CatalogMutation { key, bytes } => {
            format!("catalog {key} bytes={bytes:?}")
        }
        other => format!("unexpected {other:?}"),
    };
    format!(
        "seq={} ts={} tenant={:?} {op}",
        entry.sequence_number, entry.timestamp_ms, entry.tenant_id
    )
}

fn to_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn from_hex(hex: &str) -> Vec<u8> {
    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).expect("hex"))
        .collect()
}

/// Bytes written by data-format-2 binaries for `samples()`, in order
/// (frozen 2026-10-22).
#[rustfmt::skip]
const GOLDEN: &[&str] = &[
    "84af73657175656e63655f6e756d62657207ac74696d657374616d705f6d73cf00000199c82cc000a96f7065726174696f6e81ac5265636f726455707365727483ad636f6c6c656374696f6e5f6964a66f7264657273a67265636f7264de0018a36f6964a26f31a86c6f63616c5f6964c0a3746964c0ac766172696174696f6e5f6964c0ae7265636f72645f76657273696f6e03ac737065635f76657273696f6e01a974656e616e745f6964a27431b47065726d69747465645f7072696e636970616c7390ad726c735f706f6c6963795f6964c0ad637265617465645f61745f6e73cf17979cfe362a0000ad757064617465645f61745f6e73cf17979cfe362a0001ad76616c69645f66726f6d5f6e73c0ab76616c69645f746f5f6e73c0a66f726967696ec0a56163746f72c0a66d6574686f64c0ab6d656d6f72795f74797065c0a570726f707381a5707269636581a556616c756581a5496e7436342aa47265667390a465646765c0aa656d62656464696e677390a873657175656e6365c0a66c6162656c7390a96272616e63685f6964c0ab70726f6a656374696f6e7390a974656e616e745f6964a27431",
    "84af73657175656e63655f6e756d62657208ac74696d657374616d705f6d73cf00000199c82cc000a96f7065726174696f6e81ac5265636f726455707365727483ad636f6c6c656374696f6e5f6964a66f7264657273a67265636f7264de0018a36f6964a26f32a86c6f63616c5f6964c0a3746964c0ac766172696174696f6e5f6964c0ae7265636f72645f76657273696f6e03ac737065635f76657273696f6e01a974656e616e745f6964a0b47065726d69747465645f7072696e636970616c7390ad726c735f706f6c6963795f6964c0ad637265617465645f61745f6e73cf17979cfe362a0000ad757064617465645f61745f6e73cf17979cfe362a0001ad76616c69645f66726f6d5f6e73c0ab76616c69645f746f5f6e73c0a66f726967696ec0a56163746f72c0a66d6574686f64c0ab6d656d6f72795f74797065c0a570726f707381a5707269636581a556616c756581a5496e7436342aa47265667390a465646765c0aa656d62656464696e677390a873657175656e6365c0a66c6162656c7390a96272616e63685f6964a766656174757265ab70726f6a656374696f6e7390a974656e616e745f6964c0",
    "84af73657175656e63655f6e756d62657209ac74696d657374616d705f6d73cf00000199c82cc000a96f7065726174696f6e81ac5265636f726444656c65746583ad636f6c6c656374696f6e5f6964a66f7264657273a36f6964a26f31ab70726f6a656374696f6e7390a974656e616e745f6964a27431",
    "84af73657175656e63655f6e756d6265720aac74696d657374616d705f6d73cf00000199c82cc000a96f7065726174696f6e81b35265636f7264506172746974696f6e44726f7081ad636f6c6c656374696f6e5f6964a66f7264657273a974656e616e745f6964c0",
    "84af73657175656e63655f6e756d6265720bac74696d657374616d705f6d73cf00000199c82cc000a96f7065726174696f6e81af436174616c6f674d75746174696f6e82a36b6579a9666e3a646f75626c65a5627974657393010203a974656e616e745f6964c0",
];

#[test]
#[ignore = "generator: prints GOLDEN for a NEW format table; never to overwrite one"]
fn print_golden() {
    for entry in samples() {
        println!(
            "    \"{}\",",
            to_hex(&rmp_serde::to_vec_named(&entry).expect("encode"))
        );
    }
}

#[test]
fn format_2_bytes_still_decode_to_the_same_entries() {
    let samples = samples();
    assert_eq!(GOLDEN.len(), samples.len(), "one frozen entry per sample");
    for (hex, sample) in GOLDEN.iter().zip(&samples) {
        let decoded: CanonicalWalEntry =
            rmp_serde::from_slice(&from_hex(hex)).expect("format-2 bytes must decode");
        assert_eq!(summary(&decoded), summary(sample));
    }
}

/// What a LATER binary writes: today's entry plus a field this binary has never
/// heard of. Older binaries must ignore it (the forward half of the rule).
#[derive(Serialize)]
struct FutureEntry<'a> {
    sequence_number: u64,
    timestamp_ms: u64,
    operation: &'a CanonicalOperation,
    tenant_id: &'a Option<String>,
    added_in_a_later_format: Vec<u32>,
}

#[test]
fn an_unknown_field_from_a_newer_binary_is_ignored() {
    for sample in samples() {
        let bytes = rmp_serde::to_vec_named(&FutureEntry {
            sequence_number: sample.sequence_number,
            timestamp_ms: sample.timestamp_ms,
            operation: &sample.operation,
            tenant_id: &sample.tenant_id,
            added_in_a_later_format: vec![1, 2],
        })
        .expect("encode");
        let decoded: CanonicalWalEntry = rmp_serde::from_slice(&bytes).expect("decode");
        assert_eq!(summary(&decoded), summary(&sample));
    }
}

/// What an OLDER binary wrote: an entry without an optional field. It must
/// decode with the field's default (the backward half of the rule).
#[derive(Serialize)]
struct EntryWithoutTenant<'a> {
    sequence_number: u64,
    timestamp_ms: u64,
    operation: &'a CanonicalOperation,
}

#[test]
fn a_missing_optional_field_decodes_as_its_default() {
    let sample = &samples()[0];
    let bytes = rmp_serde::to_vec_named(&EntryWithoutTenant {
        sequence_number: sample.sequence_number,
        timestamp_ms: sample.timestamp_ms,
        operation: &sample.operation,
    })
    .expect("encode");
    let decoded: CanonicalWalEntry = rmp_serde::from_slice(&bytes).expect("decode");
    assert_eq!(decoded.tenant_id, None);
    assert_eq!(decoded.sequence_number, sample.sequence_number);
}
