//! Canonical-WAL wire-format compatibility (ADR-097 data format 2).
//!
//! Entries are `rmp_serde::to_vec_named` maps (single-entry frames and group
//! frames alike), so a field may be ADDED to `CanonicalWalEntry`,
//! `ProximaRecord` or their parts when it is `#[serde(default)]` or an
//! `Option`: old entries decode with the default and older binaries ignore
//! the new key. Two frozen-byte checks hold the rest of the format still:
//!
//! * decode — the format-2 bytes must decode, and re-encode to exactly what
//!   the same sample encodes to today (every field compared, not a summary);
//! * encode — every key and value in the frozen bytes must still appear in
//!   what today's code writes for the sample, so a rename (even one kept
//!   readable with `#[serde(alias)]`), a removal or a type change fails, while
//!   an added key does not.
//!
//! The samples populate every `Option` (a renamed optional field would
//! otherwise decode silently as `None`) and cover each value type the record
//! tier and merge-base replay depend on, projections, embeddings (custom
//! serde) and a checkpoint. Not covered here: the frame header (`PXWAL001`,
//! layout, CRC — root crate) and the opaque nested payloads (catalog deltas,
//! merge-base images), which their owners must pin.
//!
//! Never regenerate `GOLDEN` to make a failure go away. A new data format adds
//! a new table beside this one (`cargo test -p proximadb-storage-common --test
//! wal_format_compat print_golden -- --ignored --nocapture`).
//!
//!   cargo test -p proximadb-storage-common --test wal_format_compat

use proximadb_data_model::MemoryType;
use proximadb_records::{
    EdgeDirection, EdgeShape, EmbeddingCell, LabelSet, ProximaRecord, ProximaTreeNode,
    ProximaValue, TokenSequence, TypedRef,
};
use proximadb_storage_common::{
    CanonicalOperation, CanonicalWalEntry, ProjectionDirective, ProjectionFreshness,
    SnapshotManifest,
};
use serde::Serialize;

/// A record with EVERY field set, so a renamed field cannot decode silently
/// as its default. `props` holds one key: it is a HashMap, so more than one
/// has no fixed order.
fn record(oid: &str, tenant: &str, value: ProximaValue) -> ProximaRecord {
    let mut embedding = EmbeddingCell::new_fp32("m-1", "text", 2, vec![0.5, -1.0]);
    embedding.precision_epoch = Some(3);
    let mut labels = LabelSet::new();
    labels.insert("hot");
    let mut record = ProximaRecord {
        oid: oid.to_string(),
        local_id: Some(format!("{oid}-local")),
        tid: Some(77),
        variation_id: Some("orders".to_string()),
        record_version: 3,
        spec_version: 2,
        tenant_id: tenant.to_string(),
        permitted_principals: vec!["alice".to_string()],
        rls_policy_id: Some("rls-1".to_string()),
        created_at_ns: 1_700_000_000_000_000_000,
        updated_at_ns: 1_700_000_000_000_000_001,
        valid_from_ns: Some(1_700_000_000_000_000_002),
        valid_to_ns: Some(1_800_000_000_000_000_000),
        origin: Some("pgwire".to_string()),
        actor: Some("bob".to_string()),
        method: Some("insert".to_string()),
        memory_type: Some(MemoryType::Fact),
        refs: vec![
            TypedRef::ForeignKey {
                table: "customers".to_string(),
                id: "c-9".to_string(),
            },
            TypedRef::GraphEdge {
                edge_id: "e-1".to_string(),
                direction: EdgeDirection::Outgoing,
            },
        ],
        edge: Some(EdgeShape {
            source_id: "a".to_string(),
            target_id: "b".to_string(),
            edge_type: "knows".to_string(),
            weight: Some(0.25),
        }),
        embeddings: vec![embedding],
        sequence: Some(TokenSequence {
            tokens: vec![1, 2, 3],
            model_id: "tok-1".to_string(),
            offset: 4,
        }),
        labels,
        branch_id: Some("feature".to_string()),
        ..ProximaRecord::default()
    };
    record
        .props
        .insert("v".to_string(), ProximaTreeNode::Value(value));
    record
}

/// The entries `GOLDEN` was written from: every operation the record tier
/// and the catalog stores replay, the value types merge-base replay relies on
/// (String, UInt64, Binary) and the typed-column ones (Decimal, Float64).
fn samples() -> Vec<CanonicalWalEntry> {
    let entry = |sequence_number, operation, tenant_id: Option<&str>| CanonicalWalEntry {
        sequence_number,
        timestamp_ms: 1_760_000_000_000,
        operation,
        tenant_id: tenant_id.map(str::to_string),
    };
    let upsert = |oid: &str, tenant: &str, value, projections| CanonicalOperation::RecordUpsert {
        collection_id: "orders".to_string(),
        record: Box::new(record(oid, tenant, value)),
        projections,
    };
    vec![
        entry(
            7,
            upsert(
                "o1",
                "t1",
                ProximaValue::Int64(42),
                vec![
                    ProjectionDirective::DocumentJsonPathIndex {
                        collection_id: "orders".to_string(),
                        indexed_paths: vec!["$.tags[*]".to_string()],
                    },
                    ProjectionDirective::HnswIndex {
                        collection_id: "orders".to_string(),
                        embedding_field: "m-1".to_string(),
                    },
                ],
            ),
            Some("t1"),
        ),
        entry(
            8,
            upsert("o2", "", ProximaValue::String("s".to_string()), Vec::new()),
            None,
        ),
        entry(
            9,
            upsert("o3", "", ProximaValue::UInt64(u64::MAX), Vec::new()),
            None,
        ),
        entry(
            10,
            upsert("o4", "", ProximaValue::Binary(vec![0, 255]), Vec::new()),
            None,
        ),
        entry(
            11,
            upsert(
                "o5",
                "",
                ProximaValue::Decimal("1.50".to_string()),
                Vec::new(),
            ),
            None,
        ),
        entry(
            12,
            upsert("o6", "", ProximaValue::Float64(-2.5), Vec::new()),
            None,
        ),
        entry(
            13,
            CanonicalOperation::RecordDelete {
                collection_id: "orders".to_string(),
                oid: "o1".to_string(),
                projections: vec![ProjectionDirective::CsrRebuild {
                    graph_id: "g".to_string(),
                }],
            },
            Some("t1"),
        ),
        entry(
            14,
            CanonicalOperation::RecordPartitionDrop {
                collection_id: "orders".to_string(),
            },
            None,
        ),
        entry(
            15,
            CanonicalOperation::CatalogMutation {
                key: "fn:double".to_string(),
                bytes: vec![1, 2, 3],
            },
            None,
        ),
        entry(
            16,
            CanonicalOperation::Checkpoint(SnapshotManifest {
                sequence_number: 15,
                timestamp_ms: 1_760_000_000_001,
                collection_ids: vec!["orders".to_string()],
                projection_freshness: vec![ProjectionFreshness {
                    projection_name: "doc_json_path:orders".to_string(),
                    last_rebuilt_lsn: 14,
                    stale: false,
                }],
            }),
            None,
        ),
    ]
}

fn encode(entry: &CanonicalWalEntry) -> Vec<u8> {
    rmp_serde::to_vec_named(entry).expect("encode")
}

/// `new` holds everything `old` does: every key with an equal value
/// (recursively), arrays element-wise. Keys only `new` has are additions.
fn covers(new: &serde_json::Value, old: &serde_json::Value, path: &str) -> Result<(), String> {
    use serde_json::Value;
    match (new, old) {
        (Value::Object(new), Value::Object(old)) => old.iter().try_for_each(|(key, old)| {
            let path = format!("{path}.{key}");
            match new.get(key) {
                Some(new) => covers(new, old, &path),
                None => Err(format!("{path}: written by format 2, missing now")),
            }
        }),
        (Value::Array(new), Value::Array(old)) if new.len() == old.len() => new
            .iter()
            .zip(old)
            .enumerate()
            .try_for_each(|(i, (new, old))| covers(new, old, &format!("{path}[{i}]"))),
        _ if new == old => Ok(()),
        _ => Err(format!("{path}: format 2 wrote {old}, now {new}")),
    }
}

fn as_json(bytes: &[u8]) -> serde_json::Value {
    rmp_serde::from_slice(bytes).expect("msgpack as a generic value")
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
    "84af73657175656e63655f6e756d62657207ac74696d657374616d705f6d73cf00000199c82cc000a96f7065726174696f6e81ac5265636f726455707365727483ad636f6c6c656374696f6e5f6964a66f7264657273a67265636f7264de0018a36f6964a26f31a86c6f63616c5f6964a86f312d6c6f63616ca37469644dac766172696174696f6e5f6964a66f7264657273ae7265636f72645f76657273696f6e03ac737065635f76657273696f6e02a974656e616e745f6964a27431b47065726d69747465645f7072696e636970616c7391a5616c696365ad726c735f706f6c6963795f6964a5726c732d31ad637265617465645f61745f6e73cf17979cfe362a0000ad757064617465645f61745f6e73cf17979cfe362a0001ad76616c69645f66726f6d5f6e73cf17979cfe362a0002ab76616c69645f746f5f6e73cf18fae27693b40000a66f726967696ea6706777697265a56163746f72a3626f62a66d6574686f64a6696e73657274ab6d656d6f72795f74797065a466616374a570726f707381a17681a556616c756581a5496e7436342aa4726566739281aa466f726569676e4b657982a57461626c65a9637573746f6d657273a26964a3632d3981a947726170684564676582a7656467655f6964a3652d31a9646972656374696f6ea84f7574676f696e67a46564676584a9736f757263655f6964a161a97461726765745f6964a162a9656467655f74797065a56b6e6f7773a6776569676874cb3fd0000000000000aa656d62656464696e67739186a86d6f64656c5f6964a36d2d31a86d6f64616c697479a474657874a676616c75657392ca3f000000cabf800000a364696d02a9707265636973696f6e01af707265636973696f6e5f65706f636803a873657175656e636583a6746f6b656e7393010203a86d6f64656c5f6964a5746f6b2d31a66f666673657404a66c6162656c7391a3686f74a96272616e63685f6964a766656174757265ab70726f6a656374696f6e739281b5446f63756d656e744a736f6e50617468496e64657882ad636f6c6c656374696f6e5f6964a66f7264657273ad696e64657865645f706174687391a9242e746167735b2a5d81a9486e7377496e64657882ad636f6c6c656374696f6e5f6964a66f7264657273af656d62656464696e675f6669656c64a36d2d31a974656e616e745f6964a27431",
    "84af73657175656e63655f6e756d62657208ac74696d657374616d705f6d73cf00000199c82cc000a96f7065726174696f6e81ac5265636f726455707365727483ad636f6c6c656374696f6e5f6964a66f7264657273a67265636f7264de0018a36f6964a26f32a86c6f63616c5f6964a86f322d6c6f63616ca37469644dac766172696174696f6e5f6964a66f7264657273ae7265636f72645f76657273696f6e03ac737065635f76657273696f6e02a974656e616e745f6964a0b47065726d69747465645f7072696e636970616c7391a5616c696365ad726c735f706f6c6963795f6964a5726c732d31ad637265617465645f61745f6e73cf17979cfe362a0000ad757064617465645f61745f6e73cf17979cfe362a0001ad76616c69645f66726f6d5f6e73cf17979cfe362a0002ab76616c69645f746f5f6e73cf18fae27693b40000a66f726967696ea6706777697265a56163746f72a3626f62a66d6574686f64a6696e73657274ab6d656d6f72795f74797065a466616374a570726f707381a17681a556616c756581a6537472696e67a173a4726566739281aa466f726569676e4b657982a57461626c65a9637573746f6d657273a26964a3632d3981a947726170684564676582a7656467655f6964a3652d31a9646972656374696f6ea84f7574676f696e67a46564676584a9736f757263655f6964a161a97461726765745f6964a162a9656467655f74797065a56b6e6f7773a6776569676874cb3fd0000000000000aa656d62656464696e67739186a86d6f64656c5f6964a36d2d31a86d6f64616c697479a474657874a676616c75657392ca3f000000cabf800000a364696d02a9707265636973696f6e01af707265636973696f6e5f65706f636803a873657175656e636583a6746f6b656e7393010203a86d6f64656c5f6964a5746f6b2d31a66f666673657404a66c6162656c7391a3686f74a96272616e63685f6964a766656174757265ab70726f6a656374696f6e7390a974656e616e745f6964c0",
    "84af73657175656e63655f6e756d62657209ac74696d657374616d705f6d73cf00000199c82cc000a96f7065726174696f6e81ac5265636f726455707365727483ad636f6c6c656374696f6e5f6964a66f7264657273a67265636f7264de0018a36f6964a26f33a86c6f63616c5f6964a86f332d6c6f63616ca37469644dac766172696174696f6e5f6964a66f7264657273ae7265636f72645f76657273696f6e03ac737065635f76657273696f6e02a974656e616e745f6964a0b47065726d69747465645f7072696e636970616c7391a5616c696365ad726c735f706f6c6963795f6964a5726c732d31ad637265617465645f61745f6e73cf17979cfe362a0000ad757064617465645f61745f6e73cf17979cfe362a0001ad76616c69645f66726f6d5f6e73cf17979cfe362a0002ab76616c69645f746f5f6e73cf18fae27693b40000a66f726967696ea6706777697265a56163746f72a3626f62a66d6574686f64a6696e73657274ab6d656d6f72795f74797065a466616374a570726f707381a17681a556616c756581a655496e743634cfffffffffffffffffa4726566739281aa466f726569676e4b657982a57461626c65a9637573746f6d657273a26964a3632d3981a947726170684564676582a7656467655f6964a3652d31a9646972656374696f6ea84f7574676f696e67a46564676584a9736f757263655f6964a161a97461726765745f6964a162a9656467655f74797065a56b6e6f7773a6776569676874cb3fd0000000000000aa656d62656464696e67739186a86d6f64656c5f6964a36d2d31a86d6f64616c697479a474657874a676616c75657392ca3f000000cabf800000a364696d02a9707265636973696f6e01af707265636973696f6e5f65706f636803a873657175656e636583a6746f6b656e7393010203a86d6f64656c5f6964a5746f6b2d31a66f666673657404a66c6162656c7391a3686f74a96272616e63685f6964a766656174757265ab70726f6a656374696f6e7390a974656e616e745f6964c0",
    "84af73657175656e63655f6e756d6265720aac74696d657374616d705f6d73cf00000199c82cc000a96f7065726174696f6e81ac5265636f726455707365727483ad636f6c6c656374696f6e5f6964a66f7264657273a67265636f7264de0018a36f6964a26f34a86c6f63616c5f6964a86f342d6c6f63616ca37469644dac766172696174696f6e5f6964a66f7264657273ae7265636f72645f76657273696f6e03ac737065635f76657273696f6e02a974656e616e745f6964a0b47065726d69747465645f7072696e636970616c7391a5616c696365ad726c735f706f6c6963795f6964a5726c732d31ad637265617465645f61745f6e73cf17979cfe362a0000ad757064617465645f61745f6e73cf17979cfe362a0001ad76616c69645f66726f6d5f6e73cf17979cfe362a0002ab76616c69645f746f5f6e73cf18fae27693b40000a66f726967696ea6706777697265a56163746f72a3626f62a66d6574686f64a6696e73657274ab6d656d6f72795f74797065a466616374a570726f707381a17681a556616c756581a642696e6172799200ccffa4726566739281aa466f726569676e4b657982a57461626c65a9637573746f6d657273a26964a3632d3981a947726170684564676582a7656467655f6964a3652d31a9646972656374696f6ea84f7574676f696e67a46564676584a9736f757263655f6964a161a97461726765745f6964a162a9656467655f74797065a56b6e6f7773a6776569676874cb3fd0000000000000aa656d62656464696e67739186a86d6f64656c5f6964a36d2d31a86d6f64616c697479a474657874a676616c75657392ca3f000000cabf800000a364696d02a9707265636973696f6e01af707265636973696f6e5f65706f636803a873657175656e636583a6746f6b656e7393010203a86d6f64656c5f6964a5746f6b2d31a66f666673657404a66c6162656c7391a3686f74a96272616e63685f6964a766656174757265ab70726f6a656374696f6e7390a974656e616e745f6964c0",
    "84af73657175656e63655f6e756d6265720bac74696d657374616d705f6d73cf00000199c82cc000a96f7065726174696f6e81ac5265636f726455707365727483ad636f6c6c656374696f6e5f6964a66f7264657273a67265636f7264de0018a36f6964a26f35a86c6f63616c5f6964a86f352d6c6f63616ca37469644dac766172696174696f6e5f6964a66f7264657273ae7265636f72645f76657273696f6e03ac737065635f76657273696f6e02a974656e616e745f6964a0b47065726d69747465645f7072696e636970616c7391a5616c696365ad726c735f706f6c6963795f6964a5726c732d31ad637265617465645f61745f6e73cf17979cfe362a0000ad757064617465645f61745f6e73cf17979cfe362a0001ad76616c69645f66726f6d5f6e73cf17979cfe362a0002ab76616c69645f746f5f6e73cf18fae27693b40000a66f726967696ea6706777697265a56163746f72a3626f62a66d6574686f64a6696e73657274ab6d656d6f72795f74797065a466616374a570726f707381a17681a556616c756581a7446563696d616ca4312e3530a4726566739281aa466f726569676e4b657982a57461626c65a9637573746f6d657273a26964a3632d3981a947726170684564676582a7656467655f6964a3652d31a9646972656374696f6ea84f7574676f696e67a46564676584a9736f757263655f6964a161a97461726765745f6964a162a9656467655f74797065a56b6e6f7773a6776569676874cb3fd0000000000000aa656d62656464696e67739186a86d6f64656c5f6964a36d2d31a86d6f64616c697479a474657874a676616c75657392ca3f000000cabf800000a364696d02a9707265636973696f6e01af707265636973696f6e5f65706f636803a873657175656e636583a6746f6b656e7393010203a86d6f64656c5f6964a5746f6b2d31a66f666673657404a66c6162656c7391a3686f74a96272616e63685f6964a766656174757265ab70726f6a656374696f6e7390a974656e616e745f6964c0",
    "84af73657175656e63655f6e756d6265720cac74696d657374616d705f6d73cf00000199c82cc000a96f7065726174696f6e81ac5265636f726455707365727483ad636f6c6c656374696f6e5f6964a66f7264657273a67265636f7264de0018a36f6964a26f36a86c6f63616c5f6964a86f362d6c6f63616ca37469644dac766172696174696f6e5f6964a66f7264657273ae7265636f72645f76657273696f6e03ac737065635f76657273696f6e02a974656e616e745f6964a0b47065726d69747465645f7072696e636970616c7391a5616c696365ad726c735f706f6c6963795f6964a5726c732d31ad637265617465645f61745f6e73cf17979cfe362a0000ad757064617465645f61745f6e73cf17979cfe362a0001ad76616c69645f66726f6d5f6e73cf17979cfe362a0002ab76616c69645f746f5f6e73cf18fae27693b40000a66f726967696ea6706777697265a56163746f72a3626f62a66d6574686f64a6696e73657274ab6d656d6f72795f74797065a466616374a570726f707381a17681a556616c756581a7466c6f61743634cbc004000000000000a4726566739281aa466f726569676e4b657982a57461626c65a9637573746f6d657273a26964a3632d3981a947726170684564676582a7656467655f6964a3652d31a9646972656374696f6ea84f7574676f696e67a46564676584a9736f757263655f6964a161a97461726765745f6964a162a9656467655f74797065a56b6e6f7773a6776569676874cb3fd0000000000000aa656d62656464696e67739186a86d6f64656c5f6964a36d2d31a86d6f64616c697479a474657874a676616c75657392ca3f000000cabf800000a364696d02a9707265636973696f6e01af707265636973696f6e5f65706f636803a873657175656e636583a6746f6b656e7393010203a86d6f64656c5f6964a5746f6b2d31a66f666673657404a66c6162656c7391a3686f74a96272616e63685f6964a766656174757265ab70726f6a656374696f6e7390a974656e616e745f6964c0",
    "84af73657175656e63655f6e756d6265720dac74696d657374616d705f6d73cf00000199c82cc000a96f7065726174696f6e81ac5265636f726444656c65746583ad636f6c6c656374696f6e5f6964a66f7264657273a36f6964a26f31ab70726f6a656374696f6e739181aa43737252656275696c6481a867726170685f6964a167a974656e616e745f6964a27431",
    "84af73657175656e63655f6e756d6265720eac74696d657374616d705f6d73cf00000199c82cc000a96f7065726174696f6e81b35265636f7264506172746974696f6e44726f7081ad636f6c6c656374696f6e5f6964a66f7264657273a974656e616e745f6964c0",
    "84af73657175656e63655f6e756d6265720fac74696d657374616d705f6d73cf00000199c82cc000a96f7065726174696f6e81af436174616c6f674d75746174696f6e82a36b6579a9666e3a646f75626c65a5627974657393010203a974656e616e745f6964c0",
    "84af73657175656e63655f6e756d62657210ac74696d657374616d705f6d73cf00000199c82cc000a96f7065726174696f6e81aa436865636b706f696e7484af73657175656e63655f6e756d6265720fac74696d657374616d705f6d73cf00000199c82cc001ae636f6c6c656374696f6e5f69647391a66f7264657273b470726f6a656374696f6e5f66726573686e6573739183af70726f6a656374696f6e5f6e616d65b4646f635f6a736f6e5f706174683a6f7264657273b06c6173745f72656275696c745f6c736e0ea57374616c65c2a974656e616e745f6964c0",
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
        // Every field, through today's encoder: equal unless a value was lost.
        assert_eq!(
            encode(&decoded),
            encode(sample),
            "seq {}: decoded differently",
            sample.sequence_number
        );
    }
}

#[test]
fn todays_encoding_still_carries_every_format_2_key_and_value() {
    for (hex, sample) in GOLDEN.iter().zip(&samples()) {
        if let Err(drift) = covers(
            &as_json(&encode(sample)),
            &as_json(&from_hex(hex)),
            &format!("seq {}", sample.sequence_number),
        ) {
            panic!("format-2 wire drift — {drift}");
        }
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
        assert_eq!(encode(&decoded), encode(&sample));
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
