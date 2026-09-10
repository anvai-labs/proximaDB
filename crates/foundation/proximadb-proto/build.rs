//! Gated regeneration of the v2 record protobuf/gRPC stubs.
//!
//! Normal builds use the committed `src/proto/proximadb.v2.rs` (this build script
//! is a no-op). To regenerate after editing `proto/proximadb/v2/record.proto`:
//!
//! ```sh
//! PROXIMADB_PROTO_REGEN=1 cargo build -p proximadb-proto
//! ```
//!
//! Only the self-contained `proximadb.v2` package is regenerated here, so the
//! hand-maintained serde on the v1 catalog types (in `proximadb.v1.rs`) is never
//! touched. Requires `protoc` on PATH.

fn main() {
    if std::env::var_os("PROXIMADB_PROTO_REGEN").is_none() {
        return;
    }

    let manifest = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");
    let manifest_dir = std::path::Path::new(&manifest);
    // crates/foundation/proximadb-proto -> repo root
    let repo_root = manifest_dir
        .ancestors()
        .nth(3)
        .expect("repo root from crate manifest");

    // All files share the `proximadb.v2` package and are merged into the
    // single generated `proximadb.v2.rs`.
    let record_proto = repo_root.join("proto/proximadb/v2/record.proto");
    let graph_proto = repo_root.join("proto/proximadb/v2/graph.proto");
    let document_proto = repo_root.join("proto/proximadb/v2/document.proto");
    let fusion_proto = repo_root.join("proto/proximadb/v2/fusion.proto");
    let entity_proto = repo_root.join("proto/proximadb/v2/entity.proto");
    let ledger_proto = repo_root.join("proto/proximadb/v2/ledger.proto");
    let model_registry_proto = repo_root.join("proto/proximadb/v2/model_registry.proto");
    let include = repo_root.join("proto");
    let out_dir = manifest_dir.join("src/proto");

    println!(
        "cargo:warning=regenerating {} + {} + {} + {} + {} + {} + {} -> {}",
        record_proto.display(),
        graph_proto.display(),
        document_proto.display(),
        fusion_proto.display(),
        entity_proto.display(),
        ledger_proto.display(),
        model_registry_proto.display(),
        out_dir.display()
    );
    // D1: also emit the v2 file-descriptor set so gRPC server REFLECTION can
    // serve the v2 services (the checked-in v1 descriptors are the only ones
    // registered today — stale after the v1 sunset). Checked in next to the
    // v1 bins (`src/proto/`) so the runtime registers it without a build step.
    let file_descriptor_set = out_dir.join("proximadb.v2_descriptor.bin");

    tonic_prost_build::configure()
        .build_server(true)
        .build_client(true)
        .file_descriptor_set_path(&file_descriptor_set)
        .out_dir(&out_dir)
        .compile_protos(
            &[
                record_proto,
                graph_proto,
                document_proto,
                fusion_proto,
                entity_proto,
                ledger_proto,
                model_registry_proto,
            ],
            &[include],
        )
        .expect("v2 proto codegen failed");
}
