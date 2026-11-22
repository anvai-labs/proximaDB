fn main() {
    // The proto crate owns its generated sources. Keep this root build script
    // std-only so every server/test build can start without host-side codecs.
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var_os("CARGO_FEATURE_PYLIB").is_some() {
        println!("cargo:rust-cdylib=proximadb");
    }
}
