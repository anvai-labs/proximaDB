//! Build script for `proximadb-onnx-runtime`.
//!
//! ## Why this crate has a build script
//!
//! `ort`'s prebuilt macOS archive (`libonnxruntime.a`, pulled by `download-binaries`) ships the
//! CoreML execution provider. Its Objective-C++ objects guard Apple APIs with `@available`, which
//! the compiler lowers to a call to `___isPlatformVersionAtLeast`. That helper lives in
//! compiler-rt (`libclang_rt.osx.a`), not in `libSystem`. The `cc` driver adds compiler-rt to every
//! link it drives, but rustc links with `-nodefaultlibs -lSystem`, so nothing provides the symbol
//! and every final link that contains ORT fails with
//! `Undefined symbols for architecture arm64: "___isPlatformVersionAtLeast"`.
//! `ort-sys` 2.0.0-rc.15 links `c++` and the `Foundation` / `CoreML` frameworks, but not
//! compiler-rt.
//!
//! Every consumer reaches `ort` through this crate, so this is the single place the fix belongs:
//! `cargo:rustc-link-lib` and `cargo:rustc-link-search` propagate to every dependent's link.
//!
//! Only the static archive matters: under `ort/load-dynamic` (`disable-linking`) `ort-sys` links
//! nothing, and a shared `libonnxruntime` resolved ORT's own references when it was built. That is
//! why a failed lookup warns instead of failing the build — whether the symbol is needed at all is
//! not observable from here.

use std::env;
use std::error::Error;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Archive to link, both as the rustc library name (`-l static=…`) and as the file name inside
/// `<resource-dir>/lib/darwin/`.
const COMPILER_RT: &str = "clang_rt.osx";

fn main() -> Result<(), Box<dyn Error>> {
    let target = env::var("TARGET")?;
    if !target.ends_with("-apple-darwin") {
        return Ok(());
    }
    match locate_compiler_rt(&target) {
        Ok(LibclangRt { dir, file }) => {
            println!("cargo:rustc-link-search=native={}", dir.display());
            println!("cargo:rustc-link-lib=static={COMPILER_RT}");
            // An Xcode update relocates the archive (`…/clang/<version>/lib/darwin`); re-locate then.
            println!("cargo:rerun-if-changed={}", file.display());
        }
        Err(reason) => {
            println!(
                "cargo:warning=could not locate compiler-rt \
                 (lib{COMPILER_RT}.a): {reason}. Links that contain ONNX Runtime's static archive \
                 will then fail with `___isPlatformVersionAtLeast` undefined — install the Xcode \
                 Command Line Tools, or point `CC` at a clang driver."
            );
        }
    }
    Ok(())
}

/// Where compiler-rt lives on this machine.
struct LibclangRt {
    /// `<resource-dir>/lib/darwin`, for `cargo:rustc-link-search`.
    dir: PathBuf,
    /// The archive itself, for `cargo:rerun-if-changed`.
    file: PathBuf,
}

/// Ask the target's C compiler driver where its resource directory is and return compiler-rt's
/// location, provided the archive really exists there.
///
/// Compiler selection mirrors the `cc` crate (`CC_<target>` with hyphens or underscores, then
/// `TARGET_CC`, then `CC`, then `cc`) so a cross-compile toolchain is honoured.
fn locate_compiler_rt(target: &str) -> Result<LibclangRt, String> {
    let underscored = target.replace('-', "_");
    let candidates = [
        format!("CC_{target}"),
        format!("CC_{underscored}"),
        "TARGET_CC".to_string(),
        "CC".to_string(),
    ];
    for name in &candidates {
        println!("cargo:rerun-if-env-changed={name}");
    }
    let compiler = match candidates.iter().find_map(|name| env::var(name).ok()) {
        Some(configured) => configured,
        None => String::from("cc"),
    };

    let output = Command::new(&compiler)
        .arg("-print-resource-dir")
        .output()
        .map_err(|error| format!("`{compiler} -print-resource-dir` failed to start: {error}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(format!(
            "`{compiler} -print-resource-dir` exited with {}: {stderr}",
            output.status
        ));
    }
    let resource_dir = String::from_utf8(output.stdout)
        .map_err(|error| format!("`{compiler} -print-resource-dir` printed non-UTF-8: {error}"))?;
    let resource_dir = Path::new(resource_dir.trim());
    if resource_dir.as_os_str().is_empty() {
        return Err(format!("`{compiler} -print-resource-dir` printed nothing"));
    }
    let dir = resource_dir.join("lib").join("darwin");
    let file = dir.join(format!("lib{COMPILER_RT}.a"));
    if !file.is_file() {
        return Err(format!(
            "`{} -print-resource-dir` resolved to `{}`, which has no `{}`",
            compiler,
            resource_dir.display(),
            file.display()
        ));
    }
    Ok(LibclangRt { dir, file })
}
