//! Link the compiler-rt availability helper needed by the prebuilt macOS ONNX Runtime.
//!
//! The static ORT archive ships CoreML objects that reference
//! `___isPlatformVersionAtLeast`. The C driver supplies compiler-rt implicitly,
//! but rustc omits the driver's default libraries. Link it through this crate
//! so the fix reaches embedding, ranking, and their downstream binaries.

use std::env;
use std::path::PathBuf;
use std::process::Command;

const ARCHIVE: &str = "libclang_rt.osx.a";

fn main() -> Result<(), env::VarError> {
    let target = env::var("TARGET")?;
    if !target.ends_with("-apple-darwin") {
        return Ok(());
    }

    println!("cargo:rerun-if-changed=build.rs");
    for name in [
        "DOCS_RS",
        "ORT_LIB_PATH",
        "ORT_LIB_LOCATION",
        "ORT_PREFER_DYNAMIC_LINK",
        "DEVELOPER_DIR",
        "PATH",
        "SDKROOT",
    ] {
        println!("cargo:rerun-if-env-changed={name}");
    }
    // Match ort-sys: the dynamic preference applies to a user-provided ORT
    // installation. The downloaded distribution is static.
    let uses_shared_ort = (env::var("ORT_LIB_PATH").is_ok()
        || env::var("ORT_LIB_LOCATION").is_ok())
        && env::var("ORT_PREFER_DYNAMIC_LINK")
            .is_ok_and(|value| value == "1" || value.eq_ignore_ascii_case("true"));
    if env::var("DOCS_RS").is_ok() || uses_shared_ort {
        return Ok(());
    }

    let compiler = cc::Build::new().no_default_flags(true).try_get_compiler();
    let archive = compiler
        .map_err(|error| error.to_string())
        .and_then(|tool| find_archive(tool.to_command(), &target))
        .or_else(|reason| {
            // A configured compiler can build C code without providing the
            // Apple runtime. Try the selected Xcode toolchain in that case.
            let mut clang = Command::new("xcrun");
            clang.args(["--sdk", "macosx", "clang"]);
            find_archive(clang, &target).map_err(|fallback| format!("{reason}; {fallback}"))
        });

    match archive {
        Ok(path) => {
            if let Some(dir) = path.parent() {
                println!("cargo:rustc-link-search=native={}", dir.display());
                println!("cargo:rustc-link-lib=static=clang_rt.osx");
                println!("cargo:rerun-if-changed={}", path.display());
            }
        }
        Err(reason) => println!(
            "cargo:warning=Could not find {ARCHIVE}: {reason}. Install the Xcode Command Line Tools \
             or select a Clang toolchain with compiler-rt to link the prebuilt ONNX Runtime."
        ),
    }
    Ok(())
}

fn find_archive(mut compiler: Command, target: &str) -> Result<PathBuf, String> {
    let output = compiler
        .arg(format!("--target={target}"))
        .arg(format!("-print-file-name={ARCHIVE}"))
        .output()
        .map_err(|error| format!("{compiler:?}: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "{compiler:?}: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }

    let name = std::str::from_utf8(&output.stdout).map_err(|error| error.to_string())?;
    let path = PathBuf::from(name.trim());
    if path.is_absolute() && path.is_file() {
        Ok(path)
    } else {
        Err(format!("{compiler:?} returned no archive at {path:?}"))
    }
}
