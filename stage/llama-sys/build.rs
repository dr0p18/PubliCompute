//! Bind the native inference interface from headers in PUBLICOMPUTE_LLAMA_DIR
//! and link the shared libraries in PUBLICOMPUTE_LLAMA_LIB. Use headers that
//! match those libraries, since the generated types cross the C ABI.

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=PUBLICOMPUTE_LLAMA_DIR");
    println!("cargo:rerun-if-env-changed=PUBLICOMPUTE_LLAMA_LIB");
    println!("cargo:rerun-if-changed=wrapper.h");

    let dir = env::var("PUBLICOMPUTE_LLAMA_DIR").unwrap_or_else(|_| {
        panic!(
            "PUBLICOMPUTE_LLAMA_DIR must point to the engine sources reconstructed by engine/source.sh \
             (which contain include/llama.h)"
        )
    });
    let lib = env::var("PUBLICOMPUTE_LLAMA_LIB").unwrap_or_else(|_| {
        panic!("PUBLICOMPUTE_LLAMA_LIB must point to the directory containing libllama.so (for example <fork>/build-cpu/bin)")
    });
    let dir = PathBuf::from(dir);
    let header = dir.join("include/llama.h");
    println!("cargo:rerun-if-changed={}", header.display());
    assert!(header.exists(), "Missing {}: check PUBLICOMPUTE_LLAMA_DIR", header.display());
    assert!(
        PathBuf::from(&lib).join("libllama.so").exists(),
        "Missing libllama.so in {lib}. Build the engine with -DBUILD_SHARED_LIBS=ON"
    );

    println!("cargo:rustc-link-search=native={lib}");
    println!("cargo:rustc-link-lib=dylib=llama");
    // The GGUF functions used by the bindings are exported by libggml-base.
    println!("cargo:rustc-link-lib=dylib=ggml-base");
    // This search path applies to this crate's tests. Stage binaries set theirs
    // in stage/build.rs, and LD_LIBRARY_PATH can override either one.
    println!("cargo:rustc-link-arg=-Wl,-rpath,{lib}");

    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}", dir.join("include").display()))
        .clang_arg(format!("-I{}", dir.join("ggml/include").display()))
        .allowlist_function("llama_.*")
        .allowlist_function("gguf_.*")
        .allowlist_function("ggml_backend_tensor_get")
        .allowlist_type("llama_.*")
        .allowlist_type("gguf_.*")
        .allowlist_var("LLAMA_.*")
        .derive_default(true)
        .layout_tests(false)
        .generate()
        .expect("bindgen could not read llama.h. Install libclang-dev and clang.");
    let out = PathBuf::from(env::var("OUT_DIR").unwrap()).join("llama_bindings.rs");
    bindings.write_to_file(out).expect("failed to write generated bindings");
}
