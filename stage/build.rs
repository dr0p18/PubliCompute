//! Embed the native engine's library directory in the runtime search path
//! when the llama feature is enabled.
fn main() {
    println!("cargo:rerun-if-env-changed=PUBLICOMPUTE_LLAMA_LIB");
    for (fixture, cfg) in [
        ("PUBLICOMPUTE_Q35_GGUF", "qwen_parity_fixture"),
        ("PUBLICOMPUTE_Q35_MTP_GGUF", "qwen_mtp_parity_fixture"),
    ] {
        println!("cargo:rerun-if-env-changed={fixture}");
        println!("cargo:rustc-check-cfg=cfg({cfg})");
        if std::env::var_os(fixture).is_some_and(|path| !path.is_empty()) {
            println!("cargo:rustc-cfg={cfg}");
        }
    }
    if std::env::var_os("CARGO_FEATURE_LLAMA").is_some() {
        if let Ok(lib) = std::env::var("PUBLICOMPUTE_LLAMA_LIB") {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{lib}");
        }
    }
}
