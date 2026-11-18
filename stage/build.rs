//! Embed the native engine's library directory in the runtime search path
//! when the llama feature is enabled.
fn main() {
    println!("cargo:rerun-if-env-changed=PUBLICOMPUTE_LLAMA_LIB");
    if std::env::var_os("CARGO_FEATURE_LLAMA").is_some() {
        if let Ok(lib) = std::env::var("PUBLICOMPUTE_LLAMA_LIB") {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{lib}");
        }
    }
}
