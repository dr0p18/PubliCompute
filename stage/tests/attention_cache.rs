//! Set PUBLICOMPUTE_MINI_GGUF to the mini Qwen hybrid model for f16 cache tests.
//! Set PUBLICOMPUTE_QUANT_KV_GGUF to a model with separate K and V caches and
//! key and value head dimensions divisible by 32 for q8_0 and q4_0 caches.
//! PUBLICOMPUTE_TEST_NGL defaults to 0 because the mini Qwen model can crash
//! in the CUDA gated delta net kernel when layers are offloaded.
#![cfg(feature = "llama")]

use publicompute_llama_sys as sys;
use std::ffi::CString;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

fn model(variable: &str) -> Option<PathBuf> {
    let path = std::env::var_os(variable)
        .map(PathBuf::from)
        .filter(|p| p.is_file());
    if path.is_none() {
        eprintln!("{variable} fixture missing; skipping");
    }
    path
}

fn gpu_layers() -> i32 {
    std::env::var("PUBLICOMPUTE_TEST_NGL").map_or(0, |value| value.parse().unwrap())
}

fn check_context_logs(model: &Path, key_type: &str, value_type: &str, flash_attn: &str) {
    let tmp = tempfile::tempdir().unwrap();
    let prompt = tmp.path().join("prompts.txt");
    std::fs::write(&prompt, "Hello\nGood morning\n").unwrap();
    let threads = std::env::var("PUBLICOMPUTE_TEST_THREADS").unwrap_or_else(|_| "2".into());
    let output = Command::new(env!("CARGO_BIN_EXE_publicompute-stage"))
        .args(["reference", "--model"])
        .arg(model)
        .arg("--prompt")
        .arg(prompt)
        .args([
            "--n-prompt",
            "2",
            "--max-tokens",
            "1",
            "--slots",
            "2",
            "--groups",
            "2",
            "--ctx-per-slot",
            "256",
            "--n-batch",
            "64",
            "--ngl",
            &gpu_layers().to_string(),
            "--threads",
            &threads,
            "--cache-type-k",
            key_type,
            "--cache-type-v",
            value_type,
            "--flash-attn",
            flash_attn,
        ])
        .stdout(Stdio::null())
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stderr}");

    for group in 0..2 {
        let prefix = format!("[engine] group {group} main context: slots=1 ");
        let line = stderr
            .lines()
            .find(|line| line.starts_with(&prefix))
            .unwrap_or_else(|| panic!("missing log for group {group}: {stderr}"));
        for (field, value) in [
            ("attention_cache", "present"),
            ("cache_type_k_requested", key_type),
            ("cache_type_v_requested", value_type),
            ("cache_type_k_effective", key_type),
            ("cache_type_v_effective", value_type),
            ("flash_attn_requested", flash_attn),
        ] {
            let expected = format!("{field}={value}");
            assert!(
                line.split_whitespace().any(|word| word == expected),
                "{line}"
            );
        }
        let effective = line
            .split_whitespace()
            .find_map(|word| word.strip_prefix("flash_attn_effective="));
        match flash_attn {
            "auto" => assert!(matches!(effective, Some("on" | "off")), "{line}"),
            explicit => assert_eq!(effective, Some(explicit), "{line}"),
        }
    }
}

#[test]
fn null_context_reports_no_attention_cache() {
    let info = unsafe { sys::llama_get_attention_cache_info(std::ptr::null()) };
    assert_eq!(info.type_k, sys::ggml_type_GGML_TYPE_COUNT);
    assert_eq!(info.type_v, sys::ggml_type_GGML_TYPE_COUNT);
    assert!(!info.flash_attn);
    assert!(!info.has_attention_cache);
}

#[test]
fn q8_cache_with_flash_attention_is_effective() {
    let Some(model) = model("PUBLICOMPUTE_QUANT_KV_GGUF") else {
        return;
    };
    check_context_logs(&model, "q8_0", "q8_0", "on");
}

#[test]
fn q4_cache_with_flash_attention_is_effective() {
    let Some(model) = model("PUBLICOMPUTE_QUANT_KV_GGUF") else {
        return;
    };
    check_context_logs(&model, "q4_0", "q4_0", "on");
}

#[test]
fn quantized_key_and_f16_value_cache_with_flash_attention_off_are_effective() {
    let Some(model) = model("PUBLICOMPUTE_QUANT_KV_GGUF") else {
        return;
    };
    check_context_logs(&model, "q8_0", "f16", "off");
}

#[test]
fn quantized_value_cache_with_flash_attention_off_is_rejected() {
    let Some(model) = model("PUBLICOMPUTE_QUANT_KV_GGUF") else {
        return;
    };
    let path = CString::new(model.as_os_str().as_encoded_bytes()).unwrap();
    unsafe {
        sys::llama_backend_init();
        let mut model_params = sys::llama_model_default_params();
        model_params.n_gpu_layers = gpu_layers();
        let model = sys::llama_model_load_from_file(path.as_ptr(), model_params);
        assert!(!model.is_null(), "could not load the cache test model");

        let mut params = sys::llama_context_default_params();
        params.n_ctx = 256;
        params.n_batch = 64;
        params.n_ubatch = 64;
        params.type_k = sys::ggml_type_GGML_TYPE_Q8_0;
        params.type_v = sys::ggml_type_GGML_TYPE_Q8_0;
        params.flash_attn_type = sys::llama_flash_attn_type_LLAMA_FLASH_ATTN_TYPE_DISABLED;
        let ctx = sys::llama_init_from_model(model, params);
        if !ctx.is_null() {
            sys::llama_free(ctx);
        }
        sys::llama_model_free(model);
        assert!(
            ctx.is_null(),
            "the native engine accepted q8_0 V with FlashAttention off"
        );
    }
}

#[test]
fn f16_cache_with_auto_logs_the_resolved_flash_attention_mode() {
    let Some(model) = model("PUBLICOMPUTE_MINI_GGUF") else {
        return;
    };
    check_context_logs(&model, "f16", "f16", "auto");
}

#[test]
fn f16_cache_with_flash_attention_off_is_effective() {
    let Some(model) = model("PUBLICOMPUTE_MINI_GGUF") else {
        return;
    };
    check_context_logs(&model, "f16", "f16", "off");
}

#[test]
fn f16_cache_with_flash_attention_on_is_effective() {
    let Some(model) = model("PUBLICOMPUTE_MINI_GGUF") else {
        return;
    };
    check_context_logs(&model, "f16", "f16", "on");
}
