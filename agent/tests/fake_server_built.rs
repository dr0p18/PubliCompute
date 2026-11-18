//! Refer to the fixture binaries from an integration test so Cargo builds
//! them before tests try to launch them.

#[test]
fn fake_llama_server_exists() {
    assert!(std::path::Path::new(env!("CARGO_BIN_EXE_fake-llama-server")).exists());
}

#[test]
fn fake_stage_exists() {
    assert!(std::path::Path::new(env!("CARGO_BIN_EXE_fake-stage")).exists());
}
