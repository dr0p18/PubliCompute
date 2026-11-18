//! Compile the shared schemas with the Rust protobuf compiler so a build
//! needs only the schema files and Cargo dependencies.

use std::path::PathBuf;

const FILES: &[&str] = &[
    "publicompute/common/v1/common.proto",
    "publicompute/node/v1/attestation.proto",
    "publicompute/node/v1/chain.proto",
    "publicompute/node/v1/enrollment.proto",
    "publicompute/node/v1/hardware.proto",
    "publicompute/node/v1/session.proto",
    "publicompute/node/v1/work.proto",
    "publicompute/stage/service/v1/service.proto",
    "publicompute/stage/v1/stage.proto",
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    for f in FILES {
        println!("cargo:rerun-if-changed={}", root.join(f).display());
    }
    let fds = protox::compile(FILES, [&root])?;
    prost_build::Config::new().compile_fds(fds)?;
    Ok(())
}
