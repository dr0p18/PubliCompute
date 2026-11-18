#![cfg(unix)]

use std::{
    fs,
    io::{BufRead, BufReader, Write},
    os::unix::{fs::PermissionsExt, net::UnixStream},
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant},
};

#[test]
fn socket_bound_before_engine() {
    starting_socket(true);
}

#[test]
fn socket_is_0600() {
    starting_socket(false);
}

fn starting_socket(group_access: bool) {
    let dir = std::env::temp_dir().join(format!(
        "publicompute-startup-{}-{group_access}",
        std::process::id()
    ));
    fs::create_dir_all(&dir).unwrap();
    let model = dir.join("model.gguf");
    assert!(Command::new("mkfifo")
        .arg(&model)
        .status()
        .unwrap()
        .success());
    let ca = dir.join("ca.pem");
    fs::write(
        &ca,
        rcgen::generate_simple_self_signed(vec!["localhost".into()])
            .unwrap()
            .cert
            .pem(),
    )
    .unwrap();
    let data = dir.join("data");
    let mut command = Command::new(env!("CARGO_BIN_EXE_publicompute-agent"));
    command
        .arg("--ca")
        .arg(ca)
        .arg("--data-dir")
        .arg(&data)
        .arg("--model-path")
        .arg(model)
        .args([
            "--engine",
            "llama",
            "--llama-mode",
            "managed",
            "--llama-bin",
            "/missing-llama-server",
        ]);
    let socket = if group_access {
        let socket = dir.join("custom.sock");
        command.arg("--control-socket").arg(&socket);
        socket
    } else {
        data.join("agent.sock")
    };
    let mut agent = command.stdout(Stdio::null()).spawn().unwrap();

    // A FIFO with no writer stalls hashing, and the configured engine cannot start.
    let check = (|| -> anyhow::Result<()> {
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut stream = loop {
            match UnixStream::connect(&socket) {
                Ok(stream) => break stream,
                Err(e) if agent.try_wait()?.is_some() || Instant::now() >= deadline => {
                    return Err(e.into())
                }
                Err(_) => thread::sleep(Duration::from_millis(20)),
            }
        };
        stream.set_read_timeout(Some(Duration::from_secs(2)))?;
        writeln!(stream, r#"{{"op":"status"}}"#)?;
        let mut line = String::new();
        BufReader::new(stream).read_line(&mut line)?;
        let reply: serde_json::Value = serde_json::from_str(&line)?;
        anyhow::ensure!(reply["status"]["engine"] == "starting", "{reply}");
        let mode = if group_access { 0o660 } else { 0o600 };
        anyhow::ensure!(fs::metadata(&socket)?.permissions().mode() & 0o777 == mode);
        Ok(())
    })();
    let _ = agent.kill();
    agent.wait().unwrap();
    fs::remove_dir_all(dir).unwrap();
    check.unwrap();
}
