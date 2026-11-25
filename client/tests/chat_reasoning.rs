use std::{
    fs::{self, OpenOptions},
    io::{BufRead, BufReader, Read, Write},
    net::TcpListener,
    os::unix::fs::{OpenOptionsExt, PermissionsExt},
    process::{Command, Output, Stdio},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use serde_json::{json, Value};

fn chat(question: &[&str], input: &[u8], replies: &[(&str, &str)]) -> (Output, Vec<Value>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let api = format!("http://{}", listener.local_addr().unwrap());
    let config = tempfile::tempdir().unwrap();
    let dir = config.path().join("publicompute");
    fs::create_dir(&dir).unwrap();
    fs::set_permissions(&dir, fs::Permissions::from_mode(0o700)).unwrap();
    let session = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(dir.join("session"))
        .unwrap();
    let expires_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
        + 60;
    serde_json::to_writer(
        session,
        &json!({"api": api, "token": "fixture_token", "expires_at": expires_at}),
    )
    .unwrap();

    thread::scope(|scope| {
        let server = scope.spawn(move || {
            let mut requests = Vec::new();
            let deadline = Instant::now() + Duration::from_secs(5);
            for (text, reason) in replies {
                let (mut socket, _) = loop {
                    match listener.accept() {
                        Ok(connection) => break connection,
                        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock && Instant::now() < deadline => {
                            thread::sleep(Duration::from_millis(5));
                        }
                        Err(e) => panic!("test API did not receive a request: {e}"),
                    }
                };
                socket.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
                let mut reader = BufReader::new(&mut socket);
                let mut size = 0;
                loop {
                    let mut line = String::new();
                    assert_ne!(reader.read_line(&mut line).unwrap(), 0);
                    if line == "\r\n" {
                        break;
                    }
                    if let Some(value) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                        size = value.trim().parse().unwrap();
                    }
                }
                let mut body = vec![0; size];
                reader.read_exact(&mut body).unwrap();
                requests.push(serde_json::from_slice::<Value>(&body).unwrap());
                write!(socket, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n\
                    data: {}\n\ndata: {}\n\ndata: [DONE]\n\n",
                    json!({"choices":[{"delta":{"content":text}}]}),
                    json!({"choices":[{"finish_reason":reason}]})).unwrap();
            }
            requests
        });
        let mut child = Command::new(env!("CARGO_BIN_EXE_publicompute"))
            .args(["chat", "--api", &api, "--model", "qwen3.6"])
            .args(question)
            .env("XDG_CONFIG_HOME", config.path())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        child.stdin.take().unwrap().write_all(input).unwrap();
        (child.wait_with_output().unwrap(), server.join().unwrap())
    })
}

#[test]
fn token_limit_during_reasoning_exits_with_status_one() {
    for question in [&["question"][..], &[]] {
        let (output, _) = chat(
            question,
            b"question\n/exit\n",
            &[("private reasoning", "length")],
        );
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert_eq!(output.status.code(), Some(1), "{stderr}");
        assert!(output.stdout.is_empty());
        assert!(stderr.contains("the token limit was reached during reasoning"));
        assert!(!stderr.contains("private reasoning"));
    }
}

#[test]
fn an_empty_answer_leaves_no_turn_in_history() {
    let (output, requests) = chat(
        &[],
        b"first\nsecond\n/exit\n",
        &[
            ("private reasoning</think>", "stop"),
            ("other reasoning</think>answer", "stop"),
        ],
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stdout, b"answer");
    assert_eq!(
        requests[1]["messages"],
        json!([{"role":"user","content":"second"}])
    );
}
