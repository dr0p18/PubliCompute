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

#[test]
fn locked_data_dir_prevents_startup_with_a_different_socket() {
    use rustix::fs::{flock, FlockOperation};

    let dir = std::env::temp_dir().join(format!("publicompute-startup-lock-{}", std::process::id()));
    fs::create_dir(&dir).unwrap();
    let owner = fs::File::open(&dir).unwrap();
    flock(&owner, FlockOperation::NonBlockingLockExclusive).unwrap();
    let socket = dir.join("other.sock");
    let output = Command::new(env!("CARGO_BIN_EXE_publicompute-agent"))
        .arg("--data-dir").arg(&dir)
        .arg("--ca").arg(dir.join("ca.pem"))
        .arg("--control-socket").arg(&socket)
        .output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(output.status.code(), Some(1));
    assert!(stdout.contains("lock data-dir") || stderr.contains("lock data-dir"), "{stdout}{stderr}");
    assert!(!socket.exists());
    fs::remove_dir_all(dir).unwrap();
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

#[tokio::test]
async fn pause_survives_service_restart_and_is_advertised_before_work() {
    use std::sync::Arc;

    use prost::Message;
    use publicompute_proto::{
        node::{
            coordinator_message, node_message, Availability, CoordinatorMessage, EnrollResult,
            Lease, LeaseRejectReason, Welcome, WorkOffer,
        },
        wire, PROTOCOL_VERSION,
    };
    use tokio::{io::AsyncWriteExt, time::timeout};

    let dir =
        std::env::temp_dir().join(format!("publicompute-pause-startup-{}", std::process::id()));
    let data = dir.join("data");
    fs::create_dir_all(&data).unwrap();
    let ca = dir.join("ca.pem");
    let fixture = rcgen::generate_simple_self_signed(vec!["localhost".into()]).unwrap();
    fs::write(&ca, fixture.cert.pem()).unwrap();
    fs::write(data.join("key.pem"), fixture.signing_key.serialize_pem()).unwrap();
    fs::write(
        data.join("identity.bin"),
        EnrollResult {
            node_id: "test-node".into(),
            certificate_chain_der: vec![fixture.cert.der().to_vec()],
            ..Default::default()
        }
        .encode_to_vec(),
    )
    .unwrap();
    let key = rustls::pki_types::PrivatePkcs8KeyDer::from(fixture.signing_key.serialize_der());
    let mut tls = rustls::ServerConfig::builder_with_provider(Arc::new(
        rustls::crypto::ring::default_provider(),
    ))
    .with_safe_default_protocol_versions()
    .unwrap()
    .with_no_client_auth()
    .with_single_cert(vec![fixture.cert.der().clone()], key.into())
    .unwrap();
    tls.alpn_protocols = vec![b"publicompute-node/0".to_vec()];
    let server = quinn::ServerConfig::with_crypto(Arc::new(
        quinn::crypto::rustls::QuicServerConfig::try_from(tls).unwrap(),
    ));
    let endpoint = quinn::Endpoint::server(server, "127.0.0.1:0".parse().unwrap()).unwrap();

    let check = timeout(Duration::from_secs(20), async {
        for (availability, command) in [
            (Availability::Idle, Some(true)),
            (Availability::PausedByHost, Some(false)),
            (Availability::Idle, None),
        ] {
            let mut agent = tokio::process::Command::new(env!("CARGO_BIN_EXE_publicompute-agent"))
                .arg("--coordinator")
                .arg(endpoint.local_addr().unwrap().to_string())
                .arg("--ca")
                .arg(&ca)
                .arg("--data-dir")
                .arg(&data)
                .args([
                    "--model-id",
                    "test-model",
                    "--weights-sha256",
                    &"00".repeat(32),
                    "--skip-nat-probe",
                ])
                .stdout(Stdio::null())
                .kill_on_drop(true)
                .spawn()
                .unwrap();
            let connection = endpoint.accept().await.unwrap().await.unwrap();
            let (mut send, mut recv) = connection.accept_bi().await.unwrap();
            let mut incoming = bytes::BytesMut::new();
            let hello = match read_node_message(&mut recv, &mut incoming).await {
                node_message::Body::Hello(hello) => hello,
                other => panic!("expected Hello, got {other:?}"),
            };
            assert_eq!(hello.availability(), availability);

            let mut outgoing = Vec::new();
            wire::encode_frame(
                &CoordinatorMessage {
                    seq: 1,
                    body: Some(coordinator_message::Body::Welcome(Welcome {
                        protocol_version: Some(PROTOCOL_VERSION),
                        ..Default::default()
                    })),
                },
                &mut outgoing,
            )
            .unwrap();
            send.write_all(&outgoing).await.unwrap();
            if availability == Availability::PausedByHost {
                outgoing.clear();
                wire::encode_frame(
                    &CoordinatorMessage {
                        seq: 2,
                        body: Some(coordinator_message::Body::WorkOffer(WorkOffer {
                            lease: Some(Lease {
                                lease_id: "in-flight".into(),
                                ..Default::default()
                            }),
                            ..Default::default()
                        })),
                    },
                    &mut outgoing,
                )
                .unwrap();
                send.write_all(&outgoing).await.unwrap();
                match read_node_message(&mut recv, &mut incoming).await {
                    node_message::Body::LeaseReject(reply) => {
                        assert_eq!(reply.lease_id, "in-flight");
                        assert_eq!(reply.reason(), LeaseRejectReason::HostBusy);
                        assert_eq!(reply.detail, "host unavailable");
                    }
                    other => panic!("expected a rejected offer, got {other:?}"),
                }
            }

            let mut socket = tokio::net::UnixStream::connect(data.join("agent.sock"))
                .await
                .unwrap();
            assert_eq!(
                control_request(&mut socket, serde_json::json!({"op": "status"})).await["status"]
                    ["paused"],
                availability == Availability::PausedByHost
            );
            if let Some(active) = command {
                assert_eq!(
                    control_request(
                        &mut socket,
                        serde_json::json!({"op": "pause", "active": active})
                    )
                    .await["ok"],
                    true
                );
            }
            agent.kill().await.unwrap();
        }
    })
    .await;
    fs::remove_dir_all(dir).unwrap();
    check.expect("timed out waiting for the agent and coordinator");

    async fn read_node_message(
        recv: &mut quinn::RecvStream,
        buf: &mut bytes::BytesMut,
    ) -> node_message::Body {
        use publicompute_proto::node::NodeMessage;
        use tokio::io::AsyncReadExt;

        loop {
            if let Some(msg) = wire::try_decode_frame::<NodeMessage>(buf).unwrap() {
                return msg.body.unwrap();
            }
            assert_ne!(
                recv.read_buf(buf).await.unwrap(),
                0,
                "coordinator stream closed"
            );
        }
    }

    async fn control_request(
        socket: &mut tokio::net::UnixStream,
        request: serde_json::Value,
    ) -> serde_json::Value {
        use tokio::io::{AsyncBufReadExt, BufReader};

        socket
            .write_all(format!("{request}\n").as_bytes())
            .await
            .unwrap();
        let mut line = String::new();
        BufReader::new(socket).read_line(&mut line).await.unwrap();
        serde_json::from_str(&line).unwrap()
    }
}
