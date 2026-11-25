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

    let dir =
        std::env::temp_dir().join(format!("publicompute-startup-lock-{}", std::process::id()));
    fs::create_dir(&dir).unwrap();
    let owner = fs::File::open(&dir).unwrap();
    flock(&owner, FlockOperation::NonBlockingLockExclusive).unwrap();
    let socket = dir.join("other.sock");
    let output = Command::new(env!("CARGO_BIN_EXE_publicompute-agent"))
        .arg("--data-dir")
        .arg(&dir)
        .arg("--ca")
        .arg(dir.join("ca.pem"))
        .arg("--control-socket")
        .arg(&socket)
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(output.status.code(), Some(1));
    assert!(
        stdout.contains("lock data-dir") || stderr.contains("lock data-dir"),
        "{stdout}{stderr}"
    );
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
        anyhow::ensure!(reply["status"]["engine"] == "verifying weights", "{reply}");
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
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    fs::write(
        data.join("network.json"),
        serde_json::json!({
            "download_bits_per_sec": 120_000_000,
            "upload_bits_per_sec": 40_000_000,
            "bandwidth_measured_at": now,
        })
        .to_string(),
    )
    .unwrap();
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
            assert!(hello.hardware.unwrap().network.is_none());

            let mut outgoing = Vec::new();
            wire::encode_frame(
                &CoordinatorMessage {
                    seq: 1,
                    body: Some(coordinator_message::Body::Welcome(Welcome {
                        protocol_version: Some(PROTOCOL_VERSION),
                        heartbeat_interval: Some(prost_types::Duration {
                            seconds: 0,
                            nanos: 100_000_000,
                        }),
                        ..Default::default()
                    })),
                },
                &mut outgoing,
            )
            .unwrap();
            send.write_all(&outgoing).await.unwrap();
            loop {
                if let node_message::Body::Heartbeat(heartbeat) =
                    read_node_message(&mut recv, &mut incoming).await
                {
                    assert_eq!(heartbeat.network.unwrap().upload_bits_per_sec, 40_000_000);
                    break;
                }
            }
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
                loop {
                    match read_node_message(&mut recv, &mut incoming).await {
                        node_message::Body::LeaseReject(reply) => {
                            assert_eq!(reply.lease_id, "in-flight");
                            assert_eq!(reply.reason(), LeaseRejectReason::HostBusy);
                            assert_eq!(reply.detail, "host unavailable");
                            break;
                        }
                        node_message::Body::Heartbeat(_) => {}
                        other => panic!("expected a rejected offer, got {other:?}"),
                    }
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

fn cache_sparse_weights(models: &std::path::Path, chosen: &serde_json::Value) -> fs::File {
    use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
    let weights = fs::File::create(models.join(chosen["file"].as_str().unwrap())).unwrap();
    // A sparse file with a matching cache exercises the selection without reading real weights.
    weights.set_len(chosen["bytes"].as_u64().unwrap()).unwrap();
    let meta = weights.metadata().unwrap();
    let cache = fs::OpenOptions::new().create_new(true).write(true).mode(0o600)
        .open(models.join("verified.json")).unwrap();
    serde_json::to_writer(cache, &serde_json::json!([{
        "id": chosen["id"], "quant": chosen["quant"], "sha256": chosen["sha256"],
        "stamp": {
            "bytes": meta.len(), "inode": meta.ino(), "device": meta.dev(),
            "mtime_ns": meta.mtime() * 1_000_000_000 + meta.mtime_nsec(),
            "ctime_ns": meta.ctime() * 1_000_000_000 + meta.ctime_nsec()
        }
    }])).unwrap();
    weights
}

#[tokio::test]
async fn a_model_switch_exits_75_and_startup_restores_or_falls_back() {
    use serde_json::{json, Value};
    use tokio::{process::Command, time::timeout};

    let dir = std::env::temp_dir().join(format!("publicompute-switch-startup-{}-{:x}", std::process::id(), rand::random::<u64>()));
    let data = dir.join("data");
    let models = data.join("models");
    fs::create_dir_all(&models).unwrap();
    let ca = dir.join("ca.pem");
    let cert = rcgen::generate_simple_self_signed(vec!["localhost".into()]).unwrap();
    fs::write(&ca, cert.cert.pem()).unwrap();

    let manifest: Vec<Value> = serde_json::from_str(include_str!("../models.json")).unwrap();
    let chosen = &manifest[1];
    let id = chosen["id"].as_str().unwrap();
    let weights = cache_sparse_weights(&models, chosen);

    let offline = std::net::UdpSocket::bind("127.0.0.1:0").unwrap();
    let spawn = || Command::new(env!("CARGO_BIN_EXE_publicompute-agent"))
        .args(["--coordinator", &offline.local_addr().unwrap().to_string(), "--mode", "free"])
        .arg("--data-dir").arg(&data)
        .arg("--ca").arg(&ca)
        .args(["--model-id", "configured", "--weights-sha256", &"11".repeat(32), "--skip-nat-probe"])
        .stdout(Stdio::null()).kill_on_drop(true).spawn().unwrap();

    let socket = data.join("agent.sock");
    let active = data.join("active-model.json");
    let mut agent = spawn();
    let status = startup_status(&socket, |s| s["engine"] != "verifying weights").await;
    assert!(status["model_error"].is_null());
    let reply = startup_request(&socket, json!({"op":"preference", "mode":"use", "model":id})).await;
    assert_eq!(reply["state"], "switching");
    assert_eq!(timeout(Duration::from_secs(5), agent.wait()).await.unwrap().unwrap().code(), Some(75));
    assert_eq!(serde_json::from_slice::<Value>(&fs::read(&active).unwrap()).unwrap()["id"], id);

    let mut agent = spawn();
    let status = startup_status(&socket, |s| s["engine"] != "verifying weights").await;
    assert_eq!(status["preference"]["model"], id);
    assert!(status["model_error"].is_null());
    assert_eq!(startup_request(&socket, json!({"op":"preference","mode":"free"})).await["ok"], true);
    agent.kill().await.unwrap();

    let mut agent = spawn();
    let status = startup_status(&socket, |s| s["engine"] != "verifying weights").await;
    assert_eq!(status["preference"]["mode"], "free");
    agent.kill().await.unwrap();

    weights.set_len(10).unwrap();
    for corrupt in [false, true] {
        if corrupt {
            fs::write(&active, b"{").unwrap();
        }
        let mut agent = spawn();
        let status = startup_status(&socket, |s| s["engine"] != "verifying weights").await;
        assert!(status["model_error"].as_str().unwrap().contains("cannot restore the active model"));
        let models = startup_request(&socket, json!({"op":"models"})).await;
        assert_eq!(models["models"][0]["id"], "configured");
        agent.kill().await.unwrap();
    }
    fs::remove_dir_all(dir).unwrap();
}

async fn startup_request(socket: &std::path::Path, value: serde_json::Value) -> serde_json::Value {
    use tokio::{
        io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
        net::UnixStream,
        time::{sleep, timeout},
    };

    let connect = async {
        loop {
            if let Ok(stream) = UnixStream::connect(socket).await {
                return stream;
            }
            sleep(Duration::from_millis(20)).await;
        }
    };
    let mut peer = timeout(Duration::from_secs(5), connect).await.unwrap();
    peer.write_all(format!("{value}\n").as_bytes()).await.unwrap();
    let mut line = String::new();
    timeout(Duration::from_secs(3), BufReader::new(peer).read_line(&mut line)).await.unwrap().unwrap();
    serde_json::from_str(&line).unwrap()
}

async fn startup_status(
    socket: &std::path::Path,
    ready: impl Fn(&serde_json::Value) -> bool,
) -> serde_json::Value {
    use tokio::time::{sleep, timeout};
    timeout(Duration::from_secs(5), async {
        loop {
            let mut reply = startup_request(socket, serde_json::json!({"op":"status"})).await;
            let status = reply["status"].take();
            if ready(&status) {
                return status;
            }
            sleep(Duration::from_millis(20)).await;
        }
    }).await.expect("expected startup status")
}

#[tokio::test]
async fn rejects_canonical_symlink_and_accepts_use_of_the_same_model_for_repair() {
    use serde_json::{json, Value};
    use std::os::unix::fs::symlink;

    let dir = std::env::temp_dir().join(format!("publicompute-canonical-repair-{}-{:x}", std::process::id(), rand::random::<u64>()));
    let models = dir.join("models");
    fs::create_dir_all(&models).unwrap();
    let manifest: Vec<Value> = serde_json::from_str(include_str!("../models.json")).unwrap();
    let chosen = &manifest[0];
    let file = chosen["file"].as_str().unwrap();
    let path = models.join(file);
    let target = dir.join("never-read.gguf");
    assert!(Command::new("mkfifo").arg(&target).status().unwrap().success());
    symlink(&target, &path).unwrap();
    // A complete partial sends the repair straight to verification, without network access.
    fs::File::create(models.join(format!("{file}.agent.part"))).unwrap()
        .set_len(chosen["bytes"].as_u64().unwrap()).unwrap();

    let mut agent = tokio::process::Command::new(env!("CARGO_BIN_EXE_publicompute-agent"))
        .arg("--data-dir").arg(&dir)
        .arg("--ca").arg(dir.join("unused-ca.pem"))
        .arg("--model-path").arg(&path)
        .stdout(Stdio::null()).kill_on_drop(true).spawn().unwrap();

    let socket = dir.join("agent.sock");
    let status = startup_status(&socket, |s| s["engine"] == "weights error").await;
    assert!(status["model_error"].as_str().unwrap().contains("cannot verify configured model weights"));
    assert!(status["loaded_model"].is_null());
    assert_eq!(status["preference"]["model"], chosen["id"]);
    let reply = startup_request(&socket, json!({"op":"preference", "mode":"use", "model":chosen["id"]})).await;
    assert_eq!(reply["state"], "downloading");
    assert_eq!(reply["missing_bytes"], 0);
    assert_eq!(fs::read_link(&path).unwrap(), target);

    agent.kill().await.unwrap();
    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn socket_serves_status_while_active_model_is_hashed_without_cache() {
    use serde_json::{json, Value};

    let dir = std::env::temp_dir().join(format!("publicompute-active-hash-{}-{:x}", std::process::id(), rand::random::<u64>()));
    let models = dir.join("models");
    fs::create_dir_all(&models).unwrap();
    let manifest: Vec<Value> = serde_json::from_str(include_str!("../models.json")).unwrap();
    let chosen = manifest.iter().find(|model| model["id"] == "deepseek-v4-flash").unwrap();
    fs::File::create(models.join(chosen["file"].as_str().unwrap())).unwrap()
        .set_len(chosen["bytes"].as_u64().unwrap()).unwrap();
    fs::write(dir.join("active-model.json"), json!({"id":chosen["id"], "quant":chosen["quant"]}).to_string()).unwrap();
    let mut agent = tokio::process::Command::new(env!("CARGO_BIN_EXE_publicompute-agent"))
        .arg("--data-dir").arg(&dir)
        .arg("--ca").arg(dir.join("unused-ca.pem"))
        .args(["--model-id", "configured", "--weights-sha256", &"11".repeat(32)])
        .stdout(Stdio::null()).kill_on_drop(true).spawn().unwrap();

    let socket = dir.join("agent.sock");
    let check = async {
        for _ in 0..2 {
            let status = startup_request(&socket, json!({"op":"status"})).await;
            assert_eq!(status["status"]["engine"], "verifying weights");
            assert!(status["status"]["loaded_model"].is_null());
        }
    };
    tokio::time::timeout(Duration::from_secs(3), check).await.unwrap();
    assert!(!models.join("verified.json").exists());

    agent.kill().await.unwrap();
    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn stage_service_starts_free_then_prepares_the_installer_choice() {
    let dir = std::env::temp_dir().join(format!("publicompute-stage-startup-{}", std::process::id()));
    let models = dir.join("models");
    fs::create_dir_all(&models).unwrap();
    let ca = dir.join("ca.pem");
    fs::write(&ca, rcgen::generate_simple_self_signed(vec!["localhost".into()]).unwrap().cert.pem()).unwrap();
    let offline = std::net::UdpSocket::bind("127.0.0.1:0").unwrap();
    let bin = env!("CARGO_BIN_EXE_publicompute-agent");
    let spawn = || tokio::process::Command::new(bin)
        .arg("--data-dir").arg(&dir)
        .arg("--ca").arg(&ca)
        .args(["--coordinator", &offline.local_addr().unwrap().to_string(), "--skip-nat-probe"])
        .args(["--engine", "stages", "--stage-bin", "/unused-stage"])
        .stdout(Stdio::null()).kill_on_drop(true).spawn().unwrap();
    let socket = dir.join("agent.sock");
    let mut agent = spawn();
    let status = startup_status(&socket, |s| s["engine"] == "parked").await;
    assert_eq!(status["preference"]["mode"], "free");
    assert!(status["loaded_model"].is_null());
    assert!(status["download"].is_null());
    agent.kill().await.unwrap();

    let manifest: Vec<serde_json::Value> = serde_json::from_str(include_str!("../models.json")).unwrap();
    let medium = manifest.iter().find(|model| model["id"] == "qwen3.6-35b-a3b").unwrap();
    let file = medium["file"].as_str().unwrap();
    // A complete sparse partial exercises the background verifier without a network fetch.
    fs::File::create(models.join(format!("{file}.agent.part"))).unwrap()
        .set_len(medium["bytes"].as_u64().unwrap()).unwrap();
    assert!(Command::new(bin).arg("--data-dir").arg(&dir)
        .args(["install-preference", "medio"]).status().unwrap().success());
    let mut agent = spawn();
    let status = startup_status(&socket, |s| s["engine"] == "parked" && !s["download"].is_null()).await;
    assert_eq!(status["preference"]["model"], medium["id"]);
    assert_eq!(status["download"]["state"], "verifying");
    assert_eq!(status["download"]["id"], medium["id"]);
    assert!(status["loaded_model"].is_null());
    let revision = status["preference"]["revision"].as_u64().unwrap();
    agent.kill().await.unwrap();

    cache_sparse_weights(&models, medium);
    let mut agent = spawn();
    let exit = tokio::time::timeout(Duration::from_secs(5), agent.wait()).await.unwrap().unwrap();
    assert_eq!(exit.code(), Some(75));
    let mut agent = spawn();
    let status = startup_status(&socket, |s| s["engine"] == "ready").await;
    assert_eq!(status["preference"]["model"], medium["id"]);
    assert!(status["preference"]["revision"].as_u64().unwrap() > revision);
    assert!(status["download"].is_null());
    assert_eq!(status["weights_model"], medium["id"]);
    assert_eq!(status["weights_quant"], medium["quant"]);
    assert!(status["loaded_model"].is_null());
    assert!(status["loaded_quant"].is_null());
    let catalog = startup_request(&socket, serde_json::json!({"op":"models"})).await;
    assert_eq!(catalog["models"][0]["state"], "weights ready");
    assert_eq!(catalog["models"][0]["loaded"], false);
    agent.kill().await.unwrap();
    fs::remove_dir_all(dir).unwrap();
}
