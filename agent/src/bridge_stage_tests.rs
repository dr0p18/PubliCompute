//! These opt-in integration tests run a two-stage chain both over direct
//! TCP and through two agents using an in-process relay. Set
//! PUBLICOMPUTE_STAGE_BIN and PUBLICOMPUTE_STAGE_MODEL to supply the stage
//! executable and fixture weights.
//!
//! Exact output comparisons require matching batch shapes. Sequential
//! chats and batch-mode prompts make grouping deterministic. Concurrent
//! service chats are timed because their batches depend on arrival order.
//! Release tests bound the increase in median token latency. The teardown
//! test drops a bridge during generation to exercise stall handling.

use super::tests::{prossimo, tempdir};
use super::*;
use crate::engine::{
    progress::Progress,
    stages::{delta_channel, StadiEngine},
    Generation, Limits,
};
use crate::bridge::{tests::relay_di_prova, Bridge};
use publicompute_proto::{
    common::ModelRef,
    node::{ChatMessage, ChatRole, InferenceRequest, SamplingParams},
};
use sha2::{Digest, Sha256};

const PROMPT: [&str; 4] =
    ["The quick brown fox jumps over the lazy dog", "Hello world", "Another prompt for the third request", "Rust is a systems programming language"];
const TOKEN: u32 = 64;
/// Use a reserved documentation address so the fixture can demonstrate that the remote field is not dialed.
const IRRAGGIUNGIBILE: &str = "192.0.2.1";

fn ambiente() -> Option<(PathBuf, PathBuf)> {
    Some((PathBuf::from(std::env::var("PUBLICOMPUTE_STAGE_BIN").ok()?), PathBuf::from(std::env::var("PUBLICOMPUTE_STAGE_MODEL").ok()?)))
}

fn limiti(stallo_s: u64) -> Limits {
    Limits {
        prefill_stall: Duration::from_secs(60),
        decode_stall: Duration::from_secs(stallo_s),
        request_max: Duration::from_secs(300),
        call_timeout: Duration::from_secs(5),
        step_timeout: Duration::from_secs(5),
        probe_timeout: Duration::from_secs(5),
    }
}

fn request(text: &str, max_tokens: u32) -> InferenceRequest {
    InferenceRequest {
        request_id: "r".into(),
        messages: vec![ChatMessage { role: ChatRole::User as i32, content: text.into() }],
        sampling: Some(SamplingParams { max_tokens, temperature: 0.0, ..Default::default() }),
    }
}

/// Retain both stage managers, their state streams and the head service client.
struct Chain {
    tail: Arc<Stages>,
    head: Arc<Stages>,
    rx_c: mpsc::Receiver<StageStatus>,
    rx_t: mpsc::Receiver<StageStatus>,
    eng: Arc<StadiEngine>,
    log_testa: PathBuf,
    log_coda: PathBuf,
}

/// Start the chain and wait for readiness. A supplied pair of bridge endpoints
/// uses reserved remote addresses so the staged traffic must use the bridge.
async fn avvia_catena(name: &str, d: &Path, bin: &Path, model: &Path, sha: &[u8], stallo_s: u64, ponti: Option<(&Arc<Bridge>, &Arc<Bridge>, &str)>) -> Chain {
    let mk = |role: &str| {
        Stages::new(StagesConfig {
            stage_bin: bin.to_path_buf(),
            model_path: model.to_path_buf(),
            weights_sha256: sha.to_vec(),
            log_dir: d.join(format!("{name}-{role}")),
            threads: 2,
            ngl: 0,
            extra_args: vec![],
            env: vec![],
            load_timeout: Duration::from_secs(120),
            stop_grace: Duration::from_secs(10),
            kill_wait: Duration::from_secs(15),
            ca_path: None,
            tls_dir: d.join(format!("{name}-tls-{role}")),
            in_chiaro: true,
        })
    };
    let (tail, head) = (mk("tail"), mk("head"));
    let (tx_c, mut rx_c) = mpsc::channel(1024);
    let (tx_t, mut rx_t) = mpsc::channel(1024);
    tail.connect_peer(tx_c);
    head.connect_peer(tx_t);
    let (p_listen, p_token) = (porta_libera().unwrap(), porta_libera().unwrap());
    let base = |index: u32| StageAssign {
        chain_id: name.into(),
        stage_index: index,
        n_stages: 2,
        model: Some(ModelRef { model_id: "qwen-mini".into(), quantization: "f".into(), weights_sha256: sha.to_vec() }),
        layer_start: if index == 1 { 0 } else { 6 },
        layer_end: if index == 1 { 6 } else { -1 },
        slots: 4,
        ctx_per_slot: 512,
        groups: 2,
        n_batch: 64,
        state_format: "fp32".into(),
        assign_epoch: 1,
        ..Default::default()
    };
    let (mut a_testa, mut a_coda) = (base(1), base(2));
    a_coda.listen_addr = format!("127.0.0.1:{p_listen}");
    a_testa.head_token_addr = format!("127.0.0.1:{p_token}");
    match ponti {
        None => {
            a_testa.next_addr = format!("127.0.0.1:{p_listen}");
            a_coda.head_token_addr = format!("127.0.0.1:{p_token}");
        }
        Some((p_testa, p_coda, relay)) => {
            head.set_bridge(p_testa.clone());
            tail.set_bridge(p_coda.clone());
            a_testa.next_addr = format!("{IRRAGGIUNGIBILE}:{p_listen}");
            a_testa.next_node_id = p_coda.id();
            a_testa.relay_url = relay.to_string();
            a_coda.head_token_addr = format!("{IRRAGGIUNGIBILE}:{p_token}");
            a_coda.head_token_node_id = p_testa.id();
            a_coda.relay_url = relay.to_string();
        }
    }
    tail.queue_assignment(a_coda);
    head.queue_assignment(a_testa);
    assert_eq!(prossimo(&mut rx_c).await.state(), State::Loading);
    assert_eq!(prossimo(&mut rx_t).await.state(), State::Loading);
    let s = prossimo(&mut rx_c).await;
    assert_eq!(s.state(), State::Ready, "{s:?}");
    let s = prossimo(&mut rx_t).await;
    assert_eq!(s.state(), State::Ready, "{s:?}");
    println!("chain {name} ready: head started in {:.1} s", s.load_seconds);
    let eng = Arc::new(StadiEngine { stages: head.clone(), attive: Default::default(), limits: limiti(stallo_s), progress: Arc::new(Progress::default()) });
    let log = |role: &str| d.join(format!("{name}-{role}")).join(format!("stage-{name}-1.log"));
    Chain { tail, head, rx_c, rx_t, eng, log_testa: log("head"), log_coda: log("tail") }
}

async fn ferma_catena(mut c: Chain) {
    tokio::time::timeout(Duration::from_secs(40), c.head.stop_all()).await.expect("the head stops");
    tokio::time::timeout(Duration::from_secs(40), c.tail.stop_all()).await.expect("the tail stops");
    for rx in [&mut c.rx_t, &mut c.rx_c] {
        loop {
            let st = prossimo(rx).await.state();
            if matches!(st, State::Stopped | State::Failed) {
                break;
            }
        }
    }
}

/// Run four conversations concurrently and return generations and estimated
/// token intervals. Divide each delta gap by the number of tokens it covers.
async fn quattro_insieme(eng: &Arc<StadiEngine>) -> (Vec<Generation>, Vec<f64>) {
    let mut task = Vec::new();
    for p in PROMPT {
        let e = eng.clone();
        task.push(tokio::spawn(async move {
            let (dtx, mut drx, _) = delta_channel();
            let timings = tokio::spawn(async move {
                let (mut v, mut previous) = (Vec::new(), None::<Instant>);
                while let Some(d) = drx.recv().await {
                    let now = Instant::now();
                    if let (Some(p), n) = (previous, d.item.token_ids.len()) {
                        if n > 0 {
                            v.extend(std::iter::repeat((now - p).as_secs_f64() * 1000.0 / n as f64).take(n));
                        }
                    }
                    previous = Some(now);
                }
                v
            });
            let g = e.chat(&request(p, TOKEN), Some(dtx), None).await.expect("chat");
            (g, timings.await.unwrap())
        }));
    }
    let (mut gen, mut timings) = (Vec::new(), Vec::new());
    for t in task {
        let (g, v) = t.await.unwrap();
        gen.push(g);
        timings.extend(v);
    }
    (gen, timings)
}

/// Submit each conversation after the previous one finishes to keep batch shapes deterministic.
async fn quattro_in_fila(eng: &Arc<StadiEngine>) -> (Vec<Generation>, Vec<f64>) {
    let (mut gen, mut timings) = (Vec::new(), Vec::new());
    for p in PROMPT {
        let (dtx, mut drx, _) = delta_channel();
        let e = eng.clone();
        let chat = tokio::spawn(async move { e.chat(&request(p, TOKEN), Some(dtx), None).await.expect("chat") });
        let mut previous = None::<Instant>;
        while let Some(d) = drx.recv().await {
            let now = Instant::now();
            if let (Some(p), n) = (previous, d.item.token_ids.len()) {
                if n > 0 {
                    timings.extend(std::iter::repeat((now - p).as_secs_f64() * 1000.0 / n as f64).take(n));
                }
            }
            previous = Some(now);
        }
        gen.push(chat.await.unwrap());
    }
    (gen, timings)
}

fn p50(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[(s.len() - 1) / 2]
}

/// Preserve token IDs and exact probability bits for the output comparison.
fn impronta(g: &[Generation]) -> Vec<(Vec<u32>, Vec<u32>)> {
    g.iter().map(|x| (x.token_ids.clone(), x.probs.iter().map(|p| p.chosen_logprob.to_bits()).collect())).collect()
}

/// Return counts of mismatched tokens and probability bit patterns by conversation.
fn differenze(a: &[Generation], b: &[Generation]) -> (usize, usize) {
    let (mut tok, mut lp) = (0, 0);
    for (x, y) in impronta(a).iter().zip(impronta(b).iter()) {
        tok += x.0.len().abs_diff(y.0.len()) + x.0.iter().zip(&y.0).filter(|(p, q)| p != q).count();
        lp += x.1.len().abs_diff(y.1.len()) + x.1.iter().zip(&y.1).filter(|(p, q)| p != q).count();
    }
    (tok, lp)
}

/// Keep both chains running and alternate requests so they share the same
/// execution conditions.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn real_bridge_matches_direct_chain_with_small_overhead() {
    let Some((bin, model)) = ambiente() else { return };
    let d = tempdir("real-ponte");
    let sha = Sha256::digest(std::fs::read(&model).unwrap()).to_vec();
    let (server, url) = relay_di_prova().await;
    let (p_testa, p_coda) = (crate::bridge::tests::ponte_di_prova(&url).await, crate::bridge::tests::ponte_di_prova(&url).await);

    let diretta = avvia_catena("direct", &d, &bin, &model, &sha, 60, None).await;
    let bridge = avvia_catena("relayed", &d, &bin, &model, &sha, 60, Some((&p_testa, &p_coda, &url))).await;

    let giri: usize = std::env::var("PUBLICOMPUTE_BRIDGE_RECONNECT_ROUNDS").ok().and_then(|x| x.parse().ok()).unwrap_or(5);

    // Sequential requests keep both paths on identical batch shapes.
    let (rif, _) = quattro_in_fila(&diretta.eng).await;
    for (i, g) in rif.iter().enumerate() {
        println!("conversation {i}: {} tokens, {:?}", g.token_ids.len(), g.finish_reason);
        assert_eq!(g.token_ids.len(), TOKEN as usize, "conversation {i} should reach {TOKEN} tokens");
    }
    let (primo, _) = quattro_in_fila(&bridge.eng).await;
    assert_eq!(differenze(&primo, &rif), (0, 0), "bridge and direct chain returned different tokens or log probabilities");
    let (mut f_dir, mut f_ponte) = (Vec::new(), Vec::new());
    for giro in 0..giri {
        let (g, t) = quattro_in_fila(&diretta.eng).await;
        assert_eq!(differenze(&g, &rif), (0, 0), "run {giro}: the direct chain did not reproduce the reference");
        f_dir.extend(t);
        let (g, t) = quattro_in_fila(&bridge.eng).await;
        assert_eq!(differenze(&g, &rif), (0, 0), "round {giro}: bridge and direct chain returned different tokens or log probabilities");
        f_ponte.extend(t);
    }
    let confrontati: usize = rif.iter().map(|g| g.token_ids.len()).sum();
    println!(
        "Sequential requests: {} conversations × {TOKEN} tokens, {confrontati} bit-identical tokens and log probabilities over {} bridge runs",
        rif.len(),
        giri + 1
    );

    // Concurrent service batches vary with arrival order, so use these runs for timing.
    let (rif4, _) = quattro_insieme(&diretta.eng).await;
    let _ = quattro_insieme(&bridge.eng).await;
    let (mut t_dir, mut t_ponte) = (Vec::new(), Vec::new());
    let (mut ripetizione, mut contro) = ((0, 0), (0, 0));
    for _ in 0..giri {
        let (g, t) = quattro_insieme(&diretta.eng).await;
        let dd = differenze(&g, &rif4);
        t_dir.extend(t);
        let (g, t) = quattro_insieme(&bridge.eng).await;
        let dp = differenze(&g, &rif4);
        t_ponte.extend(t);
        ripetizione = (ripetizione.0 + dd.0, ripetizione.1 + dd.1);
        contro = (contro.0 + dp.0, contro.1 + dp.1);
    }
    println!(
        "Four concurrent requests over {giri} runs. Token and log probability differences: direct vs direct {ripetizione:?}, bridged vs direct {contro:?}"
    );

    let nota = if cfg!(debug_assertions) { " [debug build: informational]" } else { "" };
    let (a1, b1) = (p50(&f_dir), p50(&f_ponte));
    println!("One conversation, median time per token: direct {a1:.3} ms ({} samples), bridge {b1:.3} ms ({} samples), difference {:+.3} ms (limit +1 ms){nota}", f_dir.len(), f_ponte.len(), b1 - a1);
    let (a, b) = (p50(&t_dir), p50(&t_ponte));
    println!("Four concurrent conversations, median time per token: direct {a:.3} ms ({} samples), bridge {b:.3} ms ({} samples), difference {:+.3} ms (limit +1 ms){nota}", t_dir.len(), t_ponte.len(), b - a);
    for m in p_testa.metrics().iter().chain(p_coda.metrics().iter()) {
        println!(
            "Metric {} {} {}: path {}, first direct {}, sent {} B, received {} B, streams {}",
            m.chain_id,
            m.verso,
            if m.in_uscita { "outgoing" } else { "incoming" },
            m.percorso,
            m.primo_diretto_s.map(|s| format!("{s:.3} s")).unwrap_or_else(|| "never".into()),
            m.byte_inviati,
            m.byte_ricevuti,
            m.stream_aperti
        );
    }
    // Successful generation through reserved remote addresses verifies loopback routing.
    let m = p_testa.metrics();
    assert!(m.iter().any(|x| x.in_uscita && x.verso == "forward" && x.byte_inviati > 0), "{m:?}");
    assert!(p_coda.metrics().iter().any(|x| x.in_uscita && x.verso == "token" && x.byte_inviati > 0));
    if !cfg!(debug_assertions) {
        assert!(b - a <= 1.0, "four concurrent conversations: bridge overhead {:.3} ms exceeds 1 ms", b - a);
        assert!(b1 - a1 <= 1.0, "one conversation: bridge overhead {:.3} ms exceeds 1 ms", b1 - a1);
    }

    ferma_catena(diretta).await;
    ferma_catena(bridge).await;
    p_testa.close().await;
    p_coda.close().await;
    server.shutdown().await.ok();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires two wire-compatible stage executables and fixture weights"]
async fn mixed_stage_versions_replay_a_failed_conversation_in_both_directions() {
    let (current, model) = ambiente().expect("set PUBLICOMPUTE_STAGE_BIN and PUBLICOMPUTE_STAGE_MODEL");
    let legacy = PathBuf::from(std::env::var_os("PUBLICOMPUTE_STAGE_LEGACY_BIN").expect("set PUBLICOMPUTE_STAGE_LEGACY_BIN"));
    let dir = tempdir("mixed-stage-replay");
    for (name, head, tail) in [("legacy-head", &legacy, &current), ("legacy-tail", &current, &legacy)] {
        let binaries = BatchStages { head, tail, middle: None, tail_args: &["--test-conv-error", "1:1"] };
        let (listen, token) = (porta_libera().unwrap(), porta_libera().unwrap());
        let next = format!("127.0.0.1:{listen}");
        let token_address = format!("127.0.0.1:{token}");
        let (output, _) = corsa_a_lotti(name, &dir, &binaries, &model, listen, token, &next, &token_address).await;
        let rows = std::fs::read_to_string(&output).unwrap();
        let lengths = rows.lines().map(|row| serde_json::from_str::<serde_json::Value>(row).unwrap()["token"].as_array().unwrap().len()).collect::<Vec<_>>();
        assert_eq!(lengths, vec![TOKEN as usize; PROMPT.len()], "{name}");
        let log = std::fs::read_to_string(dir.join(format!("{name}.head.log"))).unwrap();
        assert!(log.contains("conversation fault"), "{name} did not exercise replay");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires two wire-compatible stage executables and fixture weights"]
async fn mixed_stage_versions_reconnect_a_three_stage_chain_and_replay_tokens() {
    let (current, model) = ambiente().expect("set PUBLICOMPUTE_STAGE_BIN and PUBLICOMPUTE_STAGE_MODEL");
    let legacy = PathBuf::from(std::env::var_os("PUBLICOMPUTE_STAGE_LEGACY_BIN").expect("set PUBLICOMPUTE_STAGE_LEGACY_BIN"));
    let dir = tempdir("mixed-link-replay");
    let mut expected: Vec<Vec<i32>> = Vec::new();
    // Keep the four-second outage case alongside the cases using the tail's default grace.
    let long_grace: &[&str] = &["--reconnect-grace-s", "10"];
    for (name, head, tail, outage_s, tail_args) in [
        ("baseline", &current, &current, 0, &[][..]),
        ("legacy-head", &legacy, &current, 1, &[][..]),
        ("legacy-tail", &current, &legacy, 1, &[][..]),
        ("legacy-head-long", &legacy, &current, 4, long_grace),
        ("legacy-tail-long", &current, &legacy, 4, long_grace),
    ] {
        let (listen, token, middle) = (porta_libera().unwrap(), porta_libera().unwrap(), porta_libera().unwrap());
        let downstream = format!("127.0.0.1:{listen}");
        let proxy = if outage_s == 0 {
            None
        } else {
            Some(interrupt_forward_link(&downstream, Duration::from_secs(outage_s)).await)
        };
        let middle_next = proxy.as_ref().map_or(downstream.as_str(), |(address, _, _)| address.as_str());
        let binaries = BatchStages { head, tail, middle: Some((&current, middle_next)), tail_args };
        let (output, _) = corsa_a_lotti(
            name, &dir, &binaries, &model, listen, token, &format!("127.0.0.1:{middle}"), &format!("127.0.0.1:{token}"),
        ).await;
        let rows = std::fs::read_to_string(output).unwrap();
        let tokens: Vec<Vec<i32>> = rows.lines().map(|row| {
            let row: serde_json::Value = serde_json::from_str(row).unwrap();
            serde_json::from_value(row["token"].clone()).unwrap()
        }).collect();
        assert_eq!(tokens.iter().map(Vec::len).collect::<Vec<_>>(), vec![TOKEN as usize; PROMPT.len()]);
        if let Some((_, proxy, blocks)) = proxy {
            proxy.abort();
            let log = std::fs::read_to_string(dir.join(format!("{name}.head.log"))).unwrap();
            assert!(log.contains("chain fault"), "{name}: {log}");
            assert_eq!(log.matches("final attempt 2, 1 recoveries, transcript has no gaps or duplicates").count(), PROMPT.len(), "{name}: {log}");
            let blocks = blocks.lock().unwrap();
            for (i, (actual, baseline)) in tokens.iter().zip(&expected).enumerate() {
                let id = i as u64 + 1;
                let &(prompt, first_step) = blocks.get(&(id, 1)).expect("initial prefill");
                let &(prefix, resumed_step) = blocks.get(&(id, 2)).expect("replayed prefill");
                assert_eq!((first_step, resumed_step, prefix), (0, 1, prompt + 1), "{name}, conversation {id}");
                assert_eq!(actual[0], baseline[0], "{name} changed the confirmed token for conversation {id}");
            }
        } else {
            expected = tokens;
        }
    }
}

type PrefillBlocks = Arc<Mutex<std::collections::HashMap<(u64, u32), (u32, u32)>>>;

/// Cut the forward link before the first generation round reaches the tail.
/// Record final prefill blocks before and after the cut.
async fn interrupt_forward_link(target: &str, outage: Duration) -> (String, tokio::task::JoinHandle<()>, PrefillBlocks) {
    use tokio::net::{TcpListener, TcpStream};
    use tokio::task::JoinSet;

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap().to_string();
    let target = target.to_string();
    let blocks = PrefillBlocks::default();
    let captured = blocks.clone();
    let proxy = tokio::spawn(async move {
        let (signal, mut cut) = mpsc::channel(1);
        let mut armed = true;
        let mut streams = JoinSet::new();
        loop {
            tokio::select! {
                _ = cut.recv(), if armed => {
                    armed = false;
                    streams.shutdown().await;
                    tokio::time::sleep(outage).await;
                }
                _ = streams.join_next(), if !streams.is_empty() => {}
                incoming = listener.accept() => {
                    let (from, _) = incoming.unwrap();
                    let (target, captured) = (target.clone(), captured.clone());
                    let signal = armed.then(|| signal.clone());
                    streams.spawn(async move {
                        let to = TcpStream::connect(target).await.ok()?;
                        proxy_stage_stream(from, to, signal, captured).await
                    });
                }
            }
        }
    });
    (address, proxy, blocks)
}

async fn proxy_stage_stream(
    mut from: tokio::net::TcpStream, mut to: tokio::net::TcpStream, cut: Option<mpsc::Sender<()>>, blocks: PrefillBlocks,
) -> Option<()> {
    use publicompute_proto::stage::{message::Body, Stream};
    use publicompute_proto::wire::encode_frame;
    use tokio::io::AsyncWriteExt;

    let mut buf = bytes::BytesMut::new();
    let hello = read_stage_frame(&mut from, &mut buf).await?;
    let Some(Body::Hello(h)) = &hello.body else { return None };
    let stream = h.stream;
    let mut frame = Vec::new();
    encode_frame(&hello, &mut frame).ok()?;
    to.write_all(&frame).await.ok()?;
    if stream != Stream::Prefill as i32 && !(stream == Stream::Generation as i32 && cut.is_some()) {
        to.write_all(&buf).await.ok()?;
        tokio::io::copy_bidirectional(&mut from, &mut to).await.ok()?;
        return Some(());
    }

    let (mut from_read, mut from_write) = from.split();
    let (mut to_read, mut to_write) = to.split();
    let forward = async {
        while let Some(message) = read_stage_frame(&mut from_read, &mut buf).await {
            match &message.body {
                Some(Body::Block(b)) if b.last => {
                    blocks.lock().unwrap().insert((b.conversation_id, b.attempt_id), (b.start_position + b.n_token, b.first_token_step));
                }
                Some(Body::Step(_)) if cut.as_ref().is_some_and(|cut| cut.try_send(()).is_ok()) => return Some(()),
                _ => {}
            }
            frame.clear();
            encode_frame(&message, &mut frame).ok()?;
            to_write.write_all(&frame).await.ok()?;
        }
        Some(())
    };
    tokio::select! {
        result = forward => result,
        result = tokio::io::copy(&mut to_read, &mut from_write) => result.ok().map(|_| ()),
    }
}

async fn read_stage_frame(input: &mut (impl tokio::io::AsyncRead + Unpin), buf: &mut bytes::BytesMut) -> Option<publicompute_proto::stage::Message> {
    use tokio::io::AsyncReadExt;

    loop {
        if let Some(message) = publicompute_proto::wire::try_decode_frame(buf).ok()? {
            return Some(message);
        }
        if input.read_buf(buf).await.ok()? == 0 {
            return None;
        }
    }
}


#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires the native stage executable and fixture GGUF"]
async fn head_releases_slot_on_cancel_before_first_token_and_during_reconnect() {
    use publicompute_proto::stage::{self as wire, message::Body, Stream};
    use tokio::net::{TcpListener, TcpStream};

    async fn hello(listener: &TcpListener, stream: Stream) -> TcpStream {
        let (mut socket, _) = listener.accept().await.unwrap();
        let hello = read_stage_frame(&mut socket, &mut bytes::BytesMut::new()).await.unwrap();
        assert!(matches!(hello.body, Some(Body::Hello(h)) if h.stream == stream as i32));
        socket
    }

    async fn acknowledge(socket: &mut TcpStream) {
        let message = wire::Message { body: Some(Body::Ack(wire::HelloAck { ok: true, error: String::new() })) };
        let mut frame = Vec::new();
        publicompute_proto::wire::encode_frame(&message, &mut frame).unwrap();
        socket.write_all(&frame).await.unwrap();
    }

    async fn prefill(socket: &mut TcpStream, reader: &mut bytes::BytesMut, id: u64) -> u32 {
        let open = read_stage_frame(socket, reader).await.unwrap();
        let Some(Body::Open(open)) = open.body else { panic!("expected Open, received {open:?}") };
        assert_eq!((open.conversation_id, open.attempt_id), (id, 1));
        loop {
            let block = read_stage_frame(socket, reader).await.unwrap();
            assert!(matches!(&block.body, Some(Body::Block(b)) if b.conversation_id == id));
            if matches!(block.body, Some(Body::Block(b)) if b.last) {
                return open.slot;
            }
        }
    }

    async fn cancelled(request: &mut HeadResponses) {
        let done = request.recv().await.unwrap();
        assert!(matches!(done, OutgoingLink::Done(d) if d.reason == "cancelled"));
    }

    let (bin, model) = ambiente().expect("set PUBLICOMPUTE_STAGE_BIN and PUBLICOMPUTE_STAGE_MODEL");
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let next = listener.local_addr().unwrap().to_string();
    let token = format!("127.0.0.1:{}", porta_libera().unwrap());
    let dir = tempdir("head-cancel-before-token");
    let mut child = tokio::process::Command::new(bin)
        .args(["head", "--model", model.to_str().unwrap(), "--ngl", "0", "--threads", "2", "--layer-end", "6"])
        .args(["--slots", "1", "--groups", "1", "--ctx-per-slot", "512", "--n-batch", "64"])
        .args(["--service-pipe", "--next", &next, "--listen-token", &token, "--work-timeout-s", "60"])
        .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(std::fs::File::create(dir.join("head.log")).unwrap())
        .kill_on_drop(true).spawn().unwrap();

    tokio::time::timeout(Duration::from_secs(15), async {
        let mut pf = hello(&listener, Stream::Prefill).await;
        acknowledge(&mut pf).await;
        let mut gen = hello(&listener, Stream::Generation).await;
        acknowledge(&mut gen).await;
        let mut ctl = hello(&listener, Stream::Control).await;
        acknowledge(&mut ctl).await;
        let (reader, writer, buf) = head_ready(child.stdout.take().unwrap(), child.stdin.take().unwrap()).await.unwrap();
        let link = HeadConnection::from_io(reader, writer, buf);
        let request = |id| pb::Request { id, text: "hello".into(), max_tokens: 8, ..Default::default() };

        let mut buf = bytes::BytesMut::new();
        let mut first = link.request(request(1)).await.unwrap();
        let slot = prefill(&mut pf, &mut buf, 1).await;
        link.cancel(1).await;
        tokio::time::timeout(Duration::from_secs(2), cancelled(&mut first)).await.unwrap();
        let close = read_stage_frame(&mut pf, &mut buf).await.unwrap();
        assert!(matches!(close.body, Some(Body::Close(c)) if c.conversation_id == 1 && c.attempt_id == 1));

        let mut second = link.request(request(2)).await.unwrap();
        assert_eq!(prefill(&mut pf, &mut buf, 2).await, slot);
        drop(ctl);
        let mut pf = hello(&listener, Stream::Prefill).await;
        // Hold back HelloAck until the local cancellation is complete.
        link.cancel(2).await;
        tokio::time::timeout(Duration::from_secs(2), cancelled(&mut second)).await.unwrap();
        acknowledge(&mut pf).await;
        let mut new_gen = hello(&listener, Stream::Generation).await;
        acknowledge(&mut new_gen).await;
        let mut new_ctl = hello(&listener, Stream::Control).await;
        acknowledge(&mut new_ctl).await;

        let mut third = link.request(request(3)).await.unwrap();
        buf.clear();
        assert_eq!(prefill(&mut pf, &mut buf, 3).await, slot);
        link.cancel(3).await;
        tokio::time::timeout(Duration::from_secs(2), cancelled(&mut third)).await.unwrap();
    }).await.expect("head cancellation and reconnect completed");
    child.kill().await.unwrap();
}

/// Drop the tail bridge during generation. The head must observe the link
/// failure, end the chat within its stall deadline and allow both stages
/// to shut down.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn real_bridge_disconnect_releases_stages() {
    let Some((bin, model)) = ambiente() else { return };
    let d = tempdir("real-ponte-caduto");
    let sha = Sha256::digest(std::fs::read(&model).unwrap()).to_vec();
    let (server, url) = relay_di_prova().await;
    let (p_testa, p_coda) = (crate::bridge::tests::ponte_di_prova(&url).await, crate::bridge::tests::ponte_di_prova(&url).await);
    const STALLO_S: u64 = 8;
    let c = avvia_catena("interrupted", &d, &bin, &model, &sha, STALLO_S, Some((&p_testa, &p_coda, &url))).await;

    // Verify that the chain starts in a healthy state.
    let g = c.eng.chat(&request(PROMPT[0], 16), None, None).await.unwrap();
    assert_eq!(g.token_ids.len(), 16);

    let (dtx, mut drx, _) = delta_channel();
    let e = c.eng.clone();
    let chat = tokio::spawn(async move { e.chat(&request(PROMPT[3], 400), Some(dtx), None).await });
    let mut visti = 0;
    while visti < 8 {
        let d = tokio::time::timeout(Duration::from_secs(60), drx.recv()).await.expect("token before disconnect").expect("token channel closed");
        visti += d.item.token_ids.len();
    }
    let t0 = Instant::now();
    p_coda.close().await;
    println!("tail bridge closed after {visti} tokens");

    // Inspect the head log for an independently observed link failure.
    let visto = async {
        loop {
            let log = std::fs::read_to_string(&c.log_testa).unwrap_or_default();
            if let Some(r) = log.lines().find(|r| r.contains("chain fault")) {
                break r.to_string();
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    };
    let row = tokio::time::timeout(Duration::from_secs(10), visto).await.expect("head should see the connection close within 10 s");
    println!("Head detected the disconnect after {:.3} s: {row}", t0.elapsed().as_secs_f64());

    // The active request has no more progress, so its decode stall deadline bounds completion.
    let esito = tokio::time::timeout(Duration::from_secs(STALLO_S + 10), chat).await.expect("chat should finish or return an error").unwrap();
    let elapsed = t0.elapsed().as_secs_f64();
    match &esito {
        Ok(g) => println!("Chat finished after {elapsed:.1} s with {} tokens ({:?})", g.token_ids.len(), g.finish_reason),
        Err(e) => println!("Chat failed after {elapsed:.1} s: {e:#}"),
    }
    assert!(esito.is_err() || esito.as_ref().unwrap().token_ids.len() < 400, "the response cannot finish without the tail");
    let coda_log = std::fs::read_to_string(&c.log_coda).unwrap_or_default();
    for r in coda_log.lines().rev().take(3).collect::<Vec<_>>().into_iter().rev() {
        println!("tail: {r}");
    }

    let t1 = Instant::now();
    ferma_catena(c).await;
    println!("Stages stopped in {:.1} s", t1.elapsed().as_secs_f64());
    p_testa.close().await;
    server.shutdown().await.ok();
}

struct BatchStages<'a> {
    head: &'a Path,
    tail: &'a Path,
    middle: Option<(&'a Path, &'a str)>,
    tail_args: &'a [&'a str],
}

/// Run the tail and head in batch mode to admit the four prompts together.
/// Return the head's JSONL output and elapsed time.
async fn corsa_a_lotti(name: &str, d: &Path, binaries: &BatchStages<'_>, model: &Path, p_listen: u16, p_token: u16, next: &str, testa_token: &str) -> (PathBuf, f64) {
    let prompt = d.join("prompt.txt");
    std::fs::write(&prompt, PROMPT.join("\n") + "\n").unwrap();
    let comuni = |c: &mut tokio::process::Command| {
        c.args(["--model", model.to_str().unwrap(), "--ngl", "0", "--threads", "2", "--slots", "4", "--groups", "2", "--ctx-per-slot", "512", "--n-batch", "64"]);
        c.kill_on_drop(true);
    };
    let log = |role: &str| std::fs::File::create(d.join(format!("{name}.{role}.log"))).unwrap();
    let mut tail = tokio::process::Command::new(binaries.tail);
    tail.arg("tail");
    comuni(&mut tail);
    tail.args(["--layer-start", "6", "--layer-end", "-1", "--listen", &format!("127.0.0.1:{p_listen}"), "--head-token", testa_token]);
    tail.args(binaries.tail_args);
    let mut tail = tail.stdout(Stdio::null()).stderr(log("tail")).spawn().unwrap();
    let mut middle = binaries.middle.map(|(bin, downstream)| {
        let mut command = tokio::process::Command::new(bin);
        command.arg("middle");
        comuni(&mut command);
        command.args(["--layer-start", "3", "--layer-end", "6", "--listen", next, "--next", downstream, "--timeout-s", "60"]);
        command.stdout(Stdio::null()).stderr(log("middle")).spawn().unwrap()
    });
    let head_end = if middle.is_some() { "3" } else { "6" };
    let output = d.join(name);
    let mut head = tokio::process::Command::new(binaries.head);
    head.arg("head");
    comuni(&mut head);
    head.args(["--prompt", prompt.to_str().unwrap(), "--n-prompt", "4", "--max-tokens", &TOKEN.to_string(), "--block", "64", "--temperature", "0", "--seed", "7"]);
    head.args(["--timeout-s", "60", "--work-timeout-s", "60", "--recovery-timeout-s", "60", "--layer-start", "0", "--layer-end", head_end]);
    if middle.is_some() {
        head.args(["--prefill-order", "round-robin"]);
    }
    head.args(["--next", next, "--listen-token", &format!("127.0.0.1:{p_token}"), "--output", output.to_str().unwrap()]);
    let t0 = Instant::now();
    let st = tokio::time::timeout(Duration::from_secs(180), head.stdout(Stdio::null()).stderr(log("head")).status()).await.expect("the head exits").unwrap();
    let durata = t0.elapsed().as_secs_f64();
    assert!(st.success(), "head {name}: {st:?} (log in {})", d.display());
    if let Some(middle) = &mut middle {
        let _ = middle.kill().await;
    } else {
        let _ = tokio::time::timeout(Duration::from_secs(10), tail.wait()).await;
    }
    let _ = tail.kill().await;
    (PathBuf::from(format!("{}.jsonl", output.display())), durata)
}

/// Compare the batch-mode output byte for byte across both transports.
/// The forward state and return token links both use the bridge in the
/// bridged run.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn real_bridge_matches_direct_chain_with_fixed_batches() {
    use crate::bridge::Direction;
    let Some((bin, model)) = ambiente() else { return };
    let d = tempdir("real-ponte-lotti");
    let (server, url) = relay_di_prova().await;
    let (p_testa, p_coda) = (crate::bridge::tests::ponte_di_prova(&url).await, crate::bridge::tests::ponte_di_prova(&url).await);
    let locale = |p: u16| SocketAddr::from(([127, 0, 0, 1], p));
    let binaries = BatchStages { head: &bin, tail: &bin, middle: None, tail_args: &[] };

    let (pl, pt) = (porta_libera().unwrap(), porta_libera().unwrap());
    let (diretta, s_dir) = corsa_a_lotti("diretta", &d, &binaries, &model, pl, pt, &locale(pl).to_string(), &locale(pt).to_string()).await;

    let (pl, pt) = (porta_libera().unwrap(), porta_libera().unwrap());
    let (mut g_testa, mut g_coda) = (p_testa.guard(), p_coda.guard());
    p_coda.incoming(&mut g_coda, "lotti", 1, Direction::Forward, locale(pl));
    p_testa.incoming(&mut g_testa, "lotti", 1, Direction::Token, locale(pt));
    let next = p_testa.output(&mut g_testa, "lotti", 1, Direction::Forward, &p_coda.id(), &url).await.unwrap();
    let token = p_coda.output(&mut g_coda, "lotti", 1, Direction::Token, &p_testa.id(), &url).await.unwrap();
    let (bridge, s_ponte) = corsa_a_lotti("ponte", &d, &binaries, &model, pl, pt, &next.to_string(), &token.to_string()).await;

    let rows = |f: &Path| std::fs::read_to_string(f).unwrap().lines().count();
    assert_eq!(rows(&diretta), 4);
    assert_eq!(rows(&bridge), 4);
    let token_di = |f: &Path| -> Vec<usize> {
        std::fs::read_to_string(f).unwrap().lines().map(|r| serde_json::from_str::<serde_json::Value>(r).unwrap()["token"].as_array().unwrap().len()).collect()
    };
    assert_eq!(token_di(&bridge), vec![TOKEN as usize; 4], "four conversations of {TOKEN} tokens each");
    let conf = std::process::Command::new(&bin).arg("compare").arg(&bridge).arg(&diretta).arg("--bit-identical").output().unwrap();
    let text = String::from_utf8_lossy(&conf.stdout).to_string() + &String::from_utf8_lossy(&conf.stderr);
    std::fs::write(d.join("compare.log"), &text).unwrap();
    for r in text.lines().filter(|r| r.contains("identical") || r.contains("difference")).take(4) {
        println!("compare: {r}");
    }
    assert!(conf.status.success(), "bridge and direct chain differed. See {}", d.join("compare.log").display());
    println!("Fixed batches: four concurrent conversations × {TOKEN} tokens with bit-identical tokens and log probabilities");
    println!("Head batch runtime including model load: direct {s_dir:.2} s, bridged {s_ponte:.2} s");
    for m in p_testa.metrics().iter().chain(p_coda.metrics().iter()) {
        println!("Metric {} {} {}: path {}, sent {} B, received {} B, streams {}", m.chain_id, m.verso, if m.in_uscita { "outgoing" } else { "incoming" }, m.percorso, m.byte_inviati, m.byte_ricevuti, m.stream_aperti);
    }
    drop((g_testa, g_coda));
    p_testa.close().await;
    p_coda.close().await;
    server.shutdown().await.ok();
}
