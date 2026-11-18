//! Minimal HTTP fixture for the agent's request, health and recovery paths.
//! It serves deterministic completions in both response modes. Fault switches
//! simulate slow prefill, stalled decoding, an abandoned request holding the
//! slot, or a character whose bytes span multiple tokens.

#![allow(dead_code)] // Used by both the fixture executable and the test harness.

use std::{
    net::SocketAddr,
    sync::{
        atomic::{AtomicBool, AtomicI64, AtomicU32, AtomicU64, Ordering},
        Arc,
    },
    time::Duration,
};

use serde_json::{json, Value};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
};

pub struct FakeState {
    pub model_path: String,
    /// Report HTTP 200 for health. The default is HTTP 503.
    pub health_ok: AtomicBool,
    /// Keep health available while leaving completion requests unanswered.
    pub hang_completion: AtomicBool,
    /// Stall decoding after this many tokens. A negative value disables the stall.
    pub decode_stall_after: AtomicI64,
    /// Number of prefill progress events and the interval between them.
    pub prefill_steps: AtomicU32,
    pub prefill_step_ms: AtomicU64,
    pub token_delay_ms: AtomicU64,
    /// Report a slot held by an abandoned request.
    pub busy: AtomicBool,
    /// Simulate an engine without a slot status endpoint.
    pub slots_missing: AtomicBool,
    /// Omit the first token of a split UTF-8 character.
    pub utf8_split: AtomicBool,
    /// Emit held UTF-8 tokens with their IDs and probabilities.
    pub utf8_hold: AtomicBool,
    /// Send a final text delta with a decoded count larger than the stream.
    pub incorrect_stream_count: AtomicBool,
    /// Override the decoded count on the second token chunk. Negative disables the override.
    pub second_stream_count: AtomicI64,
    /// Reject tokenization of generated text.
    pub reject_output_tokenization: AtomicBool,
    pub completions: AtomicU64,
    active: AtomicU32,
    decoded: AtomicU64,
    prompt_done: AtomicU64,
}

impl FakeState {
    pub fn new(model_path: &str) -> Arc<Self> {
        Arc::new(Self {
            model_path: model_path.to_string(),
            health_ok: AtomicBool::new(true),
            hang_completion: AtomicBool::new(false),
            decode_stall_after: AtomicI64::new(-1),
            prefill_steps: AtomicU32::new(1),
            prefill_step_ms: AtomicU64::new(0),
            token_delay_ms: AtomicU64::new(0),
            busy: AtomicBool::new(false),
            slots_missing: AtomicBool::new(false),
            utf8_split: AtomicBool::new(false),
            utf8_hold: AtomicBool::new(false),
            incorrect_stream_count: AtomicBool::new(false),
            second_stream_count: AtomicI64::new(-1),
            reject_output_tokenization: AtomicBool::new(false),
            completions: AtomicU64::new(0),
            active: AtomicU32::new(0),
            decoded: AtomicU64::new(0),
            prompt_done: AtomicU64::new(0),
        })
    }
}

pub struct Fake {
    pub state: Arc<FakeState>,
    pub addr: SocketAddr,
    task: tokio::task::JoinHandle<()>,
}

impl Fake {
    pub fn url(&self) -> String {
        format!("http://{}", self.addr)
    }
}

impl Drop for Fake {
    fn drop(&mut self) {
        self.task.abort();
    }
}

/// Start the fixture. Port zero lets the operating system pick a free port.
pub async fn spawn(port: u16, state: Arc<FakeState>) -> std::io::Result<Fake> {
    let listener = TcpListener::bind(("127.0.0.1", port)).await?;
    let addr = listener.local_addr()?;
    let st = state.clone();
    let task = tokio::spawn(async move { serve(listener, st).await });
    Ok(Fake { state, addr, task })
}

pub async fn serve(listener: TcpListener, st: Arc<FakeState>) {
    loop {
        let Ok((sock, _)) = listener.accept().await else { return };
        let st = st.clone();
        tokio::spawn(async move {
            let _ = handle(sock, st).await;
        });
    }
}

struct Tok {
    id: u32,
    text: String,
    /// Hold incomplete UTF-8 bytes until a later token makes them displayable.
    partial: bool,
    /// Hold text that ends in a possible stop string.
    held: bool,
}

fn lp(k: usize) -> f64 {
    -0.1 - 0.37 * k as f64
}

fn entry(id: u32, n: u32, chosen_k: usize) -> Value {
    let top: Vec<Value> = (0..n as usize)
        .map(|k| {
            let tid = if k == 0 { id } else { 100 + ((id - 100 + k as u32) % 200) };
            json!({"id": tid, "token": format!("<{tid}>"), "bytes": [], "logprob": lp(k)})
        })
        .collect();
    json!({"id": id, "token": format!("<{id}>"), "bytes": [], "logprob": lp(chosen_k), "top_logprobs": top})
}

fn gen_tokens(prompt: &[u32], req: &Value, split: bool, hold: bool) -> (Vec<Tok>, &'static str) {
    let seed = req["seed"].as_i64().unwrap_or(0).rem_euclid(1000) as u32;
    let temp = (req["temperature"].as_f64().unwrap_or(0.0) * 10.0) as u32;
    let base = prompt.iter().fold(0u32, |a, t| a.wrapping_mul(31).wrapping_add(*t)) % 997;
    let natural = 12usize;
    let np = req["n_predict"].as_i64().unwrap_or(-1);
    let (n, stop) = if np > 0 && (np as usize) < natural { (np as usize, "limit") } else { (natural, "eos") };
    let mut toks: Vec<Tok> = (0..n)
        .map(|i| {
            let id = 100 + (base + seed * 7 + temp * 3 + i as u32 * 13) % 200;
            Tok { id, text: format!("<{id}>"), partial: false, held: false }
        })
        .collect();
    if stop == "eos" {
        let last = toks.last_mut().unwrap();
        last.id = 151645;
        last.text = String::new();
    }
    if split && n > 6 {
        toks[3] = Tok { id: 195, text: String::new(), partial: true, held: false };
        toks[4] = Tok { id: 168, text: "è".into(), partial: false, held: false };
    }
    if hold {
        for (i, (t, id)) in toks.iter_mut().skip(1).zip([240, 159, 153, 130]).enumerate() {
            t.id = id;
            t.partial = i < 3;
            t.text = if t.partial { String::new() } else { "🙂".into() };
        }
        if n == 2 {
            // A single piece contains complete text followed by an incomplete character.
            toks[1].text = "A".into();
        }
        if stop == "eos" {
            // EOG arrives before the bytes of the next character finish.
            for (t, id) in toks[n - 3..n - 1].iter_mut().zip([240, 159]) {
                t.id = id;
                t.partial = true;
                t.text.clear();
            }
        }
        if n >= 5 && req["stop"].as_array().is_some_and(|stops| stops.iter().any(|s| s == "🙂")) {
            toks.truncate(5);
            toks[4].text.clear();
            return (toks, "word");
        }
    }
    if n > 5 && req["stop"].as_array().is_some_and(|stops| stops.iter().any(|s| s == "END")) {
        toks.truncate(5);
        // The stop starts in the middle of token 3 and ends in token 4.
        toks[3].text = "<350>".into();
        toks[3].held = true;
        toks[4].text.clear();
        return (toks, "word");
    }
    (toks, stop)
}

fn prompt_of(req: &Value) -> Vec<u32> {
    match &req["prompt"] {
        Value::Array(a) => a.iter().filter_map(|v| v.as_u64().map(|v| v as u32)).collect(),
        Value::String(s) => s.bytes().map(u32::from).collect(),
        _ => vec![],
    }
}

fn tokenize_output(mut text: &str) -> Vec<u32> {
    let mut tokens = Vec::new();
    while let Some(rest) = text.strip_prefix('<') {
        let Some((id, tail)) = rest.split_once('>') else { break };
        let Ok(id) = id.parse() else { break };
        tokens.push(id);
        text = tail;
        if let Some(tail) = text.strip_prefix('è') {
            tokens.extend([195, 168]);
            text = tail;
        }
        if let Some(tail) = text.strip_prefix('🙂') {
            tokens.extend([240, 159, 153, 130]);
            text = tail;
        }
    }
    tokens.extend(text.bytes().map(u32::from));
    tokens
}

async fn read_request(sock: &mut TcpStream) -> std::io::Result<(String, String, Vec<u8>)> {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 8192];
    let header_end = loop {
        let n = sock.read(&mut tmp).await?;
        if n == 0 {
            return Err(std::io::ErrorKind::UnexpectedEof.into());
        }
        buf.extend_from_slice(&tmp[..n]);
        if let Some(p) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break p + 4;
        }
    };
    let head = String::from_utf8_lossy(&buf[..header_end]).to_string();
    let mut lines = head.lines();
    let mut first = lines.next().unwrap_or("").split_whitespace();
    let (method, path) = (first.next().unwrap_or("").to_string(), first.next().unwrap_or("").to_string());
    let len = lines
        .filter_map(|l| l.split_once(':'))
        .find(|(k, _)| k.eq_ignore_ascii_case("content-length"))
        .and_then(|(_, v)| v.trim().parse::<usize>().ok())
        .unwrap_or(0);
    let mut body = buf[header_end..].to_vec();
    while body.len() < len {
        let n = sock.read(&mut tmp).await?;
        if n == 0 {
            break;
        }
        body.extend_from_slice(&tmp[..n]);
    }
    Ok((method, path, body))
}

async fn respond(sock: &mut TcpStream, status: u16, body: &Value) -> std::io::Result<()> {
    let b = serde_json::to_vec(body).unwrap();
    let head = format!("HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", b.len());
    sock.write_all(head.as_bytes()).await?;
    sock.write_all(&b).await?;
    sock.flush().await
}

async fn handle(mut sock: TcpStream, st: Arc<FakeState>) -> std::io::Result<()> {
    let (method, path, body) = read_request(&mut sock).await?;
    let req: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
    match (method.as_str(), path.as_str()) {
        ("GET", "/health") => {
            if st.health_ok.load(Ordering::SeqCst) {
                respond(&mut sock, 200, &json!({"status": "ok"})).await
            } else {
                respond(&mut sock, 503, &json!({"error": {"code": 503, "message": "Loading model"}})).await
            }
        }
        ("GET", "/props") => respond(&mut sock, 200, &json!({"model_path": st.model_path, "build_info": "b99999-fake"})).await,
        ("GET", "/slots") => {
            if st.slots_missing.load(Ordering::SeqCst) {
                return respond(&mut sock, 404, &json!({"error": {"code": 404, "message": "not found"}})).await;
            }
            let processing = st.busy.load(Ordering::SeqCst) || st.active.load(Ordering::SeqCst) > 0;
            let mut slot = json!({"id": 0, "n_ctx": 2048, "is_processing": processing});
            if processing {
                slot["n_prompt_tokens_processed"] = json!(st.prompt_done.load(Ordering::SeqCst));
                slot["next_token"] = json!([{"has_next_token": true, "n_decoded": st.decoded.load(Ordering::SeqCst)}]);
            }
            respond(&mut sock, 200, &json!([slot])).await
        }
        ("POST", "/tokenize") => {
            let text = req["content"].as_str().unwrap_or("");
            let tokens = if req["parse_special"].as_bool() == Some(false) {
                if st.reject_output_tokenization.load(Ordering::SeqCst) {
                    return respond(&mut sock, 400, &json!({"error": "tokenization failed"})).await;
                }
                tokenize_output(text)
            } else {
                text.bytes().map(u32::from).collect()
            };
            respond(&mut sock, 200, &json!({"tokens": tokens})).await
        }
        ("POST", "/apply-template") => {
            let mut p = String::new();
            for m in req["messages"].as_array().into_iter().flatten() {
                p.push_str(&format!("{}: {}\n", m["role"].as_str().unwrap_or(""), m["content"].as_str().unwrap_or("")));
            }
            respond(&mut sock, 200, &json!({"prompt": p})).await
        }
        ("POST", "/completion") => completion(&mut sock, &st, &req).await,
        _ => respond(&mut sock, 404, &json!({"error": {"code": 404, "message": "not found"}})).await,
    }
}

struct ActiveGuard<'a>(&'a FakeState);
impl Drop for ActiveGuard<'_> {
    fn drop(&mut self) {
        self.0.active.fetch_sub(1, Ordering::SeqCst);
    }
}

async fn forever() {
    tokio::time::sleep(Duration::from_secs(3600)).await;
}

async fn token_chunk(sock: &mut TcpStream, id: u32, text: &str, count: usize, n_probs: u32) -> std::io::Result<()> {
    let mut c = json!({"index": 0, "content": text, "tokens": [id], "stop": false, "tokens_predicted": count});
    if n_probs > 0 {
        c["completion_probabilities"] = json!([entry(id, n_probs, 0)]);
    }
    sock.write_all(format!("data: {c}\n\n").as_bytes()).await
}

async fn completion(sock: &mut TcpStream, st: &FakeState, req: &Value) -> std::io::Result<()> {
    st.completions.fetch_add(1, Ordering::SeqCst);
    // An abandoned request can still retain the single slot.
    while st.busy.load(Ordering::SeqCst) {
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    if st.hang_completion.load(Ordering::SeqCst) {
        forever().await;
    }
    let prompt = prompt_of(req);
    if prompt.len() > 5000 {
        return respond(sock, 400, &json!({"error": {"code": 400, "type": "exceed_context_size_error", "message": "the request exceeds the available context size"}})).await;
    }
    st.active.fetch_add(1, Ordering::SeqCst);
    let _g = ActiveGuard(st);
    st.decoded.store(0, Ordering::SeqCst);
    st.prompt_done.store(0, Ordering::SeqCst);
    let stream = req["stream"].as_bool().unwrap_or(false);
    let progress = req["return_progress"].as_bool().unwrap_or(false);
    let n_probs = req["n_probs"].as_u64().unwrap_or(0) as u32;
    let utf8_hold = st.utf8_hold.load(Ordering::SeqCst);
    let (toks, stop) = gen_tokens(&prompt, req, st.utf8_split.load(Ordering::SeqCst), utf8_hold);
    let steps = st.prefill_steps.load(Ordering::SeqCst).max(1);
    let total = prompt.len() as u64;

    if stream {
        sock.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n").await?;
    }
    for s in 0..=steps {
        let processed = total * u64::from(s) / u64::from(steps);
        st.prompt_done.store(processed, Ordering::SeqCst);
        if stream && progress {
            let c = json!({"index": 0, "content": "", "tokens": [0], "stop": false, "tokens_predicted": 0, "tokens_evaluated": total,
                "prompt_progress": {"total": total, "cache": 0, "processed": processed, "time_ms": 0}});
            sock.write_all(format!("data: {c}\n\n").as_bytes()).await?;
        }
        let ms = st.prefill_step_ms.load(Ordering::SeqCst);
        if s < steps && ms > 0 {
            tokio::time::sleep(Duration::from_millis(ms)).await;
        }
    }
    let stall_after = st.decode_stall_after.load(Ordering::SeqCst);
    let mut out_text = String::new();
    let incorrect_count = st.incorrect_stream_count.load(Ordering::SeqCst);
    let second_stream_count = usize::try_from(st.second_stream_count.load(Ordering::SeqCst)).ok();
    let mut held = String::new();
    let mut pending = 0;
    let mut tail = ("", Vec::new());
    for (i, t) in toks.iter().enumerate() {
        if stall_after >= 0 && i as i64 >= stall_after {
            forever().await;
        }
        let d = st.token_delay_ms.load(Ordering::SeqCst);
        if d > 0 {
            tokio::time::sleep(Duration::from_millis(d)).await;
        }
        st.decoded.fetch_add(1, Ordering::SeqCst);
        out_text.push_str(&t.text);
        if !stream {
            continue;
        }
        if t.partial {
            if utf8_hold {
                pending += 1;
                held.push_str(&t.text);
            }
            continue;
        }
        for t in &toks[i - pending..i] {
            token_chunk(sock, t.id, "", i + 1, n_probs).await?;
        }
        pending = 0;
        if incorrect_count && i + 1 == toks.len() {
            tail = (&t.text, vec![t.id]);
            break;
        }
        let text = if t.held {
            held.push_str(&t.text);
            String::new()
        } else {
            std::mem::take(&mut held) + &t.text
        };
        let count = if i == 1 { second_stream_count.unwrap_or(i + 1) } else { i + 1 };
        token_chunk(sock, t.id, &text, count, n_probs).await?;
    }
    let timings = json!({"prompt_ms": 1.5, "predicted_ms": 4.5});
    if stream {
        for (i, t) in toks.iter().enumerate().skip(toks.len() - pending) {
            let text = if i + 1 == toks.len() { &held } else { "" };
            token_chunk(sock, t.id, text, toks.len(), n_probs).await?;
        }
        let predicted = toks.len() + usize::from(incorrect_count);
        let mut c = json!({"index": 0, "content": tail.0, "tokens": tail.1, "stop": true, "stop_type": stop, "tokens_predicted": predicted, "timings": timings});
        if utf8_hold && req["return_tokens"] == true {
            c["stop_text_withheld"] = json!(stop == "word");
        }
        sock.write_all(format!("data: {c}\n\n").as_bytes()).await?;
        return sock.flush().await;
    }
    let mut r = json!({"index": 0, "content": out_text, "tokens": toks.iter().map(|t| t.id).collect::<Vec<_>>(),
        "stop": true, "stop_type": stop, "tokens_predicted": toks.len(), "timings": timings});
    if n_probs > 0 {
        r["completion_probabilities"] = json!(toks.iter().filter(|t| utf8_hold || !t.partial).map(|t| entry(t.id, n_probs, 0)).collect::<Vec<_>>());
    }
    respond(sock, 200, &r).await
}
