//! Stage process fixture for exercising supervision and the head service protocol.
//! Intermediate and tail processes bind their data listener and announce readiness.
//! The head handles concurrent requests and streams a fixed token sequence.
//! One UTF-8 character spans two tokens, so its first piece has no display text.
//! Environment variables can inject startup delay, token delay, early exit, or
//! ignored termination. A blocked request waits for cancellation.

use bytes::BytesMut;
use publicompute_proto::service as pb;
use publicompute_proto::wire::{encode_frame, try_decode_frame};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Each entry carries a token ID, its display text, and the end-of-generation flag.
const FINTI: &[(i32, &str, bool)] = &[(101, "Ciao ", false), (102, "", false), (103, "\u{e8}", false), (104, " un test", false), (2, "", true)];

fn env_ms(k: &str) -> Duration {
    Duration::from_millis(std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(0))
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let role = args.first().cloned().unwrap_or_default();
    let val = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    eprintln!("[fake-stage] {role} args: {}", args.join(" "));
    std::thread::sleep(env_ms("FAKE_STAGE_START_DELAY_MS"));
    let exit_ms = env_ms("FAKE_STAGE_EXIT_AFTER_MS");
    if !exit_ms.is_zero() {
        std::thread::spawn(move || {
            std::thread::sleep(exit_ms);
            eprintln!("[fake-stage] exiting early");
            std::process::exit(3);
        });
    }
    #[cfg(unix)]
    if std::env::var("FAKE_STAGE_IGNORE_TERM").is_ok() {
        unsafe { libc::signal(libc::SIGTERM, libc::SIG_IGN) };
    }
    match role.as_str() {
        "head" => {
            let Some(addr) = val("--service") else {
                eprintln!("error: the fake head requires --service");
                std::process::exit(2);
            };
            head(&addr);
        }
        "middle" | "tail" => {
            let Some(addr) = val("--listen") else {
                eprintln!("error: missing --listen");
                std::process::exit(2);
            };
            let _l = TcpListener::bind(&addr).unwrap_or_else(|e| {
                eprintln!("error: bind {addr}: {e}");
                std::process::exit(2);
            });
            // Keep the readiness line in sync with the stage executable because the supervisor parses it.
            eprintln!("[stage] {} layer [{}, {}) ready on {addr}", if role == "tail" { "Tail" } else { "Middle" }, val("--layer-start").unwrap_or_default(), val("--layer-end").unwrap_or_default());
            loop {
                std::thread::sleep(Duration::from_secs(3600));
            }
        }
        _ => {
            eprintln!("error: unknown role {role}");
            std::process::exit(2);
        }
    }
}

fn manda(s: &mut TcpStream, m: pb::FromHead) -> bool {
    let mut v = Vec::new();
    encode_frame(&m, &mut v).unwrap();
    s.write_all(&v).is_ok()
}

fn head(addr: &str) {
    let l = TcpListener::bind(addr).unwrap_or_else(|e| {
        eprintln!("error: bind {addr}: {e}");
        std::process::exit(2);
    });
    eprintln!("[head] fake chain ready; service listening on {addr}");
    let ritardo = env_ms("FAKE_STAGE_TOKEN_DELAY_MS");
    for conn in l.incoming() {
        let Ok(stream) = conn else { continue };
        let _ = stream.set_nodelay(true);
        // Each request owns a thread. Workers serialize writes to keep response frames intact.
        let write = Arc::new(Mutex::new(stream.try_clone().unwrap()));
        let annullate: Arc<Mutex<HashSet<u64>>> = Arc::new(Mutex::new(HashSet::new()));
        let mut stream = stream;
        let (mut buf, mut tmp) = (BytesMut::new(), vec![0u8; 4096]);
        {
            let mut w = write.lock().unwrap();
            manda(&mut w, pb::FromHead { body: Some(pb::from_head::Body::Status(pb::Status { free_slots: 4, pending: 0 })) });
        }
        'lettura: loop {
            while let Ok(Some(m)) = try_decode_frame::<pb::FromAgent>(&mut buf) {
                match m.body {
                    Some(pb::from_agent::Body::Request(r)) => {
                        let (write, annullate) = (write.clone(), annullate.clone());
                        std::thread::spawn(move || request(r, ritardo, &write, &annullate));
                    }
                    Some(pb::from_agent::Body::Cancel(a)) => {
                        // Log cancellation so the supervisor tests can observe it.
                        eprintln!("[fake-stage] Cancel received for request {}", a.id);
                        annullate.lock().unwrap().insert(a.id);
                    }
                    None => {}
                }
            }
            match stream.read(&mut tmp) {
                Ok(0) | Err(_) => break 'lettura,
                Ok(n) => buf.extend_from_slice(&tmp[..n]),
            }
        }
    }
}

/// Use the last user message to select a response, falling back to the plain-text request.
fn testo_di(r: &pb::Request) -> String {
    r.messages.iter().rev().find(|m| m.role == "user").map(|m| m.text.clone()).unwrap_or_else(|| r.text.clone())
}

fn request(r: pb::Request, ritardo: Duration, write: &Arc<Mutex<TcpStream>>, annullate: &Arc<Mutex<HashSet<u64>>>) {
    let text = testo_di(&r);
    let send = |m: pb::FromHead| {
        let mut w = write.lock().unwrap();
        manda(&mut w, m)
    };
    if text.contains("rifiuta") {
        send(pb::FromHead { body: Some(pb::from_head::Body::Done(pb::Done { id: r.id, reason: "error".into(), error: "prompt exceeds the slot context".into(), ..Default::default() })) });
        return;
    }
    // Hold this request until cancellation to simulate a prefill that never finishes.
    if text.contains("blocca") {
        while !annullate.lock().unwrap().remove(&r.id) {
            std::thread::sleep(Duration::from_millis(20));
        }
        send(pb::FromHead { body: Some(pb::from_head::Body::Done(pb::Done { id: r.id, reason: "cancelled".into(), ..Default::default() })) });
        return;
    }
    let max = if r.max_tokens == 0 { u32::MAX } else { r.max_tokens } as usize;
    let mut reason = "eog";
    let mut generati = 0u32;
    for (i, (t, testo_tok, eog)) in FINTI.iter().enumerate() {
        if i >= max {
            reason = "max_token";
            break;
        }
        std::thread::sleep(ritardo);
        if annullate.lock().unwrap().remove(&r.id) {
            reason = "cancelled";
            break;
        }
        generati += 1;
        send(pb::FromHead { body: Some(pb::from_head::Body::Token(pb::OutputToken { id: r.id, token: *t, text: testo_tok.to_string(), logprob: -0.25 * (i as f32 + 1.0), step: i as u32, eog: *eog })) });
    }
    // Count bytes across the entire conversation so callers can verify that all turns reached the head.
    let prompt: Vec<i32> = if !r.prompt_token.is_empty() {
        r.prompt_token.clone()
    } else if !r.messages.is_empty() {
        r.messages.iter().flat_map(|m| m.text.bytes()).map(i32::from).collect()
    } else {
        r.text.bytes().map(i32::from).collect()
    };
    send(pb::FromHead {
        body: Some(pb::from_head::Body::Done(pb::Done {
            id: r.id,
            reason: reason.into(),
            usage: Some(pb::Usage { prompt_tokens: prompt.len() as u32, generated_tokens: generati }),
            timings: Some(pb::Timings { wait_ms: 1.0, ttft_ms: 5.0, total_ms: 10.0 }),
            prompt_token: prompt,
            ..Default::default()
        })),
    });
}
