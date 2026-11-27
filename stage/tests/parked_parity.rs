//! Native parity over the real stage and service links. Set PUBLICOMPUTE_Q35_GGUF
//! and PUBLICOMPUTE_Q35_MTP_GGUF to Qwen3.5-MoE fixtures with a chat template.
//! The same fixtures must support q8_0 attention KV with Flash Attention.
#![cfg(feature = "llama")]

use bytes::BytesMut;
use publicompute_proto::{service as pb, wire};
use publicompute_stage::{
    engine::{self, Engine, EngineOptions, Row},
    kv_cache::CacheConfig,
    parking, sampling,
};
use std::io::{Read, Write};
use std::net::TcpListener;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::{Mutex, mpsc};
use std::time::Duration;

const QUESTION: &str = "Remember the list red, blue, green. What is the first color?";
const FOLLOW_UP: &str = "What is the last color? Keep the answer brief.";
const N_BATCH: u32 = 64;
const CONTEXT: u32 = 512;
static NATIVE_CHAIN: Mutex<()> = Mutex::new(());

#[derive(Clone, Copy)]
struct Profile<'a> {
    model: &'a str,
    stages: u32,
    mtp: bool,
    cache: &'a str,
    threads: i32,
    ngl: i32,
}

impl Profile<'_> {
    fn boundary(self) -> (usize, bool) {
        let model = Engine::load(&EngineOptions {
            model: self.model.into(),
            layer_start: 0,
            layer_end: 1,
            ngl: self.ngl,
            threads: self.threads,
            slots: 1,
            groups: 1,
            ctx_per_slot: CONTEXT,
            n_batch: N_BATCH,
            experts: false,
            max_draft_tokens: u32::from(self.mtp),
            mtp: false,
            cache: CacheConfig::parse(Some(self.cache), Some(self.cache), Some("on")).unwrap(),
        })
        .unwrap();
        assert!(model.supports_parking(), "this fixture must be qwen35moe");
        let message = message("user", QUESTION);
        let (tokens, _, _) = model.prepara_conversazione(&[&message]).unwrap();
        let boundary = model
            .user_checkpoint(&[&message], &tokens)
            .expect("missing verified user boundary");
        assert!(boundary < N_BATCH);
        (boundary as usize, model.chat_uses_think_tags())
    }

    fn command(self, role: &str, start: u32, end: i32, ttl: u32, slots: u32) -> Command {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_publicompute-stage"));
        cmd.args([
            role,
            "--model",
            self.model,
            "--layer-start",
            &start.to_string(),
            "--layer-end",
            &end.to_string(),
            "--slots",
            &slots.to_string(),
            "--groups",
            "1",
            "--ctx-per-slot",
            &CONTEXT.to_string(),
            "--n-batch",
            &N_BATCH.to_string(),
            "--threads",
            &self.threads.to_string(),
            "--ngl",
            &self.ngl.to_string(),
            "--timeout-s",
            "60",
            "--cache-type-k",
            self.cache,
            "--cache-type-v",
            self.cache,
            "--flash-attn",
            "on",
            "--park-ttl-s",
            &ttl.to_string(),
            "--draft-max-tokens",
            if self.mtp { "1" } else { "0" },
        ])
        .stdout(Stdio::null());
        if std::env::var("PUBLICOMPUTE_PARITY_LOG_LLAMA").ok().as_deref() == Some("1") {
            cmd.arg("--log-llama");
        }
        cmd
    }
}

struct Chain {
    children: Vec<Child>,
    input: ChildStdin,
    output: mpsc::Receiver<pb::from_head::Body>,
}

impl Drop for Chain {
    fn drop(&mut self) {
        for child in &mut self.children {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

impl Chain {
    fn start(profile: Profile<'_>, ttl: u32, block: usize) -> Self {
        Self::start_with_slots(profile, ttl, block, 1, 0)
    }

    fn start_with_slots(profile: Profile<'_>, ttl: u32, block: usize, slots: u32, grace: u32) -> Self {
        let listeners: Vec<_> = (0..profile.stages)
            .map(|_| TcpListener::bind("127.0.0.1:0").unwrap())
            .collect();
        let addresses: Vec<_> = listeners
            .iter()
            .map(|socket| socket.local_addr().unwrap().to_string())
            .collect();
        drop(listeners);
        let (layers, _) = engine::info_gguf(profile.model).unwrap();
        // Leave at least one trunk layer in the tail even when the GGUF
        // counts an additional draft layer.
        let first_cut = layers / profile.stages;
        let mut head = profile.command("head", 0, first_cut as i32, ttl, slots);
        head.args([
            "--service-pipe",
            "--park-grace-s",
            &grace.to_string(),
            "--next",
            &addresses[1],
            "--listen-token",
            &addresses[0],
            "--block",
            &block.to_string(),
            "--draft-source",
            if profile.mtp { "mtp" } else { "ngram" },
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped());
        let mut head = head.spawn().unwrap();
        let input = head.stdin.take().unwrap();
        let mut stdout = head.stdout.take().unwrap();
        let (tx, output) = mpsc::channel();
        std::thread::spawn(move || {
            let mut buf = BytesMut::new();
            let mut chunk = [0u8; 8192];
            while let Ok(n) = stdout.read(&mut chunk) {
                if n == 0 {
                    break;
                }
                buf.extend_from_slice(&chunk[..n]);
                while let Some(frame) = wire::try_decode_frame::<pb::FromHead>(&mut buf).unwrap() {
                    if tx.send(frame.body.unwrap()).is_err() {
                        return;
                    }
                }
            }
        });
        let mut chain = Chain {
            children: vec![head],
            input,
            output,
        };
        for index in 1..profile.stages {
            let tail = index + 1 == profile.stages;
            let mut command = profile.command(
                if tail { "tail" } else { "middle" },
                index * first_cut,
                if tail { -1 } else { ((index + 1) * first_cut) as i32 },
                ttl,
                slots,
            );
            command.args([
                "--listen",
                &addresses[index as usize],
                "--index",
                &(index + 1).to_string(),
            ]);
            if tail {
                command.args(["--head-token", &addresses[0]]);
                if profile.mtp {
                    command.arg("--mtp");
                }
            } else {
                command.args(["--next", &addresses[index as usize + 1]]);
            }
            chain.children.push(command.spawn().unwrap());
        }
        assert!(matches!(chain.recv(), pb::from_head::Body::Ready(_)));
        chain
    }

    fn recv(&self) -> pb::from_head::Body {
        self.output
            .recv_timeout(Duration::from_secs(90))
            .expect("service did not return a frame")
    }

    fn chat(&mut self, id: u64, owner: u8, messages: Vec<pb::Message>) -> Reply {
        self.chat_for_member(id, owner, 0, messages)
    }

    fn chat_for_member(&mut self, id: u64, owner: u8, member: u8, messages: Vec<pb::Message>) -> Reply {
        let mut frame = BytesMut::new();
        let request = pb::Request {
            id,
            messages,
            max_tokens: 12,
            sampling: Some(pb::Sampling {
                temperature: 0.0,
                seed: 1,
                ..Default::default()
            }),
            owner_key: vec![owner; 16],
            member_seat: member != 0,
            member_key: if member == 0 { vec![] } else { vec![member; 16] },
            ..Default::default()
        };
        wire::encode_frame(
            &pb::FromAgent {
                body: Some(pb::from_agent::Body::Request(request)),
            },
            &mut frame,
        )
        .unwrap();
        self.input.write_all(&frame).unwrap();
        let mut tokens = Vec::new();
        let mut text = String::new();
        let mut admission = None;
        loop {
            match self.recv() {
                pb::from_head::Body::Status(_) => {}
                pb::from_head::Body::Admission(event) => {
                    assert_eq!(event.id, id);
                    admission = event.slot_taken;
                }
                pb::from_head::Body::Token(token) => {
                    assert_eq!(token.id, id);
                    assert_eq!(token.step as usize, tokens.len());
                    text.push_str(&token.text);
                    tokens.push((token.token, token.logprob));
                }
                pb::from_head::Body::Done(done) => {
                    assert_eq!(done.id, id);
                    assert!(matches!(done.reason.as_str(), "eog" | "max_token"), "{done:?}");
                    assert!(!tokens.is_empty());
                    assert_eq!(
                        admission.map(|a| (a.slot, a.generation)),
                        done.slot_report.as_ref().map(|r| (r.slot, r.generation)),
                    );
                    text.push_str(&done.remaining_text);
                    return Reply {
                        tokens,
                        text,
                        cached: done.usage.unwrap().cached_prompt_tokens,
                        prompt: done.prompt_token,
                        slot_report: done.slot_report,
                    };
                }
                event => panic!("unexpected response: {event:?}"),
            }
        }
    }
}

struct Reply {
    tokens: Vec<(i32, f32)>,
    text: String,
    cached: u32,
    prompt: Vec<i32>,
    slot_report: Option<publicompute_proto::common::SlotReport>,
}

fn message(role: &str, text: &str) -> pb::Message {
    pb::Message {
        role: role.into(),
        text: text.into(),
    }
}

struct ComparisonContext<'a> {
    stages: u32,
    cache: &'a str,
    mtp: bool,
    block: usize,
    cold_ttl: u32,
    label: &'a str,
    boundary: usize,
    think_tags: bool,
    first: &'a Reply,
}

fn token_window(tokens: &[i32], center: usize) -> (usize, &[i32]) {
    let start = center.saturating_sub(4).min(tokens.len());
    let end = center.saturating_add(6).min(tokens.len());
    (start, &tokens[start..end])
}

fn planned_blocks(mut offset: usize, total: usize, limit: usize, save_at: Option<u32>) -> Vec<(usize, usize, bool)> {
    let mut out = Vec::new();
    while offset < total {
        let n = parking::block_len(offset, total, limit, save_at);
        let end = offset + n;
        let save = save_at == Some(end as u32);
        out.push((offset, end, save));
        offset = end;
    }
    out
}

fn format_diagnostics(ctx: &ComparisonContext<'_>, warm: &Reply, cold: &Reply) -> String {
    let (w_start, w_win) = token_window(&warm.prompt, ctx.boundary);
    let (c_start, c_win) = token_window(&cold.prompt, ctx.boundary);
    let prompt_diff = warm
        .prompt
        .iter()
        .zip(&cold.prompt)
        .position(|(a, b)| a != b)
        .or_else(|| (warm.prompt.len() != cold.prompt.len()).then(|| warm.prompt.len().min(cold.prompt.len())));
    let warm_blocks = planned_blocks(warm.cached as usize, warm.prompt.len(), ctx.block, None);
    let cold_blocks = planned_blocks(cold.cached as usize, cold.prompt.len(), ctx.block, None);
    let mut msg = format!(
        "\n--- parked_parity diagnostics ---\n\
         case: stages={} cache={} mtp={} block={} cold_ttl={} comparison={}\n\
         boundary: p={} think_tags={} first.prompt_len={} first.tokens={:?}\n\
         cached: warm={} cold={}\n\
         prompt: warm_len={} cold_len={} first_diff={:?}\n\
         prompt_around_p: warm[{}..]={:?} cold[{}..]={:?}\n\
         blocks_without_next_ckpt: warm={:?} cold={:?}\n\
         tokens:\n",
        ctx.stages,
        ctx.cache,
        ctx.mtp,
        ctx.block,
        ctx.cold_ttl,
        ctx.label,
        ctx.boundary,
        ctx.think_tags,
        ctx.first.prompt.len(),
        ctx.first.tokens,
        warm.cached,
        cold.cached,
        warm.prompt.len(),
        cold.prompt.len(),
        prompt_diff,
        w_start,
        w_win,
        c_start,
        c_win,
        warm_blocks,
        cold_blocks,
    );
    let steps = warm.tokens.len().max(cold.tokens.len());
    for step in 0..steps {
        let a = warm.tokens.get(step).copied();
        let b = cold.tokens.get(step).copied();
        let diff = match (a, b) {
            (Some((_, pa)), Some((_, pb))) => format!("{:.6}", (pa - pb).abs()),
            _ => "-".into(),
        };
        msg.push_str(&format!("  step {step:2}: warm={a:?} cold={b:?} |dlp|={diff}\n"));
    }
    msg
}

fn compare(ctx: &ComparisonContext<'_>, warm: &Reply, cold: &Reply, identical: bool) {
    assert_eq!(
        warm.prompt,
        cold.prompt,
        "visible history must yield the same prompt{}",
        format_diagnostics(ctx, warm, cold)
    );
    assert_eq!(
        warm.tokens.len(),
        cold.tokens.len(),
        "different reply length{}",
        format_diagnostics(ctx, warm, cold)
    );
    for (step, ((a, pa), (b, pb))) in warm.tokens.iter().zip(&cold.tokens).enumerate() {
        assert_eq!(
            a,
            b,
            "different token at step {step}{}",
            format_diagnostics(ctx, warm, cold)
        );
        if identical {
            assert_eq!(
                pa.to_bits(),
                pb.to_bits(),
                "eviction changed log probability at step {step}{}",
                format_diagnostics(ctx, warm, cold)
            );
        } else {
            assert!(
                (pa - pb).abs() <= 1e-4,
                "log probabilities at step {step}: {pa} vs {pb}{}",
                format_diagnostics(ctx, warm, cold)
            );
        }
    }
}

fn parity(stages: u32, cache: &str, mtp: bool) {
    let variable = if mtp {
        "PUBLICOMPUTE_Q35_MTP_GGUF"
    } else {
        "PUBLICOMPUTE_Q35_GGUF"
    };
    let model = std::env::var(variable).unwrap_or_else(|_| panic!("set {variable} to a Qwen fixture"));
    let _serial = NATIVE_CHAIN.lock().unwrap_or_else(|e| e.into_inner());
    let threads = std::env::var("PUBLICOMPUTE_TEST_THREADS").map_or(2, |v| v.parse().unwrap());
    let ngl = std::env::var("PUBLICOMPUTE_TEST_NGL").map_or(0, |v| v.parse().unwrap());
    let profile = Profile {
        model: &model,
        stages,
        mtp,
        cache,
        threads,
        ngl,
    };
    let (p, think_tags) = profile.boundary();
    // First align the cold and warm batch boundaries: only then the two paths
    // run the same computation, so tokens must match. Then run the normal cold
    // path, where the first batch crosses the retained boundary. A different
    // batch split changes floating-point rounding (and, in a MoE model, can
    // flip expert routing), so there only the reuse itself is checked.
    for (block, cold_ttl) in [(p, 900), (N_BATCH as usize, 0)] {
        let mut warm = Chain::start(profile, 900, block);
        let first = warm.chat(1, 1, vec![message("user", QUESTION)]);
        assert_eq!(first.cached, 0);
        let first_slot = first.slot_report.as_ref().expect("head should report a retained slot");
        assert!(first_slot.parked);
        let mut visible = first.text.clone();
        if think_tags {
            publicompute_stage::service::strip_reasoning(&mut visible);
        }
        let visible = if think_tags { visible.trim_start() } else { &visible };
        let raw = think_tags.then(|| format!("<think>Hidden reasoning for this fixture.</think> {visible}"));
        let history = |answer: &str| {
            vec![
                message("user", QUESTION),
                message("assistant", answer),
                message("user", FOLLOW_UP),
            ]
        };
        let continued = warm.chat(2, 1, history(raw.as_deref().unwrap_or(visible)));
        assert_eq!(continued.cached, p as u32);
        let continued_slot = continued.slot_report.as_ref().expect("head should report the continued slot");
        assert_eq!(continued_slot.slot, first_slot.slot);
        assert!(continued_slot.parked && continued_slot.generation > first_slot.generation);
        let fresh = Chain::start(profile, cold_ttl, block).chat(1, 2, history(visible));
        assert_eq!(fresh.cached, 0);
        if cold_ttl == 0 {
            assert!(fresh.slot_report.is_none());
        }
        if block != p {
            assert_eq!(continued.prompt, fresh.prompt);
            assert!(!continued.tokens.is_empty());
            continue;
        }
        let continued_ctx = ComparisonContext {
            stages,
            cache,
            mtp,
            block,
            cold_ttl,
            label: "continued vs fresh",
            boundary: p,
            think_tags,
            first: &first,
        };
        compare(&continued_ctx, &continued, &fresh, false);
        if cold_ttl > 0 {
            // Force eviction in the only slot. Even an identical prompt from
            // another owner must compute from zero.
            let evicted = warm.chat(3, 2, history(visible));
            assert_eq!(evicted.cached, 0);
            let evicted_ctx = ComparisonContext {
                label: "evicted vs fresh",
                ..continued_ctx
            };
            compare(&evicted_ctx, &evicted, &fresh, true);
        }
    }
}

fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn top_logits(logits: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut pairs: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    pairs.sort_by(|a, b| b.1.total_cmp(&a.1));
    pairs.truncate(k);
    pairs
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(0.0f32, |m, (&x, &y)| m.max((x - y).abs()))
}

fn load_single_engine(model: &str, layer_start: i32, layer_end: i32, cache: &str, threads: i32, ngl: i32) -> Engine {
    Engine::load(&EngineOptions {
        model: model.into(),
        layer_start,
        layer_end,
        ngl,
        threads,
        slots: 1,
        groups: 1,
        ctx_per_slot: CONTEXT,
        n_batch: N_BATCH,
        experts: false,
        max_draft_tokens: 0,
        mtp: false,
        cache: CacheConfig::parse(Some(cache), Some(cache), Some("on")).unwrap(),
    })
    .unwrap()
}

fn run_single_prefill(
    engine: &mut Engine,
    tokens: &[i32],
    start: usize,
    block: usize,
    save_at: Option<u32>,
) -> (Vec<f32>, Option<zeroize::Zeroizing<Vec<u8>>>) {
    let total = tokens.len();
    let mut off = start;
    let mut saved = None;
    while off < total {
        let n = parking::block_len(off, total, block, save_at);
        let last = off + n == total;
        let save = save_at == Some((off + n) as u32);
        let rows: Vec<Row> = (0..n)
            .map(|i| Row {
                token: tokens[off + i],
                pos: (off + i) as i32,
                seq: 0,
                logits: last && i + 1 == n,
            })
            .collect();
        engine.run(&rows, None).unwrap();
        if save {
            saved = Some(engine.save_checkpoint(0).unwrap());
        }
        off += n;
        if last {
            return (engine.logits(n - 1).unwrap().to_vec(), saved);
        }
    }
    panic!("empty prefill suffix");
}

fn run_single_turn1_gen(engine: &mut Engine, prompt: &[i32], first_logits: &[f32], max_tokens: usize) -> (Vec<i32>, String) {
    let mut generated = Vec::new();
    let mut bytes = Vec::new();
    let mut next = sampling::argmax(first_logits) as i32;
    for step in 0..max_tokens {
        generated.push(next);
        if engine.e_eog(next) {
            break;
        }
        bytes.extend_from_slice(&engine.pezzo(next));
        if step + 1 == max_tokens {
            break;
        }
        let pos = (prompt.len() + generated.len() - 1) as i32;
        let row = [Row {
            token: next,
            pos,
            seq: 0,
            logits: true,
        }];
        engine.run(&row, None).unwrap();
        next = sampling::argmax(engine.logits(0).unwrap()) as i32;
    }
    (generated, String::from_utf8_lossy(&bytes).into_owned())
}

fn run_two_stage_prefill(
    head: &mut Engine,
    tail: &mut Engine,
    tokens: &[i32],
    start: usize,
    block: usize,
    save_at: Option<u32>,
) -> (
    Vec<f32>,
    Vec<(usize, usize, Vec<f32>)>,
    Option<(zeroize::Zeroizing<Vec<u8>>, zeroize::Zeroizing<Vec<u8>>)>,
) {
    let total = tokens.len();
    let mut off = start;
    let mut activations = Vec::new();
    let mut saved = None;
    while off < total {
        let n = parking::block_len(off, total, block, save_at);
        let last = off + n == total;
        let save = save_at == Some((off + n) as u32);
        let head_rows: Vec<Row> = (0..n)
            .map(|i| Row {
                token: tokens[off + i],
                pos: (off + i) as i32,
                seq: 0,
                logits: last && i + 1 == n,
            })
            .collect();
        let state = head.run(&head_rows, None).unwrap().unwrap();
        let ck_head = save.then(|| head.save_checkpoint(0).unwrap());
        let tail_rows: Vec<Row> = (0..n)
            .map(|i| Row {
                token: 0,
                pos: (off + i) as i32,
                seq: 0,
                logits: last && i + 1 == n,
            })
            .collect();
        tail.run(&tail_rows, Some(&state)).unwrap();
        let ck_tail = save.then(|| tail.save_checkpoint(0).unwrap());
        if let (Some(h), Some(t)) = (ck_head, ck_tail) {
            saved = Some((h, t));
        }
        activations.push((off, off + n, state));
        off += n;
        if last {
            return (tail.logits(n - 1).unwrap().to_vec(), activations, saved);
        }
    }
    panic!("empty two-stage prefill suffix");
}

#[test]
#[ignore = "requires PUBLICOMPUTE_Q35_GGUF and runs checkpoint diagnostics"]
fn diagnose_checkpoint_parity() {
    let model = std::env::var("PUBLICOMPUTE_Q35_GGUF").expect("set PUBLICOMPUTE_Q35_GGUF to a Qwen fixture");
    let _serial = NATIVE_CHAIN.lock().unwrap_or_else(|e| e.into_inner());
    let threads = std::env::var("PUBLICOMPUTE_TEST_THREADS").map_or(2, |v| v.parse().unwrap());
    let ngl = std::env::var("PUBLICOMPUTE_TEST_NGL").map_or(0, |v| v.parse().unwrap());
    let cache = std::env::var("PUBLICOMPUTE_PARITY_CACHE").unwrap_or_else(|_| "f16".into());
    if std::env::var("PUBLICOMPUTE_PARITY_LOG_LLAMA").ok().as_deref() == Some("1") {
        engine::set_llama_logging(true);
    }

    let (layers, _) = engine::info_gguf(&model).unwrap();
    let mut turn1 = load_single_engine(&model, 0, -1, &cache, threads, ngl);
    assert!(turn1.supports_parking());
    let msg1 = message("user", QUESTION);
    let (prompt1, _, _) = turn1.prepara_conversazione(&[&msg1]).unwrap();
    let p = turn1.user_checkpoint(&[&msg1], &prompt1).unwrap() as usize;
    let (p1_start, p1_win) = token_window(&prompt1, p);
    eprintln!(
        "[diag] layers={layers} cache={cache} threads={threads} ngl={ngl} p={p} prompt1_len={} prompt1[{p1_start}..]={p1_win:?}",
        prompt1.len()
    );

    let (t1_logits, ck1) = run_single_prefill(&mut turn1, &prompt1, 0, p, Some(p as u32));
    let ck1 = ck1.expect("turn 1 checkpoint missing");
    let ck1_nz = ck1.iter().filter(|&&b| b != 0).count();
    eprintln!(
        "[diag] ck1 bytes={} nonzero={} fnv1a={:#018x}",
        ck1.len(),
        ck1_nz,
        fnv1a_64(&ck1)
    );
    let (t1_tokens, mut visible) = run_single_turn1_gen(&mut turn1, &prompt1, &t1_logits, 12);
    if turn1.chat_uses_think_tags() {
        publicompute_stage::service::strip_reasoning(&mut visible);
    }
    let visible = if turn1.chat_uses_think_tags() { visible.trim_start() } else { &visible };
    let history = [
        message("user", QUESTION),
        message("assistant", visible),
        message("user", FOLLOW_UP),
    ];
    let history_refs: Vec<&pb::Message> = history.iter().collect();
    let (prompt2, _, _) = turn1.prepara_conversazione(&history_refs).unwrap();
    let p2 = turn1.user_checkpoint(&history_refs, &prompt2).unwrap();
    let (p2_start, p2_win) = token_window(&prompt2, p);
    eprintln!(
        "[diag] turn1_tokens={t1_tokens:?} prompt2_len={} p2={p2} starts_with_prefix={} prompt2[{p2_start}..]={p2_win:?}",
        prompt2.len(),
        prompt2.starts_with(&prompt1[..p])
    );

    let mut cold_a_eng = load_single_engine(&model, 0, -1, &cache, threads, ngl);
    let (cold_a, _) = run_single_prefill(&mut cold_a_eng, &prompt2, 0, p, Some(p2));
    eprintln!(
        "[diag] single cold_a (block=p={p}, save_at={p2}): argmax={} top5={:?}",
        sampling::argmax(&cold_a),
        top_logits(&cold_a, 5)
    );

    let mut cold_b_eng = load_single_engine(&model, 0, -1, &cache, threads, ngl);
    let (cold_b, _) = run_single_prefill(&mut cold_b_eng, &prompt2, 0, N_BATCH as usize, None);
    eprintln!(
        "[diag] single cold_b (block=64, save_at=None): argmax={} max_diff_vs_a={:.6e} top5={:?}",
        sampling::argmax(&cold_b),
        max_abs_diff(&cold_b, &cold_a),
        top_logits(&cold_b, 5)
    );

    let mut immediate_eng = load_single_engine(&model, 0, -1, &cache, threads, ngl);
    let rows0: Vec<Row> = (0..p)
        .map(|i| Row {
            token: prompt1[i],
            pos: i as i32,
            seq: 0,
            logits: false,
        })
        .collect();
    immediate_eng.run(&rows0, None).unwrap();
    let ck_imm_before = immediate_eng.save_checkpoint(0).unwrap();
    immediate_eng.restore_checkpoint(0, p as u32, &ck_imm_before).unwrap();
    let ck_imm_after = immediate_eng.save_checkpoint(0).unwrap();
    let (warm_c, _) = run_single_prefill(&mut immediate_eng, &prompt2, p, p, Some(p2));
    eprintln!(
        "[diag] single warm_c (immediate restore at p): roundtrip_eq={} ck_eq_ck1={} argmax={} max_diff_vs_a={:.6e} top5={:?}",
        ck_imm_before.as_slice() == ck_imm_after.as_slice(),
        ck_imm_before.as_slice() == ck1.as_slice(),
        sampling::argmax(&warm_c),
        max_abs_diff(&warm_c, &cold_a),
        top_logits(&warm_c, 5)
    );

    turn1.restore_checkpoint(0, p as u32, &ck1).unwrap();
    let ck_after_t1 = turn1.save_checkpoint(0).unwrap();
    let (warm_d, _) = run_single_prefill(&mut turn1, &prompt2, p, p, Some(p2));
    eprintln!(
        "[diag] single warm_d (post-turn1 restore, block=p): roundtrip_eq={} argmax={} max_diff_vs_a={:.6e} top5={:?}",
        ck1.as_slice() == ck_after_t1.as_slice(),
        sampling::argmax(&warm_d),
        max_abs_diff(&warm_d, &cold_a),
        top_logits(&warm_d, 5)
    );

    let mut turn1_b64 = load_single_engine(&model, 0, -1, &cache, threads, ngl);
    let (t1b_logits, ck1b) = run_single_prefill(&mut turn1_b64, &prompt1, 0, N_BATCH as usize, Some(p as u32));
    let ck1b = ck1b.unwrap();
    let _ = run_single_turn1_gen(&mut turn1_b64, &prompt1, &t1b_logits, 12);
    turn1_b64.restore_checkpoint(0, p as u32, &ck1b).unwrap();
    let (warm_e, _) = run_single_prefill(&mut turn1_b64, &prompt2, p, N_BATCH as usize, Some(p2));
    eprintln!(
        "[diag] single warm_e (post-turn1 restore, block=64): argmax={} max_diff_vs_a={:.6e} max_diff_vs_b={:.6e} top5={:?}",
        sampling::argmax(&warm_e),
        max_abs_diff(&warm_e, &cold_a),
        max_abs_diff(&warm_e, &cold_b),
        top_logits(&warm_e, 5)
    );

    let mut empty_prefix_eng = load_single_engine(&model, 0, -1, &cache, threads, ngl);
    let (ctrl_f, _) = run_single_prefill(&mut empty_prefix_eng, &prompt2, p, p, Some(p2));
    eprintln!(
        "[diag] control_f (no prefix KV or recurrent state, suffix from pos p): argmax={} max_diff_vs_a={:.6e}",
        sampling::argmax(&ctrl_f),
        max_abs_diff(&ctrl_f, &cold_a)
    );

    let mut rec_only_eng = load_single_engine(&model, 0, -1, &cache, threads, ngl);
    rec_only_eng.restore_checkpoint(0, p as u32, &ck1).unwrap();
    let (ctrl_g, _) = run_single_prefill(&mut rec_only_eng, &prompt2, p, p, Some(p2));
    eprintln!(
        "[diag] control_g (recurrent checkpoint restored, no attention KV for [0..p)): argmax={} max_diff_vs_a={:.6e}",
        sampling::argmax(&ctrl_g),
        max_abs_diff(&ctrl_g, &cold_a)
    );

    // Repeat with two in-process stages to separate stage boundaries from the wire protocol.
    let cut = (layers / 2) as i32;
    let mut split_cold_h = load_single_engine(&model, 0, cut, &cache, threads, ngl);
    let mut split_cold_t = load_single_engine(&model, cut, -1, &cache, threads, ngl);
    let (split_cold, cold_acts, _) =
        run_two_stage_prefill(&mut split_cold_h, &mut split_cold_t, &prompt2, 0, p, Some(p2));

    let mut split_warm_h = load_single_engine(&model, 0, cut, &cache, threads, ngl);
    let mut split_warm_t = load_single_engine(&model, cut, -1, &cache, threads, ngl);
    let (st1_logits, _, split_ck) =
        run_two_stage_prefill(&mut split_warm_h, &mut split_warm_t, &prompt1, 0, p, Some(p as u32));
    let (ck_h, ck_t) = split_ck.expect("two-stage checkpoint missing");
    let mut next = sampling::argmax(&st1_logits) as i32;
    for step in 0..12usize {
        if split_warm_h.e_eog(next) || step + 1 == 12 {
            break;
        }
        let pos = (prompt1.len() + step) as i32;
        let h_state = split_warm_h
            .run(&[Row { token: next, pos, seq: 0, logits: true }], None)
            .unwrap()
            .unwrap();
        split_warm_t
            .run(&[Row { token: 0, pos, seq: 0, logits: true }], Some(&h_state))
            .unwrap();
        next = sampling::argmax(split_warm_t.logits(0).unwrap()) as i32;
    }
    split_warm_h.restore_checkpoint(0, p as u32, &ck_h).unwrap();
    split_warm_t.restore_checkpoint(0, p as u32, &ck_t).unwrap();
    let ck_h_after = split_warm_h.save_checkpoint(0).unwrap();
    let ck_t_after = split_warm_t.save_checkpoint(0).unwrap();
    let (split_warm, warm_acts, _) =
        run_two_stage_prefill(&mut split_warm_h, &mut split_warm_t, &prompt2, p, p, Some(p2));
    eprintln!(
        "[diag] two-stage [0,{cut})+[{cut},{layers}): ck_h_eq={} ck_t_eq={} cold_argmax={} warm_argmax={} max_diff_logits={:.6e}",
        ck_h.as_slice() == ck_h_after.as_slice(),
        ck_t.as_slice() == ck_t_after.as_slice(),
        sampling::argmax(&split_cold),
        sampling::argmax(&split_warm),
        max_abs_diff(&split_warm, &split_cold)
    );
    for (start, end, warm_state) in &warm_acts {
        if let Some((_, _, cold_state)) = cold_acts.iter().find(|(cs, ce, _)| cs == start && ce == end) {
            eprintln!(
                "[diag]   head->tail activation block [{start}..{end}): max_diff={:.6e}",
                max_abs_diff(warm_state, cold_state)
            );
        }
    }

    assert_eq!(
        sampling::argmax(&warm_d),
        sampling::argmax(&cold_a),
        "single-engine post-turn1 restore diverged at step 0"
    );
    assert_eq!(
        sampling::argmax(&split_warm),
        sampling::argmax(&split_cold),
        "in-process two-stage restore diverged at step 0"
    );
}

#[test]
#[cfg_attr(not(qwen_parity_fixture), ignore = "requires PUBLICOMPUTE_Q35_GGUF")]
fn new_member_admission_releases_grace_on_real_head() {
    let model = std::env::var("PUBLICOMPUTE_Q35_GGUF").expect("set PUBLICOMPUTE_Q35_GGUF to a Qwen fixture");
    let _serial = NATIVE_CHAIN.lock().unwrap_or_else(|e| e.into_inner());
    let profile = Profile {
        model: &model, stages: 2, mtp: false, cache: "f16",
        threads: std::env::var("PUBLICOMPUTE_TEST_THREADS").map_or(2, |v| v.parse().unwrap()),
        ngl: std::env::var("PUBLICOMPUTE_TEST_NGL").map_or(0, |v| v.parse().unwrap()),
    };
    // Leave enough grace for CPU fixtures so the old slot cannot simply age out.
    let mut chain = Chain::start_with_slots(profile, 900, N_BATCH as usize, 2, 600);
    let first = chain.chat_for_member(1, 1, 7, vec![message("user", QUESTION)]).slot_report.unwrap();
    let second = chain.chat_for_member(2, 2, 7, vec![message("user", FOLLOW_UP)]).slot_report.unwrap();
    assert!(first.parked && second.parked);
    assert_ne!(first.slot, second.slot);
    assert!(second.generation > first.generation);
    let public = chain.chat(3, 3, vec![message("user", QUESTION)]);
    assert_eq!(public.cached, 0);
    assert_eq!(public.slot_report.unwrap().slot, first.slot);
}

macro_rules! case {
    ($name:ident, $stages:literal, $cache:literal, $mtp:literal, $fixture:ident, $reason:literal) => {
        #[test]
        #[cfg_attr(not($fixture), ignore = $reason)]
        fn $name() {
            parity($stages, $cache, $mtp);
        }
    };
}

case!(two_stages_f16, 2, "f16", false, qwen_parity_fixture, "requires PUBLICOMPUTE_Q35_GGUF");
case!(three_stages_f16, 3, "f16", false, qwen_parity_fixture, "requires PUBLICOMPUTE_Q35_GGUF");
case!(two_stages_q8, 2, "q8_0", false, qwen_parity_fixture, "requires PUBLICOMPUTE_Q35_GGUF");
case!(three_stages_q8, 3, "q8_0", false, qwen_parity_fixture, "requires PUBLICOMPUTE_Q35_GGUF");
case!(two_stages_f16_mtp, 2, "f16", true, qwen_mtp_parity_fixture, "requires PUBLICOMPUTE_Q35_MTP_GGUF");
case!(three_stages_f16_mtp, 3, "f16", true, qwen_mtp_parity_fixture, "requires PUBLICOMPUTE_Q35_MTP_GGUF");
case!(two_stages_q8_mtp, 2, "q8_0", true, qwen_mtp_parity_fixture, "requires PUBLICOMPUTE_Q35_MTP_GGUF");
case!(three_stages_q8_mtp, 3, "q8_0", true, qwen_mtp_parity_fixture, "requires PUBLICOMPUTE_Q35_MTP_GGUF");
