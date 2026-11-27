//! Check recurrent state rewind with a small weight fixture. The correctness
//! check compares equal batch shapes. The diagnostic cases vary batch shape
//! and snapshot capacity to isolate their numerical and memory effects.
//! Set PUBLICOMPUTE_DS4_GGUF to the fixture path. A missing file skips the cases.
#![cfg(feature = "llama")]

use publicompute_stage::engine::{Engine, EngineOptions, Row};

fn percorso() -> Option<String> {
    let p = std::env::var("PUBLICOMPUTE_DS4_GGUF").ok()?;
    std::path::Path::new(&p).exists().then_some(p)
}

fn load(p: &str, k: u32) -> Engine {
    Engine::load(&EngineOptions { model: p.into(), layer_start: 0, layer_end: -1, ngl: 0, threads: std::env::var("PUBLICOMPUTE_TEST_THREADS").ok().and_then(|x| x.parse().ok()).unwrap_or(4), slots: 1, groups: 1, ctx_per_slot: 256, n_batch: npr().max(64) as u32, experts: false, max_draft_tokens: k, mtp: false, cache: Default::default() }).unwrap()
}

/// Choose a prompt long enough to fill a compressed attention block.
/// PUBLICOMPUTE_DS4_PROMPT_TOKENS overrides its length.
fn npr() -> usize {
    std::env::var("PUBLICOMPUTE_DS4_PROMPT_TOKENS").ok().and_then(|x| x.parse().ok()).unwrap_or(20)
}

fn tok(m: &Engine, i: usize) -> i32 {
    ((i * 37 + 11) % (m.n_vocab.min(1000) - 1) + 1) as i32
}

fn prompt(m: &Engine) -> Vec<i32> {
    (0..npr()).map(|i| tok(m, i)).collect()
}

fn prefill(m: &mut Engine) {
    let p = prompt(m);
    let rows: Vec<Row> = p.iter().enumerate().map(|(i, &t)| Row { token: t, pos: i as i32, seq: 0, logits: i + 1 == p.len() }).collect();
    m.run(&rows, None).unwrap();
}

fn uno_alla_volta(m: &mut Engine, toks: &[i32], da: i32) -> Vec<Vec<f32>> {
    toks.iter()
        .enumerate()
        .map(|(i, &t)| {
            m.run(&[Row { token: t, pos: da + i as i32, seq: 0, logits: true }], None).unwrap();
            m.logits(0).unwrap().to_vec()
        })
        .collect()
}

fn lotto(m: &mut Engine, toks: &[i32], da: i32) -> Vec<Vec<f32>> {
    let rows: Vec<Row> = toks.iter().enumerate().map(|(i, &x)| Row { token: x, pos: da + i as i32, seq: 0, logits: true }).collect();
    m.run(&rows, None).unwrap();
    (0..toks.len()).map(|i| m.logits(i).unwrap().to_vec()).collect()
}

fn scarto(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

fn argmax(a: &[f32]) -> usize {
    a.iter().enumerate().fold((0, f32::MIN), |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) }).0
}

/// A change in the leading token requires a perturbation large enough to
/// close the gap between the two largest logits.
fn margine(a: &[f32]) -> f32 {
    let mut v = a.to_vec();
    v.sort_by(|x, y| y.partial_cmp(x).unwrap());
    v[0] - v[1]
}

fn row(et: &str, a: &[Vec<f32>], b: &[Vec<f32>]) {
    for i in 0..a.len().min(b.len()) {
        println!(
            "{et} row {i}: largest difference {:e}, argmax {} (margin {:e}), logits up to {:e}",
            scarto(&a[i], &b[i]),
            if argmax(&a[i]) == argmax(&b[i]) { "match" } else { "MISMATCH" },
            margine(&a[i]),
            a[i].iter().map(|x| x.abs()).fold(0.0, f32::max)
        );
    }
}

const T: [i32; 4] = [101, 202, 303, 404];

/// Compare several rows decoded together with the same rows decoded separately.
#[test]
fn diagnose_batch_against_single_steps() {
    let Some(p) = percorso() else { return eprintln!("test model missing; skipping") };
    let mut m = load(&p, 4);
    prefill(&mut m);
    let a = uno_alla_volta(&mut m, &T, npr() as i32);
    m.svuota_posto(0);
    prefill(&mut m);
    let b = lotto(&mut m, &T, npr() as i32);
    row("four rows in one batch vs four separate calls (K=4)", &a, &b);
    // Isolate the cost of reserving recurrent snapshots with otherwise equal steps.
    let mut m0 = load(&p, 0);
    prefill(&mut m0);
    let a0 = uno_alla_volta(&mut m0, &T, npr() as i32);
    row("Reference: singoli K=0 contro singoli K=4", &a0, &a);
    // Split prefill to check whether its shape changes later logits.
    let mut m1 = load(&p, 4);
    let pr = prompt(&m1);
    let r1: Vec<Row> = pr[..npr() / 2].iter().enumerate().map(|(i, &t)| Row { token: t, pos: i as i32, seq: 0, logits: false }).collect();
    m1.run(&r1, None).unwrap();
    let r2: Vec<Row> = pr[npr() / 2..].iter().enumerate().map(|(i, &t)| Row { token: t, pos: (npr() / 2 + i) as i32, seq: 0, logits: i + 1 == npr() - npr() / 2 }).collect();
    m1.run(&r2, None).unwrap();
    let a1 = uno_alla_volta(&mut m1, &T, npr() as i32);
    row("Reference: prefill 10+10 contro 20 (poi singoli)", &a, &a1);
}

/// Use drafts of equal shape with a common first row and different suffixes.
/// Rewind the suffix and compare two following tokens to check that discarded
/// inputs have no effect. A separate run computes the first row alone to
/// show how much its batch shape influences the comparison.
#[test]
fn diagnose_rewind_at_same_batch_shape() {
    let Some(p) = percorso() else { return eprintln!("test model missing; skipping") };
    let u = [777, 888];
    let mut m = load(&p, 4);
    prefill(&mut m);
    lotto(&mut m, &T, 20);
    m.riavvolgi(0, 21).unwrap();
    let c = uno_alla_volta(&mut m, &u, 21);
    m.svuota_posto(0);
    prefill(&mut m);
    lotto(&mut m, &[T[0], 555, 666, 999], 20);
    m.riavvolgi(0, 21).unwrap();
    let e = uno_alla_volta(&mut m, &u, 21);
    m.svuota_posto(0);
    prefill(&mut m);
    lotto(&mut m, &T[..1], 20);
    let d = uno_alla_volta(&mut m, &u, 21);
    row("draft and rewind (C) vs a different draft with the same batch shape (E)", &c, &e);
    row("draft and rewind (C) vs a one-row batch without drafts (D)", &c, &d);
    assert!(scarto(&c[0], &e[0]) < 1e-4 && scarto(&c[1], &e[1]) < 1e-4, "rewind with identical batch shape must reproduce the state");
}

/// Report buffer and resident memory with zero and four rewind snapshots.
#[test]
fn context_memory_with_and_without_draft() {
    let Some(p) = percorso() else { return eprintln!("test model missing; skipping") };
    publicompute_stage::engine::set_llama_logging(true);
    let rss = || -> u64 { std::fs::read_to_string("/proc/self/status").unwrap().lines().find(|l| l.starts_with("VmRSS")).and_then(|l| l.split_whitespace().nth(1)).and_then(|x| x.parse().ok()).unwrap_or(0) };
    for k in [0u32, 4] {
        let before = rss();
        println!("MEM ---- k={k}");
        let mut m = Engine::load(&EngineOptions { model: p.clone(), layer_start: 0, layer_end: -1, ngl: 0, threads: 4, slots: 4, groups: 1, ctx_per_slot: 256, n_batch: 64, experts: false, max_draft_tokens: k, mtp: false, cache: Default::default() }).unwrap();
        prefill(&mut m);
        println!("MEM k={k}: RSS after load and prefill {} KiB (before {} KiB)", rss(), before);
    }
}
