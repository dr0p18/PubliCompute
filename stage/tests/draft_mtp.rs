//! Exercise the tail engine's draft context on small weight fixtures.
//! Point PUBLICOMPUTE_DS4_MTP_GGUF and PUBLICOMPUTE_Q35_MTP_GGUF to fixtures
//! with a draft block, and PUBLICOMPUTE_DS4_GGUF and PUBLICOMPUTE_Q35_GGUF to
//! their counterparts without it. Cases skip when required files are absent.
#![cfg(feature = "llama")]

use publicompute_stage::sampling::argmax;
use publicompute_stage::engine::{Engine, EngineOptions, Row, MtpRow};

fn file(var: &str) -> Option<String> {
    let path = std::env::var(var).ok()?;
    std::path::Path::new(&path).is_file().then_some(path)
}

/// A miniature with its MTP block, the same without it, and the cuts of a head, an intermediate and a tail stage.
struct Mini {
    name: &'static str,
    mtp: Option<String>,
    without_mtp: Option<String>,
    head: (i32, i32),
    middle: (i32, i32),
    tail: i32,
}

fn miniature() -> Vec<Mini> {
    vec![
        // Keep the hash layers in the first stage for this fixture.
        Mini {
            name: "deepseek4",
            mtp: file("PUBLICOMPUTE_DS4_MTP_GGUF"),
            without_mtp: file("PUBLICOMPUTE_DS4_GGUF"),
            head: (0, 3),
            middle: (3, 5),
            tail: 5,
        },
        Mini {
            name: "qwen35moe",
            mtp: file("PUBLICOMPUTE_Q35_MTP_GGUF"),
            without_mtp: file("PUBLICOMPUTE_Q35_GGUF"),
            head: (0, 3),
            middle: (3, 5),
            tail: 5,
        },
    ]
}

/// The miniatures with the MTP block on disk; the missing ones are reported as skipped.
fn con_mtp() -> Vec<Mini> {
    miniature()
        .into_iter()
        .filter(|m| {
            if m.mtp.is_none() {
                eprintln!("{}: test model with MTP missing; skipping", m.name);
            }
            m.mtp.is_some()
        })
        .collect()
}

fn options(p: &str, layer_start: i32, layer_end: i32, mtp: bool) -> EngineOptions {
    let threads = std::env::var("PUBLICOMPUTE_TEST_THREADS").ok().and_then(|x| x.parse().ok()).unwrap_or(4);
    EngineOptions { model: p.into(), layer_start, layer_end, ngl: 0, threads, slots: 1, groups: 1, ctx_per_slot: 256, n_batch: 64, experts: false, max_draft_tokens: 1, mtp }
}

const N: usize = 20;

fn tok(m: &Engine, i: usize) -> i32 {
    ((i * 37 + 11) % (m.n_vocab.min(1000) - 1) + 1) as i32
}

/// Scale maximum absolute error by the largest absolute reference logit.
fn errore_relativo(a: &[f32], b: &[f32]) -> f64 {
    let d = a.iter().zip(b).map(|(x, y)| (*x as f64 - *y as f64).abs()).fold(0.0, f64::max);
    let m = b.iter().map(|y| (*y as f64).abs()).fold(0.0, f64::max);
    d / m.max(f64::MIN_POSITIVE)
}

/// Run the whole layer range and prefill the draft context with the prompt
/// and first sample. Return that sample and the logits used to extend it.
fn prefill(m: &mut Engine) -> (i32, Vec<f32>) {
    let n_h = m.mtp_state_width;
    let prompt: Vec<i32> = (0..N).map(|i| tok(m, i)).collect();
    let rows: Vec<Row> = prompt.iter().enumerate().map(|(i, &t)| Row { token: t, pos: i as i32, seq: 0, logits: i + 1 == N }).collect();
    m.run(&rows, None).unwrap();
    let t0 = argmax(m.logits(N - 1).unwrap()) as i32;
    // Each token takes the preceding row's state. The first token uses zeros.
    let mut stati = vec![0f32; n_h];
    for i in 0..N {
        stati.extend_from_slice(m.mtp_state(i).unwrap());
    }
    let mut rm: Vec<MtpRow> = prompt.iter().enumerate().map(|(i, &t)| MtpRow { token: t, pos: i as i32, seq: 0, logits: false }).collect();
    rm.push(MtpRow { token: t0, pos: N as i32, seq: 0, logits: true });
    m.run_mtp(&rm, &stati).unwrap();
    (t0, m.mtp_logits(N).unwrap().to_vec())
}

/// One trunk step of one row: the sampled token (greedy) and the hidden state of the row.
fn step(m: &mut Engine, token: i32, pos: usize) -> (i32, Vec<f32>) {
    m.run(&[Row { token, pos: pos as i32, seq: 0, logits: true }], None).unwrap();
    (argmax(m.logits(0).unwrap()) as i32, m.mtp_state(0).unwrap().to_vec())
}

/// One MTP row with logits.
fn mtp(m: &mut Engine, token: i32, pos: usize, h: &[f32]) -> Vec<f32> {
    m.run_mtp(&[MtpRow { token, pos: pos as i32, seq: 0, logits: true }], h).unwrap();
    m.mtp_logits(0).unwrap().to_vec()
}

/// Compare the cache after dropping a rejected draft with one that never
/// received it. Keep the main batches identical to isolate draft state.
/// A control leaves the rejected row in place to confirm the logits can
/// distinguish the two states. A separate measurement also varies the
/// main batch shape, as speculative generation does.
#[test]
fn mtp_cache_excludes_rejected_row() {
    for m in con_mtp() {
        println!("{}", m.name);
        cache_dopo_un_rifiuto(m.mtp.as_deref().unwrap());
    }
}

fn cache_dopo_un_rifiuto(p: &str) {
    // Reference context contains only verified tokens.
    let mut b = Engine::load(&options(p, 0, -1, true)).unwrap();
    assert!(b.ha_mtp());
    let (t0, lb0) = prefill(&mut b);
    let (s0, h_n) = step(&mut b, t0, N);
    let lb1 = mtp(&mut b, s0, N + 1, &h_n);
    let (s1, h_n1) = step(&mut b, s0, N + 1);
    let lb2 = mtp(&mut b, s1, N + 2, &h_n1);
    drop(b);

    // Insert a rejected token and truncate its position before appending the verified row.
    let mut a = Engine::load(&options(p, 0, -1, true)).unwrap();
    let (t0a, la0) = prefill(&mut a);
    assert_eq!(t0a, t0);
    let (s0a, h_na) = step(&mut a, t0, N);
    assert_eq!(s0a, s0);
    let sbagliato = (s0 + 1) % a.n_vocab.min(1000) as i32;
    let l_sbagliata = mtp(&mut a, sbagliato, N + 1, &h_na);
    a.mtp_riavvolgi(0, (N + 1) as u32).unwrap();
    let la1 = mtp(&mut a, s0, N + 1, &h_na);
    let (s1a, h_n1a) = step(&mut a, s0, N + 1);
    assert_eq!(s1a, s1);
    let la2 = mtp(&mut a, s1, N + 2, &h_n1a);
    drop(a);

    // Leave the rejected token in place to show the comparison detects a stale entry.
    let mut c = Engine::load(&options(p, 0, -1, true)).unwrap();
    let (t0c, _) = prefill(&mut c);
    let (s0c, h_nc) = step(&mut c, t0c, N);
    mtp(&mut c, sbagliato, N + 1, &h_nc);
    let (s1c, h_n1c) = step(&mut c, s0c, N + 1);
    let lc2 = mtp(&mut c, s1c, N + 2, &h_n1c);
    drop(c);

    let (e0, e1, e2) = (errore_relativo(&la0, &lb0), errore_relativo(&la1, &lb1), errore_relativo(&la2, &lb2));
    let controllo = errore_relativo(&lc2, &lb2);
    println!("MTP logit relative error after rewinding a rejected row:");
    println!("first token before rejection: {e0:e}");
    println!("next token after rejection (position {}): {e1:e}; draft {} vs {}", N + 1, argmax(&la1), argmax(&lb1));
    println!("token successivo (posizione {}): {e2:e}  bozza {} / {}", N + 2, argmax(&la2), argmax(&lb2));
    println!("control with rejected row left in cache: next-token relative error {controllo:e}");
    println!("control: the rejected row's logits differ from the accepted row: {:e}", errore_relativo(&l_sbagliata, &lb1));
    assert!(e0 <= 1e-3 && e1 <= 1e-3 && e2 <= 1e-3, "MTP cache after a rejection differs from the run without the rejected row: {e0:e} {e1:e} {e2:e}");
    assert!(controllo > 1e-3, "the control must tell a cache with a wrong row from a clean one ({controllo:e})");

    // Include main-batch rounding by decoding and rewinding the speculative row there as well.
    let mut a2 = Engine::load(&options(p, 0, -1, true)).unwrap();
    let (t0d, _) = prefill(&mut a2);
    a2.run(&[Row { token: t0d, pos: N as i32, seq: 0, logits: true }, Row { token: sbagliato, pos: (N + 1) as i32, seq: 0, logits: true }], None).unwrap();
    let s0d = argmax(a2.logits(0).unwrap()) as i32;
    let h_nd = a2.mtp_state(0).unwrap().to_vec();
    let ld1 = mtp(&mut a2, s0d, N + 1, &h_nd);
    a2.riavvolgi(0, (N + 1) as u32).unwrap();
    let (s1d, h_n1d) = step(&mut a2, s0d, N + 1);
    let ld2 = mtp(&mut a2, s1d, N + 2, &h_n1d);
    println!(
        "Rejected row also decoded by the trunk (two rows then rewind): tokens {} / {}, relative error {:e} (position {}) and {:e} (position {})",
        if s0d == s0 { "match" } else { "MISMATCH" },
        if s1d == s1 { "match" } else { "MISMATCH" },
        errore_relativo(&ld1, &lb1),
        N + 1,
        errore_relativo(&ld2, &lb2),
        N + 2
    );
}

/// Loading the MTP block does not change the trunk: same logits, bit for bit, with and without `mtp`, on the
/// miniature with the MTP block (prefill and three steps).
#[test]
fn loading_mtp_does_not_change_main_model() {
    for m in con_mtp() {
        println!("{}", m.name);
        tronco_invariato(m.mtp.as_deref().unwrap());
    }
}

fn tronco_invariato(p: &str) {
    let corsa = |mtp: bool| -> Vec<Vec<f32>> {
        let mut m = Engine::load(&options(p, 0, -1, mtp)).unwrap();
        assert_eq!(m.ha_mtp(), mtp);
        let rows: Vec<Row> = (0..N).map(|i| Row { token: tok(&m, i), pos: i as i32, seq: 0, logits: i + 1 == N }).collect();
        m.run(&rows, None).unwrap();
        let mut out = vec![m.logits(N - 1).unwrap().to_vec()];
        for i in 0..3 {
            let t = argmax(out.last().unwrap()) as i32;
            m.run(&[Row { token: t, pos: (N + i) as i32, seq: 0, logits: true }], None).unwrap();
            out.push(m.logits(0).unwrap().to_vec());
        }
        out
    };
    let (a, b) = (corsa(false), corsa(true));
    let scarto = a.iter().zip(&b).flat_map(|(x, y)| x.iter().zip(y)).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max);
    println!("Trunk with and without MTP: largest logit difference {scarto:e} across {} rows", a.len());
    assert!(a == b, "the trunk logits must be bit-identical with and without the MTP block loaded");
}

/// `mtp` is refused, with an error, by a stage that is not the tail and by a file without MTP block; the tail stage of
/// the file with the block loads it.
#[test]
fn mtp_loads_only_at_tail_with_draft_weights() {
    for m in con_mtp() {
        let p = m.mtp.as_deref().unwrap();
        let (n, (t0, t1), (i0, i1), c) = (m.name, m.head, m.middle, m.tail);
        let e = Engine::load(&options(p, t0, t1, true)).err().expect("a head stage must not load the MTP block");
        println!("{n} head [{t0},{t1}) with MTP: {e}");
        assert!(e.contains("--mtp"), "{e}");
        let e = Engine::load(&options(p, i0, i1, true)).err().expect("an intermediate stage must not load the MTP block");
        println!("{n} middle stage [{i0},{i1}) with MTP: {e}");
        let tail = Engine::load(&options(p, c, -1, true)).unwrap();
        assert!(tail.ha_mtp() && tail.tail && !tail.head);
        assert_eq!(tail.mtp_state_width, tail.n_embd, "{n}: the MTP state is as wide as the state between the stages");
        println!("{n} tail [{c},n) loaded with MTP; draft state has {} values", tail.mtp_state_width);
        drop(tail);
        match &m.without_mtp {
            Some(q) => {
                let e = Engine::load(&options(q, c, -1, true)).err().expect("a file without MTP block must be refused");
                println!("{n} tail requested MTP from a file without draft weights: {e}");
                assert!(e.contains("--mtp"), "{e}");
                assert!(!Engine::load(&options(q, c, -1, false)).unwrap().ha_mtp());
            }
            None => eprintln!("{n}: test model without MTP missing; skipping"),
        }
    }
}

/// Multiple draft steps require the engine to expose output state from the draft block.
fn opzioni_k(p: &str, k: u32) -> EngineOptions {
    EngineOptions { max_draft_tokens: k, ..options(p, 0, -1, true) }
}

/// Run a draft chain from a verified row. Each later step uses the
/// previous proposal and its output state. Return logits at every depth.
fn catena(m: &mut Engine, token: i32, pos: usize, h: &[f32], k: usize) -> Vec<Vec<f32>> {
    let mut out = vec![mtp(m, token, pos, h)];
    for prof in 2..=k {
        let d = argmax(out.last().unwrap()) as i32;
        let state = m.mtp_stato_uscita(0).unwrap().to_vec();
        assert!(state.iter().all(|x| x.is_finite()) && state.iter().any(|&x| x != 0.0), "state of the MTP block empty or not finite");
        out.push(mtp(m, d, pos + prof - 1, &state));
    }
    out
}

/// Compare every proposal depth with a fresh context after dropping the
/// tentative rows from the previous step. Repeat the rewind across three
/// steps. A control using main state at depth two tests that it differs
/// from chaining through the draft block's own state.
#[test]
fn mtp_rewind_matches_fresh_state_at_every_depth() {
    for m in con_mtp() {
        println!("{}", m.name);
        catena_dopo_riavvolgimenti(m.mtp.as_deref().unwrap());
    }
}

fn catena_dopo_riavvolgimenti(p: &str) {
    const K: usize = 3;
    const PASSI: usize = 3;
    // Reuse one context across steps and truncate its tentative suffix before each update.
    let mut a = Engine::load(&opzioni_k(p, K as u32)).unwrap();
    let (mut tok_in, _) = prefill(&mut a);
    let mut verificati: Vec<(i32, Vec<f32>)> = Vec::new(); 
    let mut logit_a: Vec<Vec<Vec<f32>>> = Vec::new();
    for passo_i in 0..PASSI {
        let pos = N + passo_i;
        let (s, h) = step(&mut a, tok_in, pos);
        if passo_i > 0 {
            a.mtp_riavvolgi(0, (pos + 1) as u32).unwrap();
        }
        logit_a.push(catena(&mut a, s, pos + 1, &h, K));
        verificati.push((s, h));
        tok_in = s;
    }
    drop(a);
    // Rebuild a reference from verified rows before extending each step.
    for passo_i in 0..PASSI {
        let mut b = Engine::load(&opzioni_k(p, K as u32)).unwrap();
        let (mut t, _) = prefill(&mut b);
        let mut last = Vec::new();
        for j in 0..=passo_i {
            let pos = N + j;
            let (s, h) = step(&mut b, t, pos);
            assert!(s == verificati[j].0 && h == verificati[j].1, "the trunk must not depend on the MTP cache");
            if j < passo_i {
                mtp(&mut b, s, pos + 1, &h);
            } else {
                last = catena(&mut b, s, pos + 1, &h, K);
            }
            t = s;
        }
        for prof in 0..K {
            let e = errore_relativo(&logit_a[passo_i][prof], &last[prof]);
            println!("step {}, depth {}: relative error {e:e} (draft {} vs {})", passo_i + 1, prof + 1, argmax(&logit_a[passo_i][prof]), argmax(&last[prof]));
            assert!(e <= 1e-3, "step {} depth {}: MTP logits after the rewind of the trial rows differ from a cache rebuilt from scratch: {e:e}", passo_i + 1, prof + 1);
        }
        if passo_i == 0 {
            // Compare the second proposal against one computed with the main state.
            let d1 = argmax(&last[0]) as i32;
            let mut c = Engine::load(&opzioni_k(p, K as u32)).unwrap();
            let (t0, _) = prefill(&mut c);
            let (s0, h0) = step(&mut c, t0, N);
            mtp(&mut c, s0, N + 1, &h0);
            let con_tronco = mtp(&mut c, d1, N + 2, &h0);
            println!("Second draft fed with trunk state instead of MTP state: relative error {:e}", errore_relativo(&con_tronco, &last[1]));
        }
    }
}

/// A single-step draft does not allocate output state for chaining.
#[test]
fn mtp_output_state_supports_multiple_draft_tokens() {
    for m in con_mtp() {
        let p = m.mtp.as_deref().unwrap();
        let mut a = Engine::load(&options(p, 0, -1, true)).unwrap();
        let (t0, _) = prefill(&mut a);
        let (s, h) = step(&mut a, t0, N);
        mtp(&mut a, s, N + 1, &h);
        let e = a.mtp_stato_uscita(0).unwrap_err();
        println!("{} with K = 1: {e}", m.name);
        assert!(e.contains("--draft-max-tokens"), "{e}");
    }
}
