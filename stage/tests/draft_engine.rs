//! Exercise draft rewind with a small CPU-loaded weight fixture. Set
//! PUBLICOMPUTE_MINI_GGUF to its path. The tests skip a missing fixture.
#![cfg(feature = "llama")]

use publicompute_stage::engine::{Engine, EngineOptions, Row};

fn percorso() -> Option<String> {
    let p = std::env::var("PUBLICOMPUTE_MINI_GGUF").ok()?;
    std::path::Path::new(&p).exists().then_some(p)
}

fn load(p: &str, k: u32) -> Engine {
    Engine::load(&EngineOptions { model: p.into(), layer_start: 0, layer_end: -1, ngl: 0, threads: std::env::var("PUBLICOMPUTE_TEST_THREADS").ok().and_then(|x| x.parse().ok()).unwrap_or(2), slots: 1, groups: 1, ctx_per_slot: 256, n_batch: 64, experts: false, max_draft_tokens: k, mtp: false, cache: Default::default() }).unwrap()
}

fn prompt() -> Vec<i32> {
    (0..20).map(|i| (i * 37 + 11) % 1000 + 1).collect()
}

fn prefill(m: &mut Engine) {
    let p = prompt();
    let rows: Vec<Row> = p.iter().enumerate().map(|(i, &t)| Row { token: t, pos: i as i32, seq: 0, logits: i + 1 == p.len() }).collect();
    m.run(&rows, None).unwrap();
}

/// Collect one vocabulary row per single-token decode.
fn uno_alla_volta(m: &mut Engine, toks: &[i32], da: i32) -> Vec<Vec<f32>> {
    toks.iter()
        .enumerate()
        .map(|(i, &t)| {
            m.run(&[Row { token: t, pos: da + i as i32, seq: 0, logits: true }], None).unwrap();
            m.logits(0).unwrap().to_vec()
        })
        .collect()
}

fn scarto(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

fn norma(a: &[f32]) -> f32 {
    a.iter().map(|x| x.abs()).fold(0.0, f32::max)
}

/// After truncating the draft suffix, decode one replacement token and
/// compare with an engine that saw only the retained prefix.
#[test]
fn rewind_matches_state_without_draft() {
    let Some(p) = percorso() else { return eprintln!("test model missing; skipping") };
    let mut m = load(&p, 4);
    let t = [101, 202, 303, 404];
    prefill(&mut m);
    let c = uno_alla_volta(&mut m, &[t[0], 777, 888], 20);
    m.svuota_posto(0);
    prefill(&mut m);
    let rows: Vec<Row> = t.iter().enumerate().map(|(i, &x)| Row { token: x, pos: 20 + i as i32, seq: 0, logits: true }).collect();
    m.run(&rows, None).unwrap();
    m.riavvolgi(0, 21).unwrap();
    let b = uno_alla_volta(&mut m, &[777, 888], 21);
    println!("largest difference after rewind: {:e} (logits up to {:e})", scarto(&c[1], &b[0]).max(scarto(&c[2], &b[1])), norma(&c[1]));
    assert!(scarto(&c[1], &b[0]) < 1e-4, "token logits at position 21 after rewind");
    assert!(scarto(&c[2], &b[1]) < 1e-4, "token logits at position 22 after rewind");
}

/// Measure rounding differences between a four-row batch and four
/// single-row calls. A flat distribution makes near ties sensitive to
/// small changes even when the prompt and history agree.
#[test]
fn logit_difference_between_batch_and_single_steps() {
    let Some(p) = percorso() else { return eprintln!("test model missing; skipping") };
    let mut m = load(&p, 4);
    let t = [101, 202, 303, 404];
    prefill(&mut m);
    let a = uno_alla_volta(&mut m, &t, 20);
    m.svuota_posto(0);
    prefill(&mut m);
    let rows: Vec<Row> = t.iter().enumerate().map(|(i, &x)| Row { token: x, pos: 20 + i as i32, seq: 0, logits: true }).collect();
    m.run(&rows, None).unwrap();
    for i in 0..4 {
        println!("row {i}: largest logit difference between batched and single-row decode {:e} (logits up to {:e})", scarto(&a[i], m.logits(i).unwrap()), norma(&a[i]));
    }
}
