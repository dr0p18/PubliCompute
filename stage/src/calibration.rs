//! Estimate throughput for each number of in-flight groups. Split active
//! conversations as evenly as possible. A stage has a fixed launch cost
//! plus a cost per row, and a link adds fixed delay plus transfer time.
//! The cycle must fit both the slowest complete group round and the total
//! bottleneck compute time across groups. Total throughput is active
//! conversations divided by cycle time.
//!
//! Prefer timings from each stage because the tail also computes logits
//! and samples. Fit fixed and per-row cost across runs. When stage timings
//! are missing, approximate their costs from reference runs and layer shares.
//!
//! Recommend the smallest group count within five percent of peak estimated
//! throughput among those meeting the per-conversation minimum. If none
//! meets it, choose the fastest per-conversation rate and flag the shortfall.

use serde_json::{json, Value};
use std::collections::HashMap;
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Stage {
    pub a_ms: f64,
    pub k_ms: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Salto {
    pub fisso_ms: f64,
    pub per_riga_ms: f64,
}

impl Salto {
    /// Half RTT plus transfer: `byte_per_riga / banda` per row.
    pub fn da_rete(rtt_ms: f64, bytes_per_row: f64, banda_mbit: f64) -> Salto {
        let per_riga = if banda_mbit > 0.0 { bytes_per_row * 8.0 / (banda_mbit * 1e6) * 1000.0 } else { 0.0 };
        Salto { fisso_ms: rtt_ms / 2.0, per_riga_ms: per_riga }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Param {
    pub stages: Vec<Stage>,
    pub salti: Vec<Salto>,
    pub o_ms: f64,
}

impl Param {
    pub fn t_stadio(&self, i: usize, b: f64) -> f64 {
        self.stages[i].a_ms + self.stages[i].k_ms * b
    }
    pub fn giro(&self, b: f64) -> f64 {
        let c: f64 = (0..self.stages.len()).map(|i| self.t_stadio(i, b)).sum();
        let r: f64 = self.salti.iter().map(|s| s.fisso_ms + s.per_riga_ms * b).sum();
        c + r + self.o_ms
    }
    pub fn t_max(&self, b: f64) -> f64 {
        (0..self.stages.len()).map(|i| self.t_stadio(i, b)).fold(0.0, f64::max)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Stima {
    pub g: usize,
    pub b: usize,
    pub giro_ms: f64,
    pub t_max_ms: f64,
    pub ciclo_ms: f64,
    pub saturo: bool,
    pub tok_s_tot: f64,
    pub tok_s_conv: f64,
    /// Only for the selected `G`: no `G` meets the per-conversation minimum.
    pub sotto_minimo: bool,
}

/// Rows per group, largest first: `n mod g` groups of `ceil(n/g)`, the others of `floor`.
pub fn dimensioni_gruppi(n: usize, g: usize) -> Vec<usize> {
    let n = n.max(1);
    let g = g.max(1).min(n);
    let (q, r) = (n / g, n % g);
    (0..g).map(|j| if j < r { q + 1 } else { q }).collect()
}

pub fn stima(p: &Param, n_conv: usize, g: usize) -> Stima {
    let n_conv = n_conv.max(1);
    let dim = dimensioni_gruppi(n_conv, g);
    let g = dim.len();
    let b = dim[0]; // Largest group
    let giro = dim.iter().map(|&x| p.giro(x as f64)).fold(0.0, f64::max);
    let lavoro: f64 = dim.iter().map(|&x| p.t_max(x as f64)).sum(); // Bottleneck work per cycle
    let ciclo = giro.max(lavoro).max(1e-9);
    Stima {
        g,
        b,
        giro_ms: giro,
        t_max_ms: p.t_max(b as f64),
        ciclo_ms: ciclo,
        saturo: lavoro >= giro,
        tok_s_tot: n_conv as f64 / ciclo * 1000.0,
        tok_s_conv: 1000.0 / ciclo,
        sotto_minimo: false,
    }
}

/// Recommended `G` and estimates for `1..=min(g_max, n_conv)`.
pub fn gruppi_ottimali(p: &Param, n_conv: usize, g_max: usize, minimo_per_conv: Option<f64>) -> (usize, Vec<Stima>) {
    let tetto = g_max.max(1).min(n_conv.max(1));
    let mut st: Vec<Stima> = (1..=tetto).map(|g| stima(p, n_conv, g)).collect();
    let ammessi: Vec<usize> = match minimo_per_conv {
        None => (0..st.len()).collect(),
        Some(m) => (0..st.len()).filter(|&i| st[i].tok_s_conv >= m).collect(),
    };
    if ammessi.is_empty() {
        // No G meets the minimum. Pick the highest per-conversation tokens/s, breaking ties by lowest G.
        let mut mig = 0;
        for i in 1..st.len() {
            if st[i].tok_s_conv > st[mig].tok_s_conv * (1.0 + 1e-12) {
                mig = i;
            }
        }
        st[mig].sotto_minimo = true;
        return (st[mig].g, st);
    }
    let max = ammessi.iter().map(|&i| st[i].tok_s_tot).fold(0.0, f64::max);
    let scelto = *ammessi.iter().find(|&&i| st[i].tok_s_tot >= 0.95 * max).unwrap();
    (st[scelto].g, st)
}

// Measurements collected during a run.

/// Observe sends and replies without affecting their scheduling.
#[derive(Debug, Default)]
pub struct MisureGruppi {
    invio: Vec<Option<Instant>>,
    giri_ms: Vec<f64>,
    intervalli_ms: Vec<f64>,
    rows: u64,
    passi: u64,
    rapporti: Vec<f64>,
    con_vuoto: u64,
}

fn mediana(v: &[f64]) -> Value {
    if v.is_empty() {
        return Value::Null;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len();
    json!(if n % 2 == 1 { s[n / 2] } else { (s[n / 2 - 1] + s[n / 2]) / 2.0 })
}

impl MisureGruppi {
    /// A `Step` for group `g` is about to go out. `conteggi` holds the current member counts for each group.
    pub fn send(&mut self, g: usize, conteggi: &[usize]) {
        if self.invio.len() <= g {
            self.invio.resize(g + 1, None);
        }
        self.invio[g] = Some(Instant::now());
        self.passi += 1;
        self.rows += conteggi.get(g).copied().unwrap_or(0) as u64;
        let attivi: Vec<usize> = conteggi.iter().copied().filter(|&c| c > 0).collect();
        if !attivi.is_empty() {
            let media = attivi.iter().sum::<usize>() as f64 / attivi.len() as f64;
            self.rapporti.push(*attivi.iter().max().unwrap() as f64 / media);
        }
        if conteggi.contains(&0) && conteggi.iter().any(|&c| c > 1) {
            self.con_vuoto += 1;
        }
    }

    /// The `Token` for group `g` arrived (arrival time at the head).
    pub fn token(&mut self, g: usize, at: Instant) {
        if let Some(Some(t)) = self.invio.get_mut(g).map(|x| x.take()) {
            self.giri_ms.push(at.saturating_duration_since(t).as_secs_f64() * 1000.0);
        }
    }

    /// Interval (ms) between consecutive group `Token` messages.
    pub fn intervallo(&mut self, ms: f64) {
        self.intervalli_ms.push(ms);
    }

    pub fn json(&self, n_gruppi: usize) -> Value {
        let n = self.rapporti.len().max(1) as f64;
        json!({
            "n_groups": n_gruppi,
            "steps": self.passi,
            "round_ms_median": mediana(&self.giri_ms),
            "interval_ms_median": mediana(&self.intervalli_ms),
            "mean_rows_per_step": if self.passi > 0 { json!(self.rows as f64 / self.passi as f64) } else { Value::Null },
            "imbalance": {
                "mean_max_over_mean": self.rapporti.iter().sum::<f64>() / n,
                "peak_max_over_mean": self.rapporti.iter().cloned().fold(0.0, f64::max),
                "empty_group_step_fraction": if self.passi > 0 { self.con_vuoto as f64 / self.passi as f64 } else { 0.0 },
            },
        })
    }
}

// Fit parameters from run summaries.

/// Fit `y = a + k·b`. With a single `b`, set `a = 0` and add a note. If `a < 0`, set it to 0 and refit `k`.
pub fn adatta_retta(punti: &[(f64, f64)]) -> (f64, f64, Option<String>) {
    let n = punti.len() as f64;
    let sx: f64 = punti.iter().map(|p| p.0).sum();
    let sy: f64 = punti.iter().map(|p| p.1).sum();
    let sxx: f64 = punti.iter().map(|p| p.0 * p.0).sum();
    let sxy: f64 = punti.iter().map(|p| p.0 * p.1).sum();
    let den = n * sxx - sx * sx;
    let k_zero = || punti.iter().map(|p| p.0 * p.1).sum::<f64>() / punti.iter().map(|p| p.0 * p.0).sum::<f64>().max(1e-12);
    if punti.len() < 2 || den.abs() < 1e-9 {
        return (0.0, k_zero(), Some("reference used one row count; using a = 0 and k = time/rows".into()));
    }
    let k = (n * sxy - sx * sy) / den;
    let a = (sy - k * sx) / n;
    if a < 0.0 || k < 0.0 {
        return (0.0, k_zero(), Some("fit gave a negative a or k; using a = 0".into()));
    }
    (a, k, None)
}

/// Inputs: reference points (rows per step, median interval in ms), median single-group chain round
/// (rows, round ms), layer shares by stage and fixed per-hop delay (`rit_ms`, e.g. from ritardo.py). Split
/// reference compute across the `n` stages by layer share. The rest of the round (`giro − Σt`)
/// covers fixed delay on `n` hops (`n−1` forward plus token return), and the remainder is overhead `o`.
pub fn stima_param(rif: &[(f64, f64)], catena_g1: Option<(f64, f64)>, quote: &[f64], rit_ms: f64) -> Result<(Param, Vec<String>), String> {
    if rif.is_empty() {
        return Err("at least one reference summary needs a 'groups' key to fit a and k".into());
    }
    if quote.is_empty() {
        return Err("stage layer shares are required".into());
    }
    let (a, k, nota) = adatta_retta(rif);
    let note: Vec<String> = nota.into_iter().collect();
    let tot: f64 = quote.iter().sum();
    let stages: Vec<Stage> = quote.iter().map(|q| Stage { a_ms: a * q / tot, k_ms: k * q / tot }).collect();
    let n = stages.len();
    let p = Param { stages, salti: vec![Salto { fisso_ms: rit_ms, per_riga_ms: 0.0 }; n], o_ms: 0.0 };
    Ok(chiudi_con_catena(p, catena_g1, rit_ms, note))
}

/// Per-stage timing inputs: `punti[i]` holds (rows per step, ms per step) for stage `i`, one point per
/// chain run (see module docs). Each stage gets its own fit `a_i + k_i·b`. The rest of the single-group round
/// (`giro − Σt − ritardi`) is overhead `o`.
pub fn stima_param_stadi(punti: &[Vec<(f64, f64)>], catena_g1: Option<(f64, f64)>, rit_ms: f64) -> Result<(Param, Vec<String>), String> {
    if punti.is_empty() || punti.iter().any(|v| v.is_empty()) {
        return Err("at least one chain run needs timings for every stage".into());
    }
    let mut note = Vec::new();
    let mut stages = Vec::new();
    for (i, v) in punti.iter().enumerate() {
        let (a, k, nota) = adatta_retta(v);
        if let Some(x) = nota {
            note.push(format!("stage {i}: {x}"));
        }
        stages.push(Stage { a_ms: a, k_ms: k });
    }
    let n = stages.len();
    let p = Param { stages, salti: vec![Salto { fisso_ms: rit_ms, per_riga_ms: 0.0 }; n], o_ms: 0.0 };
    Ok(chiudi_con_catena(p, catena_g1, rit_ms, note))
}

/// Overhead `o` from the median single-group chain round (rows, round ms).
fn chiudi_con_catena(mut p: Param, catena_g1: Option<(f64, f64)>, rit_ms: f64, mut note: Vec<String>) -> (Param, Vec<String>) {
    let n = p.stages.len();
    match catena_g1 {
        Some((b, giro)) => {
            let resto = giro - p.giro(b);
            if resto < 0.0 {
                note.push(format!("chain round ({giro:.1} ms) is shorter than estimated compute and network time; using o = 0"));
            }
            p.o_ms = resto.max(0.0);
            if rit_ms > 0.0 {
                note.push(format!("fixed network delay: {n} hops × {rit_ms} ms"));
            }
        }
        None => note.push("no single-group chain run; using zero overhead, which may overestimate throughput".into()),
    }
    (p, note)
}

/// Mean step time (ms) for one stage from `timings`: (decode + logit sync + sampling) / n_decode.
pub fn ms_per_passo(timings: &Value) -> Option<f64> {
    let n = num(timings, &["n_decode"])?;
    if n <= 0.0 {
        return None;
    }
    let s = num(timings, &["decode_s"])? + num(timings, &["logits_sync_s"]).unwrap_or(0.0) + num(timings, &["sampling_s"]).unwrap_or(0.0);
    Some(s / n * 1000.0)
}

fn num(v: &Value, path: &[&str]) -> Option<f64> {
    let mut x = v;
    for p in path {
        x = x.get(*p)?;
    }
    x.as_f64()
}

/// Timings for stages after the head, from `<base>.tail.json` next to the summary (tail `timings` object or
/// array for multiple stages). Each element may also wrap its timings under `timings`.
fn tempi_coda(summary: &str) -> Option<Vec<f64>> {
    let base = summary.strip_suffix(".summary.json")?;
    let v: Value = serde_json::from_str(&std::fs::read_to_string(format!("{base}.tail.json")).ok()?).ok()?;
    let elenco = match v {
        Value::Array(a) => a,
        x => vec![x],
    };
    elenco.iter().map(|e| ms_per_passo(e.get("timings").unwrap_or(e))).collect()
}

/// Run `publicompute-stage calibrate FILE.summary.json... [options]`.
pub fn comando(file: &[String], opz: &HashMap<String, String>) -> Result<(), String> {
    if file.is_empty() {
        return Err("Usage: calibrate A.summary.json [B...] [--conversations N] [--g-max M] [--stages S | --shares 1,1] [--latency-ms X] [--min-per-conv X] [--exclude-g G] [--json FILE]\n  Place A.tail.json beside each chain summary to use tail timings when available.".into());
    }
    let parse = |k: &str, def: f64| -> Result<f64, String> {
        opz.get(k).map_or(Ok(def), |s| s.parse::<f64>().map_err(|_| format!("invalid value for --{k}: {s}")))
    };
    // --exclude-g G: omit the chain run with G groups from the fit for cross-validation, but keep it in the
    // predicted vs. measured comparison.
    let escludi = match opz.get("exclude-g") {
        Some(s) => Some(s.parse::<usize>().map_err(|_| format!("invalid value for --exclude-g: {s}"))?),
        None => None,
    };
    let mut rif = Vec::new();
    let mut catena: Vec<(usize, f64, f64, usize)> = Vec::new(); // (G, generation tok/s, rows per step, N)
    let mut catena_g1 = None;
    let mut punti_stadi: Vec<Vec<(f64, f64)>> = Vec::new();
    let mut stadi_ok = true; // All chain runs used have timings for every stage.
    let mut n_conv = None;
    for f in file {
        let v: Value = serde_json::from_str(&std::fs::read_to_string(f).map_err(|e| format!("{f}: {e}"))?).map_err(|e| format!("{f}: {e}"))?;
        let g = v.get("groups").ok_or_else(|| format!("{f}: missing \"groups\" key"))?;
        let ng = num(g, &["n_groups"]).unwrap_or(1.0) as usize;
        let rows = num(g, &["mean_rows_per_step"]).ok_or_else(|| format!("{f}: missing groups.mean_rows_per_step"))?;
        let nc = num(&v, &["n_conversations"]).unwrap_or(0.0) as usize;
        let tok_s = num(&v, &["tok_s_generation"]).unwrap_or(0.0);
        if v.get("mode").and_then(Value::as_str) == Some("chain") {
            n_conv.get_or_insert(nc);
            catena.push((ng, tok_s, rows, nc));
            if escludi == Some(ng) {
                continue;
            }
            if ng == 1 {
                if let Some(giro) = num(g, &["round_ms_median"]) {
                    catena_g1 = Some((rows, giro));
                }
            }
            let head = v.get("timings").and_then(ms_per_passo);
            match (head, tempi_coda(f)) {
                (Some(t0), Some(resto)) => {
                    let mut tutti = vec![t0];
                    tutti.extend(resto);
                    if punti_stadi.is_empty() {
                        punti_stadi = vec![Vec::new(); tutti.len()];
                    }
                    if tutti.len() == punti_stadi.len() {
                        for (i, t) in tutti.into_iter().enumerate() {
                            punti_stadi[i].push((rows, t));
                        }
                    } else {
                        stadi_ok = false;
                    }
                }
                _ => stadi_ok = false,
            }
        } else if let Some(i) = num(g, &["interval_ms_median"]) {
            n_conv.get_or_insert(nc);
            rif.push((rows, i));
        }
    }
    let n = match opz.get("conversations") {
        Some(s) => s.parse::<usize>().map_err(|_| format!("invalid value for --conversations: {s}"))?,
        None => n_conv.unwrap_or(0).max(1),
    };
    let g_max = parse("g-max", 8.0)? as usize;
    let quote: Vec<f64> = match opz.get("shares") {
        Some(s) => s.split(',').map(|x| x.trim().parse::<f64>().map_err(|_| format!("--shares: expected a number, got \"{x}\""))).collect::<Result<_, _>>()?,
        None => vec![1.0; parse("stages", 2.0)? as usize],
    };
    let rit = parse("latency-ms", 0.0)?;
    let minimo = match opz.get("min-per-conv") {
        Some(s) => Some(s.parse::<f64>().map_err(|_| format!("invalid value for --min-per-conv: {s}"))?),
        None => None,
    };
    let con_stadi = stadi_ok && !punti_stadi.is_empty();
    let (p, mut note) = if con_stadi { stima_param_stadi(&punti_stadi, catena_g1, rit)? } else { stima_param(&rif, catena_g1, &quote, rit)? };
    note.insert(0, if con_stadi { "fitted from measured stage timings".into() } else { "estimated from layer shares because tail timings are missing".into() });
    if let Some(g) = escludi {
        note.push(format!("cross-check: held out the run with G = {g}"));
    }
    let (gc, st) = gruppi_ottimali(&p, n, g_max, minimo);
    println!("Estimated parameters ({} stages, N = {n}):", p.stages.len());
    for (i, s) in p.stages.iter().enumerate() {
        println!("  stage {i}: a = {:.2} ms, k = {:.3} ms/row", s.a_ms, s.k_ms);
    }
    println!("  network: {:.2} ms per hop ({} hops), head overhead o = {:.2} ms", p.salti[0].fisso_ms, p.salti.len(), p.o_ms);
    for x in &note {
        println!("  {x}");
    }
    let mis: HashMap<usize, f64> = catena.iter().filter(|c| c.3 == n).map(|c| (c.0, c.1)).collect();
    println!("\n  G  b  round_ms  t_max_ms  cycle_ms  tok/s_tot  tok/s_conv  saturated  measured  error");
    let mut confronto = Vec::new();
    for s in &st {
        let (m, e) = match mis.get(&s.g) {
            Some(&m) if m > 0.0 => (format!("{m:9.1}"), format!("{:+6.1}%", (s.tok_s_tot - m) / m * 100.0)),
            _ => ("        -".into(), "     -".into()),
        };
        if let Some(&m) = mis.get(&s.g) {
            confronto.push(json!({"g": s.g, "predicted": s.tok_s_tot, "measured": m, "relative_error": if m > 0.0 { (s.tok_s_tot - m) / m } else { 0.0 }}));
        }
        println!(
            "{}{:3} {:2} {:9.1} {:9.1} {:9.1} {:10.1} {:11.2}  {:6} {} {}",
            if s.g == gc { ">" } else { " " },
            s.g, s.b, s.giro_ms, s.t_max_ms, s.ciclo_ms, s.tok_s_tot, s.tok_s_conv, if s.saturo { "yes" } else { "no" }, m, e
        );
    }
    let sotto = st.iter().any(|s| s.sotto_minimo);
    println!("\nRecommended G: {gc}{}", if sotto { " (closest to the requested minimum per conversation)" } else { "" });
    let (mig, ok5) = if mis.is_empty() {
        (None, None)
    } else {
        let (gm, best) = mis.iter().fold((0, 0.0), |a, (&g, &t)| if t > a.1 || (t == a.1 && g < a.0) { (g, t) } else { a });
        (Some(gm), mis.get(&gc).map(|&t| t >= 0.95 * best))
    };
    if let Some(gm) = mig {
        println!("best measured result: G = {gm}; recommended within 5% of peak: {}", match ok5 { Some(true) => "yes", Some(false) => "no", None => "not measured" });
    }
    if let Some(path) = opz.get("json") {
        let tab: Vec<Value> = st.iter().map(|s| json!({"g": s.g, "b": s.b, "round_ms": s.giro_ms, "t_max_ms": s.t_max_ms, "cycle_ms": s.ciclo_ms, "tok_s_tot": s.tok_s_tot, "tok_s_conv": s.tok_s_conv, "saturated": s.saturo})).collect();
        let v = json!({"recommended": gc, "below_minimum": sotto, "table": tab, "comparison": confronto, "best_measured": mig, "recommended_within_5pct": ok5, "o_ms": p.o_ms, "method": if con_stadi { "stages" } else { "reference" }, "stages": p.stages.iter().map(|s| json!({"a_ms": s.a_ms, "k_ms": s.k_ms})).collect::<Vec<_>>()});
        std::fs::write(path, serde_json::to_string_pretty(&v).unwrap()).map_err(|e| format!("{path}: {e}"))?;
    }
    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;

    fn p(stages: &[(f64, f64)], rit: f64, o: f64) -> Param {
        Param {
            stages: stages.iter().map(|&(a, k)| Stage { a_ms: a, k_ms: k }).collect(),
            salti: vec![Salto { fisso_ms: rit, per_riga_ms: 0.0 }; stages.len()],
            o_ms: o,
        }
    }

    #[test]
    fn single_stage_without_network_needs_one_group() {
        // t(b) = 5 + 1·b, no network: T = t_max. G·t_max rises with G at fixed b, so no gain in throughput.
        let q = p(&[(5.0, 1.0)], 0.0, 0.0);
        let (g, st) = gruppi_ottimali(&q, 8, 8, None);
        assert_eq!(g, 1);
        assert!((st[0].tok_s_tot - 8.0 / 13.0 * 1000.0).abs() < 1e-6);
        assert!(st[0].saturo);
        // With 2 groups: b = 4, cycle = 2·9 = 18 ms, worse.
        assert!(st[1].tok_s_tot < st[0].tok_s_tot);
    }

    #[test]
    fn two_lan_stages_gain_little_from_extra_groups() {
        // 2 equal stages t(b)=4+1·b, network 0.2 ms per hop, o = 1
        let q = p(&[(4.0, 1.0), (4.0, 1.0)], 0.2, 1.0);
        let s1 = stima(&q, 8, 1);
        assert!((s1.giro_ms - (2.0 * 12.0 + 0.4 + 1.0)).abs() < 1e-9);
        assert!(!s1.saturo || s1.giro_ms <= 12.0);
        let s2 = stima(&q, 8, 2);
        // b=4: t=8, T=16+0.4+1=17.4. G·t_max = 16 < 17.4, so no saturation, cycle 17.4.
        assert!((s2.ciclo_ms - 17.4).abs() < 1e-9 && !s2.saturo);
        assert!(s2.tok_s_tot > s1.tok_s_tot * 1.3);
        let (g, _) = gruppi_ottimali(&q, 8, 8, None);
        assert!(g == 2 || g == 3);
    }

    #[test]
    fn two_stages_at_25_ms_rtt_need_more_groups() {
        // RTT 25 ms → 12.5 ms per hop. Stages t(b)=3+0.5·b.
        let q = p(&[(3.0, 0.5), (3.0, 0.5)], 12.5, 1.0);
        let (g, st) = gruppi_ottimali(&q, 16, 8, None);
        // One group: T = 2·(3+8) + 25 + 1 = 48. b=8 with 2 groups: T=2·7+26=40, cycle 40.
        // 4 groups: b=4, t=5, T=36, G·t=20 → 36. 8 groups: b=2, t=4, T=34, G·t=32 → 34.
        assert!((st[0].ciclo_ms - 48.0).abs() < 1e-9);
        assert!((st[1].ciclo_ms - 40.0).abs() < 1e-9);
        assert!(g >= 2, "G = {g}");
        assert!(st[g - 1].tok_s_tot >= 0.95 * st.iter().map(|s| s.tok_s_tot).fold(0.0, f64::max));
        // Choose the smallest G that reaches 95%.
        if g > 1 {
            assert!(st[g - 2].tok_s_tot < 0.95 * st.iter().map(|s| s.tok_s_tot).fold(0.0, f64::max));
        }
    }

    #[test]
    fn three_uneven_stages_are_limited_by_bottleneck() {
        // Stage 1 is slow: t_max = 20 + b. The others take 4 + 0.5 b.
        let q = p(&[(4.0, 0.5), (20.0, 1.0), (4.0, 0.5)], 1.0, 0.5);
        let b = 4.0;
        assert!((q.t_max(b) - 24.0).abs() < 1e-9);
        assert!((q.giro(b) - (6.0 + 24.0 + 6.0 + 3.0 + 0.5)).abs() < 1e-9);
        // Already saturates at 2 groups: 2·24 = 48 > 39.5.
        let s = stima(&q, 8, 2);
        assert!(s.saturo && (s.ciclo_ms - 48.0).abs() < 1e-9);
        let (g, _) = gruppi_ottimali(&q, 8, 8, None);
        assert!(g <= 2, "G = {g}");
    }

    #[test]
    fn per_conversation_minimum_limits_groups_or_flags_shortfall() {
        let q = p(&[(3.0, 0.5), (3.0, 0.5)], 12.5, 1.0);
        // Without a minimum, G > 1. If the minimum is impossible, flag it and pick the closest.
        let (g, st) = gruppi_ottimali(&q, 16, 8, Some(1e6));
        assert!(st.iter().find(|s| s.g == g).unwrap().sotto_minimo);
        let migliore = st.iter().map(|s| s.tok_s_conv).fold(0.0, f64::max);
        assert!((st[g - 1].tok_s_conv - migliore).abs() < 1e-12);
        // Only some G values meet the minimum.
        let soglia = (st[0].tok_s_conv + st[7].tok_s_conv) / 2.0;
        let (g2, st2) = gruppi_ottimali(&q, 16, 8, Some(soglia));
        assert!(st2[g2 - 1].tok_s_conv >= soglia && !st2[g2 - 1].sotto_minimo);
    }

    #[test]
    fn group_count_does_not_exceed_conversation_count() {
        let q = p(&[(1.0, 1.0)], 5.0, 0.0);
        let (_, st) = gruppi_ottimali(&q, 3, 10, None);
        assert_eq!(st.len(), 3);
        assert_eq!(st[2].b, 1);
    }

    #[test]
    fn link_latency_from_bandwidth() {
        // 25 ms RTT, 5120 bytes per row at 100 Mbit/s = 0.4096 ms per row
        let s = Salto::da_rete(25.0, 5120.0, 100.0);
        assert!((s.fisso_ms - 12.5).abs() < 1e-9 && (s.per_riga_ms - 0.4096).abs() < 1e-9);
    }

    #[test]
    fn least_squares_fit() {
        let (a, k, n) = adatta_retta(&[(2.0, 9.0), (4.0, 13.0), (8.0, 21.0)]);
        assert!((a - 5.0).abs() < 1e-9 && (k - 2.0).abs() < 1e-9 && n.is_none());
        let (a1, k1, n1) = adatta_retta(&[(8.0, 40.0)]);
        assert!(a1 == 0.0 && (k1 - 5.0).abs() < 1e-9 && n1.is_some());
    }

    /// Compare throughput with a single-process reference using synthetic timing
    /// parameters. The second group overlaps transfer and computation.
    #[test]
    fn estimated_throughput_matches_measured_scale() {
        // A 53 ms reference step over eight conversations provides the baseline rate.
        let q = p(&[(1.0, 4.4), (1.0, 4.4)], 2.0, 8.0);
        let rif = 8.0 / 0.053 * 1000.0 / 1000.0; 
        let s1 = stima(&q, 8, 1);
        let s2 = stima(&q, 8, 2);
        assert!((s1.tok_s_tot / rif - 0.65).abs() < 0.08, "1 gruppo: {:.0}%", s1.tok_s_tot / rif * 100.0);
        assert!(s2.tok_s_tot / s1.tok_s_tot > 1.4 && (s2.tok_s_tot / rif - 1.13).abs() < 0.12, "2 groups: {:.0}%", s2.tok_s_tot / rif * 100.0);
        let (g, _) = gruppi_ottimali(&q, 8, 8, None);
        assert!(g >= 2, "G = {g}");
    }

    #[test]
    fn estimates_parameters_from_summaries() {
        // Reference interval = 6 + 2·b. One-group chain (b = 8): round 60 ms with 5 ms fixed per hop.
        let (q, _) = stima_param(&[(2.0, 10.0), (4.0, 14.0), (8.0, 22.0)], Some((8.0, 60.0)), &[1.0, 1.0], 5.0).unwrap();
        assert!((q.stages[0].a_ms - 3.0).abs() < 1e-9 && (q.stages[0].k_ms - 1.0).abs() < 1e-9);
        // Compute 22 + network 10 → o = 28
        assert!((q.o_ms - 28.0).abs() < 1e-9);
        assert!((q.giro(8.0) - 60.0).abs() < 1e-9);
        assert!(stima_param(&[], None, &[1.0], 0.0).is_err());
    }

    #[test]
    fn uneven_groups_when_conversations_do_not_divide_evenly() {
        assert_eq!(dimensioni_gruppi(8, 3), vec![3, 3, 2]);
        assert_eq!(dimensioni_gruppi(8, 4), vec![2, 2, 2, 2]);
        assert_eq!(dimensioni_gruppi(3, 8), vec![1, 1, 1]);
        // t(b) = 2 + b on one stage, no network. 3 groups of 3, 3, 2 → work 5 + 5 + 4 = 14 ms, max round 5.
        let q = p(&[(2.0, 1.0)], 0.0, 0.0);
        let s = stima(&q, 8, 3);
        assert_eq!(s.b, 3);
        assert!((s.ciclo_ms - 14.0).abs() < 1e-9 && s.saturo);
        assert!((s.tok_s_tot - 8.0 / 14.0 * 1000.0).abs() < 1e-6);
    }

    // Each fixture row gives groups, rows per step, head and tail milliseconds
    // per step, and observed aggregate tokens per second. Both stage times
    // include decode, logit synchronization, and sampling.
    const CAT_A4000: [(usize, f64, f64, f64, f64); 5] = [
        (1, 8.0, 21.540, 31.720, 104.63),
        (2, 4.0, 13.548, 20.154, 138.27),
        (3, 8.0 / 3.0, 11.706, 17.367, 145.66),
        (4, 2.0, 10.842, 16.168, 126.23),
        (8, 1.0, 9.535, 12.667, 80.47),
    ];
    const SINGLE_GROUP_A4000: (f64, f64) = (8.0, 75.29);

    fn param_a4000(escludi: Option<usize>) -> Param {
        let usate: Vec<_> = CAT_A4000.iter().filter(|c| Some(c.0) != escludi).collect();
        let head: Vec<(f64, f64)> = usate.iter().map(|c| (c.1, c.2)).collect();
        let tail: Vec<(f64, f64)> = usate.iter().map(|c| (c.1, c.3)).collect();
        let g1 = if escludi == Some(1) { None } else { Some(SINGLE_GROUP_A4000) };
        stima_param_stadi(&[head, tail], g1, 12.0).unwrap().0
    }

    fn controlla_a4000(q: &Param, soglia: f64, solo: Option<usize>) {
        for c in CAT_A4000.iter().filter(|c| solo.map_or(true, |g| c.0 == g)) {
            let e = (stima(q, 8, c.0).tok_s_tot - c.4) / c.4;
            assert!(e.abs() < soglia, "G = {}: error {:+.1}%", c.0, e * 100.0);
        }
    }

    /// The lower layer-share estimate can hide a tail bottleneck. Verify that
    /// fitting the two stages independently selects the measured optimum.
    #[test]
    fn a4000_regression_with_stage_timings() {
        // Fit the reference points using equal layer shares.
        let (vecchio, _) = stima_param(&[(8.0, 53.09), (4.0, 33.59), (2.0, 25.0), (1.0, 21.12)], Some(SINGLE_GROUP_A4000), &[1.0, 1.0], 12.0).unwrap();
        let e4 = (stima(&vecchio, 8, 4).tok_s_tot - 126.23) / 126.23;
        assert!(e4 > 0.2, "estimating by layer shares should fail here: {e4}");
        assert_ne!(gruppi_ottimali(&vecchio, 8, 8, None).0, 3);
        // Fit the head and tail timing points separately.
        let q = param_a4000(None);
        assert!(q.stages[1].a_ms > q.stages[0].a_ms && q.stages[1].k_ms > q.stages[0].k_ms, "the tail should cost more than the head: {:?}", q.stages);
        controlla_a4000(&q, 0.04, None);
        let (g, st) = gruppi_ottimali(&q, 8, 8, None);
        assert_eq!(g, 3);
        let migliore = CAT_A4000.iter().map(|c| c.4).fold(0.0, f64::max);
        let misurato_g = CAT_A4000.iter().find(|c| c.0 == g).unwrap().4;
        assert!(misurato_g >= 0.95 * migliore);
        assert!(st[2].tok_s_tot > st[1].tok_s_tot && st[2].tok_s_tot > st[3].tok_s_tot);
    }

    /// Hold out one group count at a time and compare the prediction with its measurement.
    #[test]
    fn a4000_cross_validation_holds_out_one_run() {
        for c in CAT_A4000.iter() {
            let q = param_a4000(Some(c.0));
            controlla_a4000(&q, 0.06, Some(c.0));
            assert_eq!(gruppi_ottimali(&q, 8, 8, None).0, 3, "escluso G = {}", c.0);
        }
        // Fit the outer group counts and check the interpolated cases.
        let usate: Vec<_> = CAT_A4000.iter().filter(|c| [1, 2, 8].contains(&c.0)).collect();
        let head: Vec<(f64, f64)> = usate.iter().map(|c| (c.1, c.2)).collect();
        let tail: Vec<(f64, f64)> = usate.iter().map(|c| (c.1, c.3)).collect();
        let q = stima_param_stadi(&[head, tail], Some(SINGLE_GROUP_A4000), 12.0).unwrap().0;
        controlla_a4000(&q, 0.06, Some(3));
        controlla_a4000(&q, 0.06, Some(4));
        assert_eq!(gruppi_ottimali(&q, 8, 8, None).0, 3);
    }

    #[test]
    fn derives_step_ms_from_timings() {
        let t = json!({"decode_s": 0.2, "logits_sync_s": 1.0, "sampling_s": 0.4, "n_decode": 100});
        assert!((ms_per_passo(&t).unwrap() - 16.0).abs() < 1e-9);
        assert!(ms_per_passo(&json!({"n_decode": 0, "decode_s": 1.0})).is_none());
    }

    #[test]
    fn group_metrics() {
        let mut m = MisureGruppi::default();
        m.send(0, &[3, 1]);
        m.send(1, &[3, 1]);
        m.send(0, &[2, 0]); // Group 1 empty while group 0 has more than one
        m.token(0, Instant::now());
        m.token(0, Instant::now()); // Ignored with no send in flight
        m.intervallo(10.0);
        m.intervallo(20.0);
        let j = m.json(2);
        assert_eq!(j["steps"], 3);
        assert_eq!(j["n_groups"], 2);
        assert!((j["mean_rows_per_step"].as_f64().unwrap() - 2.0).abs() < 1e-9);
        assert!((j["interval_ms_median"].as_f64().unwrap() - 15.0).abs() < 1e-9);
        assert!((j["imbalance"]["peak_max_over_mean"].as_f64().unwrap() - 1.5).abs() < 1e-9);
        // Only active groups count in the mean, so the last ratio is one.
        assert!((j["imbalance"]["mean_max_over_mean"].as_f64().unwrap() - (1.5 + 1.5 + 1.0) / 3.0).abs() < 1e-9);
        assert!((j["imbalance"]["empty_group_step_fraction"].as_f64().unwrap() - 1.0 / 3.0).abs() < 1e-9);
        assert!(j["round_ms_median"].as_f64().unwrap() >= 0.0);
    }
}
