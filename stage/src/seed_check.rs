//! Replay the tail's seeded samples with the verifier's logits and count
//! disagreements. A mismatch near a probability boundary can come from
//! rounding, so only margins above the configured tolerance count toward
//! rejection. Aggregate margins and counts across responses for the
//! sequential check.

use crate::sampling::PosSeme;
use serde::{Deserialize, Serialize};

/// Also count against fixed tolerances so summaries can be compared across runs.
pub const TOLLERANZE_RIEPILOGO: [f64; 5] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1];
/// Number of highest mismatch margins to keep per response.
const MARGINI_TENUTI: usize = 32;

#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct ConteggiSeme {
    #[serde(rename = "positions")]
    pub posizioni: usize,
    #[serde(rename = "mismatches")]
    pub discordanze: usize,
    /// Mismatches with margin above the counting tolerance.
    #[serde(rename = "suspicious")]
    pub sospette: usize,
    #[serde(rename = "tolerance")]
    pub tolleranza: f64,
    /// Largest mismatch margin (0 if none).
    #[serde(rename = "max_margin")]
    pub margine_max: f64,
    /// Suspicious mismatches for each tolerance in `TOLLERANZE_RIEPILOGO`, in the same order.
    #[serde(rename = "suspicious_by_tolerance")]
    pub sospette_per_tolleranza: Vec<usize>,
    /// Highest mismatch margins, in descending order (up to `MARGINI_TENUTI`).
    #[serde(rename = "high_margins")]
    pub margini_alti: Vec<f32>,
}

/// Count positions in a response. Negative or NaN `tolleranza` means 0, so every mismatch is suspicious.
pub fn conta(pos: &[PosSeme], tolleranza: f64) -> ConteggiSeme {
    let tol = if tolleranza.is_finite() && tolleranza > 0.0 { tolleranza } else { 0.0 };
    let mut c = ConteggiSeme { posizioni: pos.len(), tolleranza: tol, sospette_per_tolleranza: vec![0; TOLLERANZE_RIEPILOGO.len()], ..Default::default() };
    let mut margini = Vec::new();
    for p in pos.iter().filter(|p| p.discorda) {
        c.discordanze += 1;
        let m = p.margine as f64;
        if m > tol {
            c.sospette += 1;
        }
        c.margine_max = c.margine_max.max(m);
        for (i, &t) in TOLLERANZE_RIEPILOGO.iter().enumerate() {
            if m > t {
                c.sospette_per_tolleranza[i] += 1;
            }
        }
        margini.push(p.margine);
    }
    margini.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    margini.truncate(MARGINI_TENUTI);
    c.margini_alti = margini;
    c
}

/// Sum across responses (per host or series). Merge the highest margins and trim the list.
pub fn somma(a: &ConteggiSeme, b: &ConteggiSeme) -> ConteggiSeme {
    let mut m: Vec<f32> = a.margini_alti.iter().chain(&b.margini_alti).copied().collect();
    m.sort_by(|x, y| y.partial_cmp(x).unwrap_or(std::cmp::Ordering::Equal));
    m.truncate(MARGINI_TENUTI);
    let n = a.sospette_per_tolleranza.len().max(b.sospette_per_tolleranza.len());
    let per_tol = (0..n).map(|i| a.sospette_per_tolleranza.get(i).copied().unwrap_or(0) + b.sospette_per_tolleranza.get(i).copied().unwrap_or(0)).collect();
    ConteggiSeme {
        posizioni: a.posizioni + b.posizioni,
        discordanze: a.discordanze + b.discordanze,
        sospette: a.sospette + b.sospette,
        tolleranza: a.tolleranza.max(b.tolleranza),
        margine_max: a.margine_max.max(b.margine_max),
        sospette_per_tolleranza: per_tol,
        margini_alti: m,
    }
}

/// Required thresholds for the per-response and cumulative checks.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct ParamSeme {
    /// Margin above which a mismatch is suspicious.
    pub tolleranza_margine: f64,
    /// Below this many positions, defer judgment and accumulate positions per host.
    pub min_posizioni: usize,
    /// Reject if suspicious mismatches reach this count (`None` disables the rule).
    pub soglia_sospette: Option<usize>,
    /// Reject if the fraction of suspicious positions reaches this value (`None` disables the rule).
    pub soglia_frazione: Option<f64>,
    /// Reference count subtracted at every update. Quiet responses lower the cumulative score.
    pub cusum_riferimento: f64,
    /// CUSUM: cap on one response's contribution, in suspicious mismatches.
    pub cusum_tetto: f64,
    /// CUSUM: alarm when g >= this value.
    pub cusum_allarme: f64,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum EsitoRisposta {
    /// Fewer than `min_posizioni`: defer judgment.
    Corta,
    Accettata,
    Respinta,
}

/// Apply the per-response thresholds. Reject counts computed at a different
/// margin tolerance because the suspicious-position count would be incompatible.
pub fn decidi_risposta(c: &ConteggiSeme, p: &ParamSeme) -> Result<EsitoRisposta, String> {
    if (c.tolleranza - p.tolleranza_margine.max(0.0)).abs() > 1e-12 {
        return Err(format!("counts used tolerance {}, parameters specify {}", c.tolleranza, p.tolleranza_margine));
    }
    if c.posizioni < p.min_posizioni {
        return Ok(EsitoRisposta::Corta);
    }
    let per_numero = p.soglia_sospette.is_some_and(|s| c.sospette >= s);
    let per_frazione = p.soglia_frazione.is_some_and(|f| c.posizioni > 0 && c.sospette as f64 / c.posizioni as f64 >= f);
    Ok(if per_numero || per_frazione { EsitoRisposta::Respinta } else { EsitoRisposta::Accettata })
}

/// Add the capped count of suspicious positions, subtract the reference,
/// and clamp at zero. Accumulate short responses before applying this update.
pub fn cusum_passo(g: f64, c: &ConteggiSeme, p: &ParamSeme) -> f64 {
    (g + (c.sospette as f64).min(p.cusum_tetto) - p.cusum_riferimento).max(0.0)
}

pub fn allarme(g: f64, p: &ParamSeme) -> bool {
    g >= p.cusum_allarme
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::sampling::{sampling, controlla_seme, pilota, seme_conversazione};

    fn pos(d: bool, m: f32) -> PosSeme {
        PosSeme { discorda: d, margine: m }
    }

    fn param() -> ParamSeme {
        ParamSeme {
            tolleranza_margine: 1e-3,
            min_posizioni: 16,
            soglia_sospette: Some(3),
            soglia_frazione: None,
            cusum_riferimento: 0.5,
            cusum_tetto: 4.0,
            cusum_allarme: 3.0,
        }
    }

    /// Deterministic pseudorandom logit generator.
    fn logit(n: usize, s: u64, scala: f32) -> Vec<f32> {
        let mut x = s.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                ((x >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * scala
            })
            .collect()
    }

    #[test]
    fn honest_tail_matches_across_positions_and_seeds() {
        let mut tot = 0;
        for s in 0..40u64 {
            let radice = seme_conversazione(1000 + s, s + 1);
            for step in 0..50u32 {
                let l = logit(300, s * 1000 + step as u64, 8.0);
                for t in [0.4f32, 0.8, 1.0, 0.0] {
                    let (tok, _) = sampling(&l, t, radice, step).unwrap();
                    let c = controlla_seme(&l, t, radice, step, tok).unwrap();
                    assert!(!c.discorda, "s={s} step={step} t={t}");
                    tot += 1;
                }
            }
        }
        assert_eq!(tot, 8000);
    }

    #[test]
    fn small_verifier_logit_drift_is_not_suspicious() {
        // Verifier logits have 1e-4 noise (FP16 at the boundary), so any mismatch has a small margin.
        let mut posv = Vec::new();
        for s in 0..40u64 {
            let radice = seme_conversazione(7, s + 1);
            for step in 0..100u32 {
                let l = logit(300, s * 977 + step as u64, 6.0);
                let rum = logit(300, s * 31 + step as u64 + 99_999, 2e-4);
                let lv: Vec<f32> = l.iter().zip(&rum).map(|(a, b)| a + b).collect();
                let (tok, _) = sampling(&l, 0.8, radice, step).unwrap();
                posv.push(controlla_seme(&lv, 0.8, radice, step, tok).unwrap());
            }
        }
        let c = conta(&posv, 1e-2);
        assert_eq!(c.posizioni, 4000);
        assert_eq!(c.sospette, 0, "{c:?}");
    }

    #[test]
    fn detects_tail_picking_second_token_at_temperature_08() {
        let mut posv = Vec::new();
        for s in 0..20u64 {
            let radice = seme_conversazione(5, s + 1);
            for step in 0..50u32 {
                let l = logit(300, s * 1313 + step as u64, 8.0);
                // Second most probable token.
                let mut idx: Vec<usize> = (0..l.len()).collect();
                idx.sort_by(|&a, &b| l[b].partial_cmp(&l[a]).unwrap());
                posv.push(controlla_seme(&l, 0.8, radice, step, idx[1]).unwrap());
            }
        }
        let c = conta(&posv, 1e-3);
        // The seed rarely picks the second token, so almost every position differs by a large margin.
        assert!(c.discordanze > 850, "{c:?}");
        assert!(c.sospette > 800, "{c:?}");
        assert_eq!(decidi_risposta(&c, &ParamSeme { tolleranza_margine: 1e-3, ..param() }).unwrap(), EsitoRisposta::Respinta);
    }

    #[test]
    fn second_token_at_zero_temperature_mismatches_with_margin() {
        let l = [0.0f32, 3.0, 1.0, 0.0];
        let c = controlla_seme(&l, 0.0, 1, 0, 2).unwrap();
        assert!(c.discorda && c.margine > 0.5);
        assert!(!controlla_seme(&l, 0.0, 1, 0, 1).unwrap().discorda);
        // Near tie: tiny margin.
        let q = [0.0f32, 3.0, 2.99999, 0.0];
        let c = controlla_seme(&q, 0.0, 1, 0, 2).unwrap();
        assert!(c.discorda && c.margine < 1e-4, "{c:?}");
    }

    #[test]
    fn near_tie_mismatch_is_not_suspicious() {
        // Place a draw near a CDF boundary and perturb one logit. This can change
        // the selected token while leaving the margin below the counting threshold.
        let radice = 42u64;
        let base = [-50.0f32, 0.0, 0.0, -50.0];
        // Find a step where the tail picks 1.
        let step = (0..200u32).find(|&s| sampling(&base, 1.0, radice, s).unwrap().0 == 1).unwrap();
        // The verifier sees token 1 as a little less probable than token 2, shifting the boundary by ~0.5*1e-6.
        let verif = [-50.0f32, -1e-6, 0.0, -50.0];
        let c = controlla_seme(&verif, 1.0, radice, step, 1).unwrap();
        // Either it matches or the margin is tiny. Neither is suspicious at tolerance 1e-3.
        assert!(!c.discorda || c.margine < 1e-3, "{c:?}");
        // Construct explicitly: u lands just past the boundary.
        let mut trovato = false;
        for step in 0..5000u32 {
            let (a, _) = sampling(&base, 1.0, radice, step).unwrap();
            if a != 1 {
                continue;
            }
            for d in [1e-4f32, 1e-3, 1e-2] {
                let v = [-50.0f32, -d, 0.0, -50.0];
                let c = controlla_seme(&v, 1.0, radice, step, 1).unwrap();
                if c.discorda {
                    // Shift the boundary by d. The margin cannot exceed ~d/2 (probability of token 1 ~0.5).
                    assert!(c.margine as f64 <= d as f64, "d={d} {c:?}");
                    assert!(conta(&[c], 1e-1).sospette == 0);
                    trovato = true;
                }
            }
        }
        assert!(trovato, "test did not find a mismatch near a tie");
    }

    #[test]
    fn pilot_mismatches_at_requested_fraction() {
        let radice = seme_conversazione(3, 1);
        let (mut n, mut disc) = (0, 0);
        for step in 0..2000u32 {
            let l = logit(200, step as u64 + 5, 6.0);
            let (tok, _) = sampling(&l, 0.8, radice, step).unwrap();
            let riportato = pilota(&l, tok, radice, step, 0.25, 7).unwrap_or(tok);
            n += 1;
            if controlla_seme(&l, 0.8, radice, step, riportato).unwrap().discorda {
                disc += 1;
            }
        }
        let f = disc as f64 / n as f64;
        assert!((0.2..0.3).contains(&f), "{f}");
        assert!(pilota(&logit(10, 1, 1.0), 0, 1, 0, 0.0, 7).is_none());
    }

    #[test]
    fn combines_counts_and_updates_decision_and_cusum() {
        let p = ParamSeme { tolleranza_margine: 0.01, ..param() };
        let mut v: Vec<PosSeme> = (0..20).map(|_| pos(false, 0.0)).collect();
        v[3] = pos(true, 0.005); 
        v[5] = pos(true, 0.3);
        let c = conta(&v, p.tolleranza_margine);
        assert_eq!((c.posizioni, c.discordanze, c.sospette), (20, 2, 1));
        assert_eq!(c.margini_alti, vec![0.3, 0.005]);
        assert_eq!(c.sospette_per_tolleranza, vec![2, 2, 2, 1, 1]);
        assert_eq!(decidi_risposta(&c, &p).unwrap(), EsitoRisposta::Accettata);
        v[7] = pos(true, 0.2);
        v[9] = pos(true, 0.2);
        let c2 = conta(&v, p.tolleranza_margine);
        assert_eq!(decidi_risposta(&c2, &p).unwrap(), EsitoRisposta::Respinta);
        let pf = ParamSeme { soglia_sospette: None, soglia_frazione: Some(0.1), ..p.clone() };
        assert_eq!(decidi_risposta(&c2, &pf).unwrap(), EsitoRisposta::Respinta);
        assert_eq!(decidi_risposta(&c, &pf).unwrap(), EsitoRisposta::Accettata);
        assert_eq!(decidi_risposta(&conta(&v[..5], p.tolleranza_margine), &p).unwrap(), EsitoRisposta::Corta);
        assert!(decidi_risposta(&conta(&v, 0.5), &p).is_err());
        let s = somma(&c, &c2);
        assert_eq!((s.posizioni, s.discordanze, s.sospette), (40, 6, 4));
        // CUSUM: three responses with 3 suspicious and reference 0.5 cross alarm level 3 on the second.
        let mut g = 0.0;
        let tre = ConteggiSeme { sospette: 3, ..Default::default() };
        g = cusum_passo(g, &tre, &p);
        assert!(!allarme(g, &p));
        g = cusum_passo(g, &tre, &p);
        assert!(allarme(g, &p));
        // An honest response (0 suspicious) lowers the sum.
        assert_eq!(cusum_passo(0.2, &ConteggiSeme::default(), &p), 0.0);
        assert_eq!(cusum_passo(0.0, &ConteggiSeme { sospette: 100, ..Default::default() }, &p), 3.5);
    }
}
