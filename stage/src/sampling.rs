//! Shared sampling for the tail, reference, and verifier. At zero temperature
//! choose the earliest maximum. Otherwise draw from the temperature-scaled
//! cumulative distribution with a seed derived from the conversation and step.
//! Report the selected token's log probability over the full vocabulary at
//! temperature one so this score describes the underlying distribution.

pub fn argmax(l: &[f32]) -> usize {
    indice_del_massimo(l, massimo(l)).unwrap_or(0)
}

/// Reject rows without a finite maximum. Sampling cannot choose a meaningful
/// token from an empty row, all NaNs, or infinite maximum.
pub fn riga_valida(l: &[f32]) -> Result<f32, String> {
    if l.is_empty() {
        return Err("empty logit row".into());
    }
    let m = massimo(l);
    if m.is_finite() {
        Ok(m)
    } else {
        Err(format!("invalid logit row: maximum {m} (all NaN or infinite); no token to sample"))
    }
}

/// Approximate exp for nonpositive inputs with range reduction and a degree-six
/// polynomial. The loop can vectorize without a math-library call. Inputs
/// below -87 return zero.
#[inline(always)]
pub fn exp_veloce(x: f32) -> f32 {
    let x = x.max(-87.0);
    const MAGIA: f32 = 12_582_912.0; // Bias additions round the exponent using the spacing of f32 around this value.
    let n = (x * std::f32::consts::LOG2_E + MAGIA) - MAGIA;
    let r = (x - n * 0.693_145_75) - n * 1.428_606_8e-6;
    let p = 1.0 + r * (1.0 + r * (0.5 + r * (1.0 / 6.0 + r * (1.0 / 24.0 + r * (1.0 / 120.0 + r * (1.0 / 720.0))))));
    let bits = ((n as i32 + 127) as u32) << 23;
    p * f32::from_bits(bits)
}

/// The lanes let the maximum reduction vectorize. NaN values do not become maxima.
#[inline]
fn massimo(l: &[f32]) -> f32 {
    let mut acc = [f32::NEG_INFINITY; 16];
    let mut it = l.chunks_exact(16);
    for c in &mut it {
        for (a, &v) in acc.iter_mut().zip(c) {
            *a = if v > *a { v } else { *a };
        }
    }
    let mut m = f32::NEG_INFINITY;
    for &a in acc.iter().chain(it.remainder()) {
        m = if a > m { a } else { m };
    }
    m
}

/// Return the first position of a known maximum.
#[inline]
fn indice_del_massimo(l: &[f32], m: f32) -> Option<usize> {
    l.iter().position(|&v| v == m)
}

const BLOCCO: usize = 256;
const GRUPPO: usize = 4 * BLOCCO;

/// Compute shifted exponentials in blocks. Independent f32 accumulators shorten
/// dependency chains, and the f64 sum across blocks limits rounding error.
fn logsumexp_con_max(l: &[f32], m: f32) -> f64 {
    if !m.is_finite() {
        return m as f64;
    }
    let mut tot = 0.0f64;
    let mut buf = [0.0f32; GRUPPO];
    for gruppo in l.chunks(GRUPPO) {
        for (b, &v) in buf.iter_mut().zip(gruppo) {
            *b = exp_veloce(v - m);
        }
        let e = &buf[..gruppo.len()];
        if e.len() == GRUPPO {
            let (b0, r) = e.split_at(BLOCCO);
            let (b1, r) = r.split_at(BLOCCO);
            let (b2, b3) = r.split_at(BLOCCO);
            let (mut s0, mut s1, mut s2, mut s3) = (-0.0f32, -0.0f32, -0.0f32, -0.0f32);
            for i in 0..BLOCCO {
                s0 += b0[i];
                s1 += b1[i];
                s2 += b2[i];
                s3 += b3[i];
            }
            tot += s0 as f64;
            tot += s1 as f64;
            tot += s2 as f64;
            tot += s3 as f64;
        } else {
            for block in e.chunks(BLOCCO) {
                let s: f32 = block.iter().sum();
                tot += s as f64;
            }
        }
    }
    m as f64 + tot.ln()
}

pub fn logsumexp(l: &[f32]) -> f64 {
    logsumexp_con_max(l, massimo(l))
}

pub fn logprob(l: &[f32], token: usize) -> f32 {
    (l[token] as f64 - logsumexp(l)) as f32
}

/// Per-step top probabilities used to compare distributions. JSONL keys c and s
/// identify the conversation and step, and t holds token and log-probability pairs.
#[derive(serde::Serialize, serde::Deserialize, Clone, Debug, PartialEq)]
pub struct RecTop {
    pub c: u64,
    pub s: u32,
    pub t: Vec<(i32, f32)>,
}

/// Return up to k token and log-probability pairs in descending score order.
/// Lower token IDs break ties.
pub fn top_k(l: &[f32], k: usize) -> Vec<(i32, f32)> {
    let k = k.min(l.len());
    if k == 0 {
        return Vec::new();
    }
    let lse = logsumexp(l);
    let mut idx: Vec<u32> = (0..l.len() as u32).collect();
    let ord = |a: &u32, b: &u32| l[*b as usize].partial_cmp(&l[*a as usize]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(b));
    if k < idx.len() {
        idx.select_nth_unstable_by(k - 1, ord);
        idx.truncate(k);
    }
    idx.sort_by(ord);
    idx.into_iter().map(|i| (i as i32, (l[i as usize] as f64 - lse) as f32)).collect()
}

pub fn scrivi_top(path: &str, recs: &[RecTop]) -> Result<(), String> {
    let mut s = String::new();
    for r in recs {
        s.push_str(&serde_json::to_string(r).map_err(|e| e.to_string())?);
        s.push('\n');
    }
    std::fs::write(path, s).map_err(|e| format!("{path}: {e}"))
}

fn splitmix64(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Mix the process seed with the conversation ID so equal step indices
/// in two conversations give independent draws.
pub fn seme_conversazione(radice: u64, conversation_id: u64) -> u64 {
    let mut s = radice ^ conversation_id.wrapping_mul(0xA24B_AED4_963E_E407);
    let a = splitmix64(&mut s);
    let mut s2 = a ^ 0x5851_F42D_4C95_7F2D;
    splitmix64(&mut s2)
}

/// Mix the conversation seed with the token's logical step.
pub fn seed(radice: u64, step: u32) -> u64 {
    let mut s = radice;
    let a = splitmix64(&mut s);
    let mut s3 = a ^ (step as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93);
    splitmix64(&mut s3)
}

/// Optional sampling cutoffs. Greedy selection ignores both. A zero top_k
/// and a top_p outside (0, 1) leave the distribution uncut.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Tagli {
    pub top_p: f32,
    pub top_k: u32,
}

impl Tagli {
    pub fn nessuno(&self) -> bool {
        !self.attivo_k_su(usize::MAX) && !self.attivo_p()
    }
    fn attivo_k_su(&self, n: usize) -> bool {
        self.top_k > 0 && (self.top_k as usize) < n
    }
    fn attivo_p(&self) -> bool {
        self.top_p > 0.0 && self.top_p < 1.0
    }
}

/// Zero weights discarded by top_k and then by top_p. The nucleus is the
/// shortest probability-ranked prefix reaching top_p of the remaining mass.
/// Keep each weight at its original index so inverse CDF walks vocabulary order.
fn applica_tagli(l: &[f32], pesi: &mut [f64], tagli: Tagli) {
    let n = l.len();
    let (ak, ap) = (tagli.attivo_k_su(n), tagli.attivo_p());
    if !ak && !ap {
        return;
    }
    let ord = |a: &u32, b: &u32| l[*b as usize].partial_cmp(&l[*a as usize]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(b));
    let mut idx: Vec<u32> = (0..n as u32).collect();
    if ak {
        let k = tagli.top_k as usize;
        idx.select_nth_unstable_by(k - 1, ord);
        idx.truncate(k);
    }
    if ap {
        idx.sort_by(ord);
        let tot: f64 = idx.iter().map(|&i| pesi[i as usize]).sum();
        let soglia = tagli.top_p as f64 * tot;
        let mut acc = 0.0;
        let mut tenuti = idx.len();
        for (j, &i) in idx.iter().enumerate() {
            acc += pesi[i as usize];
            if acc >= soglia {
                tenuti = j + 1;
                break;
            }
        }
        idx.truncate(tenuti);
    }
    let mut tenuto = vec![false; n];
    for &i in &idx {
        tenuto[i as usize] = true;
    }
    for (w, t) in pesi.iter_mut().zip(tenuto) {
        if !t {
            *w = 0.0;
        }
    }
}

/// Sample one row and return its token and unadjusted log probability.
/// Reject invalid logits before choosing a token.
pub fn sampling(l: &[f32], temperature: f32, root_seed: u64, step: u32) -> Result<(usize, f32), String> {
    campiona_tagli(l, temperature, root_seed, step, Tagli::default())
}

/// Apply the supplied cutoffs to the draw and score against the full distribution.
pub fn campiona_tagli(l: &[f32], temperature: f32, root_seed: u64, step: u32, tagli: Tagli) -> Result<(usize, f32), String> {
    crate::timings::misura(crate::timings::Fase::Campiona, || campiona_(l, temperature, root_seed, step, tagli))
}

/// Sample without updating timers so a caller can time several rows together.
fn campiona_(l: &[f32], temperature: f32, root_seed: u64, step: u32, tagli: Tagli) -> Result<(usize, f32), String> {
    let mf = riga_valida(l)?;
    let tok = if temperature <= 0.0 {
        indice_del_massimo(l, mf).ok_or("no maximum found")?
    } else {
        let t = temperature as f64;
        let m = mf as f64;
        PESI.with(|p| {
            let mut pesi = p.borrow_mut();
            pesi.clear();
            pesi.extend(l.iter().map(|&v| ((v as f64 - m) / t).exp()));
            applica_tagli(l, &mut pesi, tagli);
            let tot: f64 = pesi.iter().sum();
            let mut st = seed(root_seed, step);
            let u = (splitmix64(&mut st) >> 11) as f64 / (1u64 << 53) as f64 * tot;
            let mut acc = 0.0;
            // A rounding residue at the end of the CDF belongs to the last surviving token.
            let mut scelto = if tagli.nessuno() { l.len() - 1 } else { pesi.iter().rposition(|&w| w > 0.0).unwrap_or(l.len() - 1) };
            for (i, w) in pesi.iter().enumerate() {
                acc += w;
                if u < acc {
                    scelto = i;
                    break;
                }
            }
            scelto
        })
    };
    Ok((tok, (l[tok] as f64 - logsumexp_con_max(l, mf)) as f32))
}

/// Result of replaying a sampled position with the verifier's logits and
/// the same deterministic seed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PosSeme {
    pub discorda: bool,
    /// On disagreement, measure the normalized distance from the draw to the
    /// reported token's CDF interval. Greedy sampling uses the difference in
    /// probabilities. A small margin can result from small numerical differences.
    pub margine: f32,
}

/// Replay the sample at step with the supplied logits and compare it with
/// riportato. Return a row error if the logits cannot be sampled.
pub fn controlla_seme(l: &[f32], temperature: f32, root_seed: u64, step: u32, riportato: usize) -> Result<PosSeme, String> {
    controlla_seme_tagli(l, temperature, root_seed, step, riportato, Tagli::default())
}

/// Replay a sample using its original temperature and cutoffs.
pub fn controlla_seme_tagli(l: &[f32], temperature: f32, root_seed: u64, step: u32, riportato: usize, tagli: Tagli) -> Result<PosSeme, String> {
    let mf = riga_valida(l)?;
    if riportato >= l.len() {
        return Ok(PosSeme { discorda: true, margine: 1.0 });
    }
    if temperature <= 0.0 {
        let am = indice_del_massimo(l, mf).ok_or("no maximum found")?;
        if am == riportato {
            return Ok(PosSeme { discorda: false, margine: 0.0 });
        }
        let lse = logsumexp_con_max(l, mf);
        let (pa, pr) = ((l[am] as f64 - lse).exp(), (l[riportato] as f64 - lse).exp());
        return Ok(PosSeme { discorda: true, margine: (pa - pr).max(0.0) as f32 });
    }
    let t = temperature as f64;
    let m = mf as f64;
    PESI.with(|p| {
        let mut pesi = p.borrow_mut();
        pesi.clear();
        pesi.extend(l.iter().map(|&v| ((v as f64 - m) / t).exp()));
        applica_tagli(l, &mut pesi, tagli);
        let tot: f64 = pesi.iter().sum();
        let mut st = seed(root_seed, step);
        let u = (splitmix64(&mut st) >> 11) as f64 / (1u64 << 53) as f64 * tot;
        let last = if tagli.nessuno() { l.len() - 1 } else { pesi.iter().rposition(|&w| w > 0.0).unwrap_or(l.len() - 1) };
        let (mut acc, mut scelto, mut lo, mut hi) = (0.0f64, last, 0.0f64, 0.0f64);
        let mut trovato = false;
        for (i, w) in pesi.iter().enumerate() {
            if i == riportato {
                lo = acc;
            }
            acc += w;
            if i == riportato {
                hi = acc;
            }
            if !trovato && u < acc {
                scelto = i;
                trovato = true;
            }
            if trovato && i >= riportato {
                break;
            }
        }
        if scelto == riportato {
            return Ok(PosSeme { discorda: false, margine: 0.0 });
        }
        let dist = if u < lo { lo - u } else { u - hi };
        Ok(PosSeme { discorda: true, margine: (dist / tot).max(0.0) as f32 })
    })
}

/// Test helper that deterministically substitutes the highest-scoring other
/// token for a fraction of positions. It reports the substituted token's
/// real log probability to exercise the seed check.
pub fn pilota(l: &[f32], scelto: usize, root_seed: u64, step: u32, frazione: f64, sale: u64) -> Option<usize> {
    if frazione <= 0.0 {
        return None;
    }
    let mut st = seed(root_seed ^ sale.wrapping_mul(0x9E37_79B9_7F4A_7C15), step ^ 0x5bd1_e995);
    let u = (splitmix64(&mut st) >> 11) as f64 / (1u64 << 53) as f64;
    if u >= frazione {
        return None;
    }
    let mut best: Option<usize> = None;
    for (i, &v) in l.iter().enumerate() {
        if i == scelto || v.is_nan() {
            continue;
        }
        if best.is_none_or(|b| v > l[b]) {
            best = Some(i);
        }
    }
    best
}

thread_local! {
    /// Keep softmax scratch space per thread so sampling can reuse the allocation.
    static PESI: std::cell::RefCell<Vec<f64>> = const { std::cell::RefCell::new(Vec::new()) };
}

pub type ParamRiga = (f32, u64, u32);

/// A zero request selects up to four workers. Bound the count by rows and
/// keep at least one worker in the calculation.
pub fn thread_effettivi(richiesti: usize, n: usize) -> usize {
    let t = if richiesti == 0 { n.min(4) } else { richiesti };
    t.min(n).max(1)
}

/// Sample independent contiguous blocks of rows. Results stay in row order
/// and match sequential sampling. Time the whole operation once, including
/// the parallel work.
pub fn campiona_righe(rows: &[&[f32]], params: &[ParamRiga], thread: usize) -> Vec<Result<(usize, f32), String>> {
    campiona_righe_tagli(rows, params, &vec![Tagli::default(); rows.len()], thread)
}

/// Supply one cutoff configuration per row.
pub fn campiona_righe_tagli(rows: &[&[f32]], params: &[ParamRiga], tagli: &[Tagli], thread: usize) -> Vec<Result<(usize, f32), String>> {
    assert_eq!(rows.len(), params.len(), "row and parameter counts differ");
    assert_eq!(rows.len(), tagli.len(), "row and cutoff counts differ");
    let n = rows.len();
    crate::timings::misura(crate::timings::Fase::Campiona, || {
        let t = thread.min(n).max(1);
        if t == 1 {
            return rows.iter().zip(params).zip(tagli).map(|((l, &(temp, seed, step)), &tg)| campiona_(l, temp, seed, step, tg)).collect();
        }
        let per = n.div_ceil(t);
        std::thread::scope(|sc| {
            let gestori: Vec<_> = rows
                .chunks(per)
                .zip(params.chunks(per))
                .zip(tagli.chunks(per))
                .map(|((rs, ps), ts)| sc.spawn(move || rs.iter().zip(ps).zip(ts).map(|((l, &(temp, seed, step)), &tg)| campiona_(l, temp, seed, step, tg)).collect::<Vec<_>>()))
                .collect();
            gestori.into_iter().flat_map(|h| h.join().expect("sampling thread panicked")).collect()
        })
    })
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn argmax_uses_first_index_on_tie() {
        assert_eq!(argmax(&[1.0, 3.0, 3.0, 2.0]), 1);
    }

    #[test]
    fn exponential_is_fast_and_accurate() {
        for i in 0..2000 {
            let x = -(i as f32) * 0.045;
            let (a, b) = (exp_veloce(x) as f64, (x as f64).exp());
            assert!((a - b).abs() <= 3e-7 * b + 1e-30, "x={x} {a} {b}");
        }
        assert!(exp_veloce(-1000.0) < 1e-37);
    }

    #[test]
    fn logprob_distribution_sums_to_one() {
        let l = [0.5f32, 1.5, -2.0, 3.0, 0.0];
        let s: f64 = (0..l.len()).map(|i| (logprob(&l, i) as f64).exp()).sum();
        assert!((s - 1.0).abs() < 1e-6, "{s}");
        assert!(logprob(&l, 3) > logprob(&l, 0));
    }

    #[test]
    fn top_k_is_sorted_and_matches_logprob() {
        let l = [0.5f32, 1.5, -2.0, 3.0, 3.0, 0.0];
        let t = top_k(&l, 3);
        assert_eq!(t.iter().map(|x| x.0).collect::<Vec<_>>(), vec![3, 4, 1]);
        assert_eq!(t[0].1, logprob(&l, 3));
        assert_eq!(t[2].1, logprob(&l, 1));
        assert_eq!(top_k(&l, 99).len(), 6);
        assert!(top_k(&l, 0).is_empty());
    }

    #[test]
    fn zero_temperature_is_greedy_with_logprob() {
        let l = [0.0f32, 2.0, 1.0];
        let (t, lp) = sampling(&l, 0.0, 1, 0).unwrap();
        assert_eq!(t, 1);
        assert_eq!(lp, logprob(&l, 1));
    }

    #[test]
    fn seed_is_deterministic_and_depends_on_inputs() {
        assert_eq!(seed(1, 3), seed(1, 3));
        assert_ne!(seed(1, 3), seed(1, 4));
        assert_ne!(seed(1, 3), seed(2, 3));
    }

    #[test]
    fn all_non_finite_row_returns_error() {
        let nan = [f32::NAN; 40];
        let neg = [f32::NEG_INFINITY; 40];
        let pos = [1.0f32, f32::INFINITY, 2.0];
        for temp in [0.0f32, 0.8] {
            assert!(sampling(&nan, temp, 1, 0).unwrap_err().contains("invalid"));
            assert!(sampling(&neg, temp, 1, 0).is_err());
            assert!(sampling(&pos, temp, 1, 0).is_err());
            assert!(sampling(&[], temp, 1, 0).is_err());
        }
        let buona = [0.0f32, 2.0, 1.0];
        let rows: Vec<&[f32]> = vec![&buona, &nan, &buona];
        let params = [(0.0f32, 1u64, 0u32); 3];
        for th in [1usize, 3] {
            let ris = campiona_righe(&rows, &params, th);
            assert_eq!(ris[0].as_ref().unwrap().0, 1);
            assert!(ris[1].is_err());
            assert_eq!(ris[2].as_ref().unwrap().0, 1);
        }
        assert_eq!(sampling(&[f32::NAN, 1.0, 0.5], 0.0, 1, 0).unwrap().0, 1);
    }

    #[test]
    fn per_conversation_seed_differs_and_matches_reference() {
        let radice = 1u64;
        assert_eq!(seme_conversazione(radice, 3), seme_conversazione(radice, 3));
        assert_ne!(seme_conversazione(radice, 1), seme_conversazione(radice, 2));
        assert_ne!(seme_conversazione(1, 5), seme_conversazione(2, 5));
        // Equal logits at the same step can yield different sequences for different conversations.
        let l: Vec<f32> = (0..50).map(|i| (i as f32 * 0.37).sin() * 0.5).collect();
        let (a, b) = (seme_conversazione(radice, 1), seme_conversazione(radice, 2));
        for step in 0..8 {
            assert_ne!(seed(a, step), seed(b, step));
        }
        let sa: Vec<usize> = (0..64).map(|s| sampling(&l, 0.8, a, s).unwrap().0).collect();
        let sb: Vec<usize> = (0..64).map(|s| sampling(&l, 0.8, b, s).unwrap().0).collect();
        assert_ne!(sa, sb);
        // Replaying a conversation reproduces the draws made by the tail.
        let sa2: Vec<usize> = (0..64).map(|s| sampling(&l, 0.8, seme_conversazione(radice, 1), s).unwrap().0).collect();
        assert_eq!(sa, sa2);
    }

    #[test]
    fn positive_temperature_is_repeatable_and_matches_distribution() {
        let l: Vec<f32> = (0..50).map(|i| (i as f32 * 0.37).sin() * 2.0).collect();
        assert_eq!(sampling(&l, 0.8, 9, 11).unwrap(), sampling(&l, 0.8, 9, 11).unwrap());
        let (mut mx, mut mn) = (0, 0);
        let (imax, imin) = (argmax(&l), l.iter().enumerate().min_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0);
        for step in 0..4000 {
            let (t, _) = sampling(&l, 1.0, 5, step).unwrap();
            mx += (t == imax) as u32;
            mn += (t == imin) as u32;
        }
        assert!(mx > mn * 3, "{mx} {mn}");
    }

    fn righe_casuali(n: usize, voci: usize, pari: bool) -> Vec<Vec<f32>> {
        let mut st = 12345u64;
        (0..n)
            .map(|_| {
                (0..voci)
                    .map(|_| {
                        let v = ((splitmix64(&mut st) >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 20.0;
                        // Quantize onto a coarse grid to generate exact ties.
                        if pari { (v * 0.5).round() * 2.0 } else { v }
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn parallel_rows_are_bit_identical_to_sequential_rows() {
        for &pari in &[false, true] {
            for &n in &[1usize, 2, 5, 8, 11] {
                let dati = righe_casuali(n, 3000, pari);
                let rows: Vec<&[f32]> = dati.iter().map(|v| v.as_slice()).collect();
                for &temp in &[0.0f32, 0.8, 1.0] {
                    let params: Vec<ParamRiga> = (0..n).map(|i| (temp, 7 + i as u64, i as u32 * 3)).collect();
                    let seq: Vec<(usize, f32)> = rows.iter().zip(&params).map(|(l, &(t, s, p))| sampling(l, t, s, p).unwrap()).collect();
                    for &th in &[1usize, 2, 3, 8] {
                        let par = campiona_righe(&rows, &params, th);
                        assert_eq!(par.len(), seq.len());
                        for (a, b) in par.iter().map(|r| r.as_ref().unwrap()).zip(&seq) {
                            assert_eq!(a.0, b.0, "token n={n} th={th} T={temp} pari={pari}");
                            assert_eq!(a.1.to_bits(), b.1.to_bits(), "logprob n={n} th={th} T={temp} pari={pari}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn empty_rows_and_effective_thread_count() {
        assert!(campiona_righe(&[], &[], 4).is_empty());
        assert_eq!(thread_effettivi(0, 8), 4);
        assert_eq!(thread_effettivi(0, 2), 2);
        assert_eq!(thread_effettivi(1, 8), 1);
        assert_eq!(thread_effettivi(16, 3), 3);
        assert_eq!(thread_effettivi(0, 0), 1);
    }

    // Sequential oracle for checking the vectorized probability calculation.
    fn argmax_vecchio(l: &[f32]) -> usize {
        let mut best = 0usize;
        let mut m = f32::NEG_INFINITY;
        for (i, &v) in l.iter().enumerate() {
            if v > m {
                m = v;
                best = i;
            }
        }
        best
    }

    fn logsumexp_vecchio(l: &[f32]) -> f64 {
        let m = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        if !m.is_finite() {
            return m as f64;
        }
        let mut tot = 0.0f64;
        for block in l.chunks(256) {
            let s: f32 = block.iter().map(|&v| exp_veloce(v - m)).sum();
            tot += s as f64;
        }
        m as f64 + tot.ln()
    }

    fn campiona_vecchio(l: &[f32], temperature: f32, root_seed: u64, step: u32) -> (usize, f32) {
        let tok = if temperature <= 0.0 {
            argmax_vecchio(l)
        } else {
            let t = temperature as f64;
            let m = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
            let pesi: Vec<f64> = l.iter().map(|&v| ((v as f64 - m) / t).exp()).collect();
            let tot: f64 = pesi.iter().sum();
            let mut st = seed(root_seed, step);
            let u = (splitmix64(&mut st) >> 11) as f64 / (1u64 << 53) as f64 * tot;
            let mut acc = 0.0;
            let mut scelto = l.len() - 1;
            for (i, w) in pesi.iter().enumerate() {
                acc += w;
                if u < acc {
                    scelto = i;
                    break;
                }
            }
            scelto
        };
        (tok, (l[tok] as f64 - logsumexp_vecchio(l)) as f32)
    }

    /// Generate rows with different numerical ranges, including ties and infinities.
    fn riga_tipo(seme_riga: u64, voci: usize, kind: usize) -> Vec<f32> {
        let mut st = seme_riga;
        (0..voci)
            .map(|_| {
                let r = splitmix64(&mut st);
                let v = ((r >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 20.0;
                match kind {
                    0 => v,
                    1 => (v * 0.5).round() * 2.0,
                    2 => if r & 7 == 0 { f32::NEG_INFINITY } else { v },
                    3 => v * 500.0 - 3000.0,
                    _ => v * 500.0 + 3000.0,
                }
            })
            .collect()
    }

    #[test]
    fn sampling_is_bit_identical_to_reference() {
        const VOCE: usize = 262_144;
        let temps = [0.0f32, 0.5, 0.8, 1.0];
        for i in 0..200usize {
            let l = riga_tipo(1000 + i as u64 * 7919, VOCE, i % 5);
            let temp = temps[(i / 5) % 4];
            let (a, b) = (sampling(&l, temp, 3 + i as u64, i as u32).unwrap(), campiona_vecchio(&l, temp, 3 + i as u64, i as u32));
            assert_eq!(a.0, b.0, "token at row {i}, T={temp}");
            assert_eq!(a.1.to_bits(), b.1.to_bits(), "log probability at row {i}, T={temp}");
            assert_eq!(argmax(&l), argmax_vecchio(&l), "argmax at row {i}");
            assert_eq!(logsumexp(&l).to_bits(), logsumexp_vecchio(&l).to_bits(), "logsumexp at row {i}");
        }
        // Exercise partial vector lanes, partial blocks, and invalid rows.
        for &n in &[1usize, 2, 15, 16, 17, 255, 256, 257, 1023, 1024, 1025, 2049, 5000] {
            for kind in 0..5 {
                let l = riga_tipo(n as u64 + kind as u64, n, kind);
                for &temp in &temps {
                    assert_eq!(sampling(&l, temp, 5, 1).unwrap(), campiona_vecchio(&l, temp, 5, 1), "n={n} tipo={kind} T={temp}");
                }
            }
            let tutte_meno_inf = vec![f32::NEG_INFINITY; n];
            let costanti = vec![0.25f32; n];
            let mut con_nan = riga_tipo(n as u64, n, 0);
            con_nan[n / 2] = f32::NAN;
            for &temp in &[0.0f32, 0.8] {
                assert!(sampling(&tutte_meno_inf, temp, 5, 1).is_err());
            }
            for l in [&costanti, &con_nan] {
                if n == 1 && l[0].is_nan() {
                    continue; 
                }
                for &temp in &[0.0f32, 0.8] {
                    let (a, b) = (sampling(l, temp, 5, 1).unwrap(), campiona_vecchio(l, temp, 5, 1));
                    assert_eq!(a.0, b.0);
                    assert_eq!(a.1.to_bits(), b.1.to_bits());
                }
            }
        }
    }

    /// Run the ignored benchmark in release mode to measure optimized sampling.
    #[test]
    #[ignore]
    fn sampling_benchmark() {
        use std::time::Instant;
        const VOCE: usize = 262_144;
        for &n in &[8usize, 16] {
            let dati = righe_casuali(n, VOCE, false);
            let rows: Vec<&[f32]> = dati.iter().map(|v| v.as_slice()).collect();
            for &temp in &[0.0f32, 0.8] {
                let params: Vec<ParamRiga> = (0..n).map(|i| (temp, 1 + i as u64, i as u32)).collect();
                let rif = campiona_righe(&rows, &params, 1);
                let ripeti = 20;
                let mut row = format!("rows={n:2} T={temp:.1}");
                for &th in &[1usize, 2, 4, 8] {
                    let t0 = Instant::now();
                    let mut last = Vec::new();
                    for _ in 0..ripeti {
                        last = campiona_righe(&rows, &params, th);
                    }
                    let ms = t0.elapsed().as_secs_f64() * 1000.0 / ripeti as f64;
                    let uguali = last.len() == rif.len() && last.iter().zip(&rif).all(|(a, b)| { let (a, b) = (a.as_ref().unwrap(), b.as_ref().unwrap()); a.0 == b.0 && a.1.to_bits() == b.1.to_bits() });
                    assert!(uguali, "results differ with {th} threads");
                    row += &format!(" | {} {:.2} ms", if th == 1 { "seq".to_string() } else { format!("{th}thr") }, ms);
                }
                println!("{row} | bit-identici: si");
            }
        }
    }

    // Sampling cutoff cases.

    fn riga_casuale(st: &mut u64, n: usize) -> Vec<f32> {
        (0..n).map(|_| (splitmix64(st) >> 40) as f32 / (1u64 << 24) as f32 * 12.0 - 6.0).collect()
    }

    /// Uncut sampling oracle for checking bit-for-bit equivalence.
    fn campiona_ante_c6(l: &[f32], temperature: f32, root_seed: u64, step: u32) -> (usize, f32) {
        let mf = riga_valida(l).unwrap();
        let tok = if temperature <= 0.0 {
            indice_del_massimo(l, mf).unwrap()
        } else {
            let (t, m) = (temperature as f64, mf as f64);
            let pesi: Vec<f64> = l.iter().map(|&v| ((v as f64 - m) / t).exp()).collect();
            let tot: f64 = pesi.iter().sum();
            let mut st = seed(root_seed, step);
            let u = (splitmix64(&mut st) >> 11) as f64 / (1u64 << 53) as f64 * tot;
            let (mut acc, mut scelto) = (0.0, l.len() - 1);
            for (i, w) in pesi.iter().enumerate() {
                acc += w;
                if u < acc {
                    scelto = i;
                    break;
                }
            }
            scelto
        };
        (tok, (l[tok] as f64 - logsumexp_con_max(l, mf)) as f32)
    }

    #[test]
    fn sampling_without_cutoffs_matches_reference_for_1000_rows() {
        let mut st = 42u64;
        for r in 0..1000u32 {
            let l = riga_casuale(&mut st, 50 + (r as usize % 300));
            let temp = [0.0f32, 0.3, 0.8, 1.0, 1.7][r as usize % 5];
            let vecchio = campiona_ante_c6(&l, temp, 1000 + r as u64, r);
            for tagli in [Tagli::default(), Tagli { top_p: 1.0, top_k: 0 }, Tagli { top_p: 2.0, top_k: 0 }, Tagli { top_p: 0.0, top_k: 100_000 }] {
                let new = campiona_tagli(&l, temp, 1000 + r as u64, r, tagli).unwrap();
                assert_eq!(new.0, vecchio.0, "row {r}");
                assert_eq!(new.1.to_bits(), vecchio.1.to_bits(), "row {r}");
            }
            let c = sampling(&l, temp, 1000 + r as u64, r).unwrap();
            assert_eq!((c.0, c.1.to_bits()), (vecchio.0, vecchio.1.to_bits()));
        }
    }

    #[test]
    fn top_k_one_is_argmax_at_any_temperature() {
        let mut st = 7u64;
        for r in 0..200u32 {
            let l = riga_casuale(&mut st, 120);
            for temp in [0.0f32, 0.5, 1.0, 2.5] {
                let (t, lp) = campiona_tagli(&l, temp, 9, r, Tagli { top_p: 0.0, top_k: 1 }).unwrap();
                assert_eq!(t, argmax(&l));
                assert_eq!(lp, logprob(&l, t));
            }
        }
        let l = [1.0f32, 5.0, 5.0, 2.0];
        assert_eq!(campiona_tagli(&l, 1.0, 3, 0, Tagli { top_p: 0.0, top_k: 1 }).unwrap().0, 1);
    }

    #[test]
    fn top_k_keeps_only_highest_scores() {
        let mut st = 11u64;
        let l = riga_casuale(&mut st, 200);
        let mut ord: Vec<usize> = (0..l.len()).collect();
        ord.sort_by(|&a, &b| l[b].partial_cmp(&l[a]).unwrap().then(a.cmp(&b)));
        let primi: std::collections::HashSet<usize> = ord[..5].iter().copied().collect();
        let mut visti = std::collections::HashSet::new();
        for step in 0..400u32 {
            let (t, _) = campiona_tagli(&l, 3.0, 5, step, Tagli { top_p: 0.0, top_k: 5 }).unwrap();
            assert!(primi.contains(&t), "token {t} is outside the top five");
            visti.insert(t);
        }
        assert!(visti.len() > 1, "high temperature should sample more than one token");
    }

    #[test]
    fn top_p_keeps_only_nucleus() {
        // Taking log probabilities as logits gives a distribution with known masses.
        let p = [0.15f64, 0.5, 0.05, 0.3];
        let l: Vec<f32> = p.iter().map(|x| x.ln() as f32).collect();
        for step in 0..100u32 {
            assert_eq!(campiona_tagli(&l, 1.0, 1, step, Tagli { top_p: 0.45, top_k: 0 }).unwrap().0, 1);
        }
        let mut visti = std::collections::HashSet::new();
        for step in 0..300u32 {
            let (t, lp) = campiona_tagli(&l, 1.0, 1, step, Tagli { top_p: 0.7, top_k: 0 }).unwrap();
            assert!(t == 1 || t == 3, "token {t} is outside the nucleus");
            // Score the selected token against the full vocabulary even when the draw uses a cutoff.
            assert!((lp as f64 - p[t].ln()).abs() < 1e-5);
            visti.insert(t);
        }
        assert_eq!(visti.len(), 2);
        // The nucleus threshold uses the top_k survivors' mass, which is 0.95 here.
        for step in 0..100u32 {
            let t = campiona_tagli(&l, 1.0, 1, step, Tagli { top_p: 0.6, top_k: 3 }).unwrap().0;
            assert!(t == 1 || t == 3);
        }
    }

    #[test]
    fn cutoffs_are_deterministic_with_same_seed() {
        let mut st = 5u64;
        let l = riga_casuale(&mut st, 300);
        let tg = Tagli { top_p: 0.9, top_k: 40 };
        for step in 0..50u32 {
            assert_eq!(campiona_tagli(&l, 0.8, 77, step, tg).unwrap(), campiona_tagli(&l, 0.8, 77, step, tg).unwrap());
        }
        assert_eq!(campiona_tagli(&l, 0.0, 77, 0, tg).unwrap(), sampling(&l, 0.0, 77, 0).unwrap());
    }

    #[test]
    fn parallel_sampling_with_cutoffs_matches_sequential() {
        let mut st = 3u64;
        let rows: Vec<Vec<f32>> = (0..9).map(|_| riga_casuale(&mut st, 150)).collect();
        let rif: Vec<&[f32]> = rows.iter().map(|r| r.as_slice()).collect();
        let params: Vec<ParamRiga> = (0..9).map(|i| (0.8, 100 + i as u64, i as u32)).collect();
        let tagli: Vec<Tagli> = (0..9).map(|i| if i % 2 == 0 { Tagli { top_p: 0.8, top_k: 20 } } else { Tagli::default() }).collect();
        let seq: Vec<_> = (0..9).map(|i| campiona_tagli(rif[i], 0.8, 100 + i as u64, i as u32, tagli[i]).unwrap()).collect();
        for th in [1usize, 3, 4] {
            let par: Vec<_> = campiona_righe_tagli(&rif, &params, &tagli, th).into_iter().map(|r| r.unwrap()).collect();
            assert_eq!(par, seq, "thread {th}");
        }
    }

    #[test]
    fn seed_check_accepts_token_sampled_with_cutoffs() {
        let mut st = 13u64;
        let tg = Tagli { top_p: 0.85, top_k: 25 };
        for step in 0..300u32 {
            let l = riga_casuale(&mut st, 180);
            let (tok, _) = campiona_tagli(&l, 0.9, 321, step, tg).unwrap();
            assert!(!controlla_seme_tagli(&l, 0.9, 321, step, tok, tg).unwrap().discorda, "passo {step}");
            let tagliato = l.iter().enumerate().min_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            assert!(controlla_seme_tagli(&l, 0.9, 321, step, tagliato, tg).unwrap().discorda);
            let a = controlla_seme(&l, 0.9, 321, step, tok).unwrap();
            let b = controlla_seme_tagli(&l, 0.9, 321, step, tok, Tagli::default()).unwrap();
            assert_eq!(a, b);
        }
    }
}
