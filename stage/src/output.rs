//! Write one JSONL record per conversation and a JSON run summary.
//! Comparisons use prompt IDs to align output. Once sampled tokens diverge,
//! log probabilities describe different prefixes. Forced replay can provide
//! a common prefix at every position.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct RecordConv {
    pub prompt_id: String,
    pub conversation_id: u64,
    #[serde(rename = "n_prompt_tokens")]
    pub n_token_prompt: u32,
    pub token: Vec<i32>,
    pub logprob: Vec<f32>,
    /// Bits of each log probability so bitwise comparison is independent of decimal formatting.
    pub logprob_bit: Vec<u32>,
    pub text: String,
    /// "eog" or "max_token".
    pub done: String,
    /// Time of the first token since the conversation started (sending `Open`), in ms.
    pub ttft_ms: f64,
    /// Time each token reaches the head since the start of the run, in ms.
    #[serde(rename = "token_times_ms")]
    pub tempi_ms: Vec<f64>,
    /// With arrival tracking, milliseconds from run start to arrival.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[serde(rename = "arrival_ms")]
    pub arrivo_ms: Option<f64>,
    /// Queue wait: from arrival to admission start (`Open`), in ms.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[serde(rename = "queue_wait_ms")]
    pub attesa_coda_ms: Option<f64>,
    /// First-word latency from arrival (first token time minus `arrivo_ms`), in ms.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[serde(rename = "arrival_ttft_ms")]
    pub ttft_arrivo_ms: Option<f64>,
    /// Store the conversation's temperature and root seed so verification can
    /// repeat the draw at each step.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub root_seed: Option<u64>,
    /// Omit unused sampling cutoffs. Missing fields decode to zero.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    /// Replay only: seeded resampling counts (see `seed_check`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<crate::seed_check::ConteggiSeme>,
    /// Present when the run records traffic classes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub class: Option<String>,
    /// Count times a batch released its slot for interactive work.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[serde(rename = "suspensions")]
    pub sospensioni: Option<u32>,
}

pub fn scrivi_jsonl(path: &str, recs: &[RecordConv]) -> Result<(), String> {
    let mut s = String::new();
    for r in recs {
        s.push_str(&serde_json::to_string(r).map_err(|e| e.to_string())?);
        s.push('\n');
    }
    std::fs::write(path, s).map_err(|e| format!("{path}: {e}"))
}

pub fn leggi_jsonl(path: &str) -> Result<Vec<RecordConv>, String> {
    let s = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let mut v = Vec::new();
    for (i, row) in s.lines().enumerate() {
        if row.trim().is_empty() {
            continue;
        }
        let mut r: RecordConv = serde_json::from_str(row).map_err(|e| format!("{path}, line {}: {e}", i + 1))?;
        if r.logprob_bit.len() != r.logprob.len() {
            r.logprob_bit = r.logprob.iter().map(|x| x.to_bits()).collect();
        }
        v.push(r);
    }
    Ok(v)
}

/// Percentile with linear interpolation. `v` is nonempty.
pub fn percentile(v: &[f64], p: f64) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let r = p / 100.0 * (s.len() - 1) as f64;
    let (lo, hi) = (r.floor() as usize, r.ceil() as usize);
    s[lo] + (s[hi] - s[lo]) * (r - lo as f64)
}

#[derive(Serialize, Debug)]
pub struct RiepConv {
    pub prompt_id: String,
    #[serde(rename = "n_tokens")]
    pub n_token: usize,
    pub tok_s: f64,
    pub ttft_ms: f64,
}

#[derive(Serialize, Debug)]
pub struct Riepilogo {
    #[serde(rename = "mode")]
    pub modo: String,
    #[serde(rename = "n_conversations")]
    pub n_conversazioni: usize,
    #[serde(rename = "total_tokens")]
    pub token_totali: usize,
    #[serde(rename = "duration_s")]
    pub durata_s: f64,
    /// Total tokens / run duration (including prefill).
    #[serde(rename = "tok_s_total")]
    pub tok_s_aggregato: f64,
    /// Tokens generated after all conversations have their first token / time until the last token.
    #[serde(rename = "tok_s_generation")]
    pub tok_s_generazione: f64,
    /// Per conversation: (n_token - 1) / (last token time - first token time).
    #[serde(rename = "tok_s_conv_median")]
    pub tok_s_conv_mediana: f64,
    pub tok_s_conv_p5: f64,
    pub tok_s_conv_min: f64,
    #[serde(rename = "ttft_ms_median")]
    pub ttft_ms_mediana: f64,
    pub ttft_ms_max: f64,
    #[serde(rename = "per_conversation")]
    pub per_conversazione: Vec<RiepConv>,
}

pub fn summary(modo: &str, recs: &[RecordConv], durata_s: f64) -> Riepilogo {
    let mut per = Vec::new();
    for r in recs {
        let n = r.token.len();
        let tok_s = if n >= 2 {
            let dt = (r.tempi_ms[n - 1] - r.tempi_ms[0]) / 1000.0;
            if dt > 0.0 { (n - 1) as f64 / dt } else { 0.0 }
        } else {
            0.0
        };
        per.push(RiepConv { prompt_id: r.prompt_id.clone(), n_token: n, tok_s, ttft_ms: r.ttft_ms });
    }
    let token_totali: usize = recs.iter().map(|r| r.token.len()).sum();
    let t_gen0 = recs.iter().filter_map(|r| r.tempi_ms.first()).cloned().fold(0.0, f64::max);
    let t_gen1 = recs.iter().filter_map(|r| r.tempi_ms.last()).cloned().fold(0.0, f64::max);
    let gen_tok: usize = recs.iter().map(|r| r.token.len().saturating_sub(1)).sum();
    let tok_s_generazione = if t_gen1 > t_gen0 { gen_tok as f64 / ((t_gen1 - t_gen0) / 1000.0) } else { 0.0 };
    let velocita: Vec<f64> = per.iter().map(|p| p.tok_s).collect();
    let ttft: Vec<f64> = per.iter().map(|p| p.ttft_ms).collect();
    let vuoto = recs.is_empty();
    Riepilogo {
        modo: modo.to_string(),
        n_conversazioni: recs.len(),
        token_totali,
        durata_s,
        tok_s_aggregato: if durata_s > 0.0 { token_totali as f64 / durata_s } else { 0.0 },
        tok_s_generazione,
        tok_s_conv_mediana: if vuoto { 0.0 } else { percentile(&velocita, 50.0) },
        tok_s_conv_p5: if vuoto { 0.0 } else { percentile(&velocita, 5.0) },
        tok_s_conv_min: velocita.iter().cloned().fold(f64::INFINITY, f64::min),
        ttft_ms_mediana: if vuoto { 0.0 } else { percentile(&ttft, 50.0) },
        ttft_ms_max: ttft.iter().cloned().fold(0.0, f64::max),
        per_conversazione: per,
    }
}

/// Group records by traffic class for latency, throughput, and suspension
/// metrics. Use arrival time for first-token latency when recorded, or
/// the conversation start otherwise.
pub fn riepilogo_classi(recs: &[RecordConv], durata_s: f64) -> serde_json::Value {
    let mut out = serde_json::Map::new();
    let mut sosp_tot = 0u64;
    for class in ["interactive", "batch"] {
        let r: Vec<&RecordConv> = recs.iter().filter(|r| r.class.as_deref() == Some(class)).collect();
        if r.is_empty() {
            continue;
        }
        let ttft: Vec<f64> = r.iter().filter_map(|x| x.ttft_arrivo_ms.or(if x.tempi_ms.is_empty() { None } else { Some(x.ttft_ms) })).collect();
        let vel: Vec<f64> = r
            .iter()
            .filter(|x| x.tempi_ms.len() >= 2)
            .filter_map(|x| {
                let n = x.tempi_ms.len();
                let dt = (x.tempi_ms[n - 1] - x.tempi_ms[0]) / 1000.0;
                (dt > 0.0).then(|| (n - 1) as f64 / dt)
            })
            .collect();
        let token: usize = r.iter().map(|x| x.token.len()).sum();
        let sosp: u64 = r.iter().map(|x| x.sospensioni.unwrap_or(0) as u64).sum();
        sosp_tot += sosp;
        let med = |v: &[f64], p: f64| if v.is_empty() { serde_json::Value::Null } else { serde_json::json!(percentile(v, p)) };
        out.insert(
            class.to_string(),
            serde_json::json!({
                "conversations": r.len(),
                "token": token,
                "tok_s": if durata_s > 0.0 { token as f64 / durata_s } else { 0.0 },
                "ttft_ms_median": med(&ttft, 50.0),
                "ttft_ms_max": if ttft.is_empty() { serde_json::Value::Null } else { serde_json::json!(ttft.iter().cloned().fold(0.0, f64::max)) },
                "tok_s_conv_median": med(&vel, 50.0),
                "tok_s_conv_p5": med(&vel, 5.0),
                "suspensions": sosp,
            }),
        );
    }
    out.insert("total_suspensions".into(), serde_json::json!(sosp_tot));
    serde_json::Value::Object(out)
}

#[derive(Debug, PartialEq)]
pub struct ConfrontoConv {
    pub prompt_id: String,
    pub token_identici: bool,
    /// First step (generated token index) where tokens differ or one output ends early.
    pub primo_passo_diverso: Option<usize>,
    pub scarto_max: f64,
    pub scarto_somma: f64,
    pub n_logprob: usize,
    pub bit_identici: bool,
}

#[derive(Debug)]
pub struct Confronto {
    pub per_conv: Vec<ConfrontoConv>,
}

impl Confronto {
    pub fn token_identici(&self) -> bool {
        self.per_conv.iter().all(|c| c.token_identici)
    }
    pub fn bit_identici(&self) -> bool {
        self.per_conv.iter().all(|c| c.bit_identici)
    }
    pub fn primo_passo_diverso(&self) -> Option<(String, usize)> {
        self.per_conv.iter().filter_map(|c| c.primo_passo_diverso.map(|p| (c.prompt_id.clone(), p))).min_by_key(|x| x.1)
    }
    /// Conversations whose tokens diverge. With a tolerance, compare log probabilities only before divergence.
    pub fn n_token_diversi(&self) -> usize {
        self.per_conv.iter().filter(|c| !c.token_identici).count()
    }
    /// Max log-probability difference is within `tol` (nats). For token differences see `n_token_diversi`.
    pub fn entro_tolleranza(&self, tol: f64) -> bool {
        self.scarto_max() <= tol
    }
    pub fn scarto_max(&self) -> f64 {
        self.per_conv.iter().map(|c| c.scarto_max).fold(0.0, f64::max)
    }
    pub fn scarto_medio(&self) -> f64 {
        let (s, n) = self.per_conv.iter().fold((0.0, 0usize), |(s, n), c| (s + c.scarto_somma, n + c.n_logprob));
        if n == 0 { 0.0 } else { s / n as f64 }
    }
}

/// Compare outputs by `prompt_id`. Compare log probabilities only before the first differing token,
/// since contexts differ after that.
pub fn compare(a: &[RecordConv], b: &[RecordConv]) -> Result<Confronto, String> {
    let mb: BTreeMap<&str, &RecordConv> = b.iter().map(|r| (r.prompt_id.as_str(), r)).collect();
    let mut per_conv = Vec::new();
    for ra in a {
        let rb = mb.get(ra.prompt_id.as_str()).ok_or_else(|| format!("prompt {} is missing from the second transcript", ra.prompt_id))?;
        let n = ra.token.len().min(rb.token.len());
        let primo = (0..n).find(|&i| ra.token[i] != rb.token[i]).or(if ra.token.len() != rb.token.len() { Some(n) } else { None });
        let done = primo.unwrap_or(n).min(n);
        let (mut max, mut somma, mut bit) = (0.0f64, 0.0f64, true);
        for i in 0..done {
            let d = (ra.logprob[i] as f64 - rb.logprob[i] as f64).abs();
            max = max.max(d);
            somma += d;
            bit &= ra.logprob_bit[i] == rb.logprob_bit[i];
        }
        per_conv.push(ConfrontoConv {
            prompt_id: ra.prompt_id.clone(),
            token_identici: primo.is_none(),
            primo_passo_diverso: primo,
            scarto_max: max,
            scarto_somma: somma,
            n_logprob: done,
            bit_identici: primo.is_none() && bit,
        });
    }
    if per_conv.len() != b.len() {
        return Err(format!("the transcripts have different conversation counts ({} vs {})", a.len(), b.len()));
    }
    Ok(Confronto { per_conv })
}

/// Compare matching token sequences across every position. To supply the
/// same context at every position, force b on the tokens in a.
#[derive(Debug)]
pub struct ScartiTutti {
    pub n_posizioni: usize,
    pub medio: f64,
    /// Nearest-rank p99.9: value at position ceil(0.999 n) in ascending order.
    pub p999: f64,
    pub max: f64,
    pub bit_identici: bool,
    /// (prompt_id, positions, mean difference, max difference) per conversation.
    pub per_conv: Vec<(String, usize, f64, f64)>,
}

pub fn scarti_tutte_le_posizioni(a: &[RecordConv], b: &[RecordConv]) -> Result<ScartiTutti, String> {
    let mb: BTreeMap<&str, &RecordConv> = b.iter().map(|r| (r.prompt_id.as_str(), r)).collect();
    if a.len() != b.len() {
        return Err(format!("the transcripts have different conversation counts ({} vs {})", a.len(), b.len()));
    }
    let (mut tutti, mut per_conv, mut bit) = (Vec::new(), Vec::new(), true);
    for ra in a {
        let rb = mb.get(ra.prompt_id.as_str()).ok_or_else(|| format!("prompt {} is missing from the second transcript", ra.prompt_id))?;
        if ra.token != rb.token {
            return Err(format!(
                "{}: transcript tokens differ; replay the first transcript to produce the second",
                ra.prompt_id
            ));
        }
        let (mut max, mut somma) = (0.0f64, 0.0f64);
        for i in 0..ra.token.len() {
            let d = (ra.logprob[i] as f64 - rb.logprob[i] as f64).abs();
            max = max.max(d);
            somma += d;
            bit &= ra.logprob_bit[i] == rb.logprob_bit[i];
            tutti.push(d);
        }
        let n = ra.token.len();
        per_conv.push((ra.prompt_id.clone(), n, if n == 0 { 0.0 } else { somma / n as f64 }, max));
    }
    if tutti.is_empty() {
        return Err("no positions to compare".into());
    }
    let n = tutti.len();
    let medio = tutti.iter().sum::<f64>() / n as f64;
    let max = tutti.iter().cloned().fold(0.0, f64::max);
    tutti.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let rango = ((0.999 * n as f64).ceil() as usize).clamp(1, n);
    Ok(ScartiTutti { n_posizioni: n, medio, p999: tutti[rango - 1], max, bit_identici: bit, per_conv })
}

#[cfg(test)]
mod test {
    use super::*;

    fn rec(id: &str, t: &[i32], lp: &[f32]) -> RecordConv {
        RecordConv {
            prompt_id: id.into(),
            token: t.to_vec(),
            logprob: lp.to_vec(),
            logprob_bit: lp.iter().map(|x| x.to_bits()).collect(),
            tempi_ms: (0..t.len()).map(|i| 1000.0 + i as f64 * 100.0).collect(),
            ttft_ms: 500.0,
            ..Default::default()
        }
    }

    /// Verify both zero-default decoding and round-trip of nonzero sampling cutoffs.
    #[test]
    fn record_sampling_cutoffs_round_trip() {
        let r = rec("p0", &[1, 2], &[-0.5, -0.25]);
        let j = serde_json::to_string(&r).unwrap();
        assert!(!j.contains("top_p") && !j.contains("top_k"));
        let letto: RecordConv = serde_json::from_str(&j).unwrap();
        assert_eq!((letto.top_p, letto.top_k), (None, None));
        let mut t = r.clone();
        t.top_p = Some(0.9);
        t.top_k = Some(40);
        let j2 = serde_json::to_string(&t).unwrap();
        assert!(j2.contains("\"top_p\"") && j2.contains("\"top_k\":40"));
        assert_eq!(serde_json::from_str::<RecordConv>(&j2).unwrap(), t);
    }

    /// Check that records without arrival tracking omit its timing fields.
    #[test]
    fn admission_fields_are_optional() {
        let r = rec("p0", &[1, 2], &[-0.5, -0.25]);
        let j = serde_json::to_string(&r).unwrap();
        assert!(!j.contains("arrival_ms") && !j.contains("queue_wait_ms") && !j.contains("arrival_ttft_ms"));
        let vecchio: RecordConv = serde_json::from_str(&j).unwrap();
        assert_eq!(vecchio.arrivo_ms, None);
        let mut n = r.clone();
        n.arrivo_ms = Some(10.0);
        n.attesa_coda_ms = Some(2.0);
        n.ttft_arrivo_ms = Some(30.0);
        let j2 = serde_json::to_string(&n).unwrap();
        assert_eq!(serde_json::from_str::<RecordConv>(&j2).unwrap(), n);
    }

    #[test]
    fn summary_by_class() {
        let mut a = rec("a", &[1, 2, 3], &[0.0; 3]);
        a.class = Some("interactive".into());
        a.sospensioni = Some(0);
        a.ttft_arrivo_ms = Some(300.0);
        let mut b = rec("b", &[1, 2, 3, 4, 5], &[0.0; 5]);
        b.class = Some("batch".into());
        b.sospensioni = Some(2);
        let mut c = rec("c", &[1, 2], &[0.0; 2]);
        c.class = Some("batch".into());
        c.sospensioni = Some(1);
        let j = riepilogo_classi(&[a, b, c], 2.0);
        assert_eq!(j["interactive"]["conversations"], 1);
        assert_eq!(j["interactive"]["ttft_ms_median"], 300.0);
        assert_eq!(j["batch"]["conversations"], 2);
        assert_eq!(j["batch"]["token"], 7);
        assert_eq!(j["batch"]["suspensions"], 3);
        assert_eq!(j["total_suspensions"], 3);
        assert!((j["batch"]["tok_s"].as_f64().unwrap() - 3.5).abs() < 1e-9);
        assert_eq!(riepilogo_classi(&[rec("x", &[1], &[0.0])], 1.0)["total_suspensions"], 0);
    }

    #[test]
    fn percentiles() {
        let v = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(percentile(&v, 50.0), 3.0);
        assert_eq!(percentile(&v, 0.0), 1.0);
        assert!((percentile(&v, 5.0) - 1.2).abs() < 1e-12);
    }

    #[test]
    fn summary_counts_tokens_per_second() {
        let r = vec![rec("a", &[1, 2, 3, 4, 5], &[0.0; 5]), rec("b", &[1, 2, 3], &[0.0; 3])];
        let s = summary("test", &r, 2.0);
        assert_eq!(s.token_totali, 8);
        assert!((s.tok_s_aggregato - 4.0).abs() < 1e-9);
        // a: 4 tokens in 0.4 s = 10 tok/s. b: 2 in 0.2 s = 10 tok/s.
        assert!((s.tok_s_conv_mediana - 10.0).abs() < 1e-9);
        assert!((s.ttft_ms_mediana - 500.0).abs() < 1e-9);
    }

    #[test]
    fn compares_identical_transcripts() {
        let a = vec![rec("p0", &[1, 2, 3], &[-0.5, -1.0, -2.0])];
        let c = compare(&a, &a.clone()).unwrap();
        assert!(c.token_identici() && c.bit_identici());
        assert_eq!(c.scarto_max(), 0.0);
        assert_eq!(c.primo_passo_diverso(), None);
    }

    #[test]
    fn compares_drift_and_divergence() {
        let a = vec![rec("p0", &[1, 2, 3, 4], &[-0.5, -1.0, -2.0, -3.0])];
        let b = vec![rec("p0", &[1, 2, 9, 4], &[-0.5, -1.25, -2.5, -3.5])];
        let c = compare(&a, &b).unwrap();
        assert!(!c.token_identici() && !c.bit_identici());
        assert_eq!(c.primo_passo_diverso(), Some(("p0".to_string(), 2)));
        assert_eq!(c.scarto_max(), 0.25);
        assert_eq!(c.scarto_medio(), 0.125);
    }

    #[test]
    fn single_bit_logprob_difference_breaks_identity() {
        let x = -1.0f32;
        let y = f32::from_bits(x.to_bits() + 1);
        let c = compare(&[rec("p", &[1], &[x])], &[rec("p", &[1], &[y])]).unwrap();
        assert!(c.token_identici());
        assert!(!c.bit_identici());
    }

    #[test]
    fn compares_with_tolerance() {
        let c = compare(&[rec("p", &[1, 2], &[-1.0, -2.0])], &[rec("p", &[1, 2], &[-1.0, -2.5])]).unwrap();
        assert!(!c.bit_identici());
        assert!(c.entro_tolleranza(0.5));
        assert!(!c.entro_tolleranza(0.4));
        assert_eq!(c.n_token_diversi(), 0);
        // After token divergence, count the token mismatch and exclude later scores.
        let d = compare(&[rec("p", &[1, 2], &[-1.0, -2.0])], &[rec("p", &[1, 3], &[-1.1, -9.0])]).unwrap();
        assert!(d.entro_tolleranza(0.2));
        assert_eq!(d.n_token_diversi(), 1);
    }

    #[test]
    fn different_lengths_count_as_divergence() {
        let c = compare(&[rec("p", &[1, 2], &[0.0, 0.0])], &[rec("p", &[1], &[0.0])]).unwrap();
        assert_eq!(c.primo_passo_diverso(), Some(("p".into(), 1)));
    }

    #[test]
    fn jsonl_roundtrip() {
        let r = vec![rec("p0", &[1, 2], &[-0.1, -0.2])];
        let p = std::env::temp_dir().join(format!("stage-uscita-{}.jsonl", std::process::id()));
        scrivi_jsonl(p.to_str().unwrap(), &r).unwrap();
        let l = leggi_jsonl(p.to_str().unwrap()).unwrap();
        std::fs::remove_file(p).ok();
        assert_eq!(l, r);
    }

    #[test]
    fn differences_across_all_positions() {
        let a = vec![rec("p0", &[1, 2, 3, 4], &[-0.5, -1.0, -2.0, -3.0]), rec("p1", &[7, 8], &[-1.0, -1.0])];
        let b = vec![rec("p0", &[1, 2, 3, 4], &[-0.5, -1.25, -2.5, -3.0]), rec("p1", &[7, 8], &[-1.0, -1.0])];
        let s = scarti_tutte_le_posizioni(&a, &b).unwrap();
        assert_eq!(s.n_posizioni, 6);
        assert!((s.medio - 0.75 / 6.0).abs() < 1e-12);
        assert_eq!(s.max, 0.5);
        assert_eq!(s.p999, 0.5); 
        assert!(!s.bit_identici);
        let id = scarti_tutte_le_posizioni(&a, &a.clone()).unwrap();
        assert!(id.bit_identici && id.max == 0.0 && id.medio == 0.0);
        let c = vec![rec("p0", &[1, 2, 9, 4], &[0.0; 4]), rec("p1", &[7, 8], &[0.0; 2])];
        assert!(scarti_tutte_le_posizioni(&a, &c).is_err());
    }
}
