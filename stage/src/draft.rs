//! Drafting puts consecutive rows of one conversation in a group step. The
//! head proposes continuation tokens from prompt lookup or from the tail.
//! A row is valid only when its input matches the sample from the previous
//! valid row. The tail uses the same per-step seed for draft verification
//! and ordinary generation. Remove rejected rows from context before decoding
//! their replacement tokens.

pub const N_MAX: usize = 4;
pub const SOGLIA_MEDIA: f64 = 0.15;
pub const PASSI_PAUSA: u32 = 16;
const PESO: f64 = 0.3;
/// The initial average lets drafting survive a few misses before pausing.
const MEDIA_INIZIALE: f64 = 1.0;
const MEDIA_DOPO_PAUSA: f64 = 0.5;

/// Match the longest available suffix against earlier tokens, preferring
/// the most recent occurrence at the same length. Return up to k tokens
/// that followed the match. No match or a zero limit gives an empty draft.
pub fn prompt_lookup(storia: &[i32], k: usize, n_max: usize) -> Vec<i32> {
    let len = storia.len();
    if k == 0 || len < 2 {
        return Vec::new();
    }
    for n in (1..=n_max.min(len - 1)).rev() {
        let suffisso = &storia[len - n..];
        // Start before the suffix so lookup cannot match it against itself.
        for i in (0..len - n).rev() {
            if &storia[i..i + n] == suffisso {
                let da = i + n;
                let a = (da + k).min(len);
                if a > da {
                    return storia[da..a].to_vec();
                }
            }
        }
    }
    Vec::new()
}

#[derive(Debug, Clone)]
pub struct DraftState {
    k_max: usize,
    k_cor: usize,
    media: f64,
    pausa: u32,
}

impl DraftState {
    pub fn new(k_max: usize) -> Self {
        DraftState { k_max, k_cor: k_max, media: MEDIA_INIZIALE, pausa: 0 }
    }

    /// Return the draft size for this step. A paused conversation returns zero
    /// and consumes one step of its pause budget.
    pub fn k_da_proporre(&mut self) -> usize {
        if self.k_max == 0 {
            return 0;
        }
        if self.pausa > 0 {
            self.pausa -= 1;
            return 0;
        }
        self.k_cor
    }

    /// Update the moving acceptance average. Double the depth after full
    /// acceptance, or try one beyond the accepted prefix after a partial match.
    /// Cap the depth at k_max. Sustained low acceptance pauses drafting.
    pub fn esito(&mut self, proposti: usize, accettati: usize) {
        if proposti == 0 || self.k_max == 0 {
            return;
        }
        let x = accettati.min(proposti) as f64;
        self.media = (1.0 - PESO) * self.media + PESO * x;
        if self.media < SOGLIA_MEDIA {
            self.pausa = PASSI_PAUSA;
            self.k_cor = 1;
            self.media = MEDIA_DOPO_PAUSA;
        } else if accettati >= proposti {
            self.k_cor = (self.k_cor * 2).clamp(1, self.k_max);
        } else {
            self.k_cor = (accettati + 1).clamp(1, self.k_max);
        }
    }

    pub fn in_pausa(&self) -> bool {
        self.pausa > 0
    }

    /// Read the depth for the next tail proposal without advancing the pause.
    pub fn k_corrente(&self) -> usize {
        self.k_cor
    }
}

/// Share a budget of extra rows by draft depth, visiting conversations in
/// order at each depth. A deeper row can pay off only if every earlier
/// proposal for the conversation is accepted.
pub fn distribute_rows(voluti: &[usize], mut extra: usize) -> Vec<usize> {
    let mut dati = vec![0; voluti.len()];
    let massimo = voluti.iter().copied().max().unwrap_or(0);
    for profondita in 1..=massimo {
        for (i, &v) in voluti.iter().enumerate() {
            if extra == 0 {
                return dati;
            }
            if v >= profondita {
                dati[i] = profondita;
                extra -= 1;
            }
        }
    }
    dati
}

/// Count valid rows in one conversation's step. Row zero always counts.
/// A subsequent row counts only when its input equals the previous sample
/// and that sample is not an end marker. The last valid row still carries
/// its sample even if it invalidates the next proposal. The head applies
/// response length and stop limits before recording the tokens.
pub fn accepted_rows(campionati: &[i32], ingressi: &[i32], eog: impl Fn(i32) -> bool) -> usize {
    if campionati.is_empty() {
        return 0;
    }
    let mut n = 1;
    while n < campionati.len() && !eog(campionati[n - 1]) && ingressi.get(n) == Some(&campionati[n - 1]) {
        n += 1;
    }
    n
}

/// Count consecutive positions and logical steps for the conversation
/// starting at da, including its mandatory row.
pub fn group_length(rows: &[(u64, u32, u32)], da: usize) -> usize {
    let (c, s, p) = rows[da];
    let mut n = 1;
    while da + n < rows.len() && rows[da + n] == (c, s + n as u32, p + n as u32) {
        n += 1;
    }
    n
}

/// Return the number of cached positions to discard when an incoming row
/// starts before the expected position. The slot allocator checks the maximum rewind.
pub fn rewind_boundary(timeout: u32, pos: u32) -> Option<u32> {
    (pos < timeout).then(|| timeout - pos)
}

// Draft state shared by the head and the tail.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DraftSource {
    /// Look up a continuation in the head's token history.
    #[default]
    Ngram,
    /// Use the continuation returned by the tail.
    Mtp,
}

impl DraftSource {
    pub fn da_testo(t: &str) -> Result<Self, String> {
        match t {
            "ngram" => Ok(DraftSource::Ngram),
            "mtp" => Ok(DraftSource::Mtp),
            altro => Err(format!("--draft-source: invalid value {altro} (ngram or mtp)")),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            DraftSource::Ngram => "ngram",
            DraftSource::Mtp => "mtp",
        }
    }
}

/// Cache action needed before appending verified rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MtpPlan {
    Aggiungi,
    /// Discard the speculative suffix before reusing its positions.
    Riavvolgi(u32),
    /// A gap cannot be filled by appending these rows. Disable drafting for this conversation.
    Buco,
}

/// Compare an incoming position against the cached half-open range [0, prossima).
pub fn piano_mtp(prossima: u32, pos: u32) -> MtpPlan {
    match pos.cmp(&prossima) {
        std::cmp::Ordering::Equal => MtpPlan::Aggiungi,
        std::cmp::Ordering::Less => MtpPlan::Riavvolgi(pos),
        std::cmp::Ordering::Greater => MtpPlan::Buco,
    }
}

/// Use a proposal only while the transcript is still at the length it had
/// when the proposal arrived. Limit it to k tokens and stop at the first
/// invalid vocabulary ID because the later tokens depend on it.
pub fn bozza_mtp(ricevuta: Option<(usize, &[i32])>, registrati: usize, k: usize, n_vocab: usize) -> Vec<i32> {
    match ricevuta {
        Some((n, d)) if n == registrati => d.iter().take(k).copied().take_while(|&t| t >= 0 && (t as usize) < n_vocab).collect(),
        _ => Vec::new(),
    }
}

/// The last histogram bin combines depths six and above.
pub const PROFONDITA_MAX: usize = 6;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ContatoriBozza {
    pub proposte: u64,
    pub token_proposti: u64,
    pub token_accettati: u64,
    /// Rows computed despite rejection of their draft prefix.
    pub righe_buttate: u64,
    pub passi_senza_bozza: u64,
    pub passi: u64,
    pub token_per_passo_somma: u64,
    /// Bin zero represents the first proposal in the step. A deeper acceptance
    /// also implies acceptance of every preceding proposal.
    pub prof_proposti: [u64; PROFONDITA_MAX],
    pub prof_accettati: [u64; PROFONDITA_MAX],
}

impl ContatoriBozza {
    pub fn proposta(&mut self, k: usize) {
        self.proposte += 1;
        self.token_proposti += k as u64;
    }

    pub fn esito(&mut self, proposti: usize, righe_tornate: usize) {
        let acc = righe_tornate.saturating_sub(1).min(proposti);
        self.token_accettati += acc as u64;
        self.righe_buttate += (proposti - acc) as u64;
        for d in 0..proposti {
            let cella = d.min(PROFONDITA_MAX - 1);
            self.prof_proposti[cella] += 1;
            if d < acc {
                self.prof_accettati[cella] += 1;
            }
        }
    }

    pub fn passo_completato(&mut self, token: usize) {
        self.passi += 1;
        self.token_per_passo_somma += token as u64;
    }

    pub fn json(&self) -> serde_json::Value {
        let frac = |a: u64, b: u64| if b == 0 { serde_json::Value::Null } else { serde_json::json!(a as f64 / b as f64) };
        // quota divides accepted by sent at the same depth. condizionata divides
        // by acceptance at the preceding depth to show the additional yield.
        let per_profondita: Vec<serde_json::Value> = (0..PROFONDITA_MAX)
            .filter(|&d| self.prof_proposti[d] > 0)
            .map(|d| {
                serde_json::json!({
                    "depth": d + 1,
                    "proposed": self.prof_proposti[d],
                    "accepted": self.prof_accettati[d],
                    "acceptance_rate": frac(self.prof_accettati[d], self.prof_proposti[d]),
                    "conditional_rate": if d == 0 { frac(self.prof_accettati[d], self.prof_proposti[d]) } else { frac(self.prof_accettati[d], self.prof_accettati[d - 1]) },
                })
            })
            .collect();
        serde_json::json!({
            "by_depth": per_profondita,
            "proposals": self.proposte,
            "proposed_tokens": self.token_proposti,
            "accepted_tokens": self.token_accettati,
            "discarded_rows": self.righe_buttate,
            "steps_without_drafts": self.passi_senza_bozza,
            "steps": self.passi,
            "accepted_per_proposal": frac(self.token_accettati, self.proposte),
            "acceptance_rate": frac(self.token_accettati, self.token_proposti),
            "tokens_per_step": frac(self.token_per_passo_somma, self.passi),
        })
    }
}

#[derive(Debug, Clone)]
pub struct BozzaConv {
    pub state: DraftState,
    /// Zero means the in-flight step contains only its mandatory row.
    pub in_volo: usize,
    /// The next row position expected by the engine. A lower position means
    /// the caller must rewind the speculative suffix.
    pub calcolata: u32,
    pub cont: ContatoriBozza,
    /// Supply known continuation tokens to exercise acceptance and rewind even
    /// with random fixture weights. The corruption interval injects mismatches.
    pub oracolo: Option<Vec<i32>>,
    /// Tail proposal paired with the recorded transcript length that makes it
    /// valid. See bozza_mtp for the freshness check.
    pub mtp: Option<(usize, Vec<i32>)>,
}

impl BozzaConv {
    pub fn new(k_max: usize) -> Self {
        BozzaConv { state: DraftState::new(k_max), in_volo: 0, calcolata: 0, cont: ContatoriBozza::default(), oracolo: None, mtp: None }
    }
}

/// Take up to k fixture tokens at index n. When rovina is positive, perturb
/// tokens at positions divisible by it.
pub fn draft_oracle(oracolo: &[i32], n: usize, k: usize, rovina: u32) -> Vec<i32> {
    (n..(n + k).min(oracolo.len())).map(|j| if rovina > 0 && j % rovina as usize == 0 { oracolo[j].wrapping_add(1) } else { oracolo[j] }).collect()
}

impl ContatoriBozza {
    pub fn somma(&mut self, a: &ContatoriBozza) {
        self.proposte += a.proposte;
        self.token_proposti += a.token_proposti;
        self.token_accettati += a.token_accettati;
        self.righe_buttate += a.righe_buttate;
        self.passi_senza_bozza += a.passi_senza_bozza;
        self.passi += a.passi;
        self.token_per_passo_somma += a.token_per_passo_somma;
        for d in 0..PROFONDITA_MAX {
            self.prof_proposti[d] += a.prof_proposti[d];
            self.prof_accettati[d] += a.prof_accettati[d];
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn parses_draft_source() {
        assert_eq!(DraftSource::da_testo("ngram").unwrap(), DraftSource::Ngram);
        assert_eq!(DraftSource::da_testo("mtp").unwrap(), DraftSource::Mtp);
        assert!(DraftSource::da_testo("unknown").unwrap_err().contains("ngram or mtp"));
        assert_eq!(DraftSource::default(), DraftSource::Ngram);
        assert_eq!(DraftSource::Mtp.name(), "mtp");
    }

    #[test]
    fn mtp_cache_continues_rewinds_or_disables() {
        assert_eq!(piano_mtp(21, 21), MtpPlan::Aggiungi);
        assert_eq!(piano_mtp(21, 20), MtpPlan::Riavvolgi(20));
        assert_eq!(piano_mtp(21, 0), MtpPlan::Riavvolgi(0));
        // A skipped interval prevents contiguous draft context.
        assert_eq!(piano_mtp(21, 23), MtpPlan::Buco);
        assert_eq!(piano_mtp(0, 0), MtpPlan::Aggiungi);
    }

    #[test]
    fn mtp_draft_is_valid_only_for_next_step() {
        assert_eq!(bozza_mtp(Some((5, &[42])), 5, 1, 100), vec![42]);
        assert_eq!(bozza_mtp(Some((5, &[42, 7, 9])), 5, 2, 100), vec![42, 7]);
        assert_eq!(bozza_mtp(Some((5, &[42, 7, 9])), 5, 4, 100), vec![42, 7, 9], "no more than the tail proposed");
        assert_eq!(bozza_mtp(Some((5, &[42, 7, 9])), 5, 1, 100), vec![42]);
        assert!(bozza_mtp(Some((5, &[42])), 5, 0, 100).is_empty());
        assert!(bozza_mtp(Some((5, &[42])), 6, 1, 100).is_empty());
        assert!(bozza_mtp(None, 5, 1, 100).is_empty());
        assert!(bozza_mtp(Some((5, &[100])), 5, 1, 100).is_empty());
        assert!(bozza_mtp(Some((5, &[-1])), 5, 1, 100).is_empty());
        assert_eq!(bozza_mtp(Some((5, &[42, 100, 9])), 5, 3, 100), vec![42]);
    }

    #[test]
    fn draft_rows_are_allocated_one_depth_at_a_time() {
        assert_eq!(distribute_rows(&[3, 2, 3], 8), vec![3, 2, 3]);
        assert_eq!(distribute_rows(&[3, 2, 3], 7), vec![3, 2, 2]);
        assert_eq!(distribute_rows(&[3, 2, 3], 5), vec![2, 2, 1]);
        assert_eq!(distribute_rows(&[3, 2, 3], 3), vec![1, 1, 1]);
        assert_eq!(distribute_rows(&[3, 2, 3], 2), vec![1, 1, 0]);
        assert_eq!(distribute_rows(&[3, 2, 3], 0), vec![0, 0, 0]);
        assert_eq!(distribute_rows(&[0, 2, 0], 5), vec![0, 2, 0]);
        assert!(distribute_rows(&[], 5).is_empty());
        assert_eq!(distribute_rows(&[1, 1, 1, 1], 3), vec![1, 1, 1, 0]);
    }

    #[test]
    fn draft_counters_track_each_depth() {
        let mut c = ContatoriBozza::default();
        c.proposta(3);
        c.esito(3, 3);
        c.proposta(3);
        c.esito(3, 1);
        c.proposta(1);
        c.esito(1, 2);
        assert_eq!(c.prof_proposti[..3], [3, 2, 2]);
        assert_eq!(c.prof_accettati[..3], [2, 1, 0]);
        let j = c.json();
        assert_eq!(j["by_depth"][0]["acceptance_rate"], serde_json::json!(2.0 / 3.0));
        assert_eq!(j["by_depth"][1]["conditional_rate"], serde_json::json!(0.5));
        assert_eq!(j["by_depth"].as_array().unwrap().len(), 3);
        let mut somma = ContatoriBozza::default();
        somma.somma(&c);
        somma.somma(&c);
        assert_eq!(somma.prof_proposti[..3], [6, 4, 4]);
    }

    #[test]
    fn lookup_finds_longest_ngram_and_continuation() {
        // The longest match is the final four-token suffix. Its earlier occurrence supplies the draft.
        let s = [5, 1, 2, 3, 4, 9, 9, 7, 1, 2, 3, 4];
        assert_eq!(prompt_lookup(&s, 3, 4), vec![9, 9, 7]);
        assert_eq!(prompt_lookup(&s, 2, 4), vec![9, 9]);
        assert_eq!(prompt_lookup(&[1, 2, 3, 1, 2], 8, 4), vec![3, 1, 2]);
    }

    #[test]
    fn lookup_tries_lengths_four_to_one_and_skips_without_match() {
        let s = [8, 3, 4, 7, 8];
        assert_eq!(prompt_lookup(&s, 2, 4), vec![3, 4]);
        assert!(prompt_lookup(&[1, 2, 3, 4], 4, 4).is_empty());
        assert!(prompt_lookup(&[1, 1, 1], 0, 4).is_empty());
        assert!(prompt_lookup(&[1], 4, 4).is_empty());
        assert!(prompt_lookup(&[], 4, 4).is_empty());
    }

    #[test]
    fn lookup_prefers_latest_match_at_equal_length() {
        let s = [1, 2, 10, 1, 2, 20, 1, 2];
        assert_eq!(prompt_lookup(&s, 1, 2), vec![20]);
    }

    #[test]
    fn adaptive_depth_starts_at_max_waits_16_steps_and_retries_at_one() {
        let mut b = DraftState::new(4);
        assert_eq!(b.k_da_proporre(), 4);
        for _ in 0..5 {
            assert!(!b.in_pausa());
            b.esito(4, 0);
        }
        assert!(!b.in_pausa());
        b.esito(1, 0);
        assert!(b.in_pausa());
        for _ in 0..PASSI_PAUSA {
            assert_eq!(b.k_da_proporre(), 0);
        }
        assert_eq!(b.k_da_proporre(), 1);
        b.esito(1, 1);
        assert_eq!(b.k_da_proporre(), 2);
        b.esito(2, 2);
        assert_eq!(b.k_da_proporre(), 4);
        b.esito(4, 4);
        assert_eq!(b.k_da_proporre(), 4);
        b.esito(4, 1);
        assert_eq!(b.k_da_proporre(), 2);
    }

    #[test]
    fn zero_draft_limit_disables_proposals() {
        let mut b = DraftState::new(0);
        assert_eq!(b.k_da_proporre(), 0);
        b.esito(3, 0);
        assert!(!b.in_pausa());
        assert_eq!(b.k_da_proporre(), 0);
    }

    #[test]
    fn tail_accepts_matching_prefix() {
        let no_eog = |_| false;
        // Two proposals match the preceding sample. The first mismatch ends the prefix.
        assert_eq!(accepted_rows(&[5, 6, 99, 1], &[10, 5, 6, 7], no_eog), 3);
        // Full acceptance yields one extra sample from the final row.
        assert_eq!(accepted_rows(&[5, 6, 7, 1], &[10, 5, 6, 7], no_eog), 4);
        assert_eq!(accepted_rows(&[9, 6, 7, 1], &[10, 5, 6, 7], no_eog), 1);
        assert_eq!(accepted_rows(&[9], &[10], no_eog), 1);
        assert_eq!(accepted_rows(&[], &[], no_eog), 0);
    }

    #[test]
    fn tail_stops_at_eog_inside_draft() {
        // An end marker is recorded even when it prevents the next matching draft from being accepted.
        let eog = |t| t == 2;
        assert_eq!(accepted_rows(&[5, 2, 7, 1], &[10, 5, 2, 7], eog), 2);
        assert_eq!(accepted_rows(&[2, 2, 7], &[10, 2, 7], eog), 1);
    }

    #[test]
    fn rows_for_same_conversation_are_consecutive() {
        let r = [(1, 5, 20), (1, 6, 21), (1, 7, 22), (2, 3, 9), (3, 4, 8), (3, 5, 9)];
        assert_eq!(group_length(&r, 0), 3);
        assert_eq!(group_length(&r, 3), 1);
        assert_eq!(group_length(&r, 4), 2);
        let r2 = [(1, 5, 20), (1, 7, 21)];
        assert_eq!(group_length(&r2, 0), 1);
    }

    #[test]
    fn rewind_discards_positions_from_cutoff() {
        assert_eq!(rewind_boundary(15, 12), Some(3));
        assert_eq!(rewind_boundary(15, 14), Some(1));
        assert_eq!(rewind_boundary(15, 15), None);
        assert_eq!(rewind_boundary(15, 16), None);
    }

    #[test]
    fn oracle_proposes_next_tokens_and_corrupts_every_nth() {
        let o = [10, 11, 12, 13, 14, 15];
        assert_eq!(draft_oracle(&o, 2, 3, 0), vec![12, 13, 14]);
        assert_eq!(draft_oracle(&o, 4, 5, 0), vec![14, 15]);
        assert_eq!(draft_oracle(&o, 2, 3, 4), vec![12, 13, 15]);
        assert!(draft_oracle(&o, 6, 3, 0).is_empty());
    }

    #[test]
    fn draft_counters_track_proposals_acceptance_and_discarded_rows() {
        let mut c = ContatoriBozza::default();
        c.proposta(4);
        c.esito(4, 3);
        c.passo_completato(3);
        c.proposta(2);
        c.esito(2, 3);
        c.passo_completato(3);
        c.passo_completato(1);
        assert_eq!((c.proposte, c.token_proposti, c.token_accettati, c.righe_buttate), (2, 6, 4, 2));
        assert_eq!(c.json()["tokens_per_step"], serde_json::json!(7.0 / 3.0));
        assert_eq!(c.json()["acceptance_rate"], serde_json::json!(4.0 / 6.0));
    }
}
