//! Each stage reserves context memory by conversation and attempt. The
//! watermark is the greatest closed attempt, starting at zero. It blocks
//! late opens and rows from attempts that have already ended.
//!
//! A live sequence has one owner. Closing or expiring a slot releases the
//! sequence only if it still belongs to that conversation and attempt.
//! Each slot also tracks its next input position so duplicate rows can
//! be dropped before computation.

use std::cell::Cell;
use std::collections::HashMap;
use std::rc::Rc;
use std::time::{Duration, Instant};

/// Monotonic time since stage startup. Can be injected for tests.
pub trait Clock {
    fn adesso(&self) -> Duration;
}

pub struct SystemClock(Instant);

impl SystemClock {
    pub fn new() -> Self {
        Self(Instant::now())
    }
}

impl Default for SystemClock {
    fn default() -> Self {
        Self::new()
    }
}

impl Clock for SystemClock {
    fn adesso(&self) -> Duration {
        self.0.elapsed()
    }
}

/// Test clock, advanced by hand. Clones share the same time.
#[derive(Clone, Default)]
pub struct ManualClock(Rc<Cell<Duration>>);

impl ManualClock {
    pub fn avanza(&self, d: Duration) {
        self.0.set(self.0.get() + d);
    }
}

impl Clock for ManualClock {
    fn adesso(&self) -> Duration {
        self.0.get()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Deadlines {
    /// Reservation lifetime before the first prefill block.
    pub riserva: Duration,
    /// Idle lifetime after a slot has seen a prefill block or generation row.
    pub inattivo: Duration,
}

impl Default for Deadlines {
    fn default() -> Self {
        Self { riserva: Duration::from_secs(30), inattivo: Duration::from_secs(120) }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Status {
    Riservato,
    Attivo,
}

#[derive(Clone, Copy, Debug)]
struct Slot {
    tentativo: u32,
    /// Context memory sequence, chosen by the head.
    seq: u32,
    state: Status,
    scadenza: Duration,
    pos_attesa: u32,
}

#[derive(Default)]
struct Conv {
    watermark: u32,
    slot: Option<Slot>,
    /// Last touch time, used to age closed conversation watermarks.
    last: Duration,
}

/// Keep a closed watermark for longer than ordinary work and idle
/// timeouts. Once it is forgotten, a late row has no slot to update.
pub const FILIGRANA_TTL: Duration = Duration::from_secs(600);
/// Cap on remembered conversations, with or without slots. Drop the oldest closed ones above it.
pub const CONV_MAX: usize = 100_000;
/// How often `scadenze()` scans the map for cleanup, unless already over the cap.
const PULIZIA_OGNI: Duration = Duration::from_secs(10);

/// Slot the caller must clear from context memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Freed {
    pub seq: u32,
    pub tentativo: u32,
}

#[derive(Debug, PartialEq, Eq)]
pub enum OpenOutcome {
    /// Slot reserved. `libera` is the slot from an older attempt closed by this `Open`.
    Riservato { libera: Option<Freed> },
    /// Same attempt already reserved or active. Keep it and repeat the `Confirmation`.
    Duplicato,
    /// Attempt at or below the watermark, or older than the one in use.
    Scartato,
    /// The sequence still belongs to another live conversation (`da`). Keep state and memory intact
    /// and send an error `Confirmation` for the opening conversation.
    Occupato { da: u64 },
    /// An open with eviction claimed a sequence whose close was lost. Clear
    /// that sequence, plus the opener's previous slot if libera is present.
    Sfrattato { da: u64, da_tentativo: u32, libera: Option<Freed> },
}

#[derive(Debug, PartialEq, Eq)]
pub enum WorkOutcome {
    /// Compute and forward.
    Compute,
    /// Position already applied. Skip recomputing it.
    Duplicate,
    /// Closed or superseded attempt. Drop without computing.
    Discarded,
    /// Position beyond the expected one (`expected`).
    UnexpectedPosition { expected: u32 },
    /// No active slot for this attempt.
    NoActiveSlot,
}

#[derive(Debug, PartialEq, Eq)]
pub struct CloseOutcome {
    /// Slot to clear, if there was one with attempt ≤ c. Forward `Close` either way.
    pub libera: Option<Freed>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Scaduto {
    pub conversation_id: u64,
    pub liberato: Freed,
    pub era_riservato: bool,
}

pub struct Slots<C: Clock> {
    orologio: C,
    scadenze: Deadlines,
    conv: HashMap<u64, Conv>,
    /// Sequence → owning (conversation, attempt).
    proprietari: HashMap<u32, (u64, u32)>,
    ttl_filigrana: Duration,
    conv_max: usize,
    ultima_pulizia: Duration,
}

impl<C: Clock> Slots<C> {
    pub fn new(orologio: C, scadenze: Deadlines) -> Self {
        Self { orologio, scadenze, conv: HashMap::new(), proprietari: HashMap::new(), ttl_filigrana: FILIGRANA_TTL, conv_max: CONV_MAX, ultima_pulizia: Duration::ZERO }
    }

    /// Change the age limit and cap for remembered closed conversations. Defaults usually suffice.
    pub fn con_pulizia(mut self, ttl_filigrana: Duration, conv_max: usize) -> Self {
        self.ttl_filigrana = ttl_filigrana;
        self.conv_max = conv_max.max(1);
        self
    }

    /// Remembered conversations, with a slot or only a watermark.
    pub fn conversazioni_ricordate(&self) -> usize {
        self.conv.len()
    }

    /// Forget conversations **without a slot** untouched for `ttl_filigrana`. If the map is over its cap,
    /// remove the oldest closed ones until it fits. Leave conversations with a slot alone until that slot
    /// expires and they become closed.
    pub fn pulisci(&mut self) {
        let adesso = self.orologio.adesso();
        let ttl = self.ttl_filigrana;
        let sopra_tetto = self.conv.len() > self.conv_max;
        if !sopra_tetto && adesso.saturating_sub(self.ultima_pulizia) < PULIZIA_OGNI {
            return;
        }
        self.ultima_pulizia = adesso;
        self.conv.retain(|_, c| c.slot.is_some() || adesso.saturating_sub(c.last) < ttl);
        if self.conv.len() > self.conv_max {
            let mut chiuse: Vec<(Duration, u64)> = self.conv.iter().filter(|(_, c)| c.slot.is_none()).map(|(&i, c)| (c.last, i)).collect();
            chiuse.sort_unstable();
            let da_togliere = (self.conv.len() - self.conv_max).min(chiuse.len());
            for (_, i) in chiuse.into_iter().take(da_togliere) {
                self.conv.remove(&i);
            }
        }
    }

    pub fn orologio(&self) -> &C {
        &self.orologio
    }

    /// Conversation watermark (0 if never closed).
    pub fn watermark(&self, conversation_id: u64) -> u32 {
        self.conv.get(&conversation_id).map_or(0, |c| c.watermark)
    }

    /// Attempt and sequence of the slot in use, if any.
    pub fn slot(&self, conversation_id: u64) -> Option<(u32, u32)> {
        self.conv.get(&conversation_id).and_then(|c| c.slot).map(|p| (p.tentativo, p.seq))
    }

    /// Owner of sequence `seq` (conversation, attempt), if any.
    pub fn proprietario(&self, seq: u32) -> Option<(u64, u32)> {
        self.proprietari.get(&seq).copied()
    }

    pub fn posti_in_uso(&self) -> usize {
        self.proprietari.len()
    }

    /// `Open` that rejects a sequence still owned by another live conversation (`EsitoApri::Occupato`).
    pub fn open(&mut self, conversation_id: u64, tentativo: u32, seq: u32, start_position: u32) -> OpenOutcome {
        self.apri_interno(conversation_id, tentativo, seq, start_position, false)
    }

    /// Reserve a slot even if a lost close left its sequence marked busy.
    /// This relies on the head assigning one live conversation to a sequence
    /// and sending opens and closes in order on the same stream. Raise the
    /// displaced owner's watermark before recycling the sequence so late
    /// messages cannot compute on it.
    pub fn apri_con_sfratto(&mut self, conversation_id: u64, tentativo: u32, seq: u32, start_position: u32) -> OpenOutcome {
        self.apri_interno(conversation_id, tentativo, seq, start_position, true)
    }

    fn apri_interno(&mut self, conversation_id: u64, tentativo: u32, seq: u32, start_position: u32, sfratta: bool) -> OpenOutcome {
        let adesso = self.orologio.adesso();
        let riserva = self.scadenze.riserva;
        // Protect the other live owner's sequence, even from a new attempt of the opening conversation.
        if let Some(&(altra, _)) = self.proprietari.get(&seq) {
            let c = self.conv.get(&conversation_id);
            let scartato = c.is_some_and(|c| tentativo <= c.watermark);
            if altra != conversation_id && !scartato {
                // Evict only if `Open` would reserve a new slot (no slot, or one from an older attempt).
                let darebbe_posto = c.and_then(|c| c.slot).map_or(true, |p| p.tentativo < tentativo);
                if !sfratta || !darebbe_posto {
                    return OpenOutcome::Occupato { da: altra };
                }
                // Eviction: remove the old owner's slot and raise its watermark to its attempt.
                let da_tentativo = self.proprietari.get(&seq).map_or(0, |x| x.1);
                if let Some(v) = self.conv.get_mut(&altra) {
                    v.watermark = v.watermark.max(da_tentativo);
                    v.last = adesso;
                    if v.slot.is_some_and(|p| p.seq == seq) {
                        v.slot = None;
                    }
                }
                self.proprietari.remove(&seq);
                return match self.apri_interno(conversation_id, tentativo, seq, start_position, true) {
                    OpenOutcome::Riservato { libera } => OpenOutcome::Sfrattato { da: altra, da_tentativo, libera },
                    altro => altro,
                };
            }
        }
        let c = self.conv.entry(conversation_id).or_default();
        c.last = adesso;
        if tentativo <= c.watermark {
            return OpenOutcome::Scartato;
        }
        let new = Slot { tentativo, seq, state: Status::Riservato, scadenza: adesso + riserva, pos_attesa: start_position };
        match c.slot {
            None => {
                c.slot = Some(new);
                self.proprietari.insert(seq, (conversation_id, tentativo));
                OpenOutcome::Riservato { libera: None }
            }
            Some(p) if p.tentativo == tentativo => OpenOutcome::Duplicato,
            Some(p) if p.tentativo > tentativo => OpenOutcome::Scartato,
            Some(p) => {
                // Close the old slot before reserving it for the new attempt.
                c.watermark = c.watermark.max(p.tentativo);
                c.slot = Some(new);
                self.proprietari.remove(&p.seq);
                self.proprietari.insert(seq, (conversation_id, tentativo));
                OpenOutcome::Riservato { libera: Some(Freed { seq: p.seq, tentativo: p.tentativo }) }
            }
        }
    }

    /// A prefill block of `n_token` tokens starting at `start_position`.
    pub fn block(&mut self, conversation_id: u64, tentativo: u32, start_position: u32, n_token: u32) -> WorkOutcome {
        self.lavoro(conversation_id, tentativo, start_position, n_token, true)
    }

    /// A `Step` row: one input token at `position`.
    pub fn step(&mut self, conversation_id: u64, tentativo: u32, position: u32) -> WorkOutcome {
        self.lavoro(conversation_id, tentativo, position, 1, false)
    }

    /// Accept a backward step within riavvolgi_max as a draft rewind. Return
    /// its position along with the compute decision. The caller must truncate
    /// context from that position before decoding the row. A zero rewind
    /// limit uses the ordinary duplicate rules.
    pub fn passo_bozza(&mut self, conversation_id: u64, tentativo: u32, position: u32, riavvolgi_max: u32) -> (WorkOutcome, Option<u32>) {
        let (e, r) = self.lavoro_r(conversation_id, tentativo, position, 1, false, riavvolgi_max);
        (e, r.then_some(position))
    }

    fn lavoro(&mut self, conversation_id: u64, tentativo: u32, pos: u32, n: u32, e_blocco: bool) -> WorkOutcome {
        self.lavoro_r(conversation_id, tentativo, pos, n, e_blocco, 0).0
    }

    fn lavoro_r(&mut self, conversation_id: u64, tentativo: u32, pos: u32, n: u32, e_blocco: bool, riavvolgi_max: u32) -> (WorkOutcome, bool) {
        let mut riavvolto = false;
        let e = self.lavoro_i(conversation_id, tentativo, pos, n, e_blocco, riavvolgi_max, &mut riavvolto);
        (e, riavvolto)
    }

    #[allow(clippy::too_many_arguments)]
    fn lavoro_i(&mut self, conversation_id: u64, tentativo: u32, pos: u32, n: u32, e_blocco: bool, riavvolgi_max: u32, riavvolto: &mut bool) -> WorkOutcome {
        let adesso = self.orologio.adesso();
        let inattivo = self.scadenze.inattivo;
        let Some(c) = self.conv.get_mut(&conversation_id) else {
            return WorkOutcome::NoActiveSlot;
        };
        c.last = adesso;
        if tentativo <= c.watermark {
            return WorkOutcome::Discarded;
        }
        let Some(p) = c.slot.as_mut() else {
            return WorkOutcome::NoActiveSlot;
        };
        if tentativo < p.tentativo {
            return WorkOutcome::Discarded;
        }
        if tentativo > p.tentativo || (!e_blocco && p.state == Status::Riservato) {
            return WorkOutcome::NoActiveSlot;
        }
        if pos < p.pos_attesa {
            if riavvolgi_max > 0 && !e_blocco && p.pos_attesa - pos <= riavvolgi_max {
                p.pos_attesa = pos;
                *riavvolto = true;
            } else {
                return WorkOutcome::Duplicate;
            }
        }
        if pos > p.pos_attesa {
            return WorkOutcome::UnexpectedPosition { expected: p.pos_attesa };
        }
        p.state = Status::Attivo;
        p.pos_attesa += n;
        p.scadenza = adesso + inattivo;
        WorkOutcome::Compute
    }

    pub fn close(&mut self, conversation_id: u64, tentativo: u32) -> CloseOutcome {
        let adesso = self.orologio.adesso();
        let c = self.conv.entry(conversation_id).or_default();
        c.last = adesso;
        c.watermark = c.watermark.max(tentativo);
        let libera = match c.slot {
            Some(p) if p.tentativo <= tentativo => {
                c.slot = None;
                if self.proprietari.get(&p.seq).is_some_and(|&(a, _)| a == conversation_id) {
                    self.proprietari.remove(&p.seq);
                }
                Some(Freed { seq: p.seq, tentativo: p.tentativo })
            }
            _ => None,
        };
        CloseOutcome { libera }
    }

    /// Free expired slots and set their watermarks to the slot attempts. The caller clears memory and sends
    /// an error `Confirmation` to the head.
    pub fn scadenze(&mut self) -> Vec<Scaduto> {
        let scaduti = self.expire_except(|_, _| false);
        self.pulisci();
        scaduti
    }

    pub fn expire_except(&mut self, pending: impl Fn(u64, u32) -> bool) -> Vec<Scaduto> {
        let adesso = self.orologio.adesso();
        let mut v = Vec::new();
        // Closed conversations can outnumber occupied slots by orders of magnitude.
        self.proprietari.retain(|&seq, &mut (id, tentativo)| {
            let c = self.conv.get_mut(&id).unwrap();
            let p = c.slot.unwrap();
            if adesso < p.scadenza || pending(id, tentativo) {
                return true;
            }
            c.last = adesso;
            c.watermark = c.watermark.max(tentativo);
            c.slot = None;
            v.push(Scaduto {
                conversation_id: id,
                liberato: Freed { seq, tentativo },
                era_riservato: p.state == Status::Riservato,
            });
            false
        });
        v.sort_by_key(|s| s.conversation_id);
        v
    }
}

#[cfg(test)]
mod test {
    use super::*;

    const S: Duration = Duration::from_secs(1);

    fn slots() -> Slots<ManualClock> {
        Slots::new(ManualClock::default(), Deadlines { riserva: 30 * S, inattivo: 120 * S })
    }

    // Slot transitions, expiry, and attempt ordering.

    #[test]
    fn stale_watermarks_are_dropped_after_expiry() {
        let orologio = ManualClock::default();
        let mut p = Slots::new(orologio.clone(), Deadlines { riserva: 30 * S, inattivo: 120 * S }).con_pulizia(100 * S, 1000);
        p.open(1, 1, 3, 0);
        p.close(1, 1);
        p.open(2, 1, 4, 0);
        assert_eq!(p.conversazioni_ricordate(), 2);
        orologio.avanza(20 * S);
        p.scadenze();
        assert_eq!(p.conversazioni_ricordate(), 2);
        // A late discarded message refreshes the closed watermark.
        assert_eq!(p.step(1, 1, 0), WorkOutcome::Discarded);
        orologio.avanza(85 * S);
        let sc = p.scadenze();
        assert_eq!(sc.len(), 1);
        assert_eq!(p.watermark(1), 1);
        assert_eq!(p.watermark(2), 1);
        orologio.avanza(30 * S);
        p.scadenze();
        assert_eq!(p.conversazioni_ricordate(), 1);
        assert_eq!(p.watermark(2), 1);
        orologio.avanza(150 * S);
        p.scadenze();
        assert_eq!(p.conversazioni_ricordate(), 0);
        assert_eq!(p.watermark(1), 0);
        // Once a watermark is forgotten, a late row reports that it has no slot.
        assert_eq!(p.step(1, 1, 0), WorkOutcome::NoActiveSlot);
    }

    #[test]
    fn watermark_cap_evicts_oldest_closed_entries_only() {
        let orologio = ManualClock::default();
        let mut p = Slots::new(orologio.clone(), Deadlines { riserva: 30 * S, inattivo: 120 * S }).con_pulizia(10_000 * S, 10);
        for i in 0..30u64 {
            orologio.avanza(S);
            p.open(i, 1, (i % 4) as u32 + 100, 0);
            p.close(i, 1);
        }
        p.open(1000, 1, 7, 0);
        p.scadenze();
        assert!(p.conversazioni_ricordate() <= 10, "{}", p.conversazioni_ricordate());
        assert_eq!(p.slot(1000), Some((1, 7)));
        assert_eq!(p.watermark(29), 1);
        assert_eq!(p.watermark(0), 0);
    }

    #[test]
    fn open_above_watermark_reserves_slot() {
        let mut p = slots();
        assert_eq!(p.open(7, 1, 3, 0), OpenOutcome::Riservato { libera: None });
        assert_eq!(p.slot(7), Some((1, 3)));
    }

    #[test]
    fn open_below_watermark_is_discarded() {
        let mut p = slots();
        p.close(7, 2);
        assert_eq!(p.open(7, 2, 0, 0), OpenOutcome::Scartato);
        assert_eq!(p.open(7, 1, 0, 0), OpenOutcome::Scartato);
        assert_eq!(p.slot(7), None);
        assert_eq!(p.open(7, 3, 0, 0), OpenOutcome::Riservato { libera: None });
    }

    #[test]
    fn same_attempt_is_duplicate() {
        let mut p = slots();
        p.open(7, 1, 0, 0);
        assert_eq!(p.open(7, 1, 0, 0), OpenOutcome::Duplicato);
        assert_eq!(p.block(7, 1, 0, 4), WorkOutcome::Compute);
        assert_eq!(p.open(7, 1, 0, 0), OpenOutcome::Duplicato);
        assert_eq!(p.slot(7), Some((1, 0)));
    }

    #[test]
    fn new_attempt_replaces_existing_reservation() {
        let mut p = slots();
        p.open(7, 1, 2, 0);
        p.block(7, 1, 0, 4);
        let e = p.open(7, 2, 5, 0);
        assert_eq!(e, OpenOutcome::Riservato { libera: Some(Freed { seq: 2, tentativo: 1 }) });
        assert_eq!(p.watermark(7), 1);
        assert_eq!(p.slot(7), Some((2, 5)));
        let e = p.open(7, 3, 6, 0);
        assert_eq!(e, OpenOutcome::Riservato { libera: Some(Freed { seq: 5, tentativo: 2 }) });
        assert_eq!(p.watermark(7), 2);
    }

    #[test]
    fn draft_rewinds_within_limit_and_plain_replay_is_duplicate() {
        let mut p = slots();
        p.open(7, 1, 0, 0);
        assert_eq!(p.block(7, 1, 0, 10), WorkOutcome::Compute);
        // Compute a speculative suffix, then ask the allocator to resume at an
        // earlier position within the configured rewind depth.
        assert_eq!(p.passo_bozza(7, 1, 10, 2), (WorkOutcome::Compute, None));
        assert_eq!(p.passo_bozza(7, 1, 11, 2), (WorkOutcome::Compute, None));
        assert_eq!(p.passo_bozza(7, 1, 12, 2), (WorkOutcome::Compute, None));
        assert_eq!(p.passo_bozza(7, 1, 11, 2), (WorkOutcome::Compute, Some(11)));
        assert_eq!(p.passo_bozza(7, 1, 12, 2), (WorkOutcome::Compute, None));
        assert_eq!(p.passo_bozza(7, 1, 10, 2), (WorkOutcome::Duplicate, None));
        assert_eq!(p.passo_bozza(7, 1, 12, 0), (WorkOutcome::Duplicate, None));
        assert_eq!(p.step(7, 1, 12), WorkOutcome::Duplicate);
        assert_eq!(p.passo_bozza(7, 1, 20, 2), (WorkOutcome::UnexpectedPosition { expected: 13 }, None));
    }

    #[test]
    fn prefill_block_activates_reserved_slot() {
        let mut p = slots();
        p.open(7, 1, 0, 0);
        assert_eq!(p.block(7, 1, 0, 16), WorkOutcome::Compute);
        assert_eq!(p.step(7, 1, 16), WorkOutcome::Compute);
    }

    #[test]
    fn active_slot_executes_and_renews_on_expected_input() {
        let mut p = slots();
        p.open(7, 1, 0, 0);
        assert_eq!(p.block(7, 1, 0, 8), WorkOutcome::Compute);
        assert_eq!(p.block(7, 1, 8, 8), WorkOutcome::Compute);
        p.orologio().avanza(100 * S);
        assert_eq!(p.step(7, 1, 16), WorkOutcome::Compute);
        // Interleave computation and the clock check so the second block must have renewed the slot.
        p.orologio().avanza(100 * S);
        assert!(p.scadenze().is_empty());
        assert_eq!(p.step(7, 1, 20), WorkOutcome::UnexpectedPosition { expected: 17 });
        assert_eq!(p.step(7, 1, 16), WorkOutcome::Duplicate);
    }

    #[test]
    fn retired_attempt_skips_execution() {
        let mut p = slots();
        p.open(7, 1, 0, 0);
        p.close(7, 1);
        assert_eq!(p.block(7, 1, 0, 4), WorkOutcome::Discarded);
        assert_eq!(p.step(7, 1, 4), WorkOutcome::Discarded);
        assert_eq!(p.open(7, 1, 0, 0), OpenOutcome::Scartato);
        p.open(7, 2, 0, 0);
        assert_eq!(p.block(7, 1, 0, 4), WorkOutcome::Discarded);
        assert_eq!(p.block(7, 2, 0, 4), WorkOutcome::Compute);
    }

    #[test]
    fn close_releases_slot_and_advances_watermark() {
        let mut p = slots();
        p.open(7, 2, 4, 0);
        p.block(7, 2, 0, 4);
        let e = p.close(7, 2);
        assert_eq!(e.libera, Some(Freed { seq: 4, tentativo: 2 }));
        assert_eq!(p.watermark(7), 2);
        assert_eq!(p.slot(7), None);
        p.open(7, 3, 4, 0);
        assert_eq!(p.close(7, 9).libera, Some(Freed { seq: 4, tentativo: 3 }));
        assert_eq!(p.watermark(7), 9);
    }

    #[test]
    fn closing_old_attempt_preserves_current_slot() {
        let mut p = slots();
        p.open(7, 3, 4, 0);
        assert_eq!(p.close(7, 2).libera, None);
        assert_eq!(p.slot(7), Some((3, 4)));
        assert_eq!(p.watermark(7), 2);
    }

    #[test]
    fn close_before_open_establishes_watermark() {
        let mut p = slots();
        // A close can arrive before its open and still fence off the attempt.
        assert_eq!(p.close(7, 1), CloseOutcome { libera: None });
        assert_eq!(p.watermark(7), 1);
        assert_eq!(p.open(7, 1, 0, 0), OpenOutcome::Scartato);
        assert_eq!(p.posti_in_uso(), 0);
        p.close(7, 5);
        p.close(7, 3);
        assert_eq!(p.watermark(7), 5);
    }

    #[test]
    fn expired_reservation_releases_slot_and_advances_watermark() {
        let mut p = slots();
        p.open(7, 1, 2, 0);
        p.orologio().avanza(29 * S);
        assert!(p.scadenze().is_empty());
        p.orologio().avanza(1 * S);
        let s = p.scadenze();
        assert_eq!(s, vec![Scaduto { conversation_id: 7, liberato: Freed { seq: 2, tentativo: 1 }, era_riservato: true }]);
        assert_eq!(p.watermark(7), 1);
        assert_eq!(p.slot(7), None);
        assert_eq!(p.open(7, 1, 2, 0), OpenOutcome::Scartato);
    }

    #[test]
    fn idle_slot_expires() {
        let mut p = slots();
        p.open(7, 1, 2, 0);
        p.block(7, 1, 0, 4);
        p.orologio().avanza(119 * S);
        assert!(p.scadenze().is_empty());
        p.orologio().avanza(1 * S);
        let s = p.scadenze();
        assert_eq!(s.len(), 1);
        assert!(!s[0].era_riservato);
        assert_eq!(p.watermark(7), 1);
        assert_eq!(p.step(7, 1, 4), WorkOutcome::Discarded);
    }

    // Slot reuse across conversations.

    /// Send closes and opens on the same stream to keep slot reuse ordered.
    #[test]
    fn slot_reuse_depends_on_close_and_open_order() {
        let mut p = slots();
        p.open(1, 1, 0, 0);
        assert_eq!(p.open(2, 1, 0, 0), OpenOutcome::Occupato { da: 1 });
        p.close(1, 1);
        assert_eq!(p.open(2, 1, 0, 0), OpenOutcome::Riservato { libera: None });
    }

    #[test]
    fn stale_open_for_active_slot_is_discarded() {
        let mut p = slots();
        p.open(7, 3, 0, 0);
        assert_eq!(p.open(7, 2, 1, 0), OpenOutcome::Scartato);
        assert_eq!(p.slot(7), Some((3, 0)));
    }

    #[test]
    fn block_and_step_require_a_slot() {
        let mut p = slots();
        assert_eq!(p.block(1, 1, 0, 4), WorkOutcome::NoActiveSlot);
        p.open(1, 1, 0, 0);
        assert_eq!(p.step(1, 1, 0), WorkOutcome::NoActiveSlot);
        assert_eq!(p.block(1, 2, 0, 4), WorkOutcome::NoActiveSlot);
    }

    #[test]
    fn position_starts_at_initial_position() {
        let mut p = slots();
        p.open(1, 2, 0, 10);
        assert_eq!(p.block(1, 2, 0, 4), WorkOutcome::Duplicate);
        assert_eq!(p.block(1, 2, 12, 4), WorkOutcome::UnexpectedPosition { expected: 10 });
        assert_eq!(p.block(1, 2, 10, 4), WorkOutcome::Compute);
    }

    #[test]
    fn conversations_are_independent() {
        let mut p = slots();
        p.close(1, 5);
        assert_eq!(p.open(2, 1, 0, 0), OpenOutcome::Riservato { libera: None });
        assert_eq!(p.posti_in_uso(), 1);
    }

    #[test]
    fn queued_work_preserves_only_the_matching_slot_attempt() {
        let clock = ManualClock::default();
        let mut slots = Slots::new(clock.clone(), Deadlines { riserva: 30 * S, inattivo: 120 * S });
        slots.open(1, 2, 0, 0);
        slots.open(2, 1, 1, 0);
        slots.open(3, 1, 2, 0);
        clock.avanza(35 * S);
        let pending = [(1, 2), (2, 2)];
        let expired = slots.expire_except(|id, attempt| pending.contains(&(id, attempt)));
        assert_eq!(expired.iter().map(|slot| slot.conversation_id).collect::<Vec<_>>(), [2, 3]);
        assert_eq!(slots.block(1, 2, 0, 16), WorkOutcome::Compute);
        assert!(slots.scadenze().is_empty());
    }

    #[test]
    fn reserved_slot_expires_without_queued_block() {
        let orologio = ManualClock::default();
        let mut p = Slots::new(orologio.clone(), Deadlines { riserva: 30 * S, inattivo: 120 * S });
        p.open(1, 1, 0, 0);
        orologio.avanza(35 * S);
        let s = p.scadenze();
        assert_eq!(s.len(), 1);
        assert!(s[0].era_riservato);
    }

    // Exclusive sequence ownership.

    #[test]
    fn open_rejects_sequence_owned_by_another_live_conversation() {
        let mut p = slots();
        assert_eq!(p.open(1, 1, 4, 0), OpenOutcome::Riservato { libera: None });
        p.block(1, 1, 0, 4);
        assert_eq!(p.open(2, 1, 4, 0), OpenOutcome::Occupato { da: 1 });
        assert_eq!(p.slot(2), None);
        assert_eq!(p.slot(1), Some((1, 4)));
        assert_eq!(p.proprietario(4), Some((1, 1)));
        assert_eq!(p.watermark(2), 0);
        assert_eq!(p.close(1, 1).libera, Some(Freed { seq: 4, tentativo: 1 }));
        assert_eq!(p.proprietario(4), None);
        assert_eq!(p.open(2, 1, 4, 0), OpenOutcome::Riservato { libera: None });
        assert_eq!(p.proprietario(4), Some((2, 1)));
    }

    #[test]
    fn late_close_does_not_clear_new_owners_state() {
        let mut p = slots();
        p.open(1, 1, 4, 0);
        assert_eq!(p.open(2, 1, 4, 0), OpenOutcome::Occupato { da: 1 });
        let e = p.close(1, 1);
        assert_eq!(e.libera, Some(Freed { seq: 4, tentativo: 1 }));
        // A duplicate close for the displaced conversation must leave its replacement alone.
        assert_eq!(p.open(2, 1, 4, 0), OpenOutcome::Riservato { libera: None });
        assert_eq!(p.close(1, 1).libera, None);
        assert_eq!(p.slot(2), Some((1, 4)));
        assert_eq!(p.proprietario(4), Some((2, 1)));
        assert_eq!(p.close(1, 9).libera, None);
        assert_eq!(p.proprietario(4), Some((2, 1)));
    }

    #[test]
    fn expiry_only_releases_the_expired_sequence() {
        let mut p = slots();
        p.open(1, 1, 4, 0);
        p.open(2, 1, 5, 0);
        p.orologio().avanza(31 * S);
        p.block(2, 1, 0, 1);
        let sc = p.scadenze();
        assert_eq!(sc.len(), 1);
        assert_eq!((sc[0].conversation_id, sc[0].liberato.seq), (1, 4));
        assert_eq!(p.proprietario(4), None);
        assert_eq!(p.proprietario(5), Some((2, 1)));
        assert_eq!(p.open(3, 1, 4, 0), OpenOutcome::Riservato { libera: None });
    }

    #[test]
    fn new_attempt_takes_ownership_of_sequence() {
        let mut p = slots();
        p.open(1, 1, 4, 0);
        p.open(2, 1, 5, 0);
        assert_eq!(p.open(1, 2, 6, 0), OpenOutcome::Riservato { libera: Some(Freed { seq: 4, tentativo: 1 }) });
        assert_eq!(p.proprietario(4), None);
        assert_eq!(p.proprietario(6), Some((1, 2)));
        assert_eq!(p.open(1, 3, 5, 0), OpenOutcome::Occupato { da: 2 });
        assert_eq!(p.slot(1), Some((2, 6)));
        assert_eq!(p.watermark(1), 1);
        p.close(1, 3);
        assert_eq!(p.open(1, 3, 5, 0), OpenOutcome::Scartato);
    }

    #[test]
    fn conversation_can_reuse_its_sequence() {
        let mut p = slots();
        p.open(1, 1, 4, 0);
        assert_eq!(p.open(1, 2, 4, 0), OpenOutcome::Riservato { libera: Some(Freed { seq: 4, tentativo: 1 }) });
        assert_eq!(p.proprietario(4), Some((1, 2)));
    }
    // Slot eviction after a lost close.

    #[test]
    fn new_open_evicts_previous_owner_after_lost_close() {
        let mut p = slots();
        p.open(1, 1, 0, 0);
        p.block(1, 1, 0, 4);
        assert_eq!(p.apri_con_sfratto(2, 1, 0, 0), OpenOutcome::Sfrattato { da: 1, da_tentativo: 1, libera: None });
        assert_eq!(p.proprietario(0), Some((2, 1)));
        assert_eq!(p.slot(2), Some((1, 0)));
        assert_eq!(p.slot(1), None);
        assert_eq!(p.watermark(1), 1);
        assert_eq!(p.step(1, 1, 4), WorkOutcome::Discarded);
        assert_eq!(p.block(1, 1, 4, 4), WorkOutcome::Discarded);
        assert_eq!(p.close(1, 1).libera, None);
        assert_eq!(p.proprietario(0), Some((2, 1)));
        assert_eq!(p.block(2, 1, 0, 4), WorkOutcome::Compute);
        p.orologio().avanza(1000 * S);
        let sc = p.scadenze();
        assert_eq!(sc.len(), 1);
        assert_eq!(sc[0].conversation_id, 2);
    }

    #[test]
    fn slot_remains_busy_without_eviction() {
        let mut p = slots();
        p.open(1, 1, 0, 0);
        assert_eq!(p.open(2, 1, 0, 0), OpenOutcome::Occupato { da: 1 });
        assert_eq!(p.proprietario(0), Some((1, 1)));
    }

    #[test]
    fn eviction_with_existing_slot_for_new_owner() {
        let mut p = slots();
        p.open(1, 1, 0, 0);
        p.open(2, 1, 5, 0);
        let e = p.apri_con_sfratto(2, 2, 0, 0);
        assert_eq!(e, OpenOutcome::Sfrattato { da: 1, da_tentativo: 1, libera: Some(Freed { seq: 5, tentativo: 1 }) });
        assert_eq!(p.proprietario(5), None);
        assert_eq!(p.proprietario(0), Some((2, 2)));
    }

    #[test]
    fn discarded_open_does_not_evict_current_owner() {
        let mut p = slots();
        p.open(1, 1, 0, 0);
        p.close(2, 3);
        assert_eq!(p.apri_con_sfratto(2, 2, 0, 0), OpenOutcome::Scartato);
        assert_eq!(p.proprietario(0), Some((1, 1)), "a rejected Open must not evict a conversation");
        p.open(3, 5, 7, 0);
        assert_eq!(p.apri_con_sfratto(3, 4, 0, 0), OpenOutcome::Occupato { da: 1 }, "attempt older than the one using the slot");
        assert_eq!(p.proprietario(0), Some((1, 1)));
    }

    /// The head tracks live owners independently and occasionally loses a close.
    /// Every assigned sequence must still have at most one live owner in the stage.
    #[test]
    fn random_lost_closes_do_not_prevent_open() {
        use crate::simulator::Rng;
        for seed in 0..300u64 {
            let mut r = Rng::new(seed);
            let mut p = slots();
            let mut vive: Vec<(u64, u32, u32)> = Vec::new();
            let mut prossima = 1u64;
            for _ in 0..200 {
                p.orologio().avanza(Duration::from_millis(r.tra(1, 3000)));
                if r.per_mille(500) && vive.len() < 4 {
                    let seq = (0..4u32).find(|s| !vive.iter().any(|v| v.2 == *s)).unwrap();
                    let id = prossima;
                    prossima += 1;
                    let e = p.apri_con_sfratto(id, 1, seq, 0);
                    assert!(matches!(e, OpenOutcome::Riservato { .. } | OpenOutcome::Sfrattato { .. }), "seme {seed}: {e:?}");
                    assert_eq!(p.proprietario(seq), Some((id, 1)));
                    vive.push((id, 1, seq));
                } else if !vive.is_empty() {
                    let (id, t, _) = vive.remove(r.sotto(vive.len() as u64) as usize);
                    if !r.per_mille(333) {
                        p.close(id, t);
                    }
                }
                let _ = p.scadenze();
                for &(id, _, seq) in &vive {
                    if let Some((a, _)) = p.proprietario(seq) {
                        assert_eq!(a, id, "seed {seed}: sequence {seq} belongs to {a}, but the head expects {id}");
                    }
                }
            }
        }
    }
}
