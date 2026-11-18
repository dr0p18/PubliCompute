//! A stage has one compute thread and separate stream readers and writers.
//! An intermediate applies its layer range and forwards residual rows. The
//! tail samples from logits and sends token replies to the head. The
//! intermediate also relays confirmations back upstream.
//!
//! Validate shape and encoding before changing a slot. A bad row reports a
//! fault for its conversation while a failed batch reports faults for its
//! members. Reopening a conversation reserves a fresh attempt and replays
//! the confirmed prefix through the chain.
//!
//! Reconnect a successor on a helper thread. Until the result arrives,
//! forwarding fails promptly so the affected conversations can recover.
//! Reader channels stay unbounded so each peer can drain its socket even
//! when all compute threads are waiting for a writer. Outgoing queues
//! have write deadlines.
//!
//! Control messages carry cumulative timings and faults. Keep the source
//! stage when relaying a confirmation and stamp it with the upstream link
//! epoch so the head can recognize retired links.

use crate::sampling::{campiona_tagli, Tagli};
use crate::experts::{self, RecEsperti};
use crate::draft::{piano_mtp, MtpPlan};
use crate::engine::{Engine, Row, MtpRow};
use crate::slot::{EventQueue, OpenOutcome, WorkOutcome, SystemClock, Slots, Prossimo, Deadlines};
use crate::network::{self, encode_state, msg, validate_prefill_block, validate_step, Writer};
use publicompute_proto::stage::{self as pb, message::Body};
use std::collections::HashMap;
use std::net::TcpStream;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Sender};
use crate::timings::StageTimings;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Role {
    Middle,
    Tail,
}

pub struct StageOptions {
    pub role: Role,
    /// Where the stage listens for streams from its predecessor.
    pub listen: String,
    /// Successor (intermediate only).
    pub next: Option<String>,
    /// Head's `Token` address (tail only).
    pub testa_token: Option<String>,
    pub scadenze: Deadlines,
    pub slots: u32,
    pub ctx_per_slot: u32,
    pub attesa_s: u64,
    /// Exit after this many seconds without prefill or generation streams from the predecessor. It can reconnect
    /// before then.
    pub grazia_s: u64,
    /// Expert log file. Load the `Engine` with `experts`.
    pub experts: Option<String>,
    /// Top-k log file (20 highest log probabilities for each token the tail samples).
    pub top: Option<String>,
    /// Sampling threads per step (`--sampling-threads`, 0 = automatic, min(rows, 4), 1 = sequential).
    pub thread_campiona: usize,
    /// TEST ONLY (off by default): `(conversation_id, n)`. On the `n`th `Step` row for that
    /// conversation, skip the computation and send an error `Confirmation` only for it, as with
    /// a rejected or expired slot. This exercises single-conversation recovery.
    pub prova_errore_conv: Option<(u64, u32)>,
    /// Fraction of positions that substitute the highest-scoring alternative
    /// to exercise seeded verification.
    pub prova_pilota: Option<f64>,
    /// Maximum speculative rows per conversation. Keep the same setting at
    /// every stage because each stage may have to rewind by this many rows
    /// when the tail rejects a draft. Zero disables drafting.
    pub max_draft_tokens: u32,
    /// Optional binary dump of the logits used for tail drafting. See RegistroMtp
    /// for the record layout.
    pub registro_mtp: Option<String>,
    /// State format an intermediate stage sends downstream (`--state-format`, default FP32).
    pub state_format: pb::StateFormat,
    /// Stage position in the chain, starting at 1 from the head (`--index`, 0 = unspecified). Goes in
    /// outgoing `Confirmation` and `Token` messages.
    pub index: u32,
    /// Input format accepted by this stage (`--input-format`). Reject an `Open` declaring
    /// another format. `None` skips that initial check, but messages still have to match their `Open`.
    pub formato_ingresso: Option<pb::StateFormat>,
}

/// Record of candidate probabilities for tokens sampled at the tail.
pub const TOP_K: usize = 20;

enum Event {
    Msg(pb::Stream, pb::Message),
    Done(pb::Stream, Option<String>),
    /// Predecessor opened the control stream with the epoch declared in its `Hello`.
    Control(Writer, u32),
    /// Predecessor opened a prefill or generation stream.
    NuovoFlusso,
    /// Successor's control stream closed (given epoch).
    AvantiCaduto(u32),
    /// Result of reconnecting the successor on a separate thread (given epoch).
    Ricollegato(Result<Forward, String>, u32),
}

/// Link to the successor (intermediate only).
struct Forward {
    prefill: Writer,
    gen: Writer,
    /// Keeps the write side of the control stream open.
    _ctl: TcpStream,
    /// The control reader sets this to `true` as soon as the successor drops.
    caduto: Arc<AtomicBool>,
    epoch: u32,
}

fn connect_forward(address: &str, timeout: Duration, tx: &Sender<Event>, epoch: u32) -> Result<Forward, String> {
    let (lp, wp) = network::connect(address, pb::Stream::Prefill, timeout, epoch)?;
    let (lg, wg) = network::connect(address, pb::Stream::Generation, timeout, epoch)?;
    let (mut lc, wc) = network::connect(address, pb::Stream::Control, timeout, epoch)?;
    let caduto = Arc::new(AtomicBool::new(false));
    let (txc, cad) = (tx.clone(), caduto.clone());
    std::thread::spawn(move || {
        while let Ok(Some(m)) = lc.read() {
            if txc.send(Event::Msg(pb::Stream::Control, m)).is_err() {
                return;
            }
        }
        cad.store(true, Ordering::SeqCst);
        let _ = txc.send(Event::AvantiCaduto(epoch));
    });
    drop((lp, lg));
    Ok(Forward { prefill: Writer::new(wp, "forward-prefill"), gen: Writer::new(wg, "forward-generation"), _ctl: wc, caduto, epoch })
}


/// The draft context pairs each token with the preceding main hidden state.
/// Verified rows are kept. Rows used to extend a speculative continuation
/// are truncated before the next verified batch.
struct MtpConv {
    /// Cached positions form the half-open range [0, prossima).
    prossima: u32,
    /// Keep the last main hidden state across prompt blocks. The first row
    /// of the next block needs it. The first block starts with a zero state.
    pendente: Vec<f32>,
}

/// Binary draft log, with little-endian fields. The header is MTPL then
/// a u32 vocabulary size. Each record holds u64 conversation ID, u32 step,
/// i32 preceding token, i32 proposal, u32 depth and rejection count, then
/// one f32 logit per vocabulary entry. The high byte of the packed field
/// is the zero-based depth. At later depths the preceding token is the
/// preceding proposal.
struct MtpTrace {
    f: std::io::BufWriter<std::fs::File>,
    n: u64,
}

impl MtpTrace {
    fn open(path: &str, n_vocab: usize) -> Result<Self, String> {
        use std::io::Write;
        let mut f = std::io::BufWriter::new(std::fs::File::create(path).map_err(|e| format!("{path}: {e}"))?);
        f.write_all(b"MTPL").and_then(|_| f.write_all(&(n_vocab as u32).to_le_bytes())).map_err(|e| format!("{path}: {e}"))?;
        Ok(MtpTrace { f, n: 0 })
    }

    fn write(&mut self, conv: u64, step: u32, token: i32, draft: i32, rifiutate: u32, profondita: u32, logit: &[f32]) {
        use std::io::Write;
        let rifiutate = rifiutate.min(0xff_ffff) | (profondita.min(0xff) << 24);
        let mut b = Vec::with_capacity(24 + logit.len() * 4);
        b.extend_from_slice(&conv.to_le_bytes());
        b.extend_from_slice(&step.to_le_bytes());
        b.extend_from_slice(&token.to_le_bytes());
        b.extend_from_slice(&draft.to_le_bytes());
        b.extend_from_slice(&rifiutate.to_le_bytes());
        for x in logit {
            b.extend_from_slice(&x.to_le_bytes());
        }
        if self.f.write_all(&b).is_ok() {
            self.n += 1;
        }
    }
}

/// State needed to extend one conversation's draft across several batches.
/// The first proposal uses the verified batch. Later proposals feed the
/// last proposed token and the corresponding output of the draft block.
struct MtpChain {
    conv: u64,
    seq: u32,
    /// Row (with logits) of the last MTP batch that produced the last draft: where its output state is read.
    row: usize,
    /// Position of the next MTP row (the one that takes the last draft).
    pos: u32,
    /// Drafts the head asked for (>= 1, at most `--draft-max-tokens` of the tail).
    k: usize,
    /// Step of the preceding token and draft rows rejected in that step (for the log).
    step: u32,
    rifiutate: u32,
    draft: Vec<i32>,
}

struct Stage {
    state_format: pb::StateFormat,
    m: Engine,
    role: Role,
    slots: Slots<SystemClock>,
    ctx_per_slot: u32,
    campion: HashMap<u64, (f32, u64, Tagli)>,
    conferme: Option<Writer>,
    /// Predecessor's control stream epoch (from `Hello`), stamped on outgoing `Confirmation` messages.
    epoca_ingresso: u32,
    /// Stage position in the chain (0 = unspecified).
    index: u32,
    next: Option<Forward>,
    /// Successor address, link epoch and last failed reconnect attempt (intermediate only).
    indirizzo_avanti: Option<String>,
    epoca_avanti: u32,
    avanti_fallito: bool,
    /// A thread is already reconnecting the successor.
    ricollegando: bool,
    timeout: Duration,
    tx: Sender<Event>,
    token: Option<Writer>,
    esp: Vec<RecEsperti>,
    registra_top: bool,
    thread_campiona: usize,
    top: Vec<crate::sampling::RecTop>,
    /// Conversation error test (see `OpzioniStadio::prova_errore_conv`): target and rows seen.
    prova_errore: Option<(u64, u32)>,
    prova_pilota: Option<f64>,
    max_draft_tokens: u32,
    /// Track draft contexts only for conversations whose prompt carried token IDs
    /// and whose tail supports drafting.
    mtp: HashMap<u64, MtpConv>,
    registro_mtp: Option<MtpTrace>,
    prova_viste: u32,
    formato_ingresso: Option<pb::StateFormat>,
    /// Format declared by `Open` for each conversation with a slot.
    formato_conv: HashMap<u64, pb::StateFormat>,
    timings: StageTimings,
    tempi_sporchi: bool,
    tempi_inviati: Instant,
}

/// Minimum interval between timing snapshots sent to the head while active.
const TEMPI_OGNI: Duration = Duration::from_secs(1);

fn accept_streams(l: std::net::TcpListener, tx: Sender<Event>) {
    network::accept(l, move |f, mut lett, mut w| match f {
        pb::Stream::Prefill | pb::Stream::Generation => {
            if network::ack_ok(&mut w).is_err() {
                return;
            }
            let _ = tx.send(Event::NuovoFlusso);
            loop {
                match lett.read() {
                    Ok(Some(m)) => {
                        if tx.send(Event::Msg(f, m)).is_err() {
                            break;
                        }
                    }
                    Ok(None) => {
                        let _ = tx.send(Event::Done(f, None));
                        break;
                    }
                    Err(e) => {
                        let _ = tx.send(Event::Done(f, Some(e)));
                        break;
                    }
                }
            }
        }
        pb::Stream::Control => {
            let Ok(w2) = w.try_clone() else { return };
            let sc = Writer::new(w2, "conferme");
            let _ = tx.send(Event::Control(sc, lett.epoca_hello));
            let _ = network::ack_ok(&mut w);
        }
        altro => eprintln!("[stage] unexpected stream {} on this listener", network::stream_name(altro)),
    });
}

/// Send on the control stream if present. Return `Err` if missing or down.
fn send_control(c: Option<&Writer>, m: &pb::Message) -> Result<(), String> {
    c.ok_or_else(|| "control stream is disconnected".to_string())?.send(m)
}

impl Stage {
    /// Send a `Confirmation` to the predecessor on control. If the stream is absent or down, log the
    /// error to stderr. The head still detects it via `T_lavoro`, and the log keeps the reason.
    fn confirmation(&self, kind: pb::AckKind, conv: u64, att: u32, ok: bool, error: &str) {
        let m = msg(Body::Confirmation(pb::Confirmation {
            kind: kind as i32,
            conversation_id: conv,
            attempt_id: att,
            ok,
            error: error.to_string(),
            stage: self.index,
            epoch: self.epoca_ingresso,
            ..Default::default()
        }));
        if let Err(e) = send_control(self.conferme.as_ref(), &m) {
            eprintln!("[stage] could not send Confirmation for conversation {conv}, attempt {att} (ok={ok}, error=\"{error}\"): {e}");
        }
    }

    fn error(&self, conv: u64, att: u32, text: String) {
        eprintln!("[stage] conversation {conv}, attempt {att}: {text}");
        self.confirmation(pb::AckKind::Error, conv, att, false, &text);
    }

    /// Chain fault: error `Confirmation` without a conversation. The head restarts all conversations.
    fn errore_catena(&self, text: &str) {
        eprintln!("[stage] chain fault: {text}");
        self.confirmation(pb::AckKind::Error, 0, 0, false, text);
    }

    /// Forward to the successor (intermediate only; the tail has none). Network wait is bounded by the
    /// socket write timeout. If the link is missing or down, start a reconnect thread (`begin_reconnect`)
    /// and fail immediately, so the caller can return a clean error to affected conversations
    /// without blocking compute for `timeout`.
    fn forward(&mut self, gen: bool, m: pb::Message) -> Result<(), String> {
        if self.indirizzo_avanti.is_none() {
            return Ok(());
        }
        if self.next.as_ref().map_or(true, |a| a.caduto.load(Ordering::SeqCst)) {
            self.begin_reconnect();
            return Err("reconnecting to the next stage".into());
        }
        let a = self.next.as_ref().ok_or("no downstream connection")?;
        match if gen { a.gen.send(&m) } else { a.prefill.send(&m) } {
            Ok(()) => Ok(()),
            Err(e) => {
                a.caduto.store(true, Ordering::SeqCst);
                self.begin_reconnect();
                Err(format!("send failed ({e}); reconnecting the next stage"))
            }
        }
    }

    fn forward_prefill(&mut self, m: pb::Message) -> Result<(), String> {
        self.forward(false, m)
    }

    fn forward_generation(&mut self, m: pb::Message) -> Result<(), String> {
        self.forward(true, m)
    }

    /// Start a thread to reconnect the successor if there is none. Its result arrives as
    /// `Evento::Ricollegato`. The first attempt waits up to `timeout`. After a failure, try for only 2 s,
    /// since each unsendable message would start a new attempt.
    fn begin_reconnect(&mut self) {
        let Some(ind) = self.indirizzo_avanti.clone() else { return };
        if self.ricollegando {
            return;
        }
        self.ricollegando = true;
        self.next = None; // The detached writers exit when their queues or sockets close.
        let timeout = if self.avanti_fallito { Duration::from_secs(2) } else { self.timeout };
        eprintln!("[stage] reconnecting to stage {ind} in the background for up to {} s", timeout.as_secs());
        self.epoca_avanti += 1;
        let (tx, epoch) = (self.tx.clone(), self.epoca_avanti);
        std::thread::spawn(move || {
            let r = connect_forward(&ind, timeout, &tx, epoch);
            let _ = tx.send(Event::Ricollegato(r, epoch));
        });
    }

    fn ricollegato(&mut self, r: Result<Forward, String>, epoch: u32) {
        self.ricollegando = false;
        match r {
            Ok(a) => {
                self.next = Some(a);
                self.avanti_fallito = false;
                eprintln!("[stage] reconnected to the next stage (epoch {epoch})");
            }
            Err(e) => {
                self.avanti_fallito = true;
                self.errore_catena(&format!("next stage unreachable: {e}"));
            }
        }
    }

    /// Send timing snapshot to the head. Use `forza` at end of run. Otherwise send at most once per second and
    /// only after activity.
    fn send_timings(&mut self, forza: bool) {
        if !self.tempi_sporchi || (!forza && self.tempi_inviati.elapsed() < TEMPI_OGNI) {
            return;
        }
        self.tempi_sporchi = false;
        self.tempi_inviati = Instant::now();
        let role = if self.role == Role::Tail { "tail" } else { "middle" };
        let m = msg(Body::Confirmation(pb::Confirmation {
            kind: pb::AckKind::Timings as i32,
            ok: true,
            stage: self.index,
            epoch: self.epoca_ingresso,
            timings_json: self.timings.json(self.index, role).to_string(),
            ..Default::default()
        }));
        // Best effort: if control is missing, the next cumulative snapshot catches up.
        let _ = send_control(self.conferme.as_ref(), &m);
    }

    /// Empty reply for a round with no computation. The head still expects a reply for the round.
    fn giro_vuoto(&mut self, p: &pb::Step) {
        let r = if self.role == Role::Tail {
            match &self.token {
                Some(t) => t.send(&msg(Body::Token(pb::Token { group_id: p.group_id, group_round: p.group_round, prefill: false, rows: vec![], stage: self.index }))),
                None => Err("no connection to the head".into()),
            }
        } else {
            self.forward_generation(msg(Body::Step(pb::Step { rows: vec![], state: vec![], ..p.clone() })))
        };
        if let Err(e) = r {
            eprintln!("[stage] could not send empty round: {e}");
        }
    }

    fn open(&mut self, a: pb::Open) -> Result<(), String> {
        if !self.m.posto_valido(a.slot) {
            self.error(a.conversation_id, a.attempt_id, format!("slot {:#x} is outside the configured range ({} groups, {} slots per group)", a.slot, self.m.groups(), self.m.per_ctx));
            return Ok(());
        }
        if a.context_limit > self.ctx_per_slot {
            self.error(a.conversation_id, a.attempt_id, format!("context {} exceeds the slot limit of {}", a.context_limit, self.ctx_per_slot));
            return Ok(());
        }
        if a.start_position != 0 {
            self.error(a.conversation_id, a.attempt_id, "recovery requires a starting position of zero".into());
            return Ok(());
        }
        let fmt = match network::declared_format(a.state_format) {
            Ok(f) => f,
            Err(e) => {
                self.error(a.conversation_id, a.attempt_id, e);
                return Ok(());
            }
        };
        if let Some(att) = self.formato_ingresso {
            if fmt != att {
                self.error(
                    a.conversation_id,
                    a.attempt_id,
                    format!("Open declares state format {}, but the stage expects {}", fmt.as_str_name(), att.as_str_name()),
                );
                return Ok(());
            }
        }
        match self.slots.apri_con_sfratto(a.conversation_id, a.attempt_id, a.slot, a.start_position) {
            esito @ (OpenOutcome::Riservato { .. } | OpenOutcome::Sfrattato { .. }) => {
                let libera = match esito {
                    OpenOutcome::Riservato { libera } => libera,
                    OpenOutcome::Sfrattato { da, da_tentativo, libera } => {
                        // Clear the old owner's memory before reserving the same sequence for its replacement.
                        eprintln!(
                            "[stage] evicted conversation {da}, attempt {da_tentativo}, from slot {:#x}; Close never arrived. Reusing the slot for conversation {}",
                            a.slot, a.conversation_id
                        );
                        self.campion.remove(&da);
                        self.formato_conv.remove(&da);
                        self.mtp.remove(&da);
                        libera
                    }
                    _ => None,
                };
                if let Some(l) = libera {
                    self.m.svuota_posto(l.seq);
                    self.m.mtp_svuota_posto(l.seq);
                }
                self.m.svuota_posto(a.slot);
                // Draft and main context must both start empty for a new attempt.
                self.mtp.remove(&a.conversation_id);
                if self.m.ha_mtp() {
                    self.m.mtp_svuota_posto(a.slot);
                    self.mtp.insert(a.conversation_id, MtpConv { prossima: 0, pendente: vec![0.0; self.m.mtp_state_width] });
                }
                let c = a.sampling.unwrap_or_default();
                self.campion.insert(a.conversation_id, sampling_from_open(&c));
                self.formato_conv.insert(a.conversation_id, fmt);
                // Forward `Open` with the format this stage will send to its successor.
                let mut next = a;
                next.state_format = self.state_format as i32;
                if let Err(e) = self.forward_prefill(msg(Body::Open(next))) {
                    self.error(a.conversation_id, a.attempt_id, format!("could not forward Open: {e}"));
                    return Ok(());
                }
                self.confirmation(pb::AckKind::Open, a.conversation_id, a.attempt_id, true, "");
            }
            OpenOutcome::Duplicato => self.confirmation(pb::AckKind::Open, a.conversation_id, a.attempt_id, true, ""),
            OpenOutcome::Scartato => {}
            OpenOutcome::Occupato { da } => {
                self.error(a.conversation_id, a.attempt_id, format!("sequence {:#x} still belongs to conversation {da}", a.slot));
            }
        }
        Ok(())
    }

    /// A `PrefillBlock` error fails only its conversation (error `Confirmation`), so the stage stays up.
    fn block(&mut self, b: pb::PrefillBlock) -> Result<(), String> {
        let (conv, att) = (b.conversation_id, b.attempt_id);
        // Check shape before touching the slot so a malformed block cannot advance the expected position.
        let formato_ok = match self.formato_conv.get(&conv) {
            Some(&d) => network::check_state_format(d, b.format),
            None => Ok(()),
        };
        let state = match formato_ok.and_then(|_| validate_prefill_block(&b, self.m.n_embd, self.m.stream_count, self.m.n_batch, self.ctx_per_slot)) {
            Ok(s) => s,
            Err(e) => {
                let vecchio = att <= self.slots.watermark(conv) || self.slots.slot(conv).is_some_and(|p| p.0 > att);
                if !vecchio {
                    self.error(conv, att, format!("malformed block: {e}"));
                }
                return Ok(());
            }
        };
        match self.slots.block(conv, att, b.start_position, b.n_token) {
            WorkOutcome::Compute => {}
            WorkOutcome::Duplicate | WorkOutcome::Discarded => return Ok(()),
            e => {
                self.error(conv, att, format!("block rejected: {e:?}"));
                return Ok(());
            }
        }
        if let Err(e) = self.blocco_calcola(&b, state) {
            self.error(conv, att, e);
        }
        Ok(())
    }

    fn blocco_calcola(&mut self, b: &pb::PrefillBlock, state: Vec<f32>) -> Result<(), String> {
        let n = b.n_token as usize;
        let seq = self.slots.slot(b.conversation_id).map(|p| p.1).ok_or("slot disappeared")?;
        let tail = self.role == Role::Tail;
        let rows: Vec<Row> =
            (0..n).map(|i| Row { token: 0, pos: (b.start_position as usize + i) as i32, seq: seq as i32, logits: tail && b.last && i + 1 == n }).collect();
        let out = self.m.run(&rows, Some(&state))?;
        match (self.role, out) {
            (Role::Middle, Some(s)) => {
                let mut f = b.clone();
                f.state = encode_state(&s, self.m.n_embd, self.m.stream_count, self.state_format);
                f.n_embd = self.m.n_embd as u32;
                f.stream_count = network::declared_stream_count(self.state_format, self.m.stream_count);
                f.format = self.state_format as i32;
                self.forward_prefill(msg(Body::Block(f)))?;
            }
            (Role::Tail, None) => {
                if b.last {
                    let (temp, seed, tagli) = *self.campion.get(&b.conversation_id).ok_or("unknown sampling mode")?;
                    let (mut t, mut lp) = campiona_tagli(self.m.logits(n - 1)?, temp, seed, b.first_token_step, tagli)?;
                    if let Some(f) = self.prova_pilota {
                        let l = self.m.logits(n - 1)?;
                        if let Some(alt) = crate::sampling::pilota(l, t, seed, b.first_token_step, f, 0) {
                            t = alt;
                            lp = crate::sampling::logprob(l, alt);
                        }
                    }
                    if self.registra_top {
                        let t20 = crate::sampling::top_k(self.m.logits(n - 1)?, TOP_K);
                        self.top.push(crate::sampling::RecTop { c: b.conversation_id, s: b.first_token_step, t: t20 });
                    }
                    // Put the verified prompt rows into the draft context. On the final block
                    // also use the first sampled token to propose its continuation.
                    let draft = self.mtp_blocco(b, seq, Some(t as i32));
                    let tok = pb::Token {
                        group_id: b.group_id,
                        group_round: 0,
                        prefill: true,
                        rows: vec![pb::TokenRow { conversation_id: b.conversation_id, attempt_id: b.attempt_id, conversation_step: b.first_token_step, token: t as i32, logprob: lp, draft }],
                        stage: self.index,
                    };
                    self.token.as_ref().ok_or("no connection to the head")?.send(&msg(Body::Token(tok)))?;
                } else {
                    self.mtp_blocco(b, seq, None);
                }
            }
            _ => return Err("engine output does not match the stage role; check the layer range".into()),
        }
        Ok(())
    }

    /// One `Step` with rows from different conversations. A rejected row or failed computation
    /// fails only the affected conversations. Always reply to the round, even if empty.
    fn step(&mut self, p: pb::Step) -> Result<(), String> {
        let n_embd = self.m.n_embd;
        let state = match validate_step(&p, n_embd, self.m.stream_count, self.m.n_batch) {
            Ok(s) => s,
            Err(e) => {
                for r in &p.rows {
                    self.error(r.conversation_id, r.attempt_id, format!("passo malformato: {e}"));
                }
                self.giro_vuoto(&p);
                return Ok(());
            }
        };
        let mut tenute = Vec::new();
        // Draft rewinds: (row index, position to start clearing from memory).
        let mut riavvolgi: Vec<(usize, u32)> = Vec::new();
        for (i, r) in p.rows.iter().enumerate() {
            if let Some((bersaglio, n)) = self.prova_errore {
                if r.conversation_id == bersaglio {
                    self.prova_viste += 1;
                    if self.prova_viste == n {
                        self.error(r.conversation_id, r.attempt_id, "injected fault (--test-conv-error)".into());
                        continue;
                    }
                }
            }
            if let Some(&d) = self.formato_conv.get(&r.conversation_id) {
                if let Err(e) = network::check_state_format(d, p.format) {
                    self.error(r.conversation_id, r.attempt_id, e);
                    continue;
                }
            }
            if r.position >= self.ctx_per_slot {
                self.error(r.conversation_id, r.attempt_id, format!("position {} exceeds the slot context ({})", r.position, self.ctx_per_slot));
                continue;
            }
            let (esito, riavv) = self.slots.passo_bozza(r.conversation_id, r.attempt_id, r.position, self.max_draft_tokens);
            match esito {
                WorkOutcome::Compute => {
                    if let Some(pos) = riavv {
                        riavvolgi.push((i, pos));
                    }
                    tenute.push(i)
                }
                WorkOutcome::Duplicate | WorkOutcome::Discarded => {}
                e => self.error(r.conversation_id, r.attempt_id, format!("step rejected: {e:?}")),
            }
        }
        if tenute.is_empty() {
            self.giro_vuoto(&p);
            return Ok(());
        }
        let mut ingresso = Vec::with_capacity(tenute.len() * n_embd);
        let mut rows = Vec::new();
        let mut kept = Vec::new();
        let tail = self.role == Role::Tail;
        // If a rewind failed for a conversation, drop all its rows in this step.
        let mut fallite: Vec<u64> = Vec::new();
        for &i in &tenute {
            let r = &p.rows[i];
            if fallite.contains(&r.conversation_id) {
                continue;
            }
            let Some(seq) = self.slots.slot(r.conversation_id).map(|x| x.1) else {
                self.error(r.conversation_id, r.attempt_id, "slot disappeared".into());
                continue;
            };
            if let Some(&(_, pos)) = riavvolgi.iter().find(|(k, _)| *k == i) {
                // The tail rejected part of the draft. Clear positions from `pos` onward (at most K) before computing.
                if let Err(e) = self.m.riavvolgi(seq, pos) {
                    self.error(r.conversation_id, r.attempt_id, e);
                    fallite.push(r.conversation_id);
                    continue;
                }
            }
            ingresso.extend_from_slice(&state[i * n_embd..(i + 1) * n_embd]);
            rows.push(Row { token: 0, pos: r.position as i32, seq: seq as i32, logits: tail });
            kept.push(*r);
        }
        if kept.is_empty() {
            self.giro_vuoto(&p);
            return Ok(());
        }
        let out = match self.m.run(&rows, Some(&ingresso)) {
            Ok(o) => o,
            Err(e) => {
                // Batch computation failed for all its conversations. Each recovers independently.
                for r in &kept {
                    self.error(r.conversation_id, r.attempt_id, format!("decode failed: {e}"));
                }
                self.giro_vuoto(&p);
                return Ok(());
            }
        };
        for (row, layer, mut e) in self.m.prendi_esperti() {
            e.sort_unstable();
            self.esp.push(RecEsperti { c: kept[row].conversation_id, s: kept[row].conversation_step, l: layer, e });
        }
        match (self.role, out) {
            (Role::Middle, Some(s)) => {
                let m = msg(Body::Step(pb::Step {
                    group_id: p.group_id,
                    group_round: p.group_round,
                    rows: kept.clone(),
                    n_embd: n_embd as u32,
                    format: self.state_format as i32,
                    state: encode_state(&s, n_embd, self.m.stream_count, self.state_format),
                    stream_count: network::declared_stream_count(self.state_format, self.m.stream_count),
                }));
                if let Err(e) = self.forward_generation(m) {
                    for r in &kept {
                        self.error(r.conversation_id, r.attempt_id, format!("could not forward step: {e}"));
                    }
                }
            }
            (Role::Tail, None) => {
                let out_righe = self.sample_step(&kept);
                let invio = match &self.token {
                    Some(t) => t.send(&msg(Body::Token(pb::Token { group_id: p.group_id, group_round: p.group_round, prefill: false, rows: out_righe, stage: self.index }))),
                    None => Err("no connection to the head".to_string()),
                };
                if let Err(e) = invio {
                    for r in &kept {
                        self.error(r.conversation_id, r.attempt_id, format!("could not send tokens: {e}"));
                    }
                }
            }
            _ => {
                for r in &kept {
                    self.error(r.conversation_id, r.attempt_id, "engine output does not match the stage role; check the layer range".into());
                }
            }
        }
        Ok(())
    }

    /// Sample step rows: collect all logits, sample in parallel (`--sampling-threads`,
    /// bit-identical to sequential), then build tokens. A row error (unknown
    /// sampling settings, null logits) fails only that conversation.
    fn sample_step(&mut self, kept: &[pb::StepRow]) -> Vec<pb::TokenRow> {
        let mut idx: Vec<usize> = Vec::with_capacity(kept.len());
        let mut row_errors: Vec<(usize, String)> = Vec::new();
        let mut params = Vec::with_capacity(kept.len());
        let mut tagli = Vec::with_capacity(kept.len());
        let mut rows: Vec<&[f32]> = Vec::with_capacity(kept.len());
        for (j, r) in kept.iter().enumerate() {
            let (p, tg) = match self.campion.get(&r.conversation_id) {
                Some(&(temp, seed, tg)) => ((temp, seed, r.conversation_step), tg),
                None => {
                    row_errors.push((j, "unknown sampling mode".into()));
                    continue;
                }
            };
            match self.m.logits(j) {
                Ok(l) => {
                    idx.push(j);
                    params.push(p);
                    tagli.push(tg);
                    rows.push(l);
                }
                Err(e) => row_errors.push((j, e)),
            }
        }
        let th = crate::sampling::thread_effettivi(self.thread_campiona, rows.len());
        let mut ris = crate::sampling::campiona_righe_tagli(&rows, &params, &tagli, th);
        if let Some(f) = self.prova_pilota {
            for ((r, l), p) in ris.iter_mut().zip(&rows).zip(&params) {
                if let Ok((t, lp)) = r {
                    if let Some(alt) = crate::sampling::pilota(l, *t, p.1, p.2, f, 0) {
                        *t = alt;
                        *lp = crate::sampling::logprob(l, alt);
                    }
                }
            }
        }
        // Result per batch row (None = row error already in `row_errors`, or row not sampled).
        let mut per_riga: Vec<Option<(usize, f32)>> = vec![None; kept.len()];
        for (&j, r) in idx.iter().zip(ris.drain(..)) {
            match r {
                Ok(x) => per_riga[j] = Some(x),
                Err(e) => row_errors.push((j, e)),
            }
        }
        // Compare samples with the following rows' input tokens and keep the
        // matching prefix. A discarded draft must not publish a token, a row
        // error, or a top-probability entry.
        let mut accettate = vec![false; kept.len()];
        // Keep accepted row ranges for updating each conversation's draft context.
        let mut gruppi_acc: Vec<(usize, usize, usize)> = Vec::new();
        let chiavi: Vec<(u64, u32, u32)> = kept.iter().map(|r| (r.conversation_id, r.conversation_step, r.position)).collect();
        let mut j = 0;
        while j < kept.len() {
            let gl = if self.max_draft_tokens > 0 {
                crate::draft::group_length(&chiavi, j)
            } else {
                1
            };
            // Shorten the group at the first row without a token (error).
            let valide = (0..gl).take_while(|&i| per_riga[j + i].is_some()).count();
            let n = if valide == 0 {
                1 
            } else {
                let campionati: Vec<i32> = (0..valide).map(|i| per_riga[j + i].unwrap().0 as i32).collect();
                let ingressi: Vec<i32> = (0..valide).map(|i| kept[j + i].draft_token).collect();
                crate::draft::accepted_rows(&campionati, &ingressi, |t| self.m.e_eog(t))
            };
            for i in 0..n {
                accettate[j + i] = true;
            }
            gruppi_acc.push((j, n, gl));
            // Ignore errors on discarded draft rows.
            row_errors.retain(|(k, _)| !(*k >= j + n && *k < j + gl));
            j += gl;
        }
        let mut top_nuovi = Vec::new();
        if self.registra_top {
            for (&j, l) in idx.iter().zip(&rows) {
                if accettate[j] && per_riga[j].is_some() {
                    top_nuovi.push(crate::sampling::RecTop { c: kept[j].conversation_id, s: kept[j].conversation_step, t: crate::sampling::top_k(l, TOP_K) });
                }
            }
        }
        drop(rows);
        self.top.extend(top_nuovi);
        for (j, e) in row_errors {
            let r = &kept[j];
            self.error(r.conversation_id, r.attempt_id, e);
        }
        // Update draft context with accepted rows and propose a continuation from
        // each conversation's last surviving sample.
        let bozze = self.mtp_passo(kept, &gruppi_acc, &per_riga);
        (0..kept.len())
            .filter(|&j| accettate[j])
            .filter_map(|j| {
                let (t, lp) = per_riga[j]?;
                let r = &kept[j];
                let draft = bozze.iter().find(|(k, _)| *k == j).map(|(_, d)| d.clone()).unwrap_or_default();
                Some(pb::TokenRow { conversation_id: r.conversation_id, attempt_id: r.attempt_id, conversation_step: r.conversation_step, token: t as i32, logprob: lp, draft })
            })
            .collect()
    }

    /// A cache gap, failed MTP batch or out-of-vocabulary token stops the tail's drafts for this
    /// conversation. Log the reason and keep the conversation running, since drafts are optional.
    fn mtp_spegni(&mut self, conv: u64, seq: u32, reason: &str) {
        if self.mtp.remove(&conv).is_some() {
            self.m.mtp_svuota_posto(seq);
            eprintln!("[stage] disabled MTP drafting for conversation {conv}: {reason}");
        }
    }

    /// Prefill draft context with each input token and the preceding main
    /// hidden state. The first row of a block uses the state saved from the
    /// previous block. On the final block, append the sampled token with the
    /// final prompt state and request logits for its continuation. Return up
    /// to the requested depth, or an empty draft if the prompt omitted token
    /// IDs or drafting is unavailable.
    fn mtp_blocco(&mut self, b: &pb::PrefillBlock, seq: u32, primo: Option<i32>) -> Vec<i32> {
        let conv = b.conversation_id;
        if !self.mtp.contains_key(&conv) {
            return Vec::new();
        }
        if b.token.is_empty() {
            self.mtp.remove(&conv);
            return Vec::new();
        }
        match self.mtp_blocco_calcola(b, seq, primo) {
            Ok(d) => d,
            Err(e) => {
                self.mtp_spegni(conv, seq, &e);
                Vec::new()
            }
        }
    }

    fn mtp_blocco_calcola(&mut self, b: &pb::PrefillBlock, seq: u32, primo: Option<i32>) -> Result<Vec<i32>, String> {
        let (conv, n, n_h, p0) = (b.conversation_id, b.n_token as usize, self.m.mtp_state_width, b.start_position);
        let st = self.mtp.get(&conv).ok_or("cache MTP sparita")?;
        if piano_mtp(st.prossima, p0) != MtpPlan::Aggiungi {
            return Err(format!("block at position {p0} with the MTP cache at {}", st.prossima));
        }
        if b.token.len() != n || b.token.iter().any(|&t| t < 0 || t as usize >= self.m.n_vocab) {
            return Err("block token IDs are missing or outside the vocabulary".into());
        }
        // The draft batch sees one state per input row, starting with the saved
        // state and ending with the state for the sampled token.
        let mut stati = Vec::with_capacity((n + 1) * n_h);
        stati.extend_from_slice(&st.pendente);
        for i in 0..n {
            stati.extend_from_slice(self.m.mtp_state(i)?);
        }
        let mut rows: Vec<MtpRow> = (0..n).map(|i| MtpRow { token: b.token[i], pos: (p0 as usize + i) as i32, seq: seq as i32, logits: false }).collect();
        let primo = primo.filter(|&t| !self.m.e_eog(t) && p0 + (n as u32) < self.ctx_per_slot);
        if let Some(t) = primo {
            rows.push(MtpRow { token: t, pos: (p0 as usize + n) as i32, seq: seq as i32, logits: true });
        }
        self.m.run_mtp(&rows, &stati[..rows.len() * n_h])?;
        let done = p0 + rows.len() as u32;
        let mut draft = Vec::new();
        let mut trial = 0;
        if let Some(t) = primo {
            let l = self.m.mtp_logits(n)?;
            let d = crate::sampling::argmax(l) as i32;
            if let Some(r) = &mut self.registro_mtp {
                r.write(conv, b.first_token_step, t, d, 0, 0, l);
            }
            let mut catene = [MtpChain { conv, seq, row: n, pos: done, k: self.profondita(b.draft_count), step: b.first_token_step, rifiutate: 0, draft: vec![d] }];
            self.mtp_catena(&mut catene);
            draft = std::mem::take(&mut catene[0].draft);
            trial = draft.len().saturating_sub(1) as u32;
        }
        // Deeper proposals occupy tentative cache positions beyond fine. The
        // next verified step truncates them.
        let st = self.mtp.get_mut(&conv).ok_or("cache MTP sparita")?;
        st.prossima = done + trial;
        st.pendente = stati[n * n_h..].to_vec();
        Ok(draft)
    }

    /// Clamp the requested depth to at least one and the tail's configured maximum.
    fn profondita(&self, request: u32) -> usize {
        (request.max(1) as usize).min(self.max_draft_tokens.max(1) as usize)
    }

    /// Extend the surviving drafts one depth at a time. A batch at each depth
    /// includes one row for every conversation that can still extend. Stop a
    /// sequence at the requested depth, an end marker, or the context limit.
    /// Write these positions speculatively so the next verified batch can
    /// replace them. If a batch fails, disable further drafts for its rows
    /// and keep any proposals already produced.
    fn mtp_catena(&mut self, catene: &mut [MtpChain]) {
        let k_max = catene.iter().map(|c| c.k).max().unwrap_or(1);
        for prof in 2..=k_max {
            let mut rows: Vec<MtpRow> = Vec::new();
            let mut stati: Vec<f32> = Vec::new();
            let mut scelte: Vec<usize> = Vec::new();
            for (i, c) in catene.iter().enumerate() {
                let Some(&ultima) = c.draft.last() else { continue };
                if c.draft.len() != prof - 1 || c.draft.len() >= c.k || self.m.e_eog(ultima) || c.pos >= self.ctx_per_slot {
                    continue;
                }
                if ultima < 0 || ultima as usize >= self.m.n_vocab {
                    continue;
                }
                match self.m.mtp_stato_uscita(c.row) {
                    Ok(h) => stati.extend_from_slice(h),
                    Err(_) => continue,
                }
                rows.push(MtpRow { token: ultima, pos: c.pos as i32, seq: c.seq as i32, logits: true });
                scelte.push(i);
            }
            if rows.is_empty() {
                break;
            }
            if let Err(e) = self.m.run_mtp(&rows, &stati) {
                for &i in &scelte {
                    let (conv, seq) = (catene[i].conv, catene[i].seq);
                    self.mtp_spegni(conv, seq, &e);
                    catene[i].k = 0;
                }
                break;
            }
            for (r, &i) in scelte.iter().enumerate() {
                let c = &mut catene[i];
                match self.m.mtp_logits(r) {
                    Ok(l) => {
                        let d = crate::sampling::argmax(l) as i32;
                        if let Some(reg) = &mut self.registro_mtp {
                            reg.write(c.conv, c.step, rows[r].token, d, c.rifiutate, prof as u32 - 1, l);
                        }
                        c.draft.push(d);
                        c.row = r;
                        c.pos += 1;
                    }
                    Err(e) => {
                        let (conv, seq) = (c.conv, c.seq);
                        c.k = 0;
                        self.mtp_spegni(conv, seq, &e);
                    }
                }
            }
        }
    }

    /// Update draft context during generation using only the accepted prefix of
    /// each conversation. Row i supplies the hidden state for its sampled
    /// token at the following position. Rewind overlapping cached rows before
    /// appending verified ones. Request logits on each conversation's last
    /// accepted row, then extend to its requested depth. Return the row index
    /// and proposals so the reply attaches them to the correct sample.
    fn mtp_passo(&mut self, kept: &[pb::StepRow], groups: &[(usize, usize, usize)], per_riga: &[Option<(usize, f32)>]) -> Vec<(usize, Vec<i32>)> {
        if self.mtp.is_empty() || !self.m.ha_mtp() {
            return Vec::new();
        }
        let n_h = self.m.mtp_state_width;
        let mut rows: Vec<MtpRow> = Vec::new();
        let mut stati: Vec<f32> = Vec::new();
        // Per-conversation draft metadata, indexed by the row supplying continuation logits.
        let mut attese: Vec<(u64, u32, usize, usize, u32, u32, usize)> = Vec::new();
        for &(j, n, gl) in groups {
            let conv = kept[j].conversation_id;
            let Some(prossima) = self.mtp.get(&conv).map(|s| s.prossima) else { continue };
            let Some(seq) = self.slots.slot(conv).map(|p| p.1) else { continue };
            let campionati: Option<Vec<i32>> = (0..n).map(|i| per_riga[j + i].map(|x| x.0 as i32)).collect();
            let Some(campionati) = campionati else {
                self.mtp_spegni(conv, seq, "row has no token");
                continue;
            };
            // Drop draft state when no following position can be used.
            let pos0 = kept[j].position + 1;
            if self.m.e_eog(campionati[n - 1]) || pos0 + n as u32 > self.ctx_per_slot {
                self.mtp.remove(&conv);
                continue;
            }
            match piano_mtp(prossima, pos0) {
                MtpPlan::Aggiungi => {}
                MtpPlan::Riavvolgi(p) => {
                    if let Err(e) = self.m.mtp_riavvolgi(seq, p) {
                        self.mtp_spegni(conv, seq, &e);
                        continue;
                    }
                }
                MtpPlan::Buco => {
                    self.mtp_spegni(conv, seq, &format!("row at position {} with the MTP cache at {prossima}", pos0 - 1));
                    continue;
                }
            }
            let mut ok = true;
            for (i, &t) in campionati.iter().enumerate() {
                match self.m.mtp_state(j + i) {
                    Ok(h) => stati.extend_from_slice(h),
                    Err(_) => {
                        ok = false;
                        break;
                    }
                }
                rows.push(MtpRow { token: t, pos: (pos0 as usize + i) as i32, seq: seq as i32, logits: i + 1 == n });
            }
            if !ok {
                // Remove queued rows for this conversation before decoding the batch.
                rows.retain(|r| r.seq != seq as i32);
                stati.truncate(rows.len() * n_h);
                self.mtp_spegni(conv, seq, "missing row state");
                continue;
            }
            attese.push((conv, seq, j + n - 1, rows.len() - 1, pos0 + n as u32, (gl - n) as u32, self.profondita(kept[j].draft_count)));
        }
        if rows.is_empty() {
            return Vec::new();
        }
        if let Err(e) = self.m.run_mtp(&rows, &stati) {
            for &(conv, seq, ..) in &attese {
                self.mtp_spegni(conv, seq, &e);
            }
            return Vec::new();
        }
        let mut catene: Vec<MtpChain> = Vec::with_capacity(attese.len());
        let mut ultime: Vec<usize> = Vec::with_capacity(attese.len());
        for (conv, seq, ultima, riga_mtp, done, rifiutate, k) in attese {
            if let Some(st) = self.mtp.get_mut(&conv) {
                st.prossima = done;
            }
            match self.m.mtp_logits(riga_mtp) {
                Ok(l) => {
                    let d = crate::sampling::argmax(l) as i32;
                    let step = kept[ultima].conversation_step;
                    if let Some(r) = &mut self.registro_mtp {
                        r.write(conv, step, rows[riga_mtp].token, d, rifiutate, 0, l);
                    }
                    catene.push(MtpChain { conv, seq, row: riga_mtp, pos: done, k, step, rifiutate, draft: vec![d] });
                    ultime.push(ultima);
                }
                Err(e) => self.mtp_spegni(conv, seq, &e),
            }
        }
        self.mtp_catena(&mut catene);
        let mut bozze = Vec::with_capacity(catene.len());
        for (c, ultima) in catene.into_iter().zip(ultime) {
            // Include tentative positions in the cache boundary so the next step can rewind them.
            if let Some(st) = self.mtp.get_mut(&c.conv) {
                st.prossima += c.draft.len().saturating_sub(1) as u32;
            }
            bozze.push((ultima, c.draft));
        }
        bozze
    }

    fn close(&mut self, c: pb::Close) -> Result<(), String> {
        let e = self.slots.close(c.conversation_id, c.attempt_id);
        if let Some(l) = e.libera {
            // `l.seq` was owned by this conversation, so clearing it cannot touch another one (see `Slots::proprietario`).
            self.m.svuota_posto(l.seq);
            self.m.mtp_svuota_posto(l.seq);
            self.mtp.remove(&c.conversation_id);
            self.campion.remove(&c.conversation_id);
            self.formato_conv.remove(&c.conversation_id);
        }
        // Same stream as `Open` (prefill), see `rete::su_generazione`.
        if let Err(e) = self.forward_prefill(msg(Body::Close(c))) {
            eprintln!("[stage] could not forward Close: {e}");
        }
        Ok(())
    }

    fn evento(&mut self, f: pb::Stream, m: pb::Message) -> Result<(), String> {
        match m.body {
            Some(Body::Open(a)) => self.open(a),
            Some(Body::Block(b)) => self.block(b),
            Some(Body::Step(p)) => self.step(p),
            Some(Body::Close(c)) => self.close(c),
            Some(Body::Confirmation(c)) => {
                // Relay from successor to predecessor using our incoming control epoch, which the head
                // knows, and the source index. Log if sending fails.
                let mut c = c;
                c.epoch = self.epoca_ingresso;
                let (conv, att, da) = (c.conversation_id, c.attempt_id, c.stage);
                if let Err(e) = send_control(self.conferme.as_ref(), &msg(Body::Confirmation(c))) {
                    eprintln!("[stage] could not forward Confirmation from stage {da} for conversation {conv}, attempt {att}: {e}");
                }
                Ok(())
            }
            altro => {
                // An unrelated message does not stop the stage.
                eprintln!("[stage] ignoring unexpected message on stream {}: {altro:?}", network::stream_name(f));
                Ok(())
            }
        }
    }
}

pub fn run(m: Engine, o: &StageOptions) -> Result<(), String> {
    match o.role {
        Role::Tail if !m.tail => return Err("tail must include the final layer".into()),
        Role::Middle if m.tail || m.head => return Err("a middle stage must exclude the first and final layers".into()),
        _ => {}
    }
    let timeout = Duration::from_secs(o.attesa_s);
    let listener = network::listen(&o.listen)?;
    let (tx, rx) = channel::<Event>();

    let mut next = None;
    let mut token = None;
    match o.role {
        Role::Middle => {
            let a = o.next.as_ref().ok_or("middle stage requires --next")?;
            next = Some(connect_forward(a, timeout, &tx, 1)?);
        }
        Role::Tail => {
            let t = o.testa_token.as_ref().ok_or("tail requires --head-token")?;
            // Advertise draft support during the token handshake so the head can reject
            // a tail that cannot provide the configured draft source.
            let (_l, w) = network::connect_tokens(t, timeout, 1, m.ha_mtp())?;
            token = Some(Writer::new(w, "token"));
        }
    }
    if m.ha_mtp() && o.role != Role::Tail {
        return Err("--mtp requires the tail".into());
    }
    let registro_mtp = match (&o.registro_mtp, m.ha_mtp()) {
        (Some(p), true) => Some(MtpTrace::open(p, m.n_vocab)?),
        (Some(_), false) => return Err("--log-mtp requires the tail with --mtp".into()),
        (None, _) => None,
    };
    let txa = tx.clone();
    std::thread::spawn(move || accept_streams(listener, txa));

    let mut s = Stage {
        m,
        role: o.role,
        state_format: o.state_format,
        slots: Slots::new(SystemClock::new(), o.scadenze),
        ctx_per_slot: o.ctx_per_slot,
        campion: HashMap::new(),
        conferme: None,
        epoca_ingresso: 0,
        index: o.index,
        next,
        indirizzo_avanti: if o.role == Role::Middle { o.next.clone() } else { None },
        epoca_avanti: 1,
        avanti_fallito: false,
        ricollegando: false,
        timeout,
        tx,
        token,
        esp: Vec::new(),
        registra_top: o.top.is_some(),
        thread_campiona: o.thread_campiona,
        top: Vec::new(),
        prova_errore: o.prova_errore_conv,
        prova_pilota: o.prova_pilota,
        max_draft_tokens: o.max_draft_tokens,
        mtp: HashMap::new(),
        registro_mtp,
        prova_viste: 0,
        formato_ingresso: o.formato_ingresso,
        formato_conv: HashMap::new(),
        timings: StageTimings::default(),
        tempi_sporchi: false,
        tempi_inviati: Instant::now(),
    };
    eprintln!("[stage] {:?} layer [{}, {}) ready on {}", o.role, s.m.layer_start, s.m.layer_end, o.listen);
    if s.m.ha_mtp() {
        eprintln!("[stage] MTP block loaded; drafting requires prompt token IDs from the head");
    }
    // Open prefill/generation streams from the predecessor, and when the last one closed after seeing any.
    let (mut vivi, mut visti, mut senza_flussi_da): (i64, u64, Option<Instant>) = (0, 0, None);
    let grazia = Duration::from_secs(o.grazia_s);
    let mut coda_ev = EventQueue::new(rx);
    // Wait since the last work message, summing 500 ms timeouts. This includes the wait for the next step.
    let mut attesa_acc = Duration::ZERO;
    let esito: Result<(), String> = loop {
        let t_att = Instant::now();
        let r = coda_ev.prossimo(Duration::from_millis(500));
        crate::timings::aggiungi(crate::timings::Fase::Attesa, t_att.elapsed());
        attesa_acc += t_att.elapsed();
        match r {
            Prossimo::Event(Event::Control(sc, epoch)) => {
                s.conferme = Some(sc);
                s.epoca_ingresso = epoch;
            }
            Prossimo::Event(Event::NuovoFlusso) => {
                vivi += 1;
                visti += 1;
                senza_flussi_da = None;
            }
            Prossimo::Event(Event::AvantiCaduto(ep)) => {
                if s.next.as_ref().is_some_and(|a| a.epoch == ep) {
                    s.errore_catena("next stage closed its control stream");
                    s.begin_reconnect();
                }
            }
            Prossimo::Event(Event::Ricollegato(r, ep)) => s.ricollegato(r, ep),
            Prossimo::Event(Event::Msg(f, m)) => {
                let t0 = Instant::now();
                let (rows, lavoro) = match &m.body {
                    Some(Body::Step(p)) => (p.rows.len() as u32, true),
                    Some(Body::Open(_) | Body::Block(_) | Body::Close(_)) => (0, true),
                    _ => (0, false),
                };
                let step = matches!(m.body, Some(Body::Step(_)));
                if let Err(e) = s.evento(f, m) {
                    break Err(e);
                }
                if step && rows > 0 {
                    s.timings.step(rows, attesa_acc, t0);
                } else if lavoro {
                    s.timings.lavoro(t0);
                }
                if lavoro {
                    s.tempi_sporchi = true;
                    attesa_acc = Duration::ZERO;
                }
            }
            Prossimo::Event(Event::Done(f, err)) => {
                if let Some(e) = err {
                    eprintln!("[stage] stream {} closed with error: {e}", network::stream_name(f));
                }
                vivi -= 1;
                if vivi <= 0 && visti >= 2 {
                    senza_flussi_da.get_or_insert_with(Instant::now);
                }
            }
            Prossimo::Vuoto => s.send_timings(true), // Idle is a chance to publish final timings before the upstream link closes.
            Prossimo::Chiuso => break Ok(()),
        }
        s.send_timings(false);
        if senza_flussi_da.is_some_and(|t| t.elapsed() >= grazia) {
            break Ok(());
        }
        // Check expiry only when the queue is empty, so a `PrefillBlock` that arrived during computation can
        // renew its slot first (see `EventQueue`).
        if !coda_ev.nessuno_in_attesa() {
            continue;
        }
        for sc in s.slots.scadenze() {
            s.m.svuota_posto(sc.liberato.seq);
            s.m.mtp_svuota_posto(sc.liberato.seq);
            s.mtp.remove(&sc.conversation_id);
            s.campion.remove(&sc.conversation_id);
            s.formato_conv.remove(&sc.conversation_id);
            s.error(sc.conversation_id, sc.liberato.tentativo, "slot expired".into());
        }
    };
    if let Some(p) = &o.experts {
        match experts::write(p, &s.esp) {
            Ok(()) => eprintln!("[stage] wrote {} expert trace rows to {p} ({} batches with missing rows)", s.esp.len(), s.m.esperti_persi),
            Err(e) => eprintln!("[stage] could not write the expert trace: {e}"),
        }
    }
    if let Some(p) = &o.top {
        match crate::sampling::scrivi_top(p, &s.top) {
            Ok(()) => eprintln!("[stage] wrote {} top-{TOP_K} trace rows to {p}", s.top.len()),
            Err(e) => eprintln!("[stage] could not write the top token trace: {e}"),
        }
    }
    if let Some(r) = &mut s.registro_mtp {
        use std::io::Write;
        match r.f.flush() {
            Ok(()) => eprintln!("[stage] wrote {} MTP drafts to {}", r.n, o.registro_mtp.as_deref().unwrap_or("")),
            Err(e) => eprintln!("[stage] could not write the MTP trace: {e}"),
        }
    }
    eprintln!("[stage] timings {}", crate::timings::istantanea());
    s.tempi_sporchi = true;
    s.send_timings(true);
    eprintln!(
        "[stage] timings-stadio {}",
        s.timings.json(s.index, if s.role == Role::Tail { "tail" } else { "middle" })
    );
    if let Some(a) = s.next.take() {
        a.prefill.terminate();
        a.gen.terminate();
    }
    if let Some(t) = s.token.take() {
        t.terminate();
    }
    if let Some(c) = s.conferme.take() {
        c.terminate();
    }
    esito
}

/// Conversation sampling as declared by `Open`: temperature, root seed and cutoffs (0 = no cutoff).
fn sampling_from_open(c: &pb::Sampling) -> (f32, u64, Tagli) {
    (c.temperature, c.root_seed, Tagli { top_p: c.top_p, top_k: c.top_k })
}

#[cfg(test)]
mod test {
    use super::*;

    /// The tail uses the cutoffs declared by Open. A top_k of one forces
    /// the top token even at a nonzero temperature.
    #[test]
    fn tail_reads_sampling_cutoffs_from_open() {
        use prost::Message;
        let open = pb::Open { sampling: Some(pb::Sampling { temperature: 1.5, root_seed: 9, top_p: 0.0, top_k: 1 }), ..Default::default() };
        let letto = pb::Open::decode(open.encode_to_vec().as_slice()).unwrap();
        let (temp, seed, tagli) = sampling_from_open(&letto.sampling.unwrap());
        assert_eq!(tagli, Tagli { top_p: 0.0, top_k: 1 });
        let l: Vec<f32> = (0..64).map(|i| ((i * 37 % 64) as f32) * 0.1).collect();
        let am = crate::sampling::argmax(&l);
        for step in 0..100u32 {
            assert_eq!(campiona_tagli(&l, temp, seed, step, tagli).unwrap().0, am);
        }
        // With cutoffs absent, seeded sampling uses the full distribution.
        let vecchio = pb::Sampling { temperature: 1.5, root_seed: 9, ..Default::default() };
        let (t2, s2, tg2) = sampling_from_open(&vecchio);
        assert!(tg2.nessuno());
        for step in 0..50u32 {
            assert_eq!(campiona_tagli(&l, t2, s2, step, tg2).unwrap(), crate::sampling::sampling(&l, 1.5, 9, step).unwrap());
        }
    }

    #[test]
    fn confirmation_without_control_stream_returns_error() {
        let m = msg(Body::Confirmation(pb::Confirmation { kind: pb::AckKind::Error as i32, conversation_id: 1, attempt_id: 1, ok: false, error: "x".into(), ..Default::default() }));
        let e = send_control(None, &m).unwrap_err();
        assert!(e.contains("control"), "{e}");
    }
}
