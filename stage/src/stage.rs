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
//! Ingress budgets bound how much data peers can queue while compute is busy.
//!
//! Control messages carry cumulative timings and faults. Keep the source
//! stage when relaying a confirmation and stamp it with the upstream link
//! epoch so the head can recognize retired links.

use crate::sampling::{campiona_tagli, Tagli};
use crate::experts::{self, RecEsperti};
use crate::draft::{piano_mtp, MtpPlan};
use crate::engine::{Engine, Row, MtpRow};
use crate::slot::{OpenOutcome, WorkOutcome, SystemClock, Slots, Deadlines};
use crate::network::{self, encode_state, msg, validate_prefill_block, validate_step, IngressBudget, IngressPermit, Writer};
use publicompute_proto::stage::{self as pb, message::Body};
use std::collections::HashMap;
use std::net::{Shutdown, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, SendError, Sender};
use crate::timings::StageTimings;
use std::sync::{Arc, Mutex};
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
    pub ingress_max_mib: usize,
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
    Work(u64, pb::Message, IngressPermit, QueuedWork),
    Confirmation(u32, pb::Confirmation, IngressPermit),
    Done(IngressPermit),
    /// Predecessor opened the control stream with the epoch declared in its `Hello`.
    Control(Writer, u64, u32, IngressPermit),
    /// Predecessor opened a prefill or generation stream.
    NuovoFlusso,
    /// Successor's control stream closed (given epoch).
    AvantiCaduto(u32),
    /// Result of reconnecting an outgoing link on a separate thread.
    Ricollegato(Result<Downstream, String>, u32),
}

#[derive(Clone)]
struct EventTx {
    tx: Sender<Event>,
    compute: std::thread::Thread,
}

impl EventTx {
    fn channel() -> (Self, Receiver<Event>) {
        let (tx, rx) = channel();
        (Self { tx, compute: std::thread::current() }, rx)
    }

    fn send(&self, event: Event) -> Result<(), SendError<Event>> {
        self.tx.send(event)?;
        self.compute.unpark();
        Ok(())
    }
}

const INGRESS_WAIT: Duration = Duration::from_secs(15);
const FORWARD_WAIT: Duration = Duration::from_millis(100);
const MAX_DATA_EVENTS: usize = 1024;
const MAX_DATA_READERS: usize = 8;

type Attempts = Arc<Mutex<HashMap<(u64, u32), usize>>>;

struct QueuedWork {
    attempts: Attempts,
    keys: Vec<(u64, u32)>,
}

impl QueuedWork {
    fn new(attempts: &Attempts, message: &pb::Message) -> Self {
        let keys: Vec<_> = match message.body.as_ref() {
            Some(Body::Open(open)) => vec![(open.conversation_id, open.attempt_id)],
            Some(Body::Block(block)) => vec![(block.conversation_id, block.attempt_id)],
            Some(Body::Close(close)) => vec![(close.conversation_id, close.attempt_id)],
            Some(Body::Step(step)) => step.rows.iter().map(|row| (row.conversation_id, row.attempt_id)).collect(),
            _ => Vec::new(),
        };
        let mut pending = attempts.lock().unwrap();
        for &key in &keys {
            *pending.entry(key).or_default() += 1;
        }
        Self { attempts: attempts.clone(), keys }
    }
}

impl Drop for QueuedWork {
    fn drop(&mut self) {
        let mut pending = self.attempts.lock().unwrap();
        for &key in &self.keys {
            let count = pending.get_mut(&key).unwrap();
            *count -= 1;
            if *count == 0 {
                pending.remove(&key);
            }
        }
    }
}

#[derive(Default)]
struct Upstream {
    ready: bool,
    downstream_epoch: u32,
    link: u64,
    epoch: u32,
    failed: bool,
    draining: bool,
    sockets: [Option<TcpStream>; 3],
}

impl Upstream {
    fn close(&mut self) {
        for socket in &mut self.sockets {
            if let Some(socket) = socket.take() {
                let _ = socket.shutdown(Shutdown::Both);
            }
        }
    }

    fn attach(&mut self, kind: pb::Stream, epoch: u32, socket: &TcpStream) -> Result<u64, String> {
        if !self.ready {
            return Err("stage link is unavailable".into());
        }
        let no_data = self.sockets[..2].iter().all(Option::is_none);
        // A restarted peer can start its wire epoch from one again.
        if kind == pb::Stream::Prefill && (epoch > self.epoch || no_data) {
            self.close();
            self.link += 1;
            self.epoch = epoch;
            self.failed = false;
            self.draining = false;
        }
        let dest = &mut self.sockets[kind as usize - 1];
        if self.failed || self.draining || epoch != self.epoch || dest.is_some() {
            return Err("stage stream already in use or retired".into());
        }
        *dest = Some(socket.try_clone().map_err(|e| e.to_string())?);
        Ok(self.link)
    }

    fn active(&self, link: u64) -> bool {
        self.can_drain(link) && !self.draining
    }

    fn can_drain(&self, link: u64) -> bool {
        self.ready && !self.failed && self.link == link
    }

    fn data_eof(&mut self, kind: pb::Stream, link: u64) {
        if !self.can_drain(link) {
            return;
        }
        self.sockets[kind as usize - 1] = None;
        self.draining = true;
        // Wake the predecessor at once. Frames already sent on either data stream
        // may still drain until this link is replaced, including final Closes.
        if let Some(control) = self.sockets[pb::Stream::Control as usize - 1].take() {
            let _ = control.shutdown(Shutdown::Both);
        }
    }

    fn fail(&mut self, link: u64) {
        if self.link == link {
            self.failed = true;
            self.close();
        }
    }

    fn control_failed(&mut self, link: u64) {
        if self.active(link) {
            self.fail(link);
        }
    }

    fn downstream_lost(&mut self, epoch: u32) {
        if self.downstream_epoch == epoch {
            self.ready = false;
            self.fail(self.link);
        }
    }
}

/// Link to the successor (intermediate only).
struct Forward {
    prefill: Writer,
    gen: Writer,
    /// Keeps the write side of the control stream open.
    ctl: TcpStream,
    /// The control reader sets this to `true` as soon as the successor drops.
    caduto: Arc<AtomicBool>,
    epoch: u32,
}

enum Downstream {
    Forward(Forward),
    Token(Writer),
}

fn connect_forward(address: &str, timeout: Duration, tx: &EventTx, epoch: &mut u32, control_budget: Arc<IngressBudget>, upstream: Arc<Mutex<Upstream>>) -> Result<Forward, String> {
    let [(lp, wp), (lg, wg), (mut lc, wc)] = network::connect_forward(address, timeout, epoch)?;
    let epoch = *epoch;
    upstream.lock().unwrap().downstream_epoch = epoch;
    let caduto = Arc::new(AtomicBool::new(false));
    let (txc, cad) = (tx.clone(), caduto.clone());
    std::thread::spawn(move || {
        while let Ok(Some((message, permit))) = lc.read_admitted(&control_budget, INGRESS_WAIT, pb::Stream::Control, 0) {
            let Some(Body::Confirmation(c)) = message.body else { break };
            if c.kind == pb::AckKind::Error as i32 && !c.ok && c.conversation_id == 0 {
                eprintln!("[stage] next stage reported a chain fault: {}", c.error);
                break;
            }
            if txc.send(Event::Confirmation(epoch, c, permit)).is_err() {
                return;
            }
        }
        cad.store(true, Ordering::SeqCst);
        upstream.lock().unwrap().downstream_lost(epoch);
        let _ = txc.send(Event::AvantiCaduto(epoch));
    });
    drop((lp, lg));
    Ok(Forward { prefill: Writer::new(wp, "forward-prefill"), gen: Writer::new(wg, "forward-generation"), ctl: wc, caduto, epoch })
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
    upstream: Arc<Mutex<Upstream>>,
    /// Predecessor's control stream epoch (from `Hello`), stamped on outgoing `Confirmation` messages.
    epoca_ingresso: u32,
    link_ingresso: u64,
    /// Stage position in the chain (0 = unspecified).
    index: u32,
    next: Option<Forward>,
    /// Address and epoch of the outgoing link, to the successor or to the head.
    indirizzo_avanti: String,
    epoca_avanti: u32,
    avanti_fallito: bool,
    /// A thread is already reconnecting the successor.
    ricollegando: bool,
    timeout: Duration,
    tx: EventTx,
    control_budget: Arc<IngressBudget>,
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

struct WorkIngress {
    budget: Arc<IngressBudget>,
    attempts: Attempts,
    n_batch: usize,
    wait: Duration,
}

fn accept_streams(
    l: std::net::TcpListener, tx: EventTx, work_tx: EventTx,
    upstream: Arc<Mutex<Upstream>>, work: WorkIngress,
) {
    let data_readers = IngressBudget::new(0, MAX_DATA_READERS);
    let control_readers = IngressBudget::new(0, 4);
    network::accept(l, move |f, mut lett, mut w| {
        let epoch = lett.epoca_hello;
        match f {
            pb::Stream::Prefill | pb::Stream::Generation => {
                let Ok(reader) = data_readers.reserve(0, Duration::ZERO) else { return };
                let Ok(link) = upstream.lock().unwrap().attach(f, epoch, &w) else { return };
                if network::ack_ok(&mut w).is_err() {
                    upstream.lock().unwrap().fail(link);
                    return;
                }
                let _ = tx.send(Event::NuovoFlusso);
                loop {
                    match lett.read_admitted(&work.budget, work.wait, f, work.n_batch) {
                        Ok(Some((message, permit))) => {
                            let work = QueuedWork::new(&work.attempts, &message);
                            if work_tx.send(Event::Work(link, message, permit, work)).is_err() {
                                break;
                            }
                        }
                        Ok(None) => {
                            upstream.lock().unwrap().data_eof(f, link);
                            break;
                        }
                        Err(error) => {
                            upstream.lock().unwrap().fail(link);
                            eprintln!("[stage] stream {} closed with error: {error}", network::stream_name(f));
                            break;
                        }
                    }
                }
                let _ = w.shutdown(Shutdown::Both);
                // Start the reconnect grace after this reader's queued work drains.
                let _ = work_tx.send(Event::Done(reader));
            }
            pb::Stream::Control => {
                let Ok(reader) = control_readers.reserve(0, Duration::ZERO) else { return };
                let Ok(link) = upstream.lock().unwrap().attach(f, epoch, &w) else { return };
                let _ = tx.send(Event::Control(Writer::new(w, "conferme"), link, epoch, reader));
            }
            altro => eprintln!("[stage] unexpected stream {} on this listener", network::stream_name(altro)),
        }
    });
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
        if let Err(e) = self.send_control(&m, FORWARD_WAIT) {
            self.upstream.lock().unwrap().control_failed(self.link_ingresso);
            eprintln!("[stage] could not send Confirmation for conversation {conv}, attempt {att} (ok={ok}, error=\"{error}\"): {e}");
        }
    }

    fn error(&self, conv: u64, att: u32, text: String) {
        eprintln!("[stage] conversation {conv}, attempt {att}: {text}");
        self.confirmation(pb::AckKind::Error, conv, att, false, &text);
    }

    fn send_control(&self, m: &pb::Message, wait: Duration) -> Result<(), String> {
        let upstream = self.upstream.lock().unwrap();
        if !upstream.can_drain(self.link_ingresso) {
            return Err("upstream link retired".into());
        }
        if upstream.draining {
            return Ok(());
        }
        drop(upstream);
        self.conferme.as_ref().ok_or("control stream is disconnected")?.send_with_wait(m, wait)
    }

    fn send_token(&self, token: pb::Token) -> Result<(), String> {
        let writer = self.token.as_ref().ok_or("no connection to the head")?;
        if let Err(e) = writer.send_with_wait(&msg(Body::Token(token)), FORWARD_WAIT) {
            writer.abort();
            self.upstream.lock().unwrap().downstream_lost(self.epoca_avanti);
            return Err(e);
        }
        Ok(())
    }

    /// Queue output to the successor while leaving compute time for control and slot expiry.
    fn forward(&mut self, gen: bool, m: pb::Message) -> Result<(), String> {
        if self.role == Role::Tail {
            return Ok(());
        }
        if self.next.as_ref().map_or(true, |a| a.caduto.load(Ordering::SeqCst)) {
            self.begin_reconnect();
            return Err("reconnecting to the next stage".into());
        }
        let a = self.next.as_ref().ok_or("no downstream connection")?;
        match if gen { a.gen.send_with_wait(&m, FORWARD_WAIT) } else { a.prefill.send_with_wait(&m, FORWARD_WAIT) } {
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

    /// Reopen upstream handshakes only after the outgoing link is ready.
    fn begin_reconnect(&mut self) {
        if self.ricollegando {
            return;
        }
        self.ricollegando = true;
        self.upstream.lock().unwrap().downstream_lost(self.epoca_avanti);
        if let Some(old) = self.next.take() {
            let _ = old.ctl.shutdown(Shutdown::Both);
            old.prefill.abort();
            old.gen.abort();
        }
        if let Some(token) = self.token.take() {
            token.abort();
        }
        let ind = self.indirizzo_avanti.clone();
        let timeout = if self.avanti_fallito { Duration::from_secs(2) } else { self.timeout };
        eprintln!("[stage] reconnecting to {ind} in the background for up to {} s", timeout.as_secs());
        self.epoca_avanti += 1;
        let (tx, mut epoch) = (self.tx.clone(), self.epoca_avanti);
        let control_budget = self.control_budget.clone();
        let upstream = self.upstream.clone();
        upstream.lock().unwrap().downstream_epoch = epoch;
        let (role, mtp) = (self.role, self.m.ha_mtp());
        std::thread::spawn(move || {
            let r = match role {
                Role::Middle => connect_forward(&ind, timeout, &tx, &mut epoch, control_budget, upstream).map(Downstream::Forward),
                Role::Tail => network::connect_tokens(&ind, timeout, epoch, mtp).map(|(_, w)| Downstream::Token(Writer::new(w, "token"))),
            };
            let _ = tx.send(Event::Ricollegato(r, epoch));
        });
    }

    fn ricollegato(&mut self, r: Result<Downstream, String>, epoch: u32) {
        self.ricollegando = false;
        self.epoca_avanti = epoch;
        match r {
            Ok(Downstream::Forward(a)) => {
                self.upstream.lock().unwrap().ready = !a.caduto.load(Ordering::SeqCst);
                self.next = Some(a);
            }
            Ok(Downstream::Token(writer)) => {
                self.token = Some(writer);
                self.upstream.lock().unwrap().ready = true;
            }
            Err(e) => {
                self.avanti_fallito = true;
                eprintln!("[stage] outgoing link unreachable: {e}");
                return;
            }
        }
        self.avanti_fallito = false;
        eprintln!("[stage] reconnected outgoing link (epoch {epoch})");
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
        let _ = self.send_control(&m, Duration::ZERO);
    }

    /// Empty reply for a round with no computation. The head still expects a reply for the round.
    fn giro_vuoto(&mut self, p: &pb::Step) {
        let r = if self.role == Role::Tail {
            self.send_token(pb::Token { group_id: p.group_id, group_round: p.group_round, prefill: false, rows: vec![], stage: self.index })
        } else {
            self.forward_generation(msg(Body::Step(pb::Step { rows: vec![], state: vec![], ..p.clone() })))
        };
        if let Err(e) = r {
            eprintln!("[stage] could not send empty round: {e}");
        }
    }

    fn open(&mut self, a: pb::Open) -> Result<(), String> {
        if !self.m.posto_valido(a.slot) {
            self.error(a.conversation_id, a.attempt_id, format!("slot {:#x} is outside the configured range (slots per group {:?})", a.slot, self.m.per_ctx));
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
    fn block(&mut self, mut b: pb::PrefillBlock) -> Result<(), String> {
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
        b.state = Vec::new();
        if let Err(e) = self.blocco_calcola(b, state) {
            self.error(conv, att, e);
        }
        Ok(())
    }

    fn blocco_calcola(&mut self, b: pb::PrefillBlock, state: Vec<f32>) -> Result<(), String> {
        let n = b.n_token as usize;
        let seq = self.slots.slot(b.conversation_id).map(|p| p.1).ok_or("slot disappeared")?;
        let tail = self.role == Role::Tail;
        let rows: Vec<Row> =
            (0..n).map(|i| Row { token: 0, pos: (b.start_position as usize + i) as i32, seq: seq as i32, logits: tail && b.last && i + 1 == n }).collect();
        let out = self.m.run(&rows, Some(&state))?;
        match (self.role, out) {
            (Role::Middle, Some(s)) => {
                let block = network::with_prefill_state(b, &s, self.m.n_embd, self.m.stream_count, self.state_format);
                self.forward_prefill(msg(Body::Block(block)))?;
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
                    let draft = self.mtp_blocco(&b, seq, Some(t as i32));
                    let tok = pb::Token {
                        group_id: b.group_id,
                        group_round: 0,
                        prefill: true,
                        rows: vec![pb::TokenRow { conversation_id: b.conversation_id, attempt_id: b.attempt_id, conversation_step: b.first_token_step, token: t as i32, logprob: lp, draft }],
                        stage: self.index,
                    };
                    self.send_token(tok)?;
                } else {
                    self.mtp_blocco(&b, seq, None);
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
                let invio = self.send_token(pb::Token {
                    group_id: p.group_id, group_round: p.group_round, prefill: false, rows: out_righe, stage: self.index,
                });
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

    fn evento(&mut self, m: pb::Message) -> Result<(), String> {
        match m.body {
            Some(Body::Open(a)) => self.open(a),
            Some(Body::Block(b)) => self.block(b),
            Some(Body::Step(p)) => self.step(p),
            Some(Body::Close(c)) => self.close(c),
            _ => Ok(()),
        }
    }

    fn relay(&self, mut c: pb::Confirmation) {
        c.epoch = self.epoca_ingresso;
        let (conv, att, da) = (c.conversation_id, c.attempt_id, c.stage);
        let wait = if c.kind == pb::AckKind::Timings as i32 { Duration::ZERO } else { FORWARD_WAIT };
        if let Err(e) = self.send_control(&msg(Body::Confirmation(c)), wait) {
            if !wait.is_zero() {
                self.upstream.lock().unwrap().control_failed(self.link_ingresso);
            }
            eprintln!("[stage] could not forward Confirmation from stage {da} for conversation {conv}, attempt {att}: {e}");
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
    let (tx, rx) = EventTx::channel();
    let (work_tx, work_rx) = EventTx::channel();
    let upstream = Arc::new(Mutex::new(Upstream { downstream_epoch: 1, ..Default::default() }));
    let ingress_bytes = o.ingress_max_mib.checked_mul(1 << 20).filter(|&bytes| bytes > 0).ok_or("invalid --ingress-max-mib")?;
    let budget = IngressBudget::new(ingress_bytes, MAX_DATA_EVENTS);
    let control_budget = IngressBudget::new(network::CONTROL_INGRESS_BYTES, 128);
    let attempts = Attempts::default();

    let mut next = None;
    let mut token = None;
    let mut epoca_avanti = 1;
    let address;
    match o.role {
        Role::Middle => {
            address = o.next.as_ref().ok_or("middle stage requires --next")?;
            let forward = connect_forward(address, timeout, &tx, &mut epoca_avanti, control_budget.clone(), upstream.clone())?;
            upstream.lock().unwrap().ready = !forward.caduto.load(Ordering::SeqCst);
            next = Some(forward);
        }
        Role::Tail => {
            address = o.testa_token.as_ref().ok_or("tail requires --head-token")?;
            // Advertise draft support during the token handshake so the head can reject
            // a tail that cannot provide the configured draft source.
            let (_l, w) = network::connect_tokens(address, timeout, 1, m.ha_mtp())?;
            token = Some(Writer::new(w, "token"));
            upstream.lock().unwrap().ready = true;
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
    let reader_upstream = upstream.clone();
    let ingress = WorkIngress { budget, attempts: attempts.clone(), n_batch: m.n_batch, wait: INGRESS_WAIT };
    std::thread::spawn(move || accept_streams(listener, txa, work_tx, reader_upstream, ingress));

    let mut s = Stage {
        m,
        role: o.role,
        state_format: o.state_format,
        slots: Slots::new(SystemClock::new(), o.scadenze),
        ctx_per_slot: o.ctx_per_slot,
        campion: HashMap::new(),
        conferme: None,
        upstream,
        epoca_ingresso: 0,
        link_ingresso: 0,
        index: o.index,
        next,
        indirizzo_avanti: address.clone(),
        epoca_avanti,
        avanti_fallito: false,
        ricollegando: false,
        timeout,
        tx,
        control_budget,
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
    let mut last_expiry = Instant::now();
    let mut last_reconnect = Instant::now();
    // Wait since the last work message, including the wait for the next step.
    let mut attesa_acc = Duration::ZERO;
    let esito: Result<(), String> = loop {
        let t_att = Instant::now();
        let next_event = || rx.try_recv().or_else(|_| work_rx.try_recv()).ok();
        let r = next_event().or_else(|| {
            std::thread::park_timeout(Duration::from_millis(100));
            next_event()
        });
        crate::timings::aggiungi(crate::timings::Fase::Attesa, t_att.elapsed());
        attesa_acc += t_att.elapsed();
        match r {
            Some(Event::Control(sc, link, epoch, _permit)) => {
                if s.upstream.lock().unwrap().active(link) {
                    s.link_ingresso = link;
                    s.epoca_ingresso = epoch;
                    // Install the epoch before the writer can acknowledge it to the predecessor.
                    let ack = msg(Body::Ack(pb::HelloAck { ok: true, error: String::new() }));
                    if sc.send_with_wait(&ack, Duration::ZERO).is_ok() {
                        s.conferme = Some(sc);
                    } else {
                        s.upstream.lock().unwrap().control_failed(link);
                    }
                }
            }
            Some(Event::NuovoFlusso) => {
                vivi += 1;
                visti += 1;
                senza_flussi_da = None;
            }
            Some(Event::AvantiCaduto(ep)) => {
                if s.next.as_ref().is_some_and(|a| a.epoch == ep) {
                    eprintln!("[stage] next stage closed its control stream");
                    s.begin_reconnect();
                }
            }
            Some(Event::Ricollegato(r, ep)) => {
                if r.is_ok() && senza_flussi_da.is_some() {
                    senza_flussi_da = Some(Instant::now());
                }
                s.ricollegato(r, ep);
            }
            Some(Event::Confirmation(ep, c, _permit)) => {
                if s.next.as_ref().is_some_and(|a| a.epoch == ep && !a.caduto.load(Ordering::SeqCst)) {
                    s.relay(c);
                }
            }
            Some(Event::Work(link, m, _permit, _work)) => {
                if s.link_ingresso != link || !s.upstream.lock().unwrap().can_drain(link) {
                    drop((_permit, _work));
                } else {
                    let t0 = Instant::now();
                    let rows = match &m.body {
                        Some(Body::Step(p)) => p.rows.len() as u32,
                        _ => 0,
                    };
                    if let Err(e) = s.evento(m) {
                        break Err(e);
                    }
                    if rows > 0 {
                        s.timings.step(rows, attesa_acc, t0);
                    } else {
                        s.timings.lavoro(t0);
                    }
                    s.tempi_sporchi = true;
                    attesa_acc = Duration::ZERO;
                }
            }
            Some(Event::Done(_reader)) => {
                vivi -= 1;
                if vivi <= 0 && visti >= 2 {
                    senza_flussi_da.get_or_insert_with(Instant::now);
                }
            }
            None if attesa_acc >= Duration::from_millis(500) => s.send_timings(true),
            None => {}
        }
        let token_lost = s.token.as_ref().is_some_and(|writer| writer.failed() || !s.upstream.lock().unwrap().ready);
        if token_lost || s.next.as_ref().is_some_and(|a| a.caduto.load(Ordering::SeqCst) || a.prefill.failed() || a.gen.failed()) {
            s.begin_reconnect();
        }
        if s.avanti_fallito && last_reconnect.elapsed() >= Duration::from_millis(200) {
            s.begin_reconnect();
            last_reconnect = Instant::now();
        }
        if s.conferme.as_ref().is_some_and(Writer::failed) {
            s.upstream.lock().unwrap().control_failed(s.link_ingresso);
            s.conferme = None;
        }
        s.send_timings(false);
        let grace = if s.ricollegando || s.avanti_fallito { grazia.max(timeout) } else { grazia };
        if senza_flussi_da.is_some_and(|t| t.elapsed() >= grace) {
            break Ok(());
        }
        if last_expiry.elapsed() < Duration::from_millis(100) {
            continue;
        }
        last_expiry = Instant::now();
        let expired = {
            let pending = attempts.lock().unwrap();
            s.slots.expire_except(|id, attempt| pending.contains_key(&(id, attempt)))
        };
        s.slots.pulisci();
        for sc in expired {
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
        let _ = a.ctl.shutdown(Shutdown::Both);
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
    fn saturated_reader_closes_control_while_work_is_still_queued() {
        check_data_loss(false);
    }

    #[test]
    fn data_eof_retires_control_and_preserves_queued_work() {
        check_data_loss(true);
    }

    fn check_data_loss(eof: bool) {
        let listener = network::listen("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap().to_string();
        let (tx, rx) = EventTx::channel();
        let (work_tx, work_rx) = EventTx::channel();
        let upstream = Arc::new(Mutex::new(Upstream { ready: true, ..Default::default() }));
        let ingress = WorkIngress {
            budget: IngressBudget::new(1024, 1),
            attempts: Attempts::default(),
            n_batch: 4,
            wait: Duration::from_millis(20),
        };
        let reader_upstream = upstream.clone();
        std::thread::spawn(move || accept_streams(listener, tx, work_tx, reader_upstream, ingress));

        let timeout = Duration::from_secs(2);
        let (_read, mut prefill) = network::connect(&address, pb::Stream::Prefill, timeout, 1).unwrap();
        let _generation = network::connect(&address, pb::Stream::Generation, timeout, 1).unwrap();
        let mut control = TcpStream::connect(&address).unwrap();
        network::write_sync(&mut control, &msg(Body::Hello(pb::Hello {
            protocol_major: network::WIRE_VERSION.0,
            protocol_minor: network::WIRE_VERSION.1,
            stream: pb::Stream::Control as i32,
            epoch: 1,
            mtp: false,
        }))).unwrap();
        loop {
            if let Event::Control(writer, link, _, _) = rx.recv_timeout(timeout).unwrap() {
                writer.send_with_wait(&msg(Body::Ack(pb::HelloAck { ok: true, error: String::new() })), Duration::ZERO).unwrap();
                let raw = control.try_clone().unwrap();
                let mut reader = network::Reader::new(control);
                raw.set_read_timeout(Some(timeout)).unwrap();
                assert!(matches!(reader.read().unwrap().unwrap().body, Some(Body::Ack(_))));

                let open = msg(Body::Open(pb::Open { conversation_id: 1, attempt_id: 1, ..Default::default() }));
                network::write_sync(&mut prefill, &open).unwrap();
                if eof {
                    prefill.shutdown(Shutdown::Write).unwrap();
                } else {
                    network::write_sync(&mut prefill, &open).unwrap();
                }
                assert!(reader.read().unwrap().is_none());
                {
                    let upstream = upstream.lock().unwrap();
                    assert!(!upstream.active(link));
                    assert_eq!(upstream.can_drain(link), eof);
                }
                assert!(matches!(work_rx.try_recv().unwrap(), Event::Work(_, _, _, _)));
                if eof {
                    assert!(matches!(work_rx.recv_timeout(timeout).unwrap(), Event::Done(_)));
                }
                break;
            }
        }
    }

    #[test]
    fn reconnect_retries_all_streams_after_a_partial_handshake() {
        let listener = network::listen("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap().to_string();
        let peer = std::thread::spawn(move || {
            let mut upstream = Upstream { ready: true, ..Default::default() };
            let accept = || network::read_hello(listener.accept().unwrap().0).unwrap();
            let (stream, reader, mut socket) = accept();
            assert_eq!(stream, pb::Stream::Prefill);
            let old = upstream.attach(stream, reader.epoca_hello, &socket).unwrap();
            network::ack_ok(&mut socket).unwrap();

            let (stream, reader, socket) = accept();
            assert_eq!(stream, pb::Stream::Generation);
            assert_eq!(reader.epoca_hello, 1);
            upstream.fail(old);
            socket.shutdown(Shutdown::Both).unwrap();

            for expected in [pb::Stream::Prefill, pb::Stream::Generation, pb::Stream::Control] {
                let (stream, reader, mut socket) = accept();
                assert_eq!(stream, expected);
                assert_eq!(reader.epoca_hello, 2);
                assert_ne!(upstream.attach(stream, reader.epoca_hello, &socket).unwrap(), old);
                network::ack_ok(&mut socket).unwrap();
            }
        });
        let mut epoch = 1;
        let _streams = network::connect_forward(&address, Duration::from_secs(2), &mut epoch).unwrap();
        assert_eq!(epoch, 2);
        peer.join().unwrap();
    }

    #[test]
    fn retired_data_stream_cannot_close_new_control_stream() {
        let listener = network::listen("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let peer = TcpStream::connect(address).unwrap();
        let (control, _) = listener.accept().unwrap();
        let mut upstream = Upstream { ready: true, downstream_epoch: 7, ..Default::default() };
        let old = upstream.attach(pb::Stream::Prefill, 2, &control).unwrap();
        upstream.data_eof(pb::Stream::Prefill, old);
        let new = upstream.attach(pb::Stream::Prefill, 1, &control).unwrap();
        upstream.attach(pb::Stream::Control, 1, &control).unwrap();
        assert_ne!(old, new);
        upstream.data_eof(pb::Stream::Prefill, old);
        upstream.fail(old);
        upstream.downstream_lost(6);
        assert!(upstream.active(new));
        upstream.downstream_lost(7);
        assert!(!upstream.active(new));
        peer.set_read_timeout(Some(Duration::from_secs(1))).unwrap();
        assert_eq!(std::io::Read::read(&mut &peer, &mut [0]).unwrap(), 0);
        assert!(upstream.attach(pb::Stream::Prefill, 2, &control).is_err());
    }
}
