//! The head assigns slots, sends prefill blocks and generation steps, and
//! commits tokens returned by the tail. Its transcript accepts only the
//! active attempt's next logical step. Reference mode uses the same
//! scheduler in one process. Forced replay keeps the supplied tokens and
//! recomputes their probabilities.
//!
//! Each group has at most one generation step in flight. The scheduler
//! consumes group replies in send order to keep batch membership independent
//! of network timing. Admission can send one prompt block between steps,
//! and consumes the first token at a configured round. Round-robin startup
//! prefill can instead finish every initial prompt before generation.
//!
//! A conversation fault restarts only that conversation. A link fault or an
//! overdue reply triggers recovery for the whole chain. Recovery opens a
//! fresh attempt and replays the confirmed prefix before joining a group.
//! If a reply omits a live row, wait for its control error. Resending the
//! row would reuse a position the head has already computed.
//!
//! Service mode accepts local requests through the same admission path and
//! emits text as tokens are committed. It also releases completed request
//! state so memory does not grow with total requests served.

use crate::draft::{self, BozzaConv, ContatoriBozza, DraftSource};
use crate::admission::{parse_arrivals, parse_classes, Admission, Class};
use crate::sampling::{campiona_tagli, controlla_seme_tagli, logprob, seme_conversazione, PosSeme, Tagli};
use crate::conversation::{Action, Config, Conversation, TokenOutcome, Status};
use crate::experts::{self, RecEsperti};
use crate::engine::{Engine, Row};
use crate::safeguards::{conferma_catena_vecchia, Protezioni};
use crate::network::{self, encode_state, msg, Writer};
use crate::service::{Event as EvServ, Utf8Fragments, Service};
use crate::output::{self, RecordConv};
use publicompute_proto::service as pbs;
use publicompute_proto::stage::{self as pb, message::Body};
use std::collections::VecDeque;
use std::net::TcpStream;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{channel, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Order in which the prompts present at start-up (or arriving together in service mode) are prefilled.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PrefillOrder {
    /// One conversation at a time in prompt order (same machine as `ammissione()`): the first one starts generating
    /// as soon as it has its first token, the others catch up while generation goes on.
    #[default]
    Fifo,
    /// Prefill one block per conversation in turn, then read their first tokens.
    Turno,
}

impl PrefillOrder {
    pub fn da_testo(t: &str) -> Result<Self, String> {
        match t {
            "fifo" => Ok(PrefillOrder::Fifo),
            "round-robin" => Ok(PrefillOrder::Turno),
            _ => Err(format!("--prefill-order: invalid value \"{t}\" (fifo or round-robin)")),
        }
    }
}

pub struct HeadOptions {
    pub prompt: String,
    pub n_prompt: usize,
    /// Prompts to skip at the start of the file.
    pub salta: usize,
    pub max_tokens: u32,
    pub block: usize,
    pub groups: usize,
    pub temperature: f32,
    pub seed: u64,
    pub ctx_per_slot: u32,
    pub output: Option<String>,
    /// Maximum wait to connect the stages.
    pub attesa_s: u64,
    /// `T_lavoro`: an overdue reply triggers fault recovery instead of failing the run.
    pub t_lavoro_s: u64,
    /// `T_ripresa` in the conversation state machine.
    pub t_ripresa_s: u64,
    /// `K_riprese` in the conversation state machine.
    pub k_riprese: u32,
    /// Forced reference: JSONL with tokens to force (by `prompt_id`).
    pub forzati: Option<String>,
    /// Expert log file (requires `Engine` loaded with `experts`).
    pub experts: Option<String>,
    /// Top-k log file (20 highest log probabilities per step in reference and forced reference).
    pub top: Option<String>,
    /// Sampling threads per step (`--sampling-threads`, 0 = automatic, min(rows, 4), 1 = sequential).
    pub thread_campiona: usize,
    /// Degrade the residual to per-row int8 precision while sending it in
    /// float32. This exercises verification of unreported quantization.
    pub prova_stato_int8: bool,
    /// Wire encoding the head declares and sends on the downstream link.
    pub state_format: pb::StateFormat,
    /// Class list in prompt order. Omitted entries default to interactive,
    /// which can preempt an admitted batch.
    pub classi: Option<String>,
    /// Slots to keep free for interactive work when admitting a batch (`--interactive-reserve`, default 0).
    pub riserva_interattive: usize,
    /// In forced replay, a resampled token mismatch above this margin counts
    /// as suspicious. The total includes mismatches at every margin.
    pub tol_seme: f64,
    /// `forza` only: recompute in the shape of a production verifier (see `forma_verificatore`).
    pub forma_verificatore: bool,
    /// Deterministic arrivals (`--arrivals`): number of completed generation steps before each conversation arrives.
    pub arrivi: Option<String>,
    /// Number of admission rounds between the final prefill block and consuming
    /// its first token. Zero chooses twice the group count, with a minimum of two.
    /// One waits for the token in the next admission round.
    pub ritardo_primo: u64,
    /// Choose FIFO admission or round-robin startup prefill. The reference uses
    /// the same setting to match batch shapes.
    pub prefill_order: PrefillOrder,
    /// A loopback address enables persistent service. Agent requests enter
    /// admission instead of prompts read from a file.
    pub service: Option<String>,
    pub service_pipe: bool,
    /// Maximum speculative rows per conversation and step. Zero disables drafts.
    /// Set the same limit on every stage so recurrent snapshots cover any rewind.
    pub max_draft_tokens: u32,
    /// Draft no more rows once a `Step` hits this row cap (`--draft-max-rows`, default 64).
    pub max_draft_rows: usize,
    /// TEST ONLY: use a no-draft run's JSONL tokens (same prompts, temperature and seed) as drafts
    /// instead of prompt lookup. Optionally corrupt every Nth token (0 = none). See `BozzaConv::oracolo`.
    pub draft_oracle: Option<String>,
    pub draft_oracle_corrupt_every: u32,
    /// Choose prompt lookup or tail proposals. Tail drafting requires the
    /// capability in the token handshake and the prompt token IDs in Blocco.
    pub draft_source: DraftSource,
    /// TEST ONLY (`--test-mtp-depth N`): ask for N MTP drafts each step regardless of the head's row cap or adaptive
    /// K. With `--draft-max-rows 1`, the tail builds the full chain and the head sends no draft rows.
    /// The cache then holds only verified rows, giving reference logits for a run with rejections. 0 = off.
    pub mtp_k_prova: u32,
}

/// Quantize `state` (rows × `n_embd`) to int8 with a scale per row (max|x| / 127, round to nearest) and
/// convert back to FP32. This simulates a stage using 8-bit state to save bandwidth. An all-zero row stays zero.
pub fn degrada_int8(state: &[f32], n_embd: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(state.len());
    for row in state.chunks(n_embd.max(1)) {
        let m = row.iter().fold(0f32, |a, &x| if x.is_finite() { a.max(x.abs()) } else { a });
        if m == 0.0 {
            out.extend(std::iter::repeat(0.0).take(row.len()));
            continue;
        }
        let scala = m / 127.0;
        out.extend(row.iter().map(|&x| ((x / scala).round().clamp(-127.0, 127.0) as i8) as f32 * scala));
    }
    out
}

pub struct HeadLinks {
    /// First downstream stage, which takes all three connections.
    pub next: String,
    /// Where the tail connects for `Token`.
    pub ascolta_token: String,
}

/// Source of a dropped-link notice. The number identifies the connection so notices from links already
/// replaced during recovery can be ignored.
#[derive(Clone, Copy, Debug)]
enum Fonte {
    Control(u32),
    Token(u64),
}

enum HeadEvent {
    Token(pb::Token, Instant),
    Confirmation(pb::Confirmation),
    Guasto(Fonte, String),
}

/// Result of head work. A chain fault is recoverable, while other errors stop the run.
enum Fail {
    /// Chain fault: recover all live conversations.
    Guasto(String),
    /// Single-conversation fault (index in `convs`): recover only that conversation.
    GuastoConv(usize, String),
    Fatale(String),
}

impl From<String> for Fail {
    fn from(s: String) -> Self {
        Fail::Fatale(s)
    }
}

impl From<&str> for Fail {
    fn from(s: &str) -> Self {
        Fail::Fatale(s.to_string())
    }
}

type R<T = ()> = Result<T, Fail>;

struct Connected {
    prefill: Writer,
    gen: Writer,
    /// Keeps the write side of the control stream open.
    control: TcpStream,
    caduto: Arc<AtomicBool>,
    epoch: u32,
}

enum Downstream {
    Rete(Box<DownstreamLinks>),
    Locale,
}

struct DownstreamLinks {
    col: Connected,
    /// Last wire epoch attempted, even if the handshake failed.
    last_epoch: u32,
    rx: Receiver<HeadEvent>,
    tx: Sender<HeadEvent>,
    /// Number of the last accepted `Token` connection.
    tok_serial: Arc<AtomicU64>,
    /// `Token` drop notices with lower numbers belong to superseded connections.
    tok_min: u64,
}

/// Listener of the tokens: events, their sender, serial of the token connections and, with `--draft-source mtp`, the
/// reason why a tail was refused (it did not declare the MTP block).
type TokenListener = (Receiver<HeadEvent>, Sender<HeadEvent>, Arc<AtomicU64>, Arc<Mutex<Option<String>>>);

struct Conv {
    prompt_id: String,
    conv_id: u64,
    gruppo: usize,
    slot: u32,
    prompt: Vec<i32>,
    /// Conversation state machine (state, attempt, authoritative transcript).
    c: Conversation,
    /// `Open` to send (attempt, initial position), chosen by the state machine.
    da_aprire: Option<(u32, u32)>,
    /// Tokens to send through the stages for this attempt: prompt, plus recorded tokens on recovery.
    da_inviare: Vec<i32>,
    /// `conversation_step` of the token sampled on the last block (0, or the number already recorded).
    passo_primo: u32,
    inviati: usize,
    token: Vec<i32>,
    logprob: Vec<f32>,
    tempi_ms: Vec<f64>,
    pezzi: Vec<u8>,
    done: Option<&'static str>,
    /// The in-flight step returned without this row. A stage dropped it and sent an error `Confirmation`, which
    /// can arrive after the round's `Token`. Wait for the error because the same position cannot be resent.
    orfana: bool,
    /// Forced tokens (forced reference).
    forzati: Option<Vec<i32>>,
    /// Admitted (group and slot assigned, machine created). Faults and recovery leave unadmitted work alone.
    ammessa: bool,
    /// Arrival and admission start (`Open`) in ms since start, plus whether admitted during
    /// generation.
    t_arrivo_ms: Option<f64>,
    t_ammissione_ms: Option<f64>,
    in_volo_amm: bool,
    /// (Temperature, conversation root seed) used to choose tokens. Forced reference takes them
    /// from the input record (`temperature`, `root_seed`) or options, and uses them to resample.
    seme_ctl: (f32, u64),
    /// Traffic class and number of times batch work yielded its slot.
    class: Class,
    sospensioni: u32,
    /// Service mode: agent request ID (0 for test conversations), its token limit, requested
    /// cancellation (close after the next recorded token), and UTF-8 buffering for output text.
    ext_id: u64,
    chat_id: String,
    max_tokens: Option<u32>,
    utf8: Utf8Fragments,
    /// Sampling cutoffs are set by the service request.
    tagli: Tagli,
    /// Stop strings and text buffered in case it completes one of them.
    filtro: StopFilter,
    /// Draft depth policy, next computed position, and counters.
    bz: BozzaConv,
}

/// Scan complete UTF-8 for stop strings. Emit the text before the first
/// match, or hold any trailing prefix that could match after more tokens.
/// The caller flushes that prefix when output ends without a stop.
#[derive(Default, Debug, Clone)]
pub struct StopFilter {
    stop: Vec<String>,
    sospeso: String,
}

impl StopFilter {
    /// Ignore empty strings.
    pub fn new(stop: &[String]) -> Self {
        Self { stop: stop.iter().filter(|s| !s.is_empty()).cloned().collect(), sospeso: String::new() }
    }

    pub fn attivo(&self) -> bool {
        !self.stop.is_empty()
    }

    /// (Text to send now, stop triggered).
    pub fn passa(&mut self, new: &str) -> (String, bool) {
        if self.stop.is_empty() {
            return (new.to_string(), false);
        }
        let mut buf = std::mem::take(&mut self.sospeso);
        buf.push_str(new);
        // Earliest stop string, longest on ties.
        let trovato = self.stop.iter().filter_map(|st| buf.find(st.as_str()).map(|p| (p, st.len()))).min_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)));
        if let Some((p, _)) = trovato {
            buf.truncate(p);
            return (buf, true);
        }
        // Longest suffix of `buf` that is a proper prefix of a stop string.
        let mut trattieni = 0;
        for st in &self.stop {
            let massimo = (st.len() - 1).min(buf.len());
            for k in (trattieni + 1..=massimo).rev() {
                if st.is_char_boundary(k) && buf.ends_with(&st[..k]) {
                    trattieni = k;
                    break;
                }
            }
        }
        self.sospeso = buf.split_off(buf.len() - trattieni);
        (buf, false)
    }

    /// At the end without a stop, return the buffered text. It did not start a stop string.
    pub fn clear(&mut self) -> String {
        std::mem::take(&mut self.sospeso)
    }
}

/// Token and log probability for conversation step `step`, sampled or forced (forced reference).
fn scegli(convs: &[Conv], o: &HeadOptions, conv_id: u64, l: &[f32], step: u32, ctl: &mut Vec<(u64, u32, PosSeme)>) -> Result<(usize, f32), String> {
    let cv = convs.iter().find(|c| c.conv_id == conv_id);
    let forzato = cv.and_then(|c| c.forzati.as_ref()).and_then(|f| f.get(step as usize));
    match forzato {
        Some(&t) => {
            // Repeating the draw with the same seed lets the verifier separate a
            // matching sample from a forced token that its distribution would not pick.
            if let Some(c) = cv {
                ctl.push((conv_id, step, controlla_seme_tagli(l, c.seme_ctl.0, c.seme_ctl.1, step, t as usize, c.tagli).map_err(|e| format!("conversation {conv_id}, step {step}: {e}"))?));
            }
            Ok((t as usize, logprob(l, t as usize)))
        }
        None => campiona_tagli(l, o.temperature, seme_conversazione(o.seed, conv_id), step, cv.map_or(Tagli::default(), |c| c.tagli)).map_err(|e| format!("conversation {conv_id}, step {step}: {e}")),
    }
}

/// Like `scegli` for several rows in one step. Sample in parallel (`campiona_righe`,
/// bit-identical to sequential), and process forced tokens sequentially.
fn scegli_righe(convs: &[Conv], o: &HeadOptions, chiavi: &[(u64, u32)], logit: &[&[f32]], ctl: &mut Vec<(u64, u32, PosSeme)>) -> Result<Vec<(usize, f32)>, String> {
    let mut out: Vec<(usize, f32)> = vec![(0, 0.0); chiavi.len()];
    let mut idx = Vec::new();
    let mut rows: Vec<&[f32]> = Vec::new();
    let mut params = Vec::new();
    let mut tagli = Vec::new();
    for (i, &(conv_id, step)) in chiavi.iter().enumerate() {
        let cv = convs.iter().find(|c| c.conv_id == conv_id);
        let forzato = cv.and_then(|c| c.forzati.as_ref()).and_then(|f| f.get(step as usize));
        match forzato {
            Some(&t) => {
                if let Some(c) = cv {
                    ctl.push((conv_id, step, controlla_seme_tagli(logit[i], c.seme_ctl.0, c.seme_ctl.1, step, t as usize, c.tagli).map_err(|e| format!("conversation {conv_id}, step {step}: {e}"))?));
                }
                out[i] = (t as usize, logprob(logit[i], t as usize))
            }
            None => {
                idx.push(i);
                rows.push(logit[i]);
                params.push((o.temperature, seme_conversazione(o.seed, conv_id), step));
                tagli.push(cv.map_or(Tagli::default(), |c| c.tagli));
            }
        }
    }
    let th = crate::sampling::thread_effettivi(o.thread_campiona, rows.len());
    for (i, r) in idx.into_iter().zip(crate::sampling::campiona_righe_tagli(&rows, &params, &tagli, th)) {
        out[i] = r.map_err(|e| format!("conversation {}, step {}: {e}", chiavi[i].0, chiavi[i].1))?;
    }
    Ok(out)
}

/// Use the next transcript step for the first token after prefill. A replay
/// starts at the recorded count so the new token does not collide with a
/// step already committed.
fn token_prefill_locale(gruppo: u32, conv_id: u64, tentativo: u32, passo_primo: u32, token: usize, logprob: f32) -> pb::Token {
    pb::Token {
        group_id: gruppo,
        group_round: 0,
        prefill: true,
        rows: vec![pb::TokenRow { conversation_id: conv_id, attempt_id: tentativo, conversation_step: passo_primo, token: token as i32, logprob, draft: Vec::new() }],
        stage: 0,
    }
}

/// Conversations still without `done` at end of run (rebuilding, suspended or unfinished): "name (state, n tokens)".
fn non_concluse(convs: &[Conv]) -> Vec<String> {
    convs.iter().filter(|c| c.done.is_none()).map(|c| format!("{} ({:?}, {} tokens)", c.prompt_id, c.c.state(), c.token.len())).collect()
}

/// Reserve the low 32 bits for rounds and the high 32 for recovery episodes
/// so late replies cannot collide with rounds in the next episode.
pub fn initial_round(episodio: u32) -> u64 {
    ((episodio as u64) << 32) + 1
}

/// A `Token` nobody is waiting for: a generation round below the group's expected round (already consumed or from
/// an older episode), or a message older than `limite`.
fn stale_token(t: &pb::Token, round: &[u64], gen_avviata: bool, eta: Duration, limite: Duration) -> bool {
    if eta > limite {
        return true;
    }
    !t.prefill && gen_avviata && round.get(t.group_id as usize).is_some_and(|&expected| t.group_round < expected)
}

/// What `Confirmation` means to the head.
#[derive(Debug, PartialEq, Eq)]
enum ConfirmationClass {
    /// Success or a late error from a fault already handled (stale attempt or finished conversation).
    Ignora,
    /// Error for one conversation (index in `convs`): recover only that one.
    Conv(usize),
    /// Error without a valid `conversation_id`: chain fault.
    Chain,
}

fn classify_confirmation(convs: &[Conv], c: &pb::Confirmation) -> ConfirmationClass {
    if c.kind != pb::AckKind::Error as i32 && c.ok {
        return ConfirmationClass::Ignora;
    }
    match convs.iter().position(|x| x.conv_id == c.conversation_id) {
        None => ConfirmationClass::Chain,
        Some(i) => {
            let cv = &convs[i];
            // Ignore failures from in-flight work retired by voluntary suspension.
            if cv.done.is_some() || cv.c.state().terminale() || cv.c.sospesa_volontaria() || c.attempt_id != cv.c.tentativo() {
                ConfirmationClass::Ignora
            } else {
                ConfirmationClass::Conv(i)
            }
        }
    }
}

/// Prompts from a file: JSONL `{"id": ..., "prompt": ...}` or plain text lines. Skip `#` and blank lines.
pub fn leggi_prompt(path: &str, salta: usize, n: usize) -> Result<Vec<(String, String)>, String> {
    let s = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let mut v = Vec::new();
    let mut index = 0usize;
    for row in s.lines() {
        let t = row.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        let (id, text) = if t.starts_with('{') {
            let j: serde_json::Value = serde_json::from_str(t).map_err(|e| format!("{path}: invalid JSON: {e}"))?;
            let text = j.get("prompt").and_then(|x| x.as_str()).ok_or("missing \"prompt\" field")?.to_string();
            let id = j.get("id").map(|x| x.as_str().map(String::from).unwrap_or_else(|| x.to_string())).unwrap_or_else(|| format!("p{index}"));
            (id, text)
        } else {
            (format!("p{index}"), t.to_string())
        };
        if index >= salta && v.len() < n {
            v.push((id, text));
        }
        index += 1;
    }
    if v.is_empty() {
        return Err(format!("no prompts in {path}"));
    }
    Ok(v)
}

/// Listen for tail `Token` and faults once per run. A restarted tail reconnects here.
fn ascolta_token(r: &HeadLinks, richiede_mtp: bool) -> Result<TokenListener, String> {
    let ascolto = network::listen(&r.ascolta_token)?;
    let (tx, rx) = channel::<HeadEvent>();
    let serial = Arc::new(AtomicU64::new(0));
    let rifiuto = Arc::new(Mutex::new(None::<String>));
    let (tx_tok, ser, rif) = (tx.clone(), serial.clone(), rifiuto.clone());
    // Per-connection threads with a deadline for `Hello` keep silent clients from blocking accept (`rete::accetta`).
    std::thread::spawn(move || {
        network::accept(ascolto, move |f, mut lett, mut w| match f {
            pb::Stream::Token => {
                if richiede_mtp && !lett.mtp_hello {
                    // Reject a tail without the required draft capability during the handshake.
                    // The failure reaches startup immediately and recovery as a link fault.
                    let reason = "--draft-source mtp requires the tail to load MTP weights with --mtp".to_string();
                    eprintln!("[head] tail rejected: {reason}");
                    let _ = network::write_sync(&mut w, &msg(Body::Ack(pb::HelloAck { ok: false, error: reason.clone() })));
                    let _ = w.shutdown(std::net::Shutdown::Both);
                    *rif.lock().unwrap() = Some(reason);
                    return;
                }
                let n = ser.fetch_add(1, Ordering::SeqCst) + 1;
                if network::ack_ok(&mut w).is_err() {
                    return;
                }
                loop {
                    match lett.read() {
                        Ok(Some(pb::Message { body: Some(Body::Token(t)) })) => {
                            let _ = tx_tok.send(HeadEvent::Token(t, Instant::now()));
                        }
                        Ok(Some(_)) => {}
                        Ok(None) => {
                            let _ = tx_tok.send(HeadEvent::Guasto(Fonte::Token(n), "tail closed the token stream".into()));
                            break;
                        }
                        Err(e) => {
                            let _ = tx_tok.send(HeadEvent::Guasto(Fonte::Token(n), format!("token stream: {e}")));
                            break;
                        }
                    }
                }
            }
            altro => eprintln!("[head] unexpected connection on the {} stream", network::stream_name(altro)),
        });
    });
    Ok((rx, tx, serial, rifiuto))
}

/// Connect to the first downstream stage on all three streams, retrying up to `timeout`.
fn connect_forward(next: &str, timeout: Duration, tx: &Sender<HeadEvent>, epoch: &mut u32) -> Result<Connected, String> {
    let [(lp, wp), (lg, wg), (mut lc, wc)] = network::connect_forward(next, timeout, epoch)?;
    let epoch = *epoch;
    let caduto = Arc::new(AtomicBool::new(false));
    let lost = caduto.clone();
    let tx_c = tx.clone();
    std::thread::spawn(move || {
        let reason = loop {
            match lc.read() {
                Ok(Some(pb::Message { body: Some(Body::Confirmation(c)) })) => {
                    let _ = tx_c.send(HeadEvent::Confirmation(c));
                }
                Ok(Some(_)) => {}
                Ok(None) => break "stage closed the control stream".into(),
                Err(e) => break format!("control stream: {e}"),
            }
        };
        lost.store(true, Ordering::SeqCst);
        let _ = tx_c.send(HeadEvent::Guasto(Fonte::Control(epoch), reason));
    });
    drop((lp, lg));
    Ok(Connected { prefill: Writer::new(wp, "prefill"), gen: Writer::new(wg, "generation"), control: wc, caduto, epoch })
}

fn connect_downstream(r: &HeadLinks, timeout: Duration, richiede_mtp: bool) -> Result<Downstream, String> {
    let (rx, tx, tok_serial, rifiuto) = ascolta_token(r, richiede_mtp)?;
    let mut epoch = 1;
    let col = if richiede_mtp {
        // The chain is connected on a helper thread: a tail refused because it has no MTP block ends the head at once
        // with that reason, instead of after the whole `timeout` of a chain that will never answer.
        let (txc, rxc) = channel();
        let (next, tx2) = (r.next.clone(), tx.clone());
        std::thread::spawn(move || {
            let _ = txc.send(connect_forward(&next, timeout, &tx2, &mut epoch));
        });
        let esito = loop {
            if let Some(reason) = rifiuto.lock().unwrap().clone() {
                return Err(reason);
            }
            match rxc.recv_timeout(Duration::from_millis(100)) {
                Ok(c) => break c,
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break Err("chain connection closed".to_string()),
            }
        };
        if let Some(reason) = rifiuto.lock().unwrap().clone() {
            return Err(reason);
        }
        let col = esito?;
        // The tail opens the token stream before acknowledging the stage links.
        if tok_serial.load(Ordering::SeqCst) == 0 {
            return Err("--draft-source mtp: the tail did not open a token stream to confirm MTP support".into());
        }
        col
    } else {
        connect_forward(&r.next, timeout, &tx, &mut epoch)?
    };
    let tok_min = tok_serial.load(Ordering::SeqCst);
    Ok(Downstream::Rete(Box::new(DownstreamLinks { last_epoch: col.epoch, col, rx, tx, tok_serial, tok_min })))
}

/// Compact completed request indices only when idle and above this count.
const SOGLIA_COMPATTAZIONE: usize = 1000;
/// In service mode, `int_con` and `int_senza` retain at most this many recent intervals.
const MAX_INTERVALLI: usize = 4096;

/// Append `x` to `v`. If `v` already has `tetto` values, remove the oldest.
fn spingi_limitato(v: &mut Vec<f64>, x: f64, tetto: usize) {
    if v.len() >= tetto {
        v.remove(0);
    }
    v.push(x);
}

impl Conv {
    /// Release heap allocations after the final service response. Request entries
    /// remain indexable until idle compaction. File-based outputs still need their data.
    fn clear_weights(&mut self) {
        if self.ext_id == 0 {
            return;
        }
        self.prompt = Vec::new();
        self.da_inviare = Vec::new();
        self.token = Vec::new();
        self.logprob = Vec::new();
        self.tempi_ms = Vec::new();
        self.pezzi = Vec::new();
        self.forzati = None;
        self.c.clear_transcript();
    }
}

/// Service is idle: no live conversations, no steps in flight and no admission in progress.
#[allow(clippy::too_many_arguments)]
fn service_idle(convs: &[Conv], serv_vive: &[usize], in_volo: &VecDeque<usize>, corrente: Option<usize>, attesa_primo: &VecDeque<(usize, u64)>, amm: &Admission) -> bool {
    serv_vive.is_empty()
        && in_volo.is_empty()
        && corrente.is_none()
        && attesa_primo.is_empty()
        && !amm.pendenti()
        && amm.occupati() == 0
        && convs.iter().all(|c| c.done.is_some())
}

struct Head<'a> {
    m: &'a mut Engine,
    o: &'a HeadOptions,
    network: Option<&'a HeadLinks>,
    convs: Vec<Conv>,
    valle: Downstream,
    pronti: Vec<(pb::Token, Instant)>,
    t0: Instant,
    esp: Vec<RecEsperti>,
    top: Vec<crate::sampling::RecTop>,
    /// Seeded resampling results (forced reference only): (conversation, step, result).
    ctl_seme: Vec<(u64, u32, PosSeme)>,
    /// Recovery number (0 = normal run). `group_round` starts at `round_iniziale(episodio)` (episode in high 32 bits).
    episodio: u32,
    /// Latest timings snapshot from each stage (`Confirmation` of type `TIMINGS`), by stage index.
    tempi_stadi: std::collections::BTreeMap<u32, serde_json::Value>,
    /// A link dropped, so recovery must reconnect the chain.
    rotta: bool,
    registrati: u64,
    /// Round of each group's next or in-flight step. Keep these across `giro` calls so
    /// single-conversation recovery can rejoin while other groups' steps are in transit.
    round: Vec<u64>,
    /// Groups with steps in flight, oldest first.
    in_volo: VecDeque<usize>,
    /// Per group, conversations (index, attempt) in the in-flight step. A missing reply row marks one `orfana`.
    membri_volo: Vec<Vec<(usize, u32)>>,
    /// Tight-round generation has started. Reset on chain recovery.
    gen_avviata: bool,
    /// Admission rule (pure module).
    amm: Admission,
    /// Completed generation steps (group Token messages consumed), across all groups.
    passi: u64,
    /// Admission prefilling: `Open` sent, blocks remain.
    corrente: Option<usize>,
    /// Admissions with all blocks sent and awaiting first token: (conversation, admission round).
    attesa_primo: VecDeque<(usize, u64)>,
    giro_amm: u64,
    /// Measurements: time of last consumed group Token, whether a block was sent or first token awaited between group
    /// Tokens, and intervals (ms) between consecutive Tokens with and without interleaved prefill.
    ultimo_cons: Option<Instant>,
    lavoro_amm: bool,
    int_con: Vec<f64>,
    int_senza: Vec<f64>,
    /// Observe round timing and group membership for throughput calibration.
    mis: crate::calibration::MisureGruppi,
    /// Counters for tokens without a conversation and confirmations from retired links.
    prot: Protezioni,
    /// Service mode: local socket, open agent requests (indices in `convs`), last sent `Status`,
    /// next `conversation_id`, and slot count.
    serv: Option<Service>,
    serv_vive: Vec<usize>,
    serv_stato: Option<(u32, u32)>,
    prossimo_conv: u64,
    posti_tot: u32,
    /// Aggregate draft counters survive request compaction.
    bozza_tot: ContatoriBozza,
}

impl<'a> Head<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(m: &'a mut Engine, o: &'a HeadOptions, network: Option<&'a HeadLinks>, convs: Vec<Conv>, valle: Downstream, amm: Admission, slots: u32) -> Self {
        let prossimo_conv = convs.len() as u64 + 1;
        Head { m, o, network, convs, valle, pronti: Vec::new(), t0: Instant::now(), esp: Vec::new(), top: Vec::new(), ctl_seme: Vec::new(), episodio: 0, tempi_stadi: Default::default(), rotta: false, registrati: 0, round: Vec::new(), in_volo: VecDeque::new(), membri_volo: Vec::new(), gen_avviata: false, amm, passi: 0, corrente: None, attesa_primo: VecDeque::new(), giro_amm: 0, ultimo_cons: None, lavoro_amm: false, int_con: Vec::new(), int_senza: Vec::new(), mis: Default::default(), prot: Protezioni::default(), serv: None, serv_vive: Vec::new(), serv_stato: None, prossimo_conv, posti_tot: slots, bozza_tot: ContatoriBozza::default() }
    }
}

impl Head<'_> {
    fn ms(&self, t: Instant) -> f64 {
        t.duration_since(self.t0).as_secs_f64() * 1000.0
    }

    fn adesso(&self) -> Duration {
        self.t0.elapsed()
    }

    /// Send on prefill (`gen = false`) or generation. A send error means the link is down.
    fn manda(&mut self, gen: bool, m: pb::Message, still_needed: impl Fn(&Self) -> bool) -> R {
        let mut frame = network::encode(&m).map_err(|e| {
            self.rotta = true;
            Fail::Guasto(e)
        })?;
        let timeout = Writer::queue_timeout();
        let done = Instant::now() + timeout;
        let reason = loop {
            let Downstream::Rete(v) = &self.valle else { return Ok(()) };
            if self.rotta || v.col.caduto.load(Ordering::SeqCst) {
                break "stage closed the control stream".into();
            }
            let writer = if gen { &v.col.gen } else { &v.col.prefill };
            let wait = done.saturating_duration_since(Instant::now()).min(Duration::from_millis(50));
            match writer.send_frame_with_wait(frame, wait) {
                Ok(None) => return Ok(()),
                Ok(Some(bytes)) => frame = bytes,
                Err(e) => break e,
            }
            // Keep servicing Cancel while the writer waits for the peer to read.
            self.poll_service();
            if self.serv.as_ref().is_some_and(Service::pipe_closed) {
                return Err(Fail::Fatale("agent pipe closed".into()));
            }
            if !still_needed(self) {
                return Ok(());
            }
            if Instant::now() >= done {
                break format!("stage writer queue full for {timeout:?}");
            }
        };
        self.rotta = true;
        Err(Fail::Guasto(reason))
    }

    /// Apply the state machine's actions. `Close` is best effort because a dropped link has no receiver and
    /// the new stage has nothing to close.
    fn applica(&mut self, c: usize, azioni: Vec<Action>) {
        for a in azioni {
            match a {
                Action::Open { tentativo, start_position } => self.convs[c].da_aprire = Some((tentativo, start_position)),
                Action::Close { tentativo } => {
                    if let Downstream::Rete(v) = &self.valle {
                        if !self.rotta {
                            // Close shares the prefill stream so Open cannot overtake it.
                            let close = msg(Body::Close(pb::Close {
                                conversation_id: self.convs[c].conv_id, attempt_id: tentativo,
                            }));
                            if let Err(e) = v.col.prefill.send_with_wait(&close, Duration::from_millis(100)) {
                                self.rotta = true;
                                let _ = v.col.control.shutdown(std::net::Shutdown::Both);
                                eprintln!("[head] could not send Close: {e}");
                            }
                        }
                    }
                }
                Action::ErroreClient(reason) => {
                    eprintln!("[head] conversation {} failed ({:?})", self.convs[c].prompt_id, reason);
                    self.convs[c].done = Some("failed");
                    let slot = self.convs[c].slot;
                    self.m.svuota_posto(slot);
                    self.amm.libera(c);
                }
            }
        }
    }

    /// Send `Open` for the attempt chosen by the state machine. The caller then prefills the confirmed prefix.
    fn open(&mut self, c: usize) -> R {
        let Some((tentativo, pos)) = self.convs[c].da_aprire.take() else { return Ok(()) };
        let cv = &mut self.convs[c];
        // Prompt + recorded tokens (prompt alone on first open).
        cv.da_inviare = cv.prompt.iter().chain(cv.token.iter()).copied().collect();
        debug_assert_eq!(cv.da_inviare.len() as u32, cv.c.prefisso_len());
        cv.passo_primo = cv.c.expected_step();
        cv.inviati = pos as usize;
        // The slot is cleared below, so there is nothing to rewind. Any in-flight draft is lost.
        cv.bz.calcolata = 0;
        cv.bz.in_volo = 0;
        cv.bz.mtp = None;
        let (conv_id, gruppo, slot) = (cv.conv_id, cv.gruppo, cv.slot);
        self.m.svuota_posto(slot);
        let ap = pb::Open {
            conversation_id: conv_id,
            attempt_id: tentativo,
            group_id: gruppo as u32,
            slot,
            context_limit: self.o.ctx_per_slot,
            start_position: pos,
            // Conversation (temperature, seed) comes from options in tests and the request in service mode.
            sampling: Some(pb::Sampling { temperature: self.convs[c].seme_ctl.0, root_seed: self.convs[c].seme_ctl.1, top_p: self.convs[c].tagli.top_p, top_k: self.convs[c].tagli.top_k }),
            state_format: self.o.state_format as i32,
        };
        self.manda(false, msg(Body::Open(ap)), |head| head.convs[c].done.is_none())
    }

    /// Compute and send the next prefill block for conversation `c`.
    fn block(&mut self, c: usize) -> R {
        let (conv_id, gruppo, slot, off, tot, tentativo, passo_primo) = {
            let cv = &self.convs[c];
            (cv.conv_id, cv.gruppo, cv.slot, cv.inviati, cv.da_inviare.len(), cv.c.tentativo(), cv.passo_primo)
        };
        let n = self.o.block.min(tot - off);
        let last = off + n == tot;
        let rows: Vec<Row> = (0..n)
            .map(|i| Row { token: self.convs[c].da_inviare[off + i], pos: (off + i) as i32, seq: slot as i32, logits: last && i + 1 == n })
            .collect();
        let state = self.m.run(&rows, None)?;
        let state = if self.o.prova_stato_int8 { state.map(|s| degrada_int8(&s, self.m.n_embd)) } else { state };
        // Send input token IDs alongside the residuals when the tail fills a draft context.
        let token_blocco: Vec<i32> = if self.o.draft_source == DraftSource::Mtp { self.convs[c].da_inviare[off..off + n].to_vec() } else { Vec::new() };
        // Only the final block asks the tail to draft after its first sample.
        let k_blocco = if self.o.draft_source != DraftSource::Mtp {
            0
        } else if self.o.mtp_k_prova > 0 {
            self.o.mtp_k_prova
        } else {
            self.convs[c].bz.state.k_corrente().max(1) as u32
        };
        self.convs[c].inviati += n;
        match (&self.valle, state) {
            (Downstream::Rete(_), Some(s)) => self.manda(
                false,
                msg(Body::Block(pb::PrefillBlock {
                    conversation_id: conv_id,
                    attempt_id: tentativo,
                    group_id: gruppo as u32,
                    start_position: off as u32,
                    n_token: n as u32,
                    n_embd: self.m.n_embd as u32,
                    format: self.o.state_format as i32,
                    state: encode_state(&s, self.m.n_embd, self.m.stream_count, self.o.state_format),
                    stream_count: crate::network::declared_stream_count(self.o.state_format, self.m.stream_count),
                    last,
                    first_token_step: passo_primo,
                    token: token_blocco,
                    draft_count: k_blocco,
                })),
                |head| head.convs[c].done.is_none(),
            ),
            (Downstream::Locale, None) => {
                if last {
                    let l = self.m.logits(n - 1)?;
                    let (t, lp) = scegli(&self.convs, self.o, conv_id, l, passo_primo, &mut self.ctl_seme)?;
                    if self.o.top.is_some() && self.serv.is_none() {
                        self.top.push(crate::sampling::RecTop { c: conv_id, s: passo_primo, t: crate::sampling::top_k(l, crate::stage::TOP_K) });
                    }
                    self.pronti.push((token_prefill_locale(gruppo as u32, conv_id, tentativo, passo_primo, t, lp), Instant::now()));
                }
                Ok(())
            }
            _ => Err("expected network state from the head or logits from the reference; check layer_end".into()),
        }
    }

    fn membri(&self, g: usize) -> Vec<usize> {
        (0..self.convs.len())
            .filter(|&c| {
                let cv = &self.convs[c];
                cv.gruppo == g && cv.done.is_none() && !cv.orfana && !cv.token.is_empty() && cv.c.state() == Status::Active
            })
            .collect()
    }

    fn step(&mut self, g: usize, round: u64, membri: &[usize]) -> R {
        let mut rows = Vec::new();
        let mut rp = Vec::new();
        if self.membri_volo.len() <= g {
            self.membri_volo.resize(g + 1, Vec::new());
        }
        let conteggi: Vec<usize> = (0..self.o.groups.max(1).min(self.convs.len()).max(g + 1)).map(|h| if h == g { membri.len() } else { self.membri(h).len() }).collect();
        self.mis.send(g, &conteggi);
        self.membri_volo[g] = membri.iter().map(|&c| (c, self.convs[c].c.tentativo())).collect();
        // The spare capacity is the row limit minus one required row per conversation.
        let bozza_accesa = self.o.max_draft_tokens > 0 && matches!(self.valle, Downstream::Rete(_));
        let tetto = self.o.max_draft_rows.min(self.m.n_batch);
        let mut extra = tetto.saturating_sub(membri.len());
        let mtp = bozza_accesa && self.o.draft_source == DraftSource::Mtp;
        // Rewind rejected positions and find how many candidate tokens each
        // conversation can propose within its length and context limits.
        let mut requested_depths: Vec<usize> = Vec::with_capacity(membri.len());
        let mut candidate: Vec<Vec<i32>> = Vec::with_capacity(membri.len());
        let mut voluti: Vec<usize> = Vec::with_capacity(membri.len());
        for &c in membri {
            // K for the step, taken once per step so pause steps are counted.
            let k_prop = if bozza_accesa { self.convs[c].bz.state.k_da_proporre() } else { 0 };
            let cv = &self.convs[c];
            let pos = (cv.prompt.len() + cv.token.len() - 1) as i32;
            // Rewind the head's memory when starting again at a computed position after draft rejection.
            if (pos as u32) < cv.bz.calcolata {
                if let Err(e) = self.m.riavvolgi(cv.slot, pos as u32) {
                    return Err(Fail::GuastoConv(c, format!("conversation {}: {e}", cv.conv_id)));
                }
            }
            let (lista, volute) = if bozza_accesa {
                // Bound by remaining tokens (row 0 produces one) and by the slot context.
                let limite = cv.max_tokens.unwrap_or(self.o.max_tokens) as usize;
                let restanti = limite.saturating_sub(cv.token.len());
                let ctx_libero = (self.o.ctx_per_slot as usize).saturating_sub(pos as usize + 1);
                let k = k_prop.min(restanti.saturating_sub(1)).min(ctx_libero);
                // Each source returns a candidate prefix up to the requested depth.
                let lista = if k == 0 {
                    Vec::new()
                } else if let Some(or) = &cv.bz.oracolo {
                    draft::draft_oracle(or, cv.token.len(), k, self.o.draft_oracle_corrupt_every)
                } else if self.o.draft_source == DraftSource::Mtp {
                    // Use the tail proposal paired with the last committed token.
                    draft::bozza_mtp(cv.bz.mtp.as_ref().map(|(n, d)| (*n, d.as_slice())), cv.token.len(), k, self.m.n_vocab)
                } else {
                    let storia: Vec<i32> = cv.prompt.iter().chain(cv.token.iter()).copied().collect();
                    draft::prompt_lookup(&storia, k, draft::N_MAX)
                };
                (lista, k)
            } else {
                (Vec::new(), 0)
            };
            voluti.push(lista.len());
            candidate.push(lista);
            requested_depths.push(volute);
        }
        // Allocate spare rows across depth for tail proposals. Prompt lookup
        // consumes the budget in conversation order.
        let budget = extra;
        let quote: Vec<usize> = if mtp {
            draft::distribute_rows(&voluti, extra)
        } else {
            voluti.iter().map(|&v| { let q = v.min(extra); extra -= q; q }).collect()
        };
        // Drafts the head asks the tail for per conversation (`RigaPasso.bozza_k`). Ask for all desired drafts if
        // the rows fit, otherwise only what the row budget allows, so the tail does not compute unused drafts
        // under load. At least 1.
        let chiede_tutto = requested_depths.iter().sum::<usize>() <= budget;
        for (i, &c) in membri.iter().enumerate() {
            let pos = (self.convs[c].prompt.len() + self.convs[c].token.len() - 1) as i32;
            let draft: Vec<i32> = candidate[i][..quote[i]].to_vec();
            let k_chiesta = if !mtp {
                0
            } else if self.o.mtp_k_prova > 0 {
                self.o.mtp_k_prova
            } else if chiede_tutto {
                requested_depths[i].max(1) as u32
            } else {
                quote[i].max(1) as u32
            };
            let cv = &mut self.convs[c];
            if bozza_accesa {
                if draft.is_empty() {
                    cv.bz.cont.passi_senza_bozza += 1;
                    self.bozza_tot.passi_senza_bozza += 1;
                } else {
                    cv.bz.cont.proposta(draft.len());
                    self.bozza_tot.proposta(draft.len());
                }
            }
            cv.bz.in_volo = draft.len();
            cv.bz.calcolata = pos as u32 + 1 + draft.len() as u32;
            let cv = &self.convs[c];
            rows.push(Row { token: *cv.token.last().unwrap(), pos, seq: cv.slot as i32, logits: true });
            rp.push(pb::StepRow { conversation_id: cv.conv_id, attempt_id: cv.c.tentativo(), conversation_step: cv.token.len() as u32, position: pos as u32, draft_token: 0, draft_count: k_chiesta });
            for (i, &d) in draft.iter().enumerate() {
                rows.push(Row { token: d, pos: pos + 1 + i as i32, seq: cv.slot as i32, logits: true });
                rp.push(pb::StepRow {
                    conversation_id: cv.conv_id,
                    attempt_id: cv.c.tentativo(),
                    conversation_step: (cv.token.len() + 1 + i) as u32,
                    position: (pos + 1 + i as i32) as u32,
                    draft_token: d,
                    draft_count: 0,
                });
            }
        }
        let state = self.m.run(&rows, None)?;
        let state = if self.o.prova_stato_int8 { state.map(|s| degrada_int8(&s, self.m.n_embd)) } else { state };
        for (row, layer, mut e) in self.m.prendi_esperti() {
            e.sort_unstable();
            // Expert traces would grow with service lifetime, so collect them only for finite runs.
            if self.serv.is_none() {
                self.esp.push(RecEsperti { c: rp[row].conversation_id, s: rp[row].conversation_step, l: layer, e });
            }
        }
        match (&self.valle, state) {
            (Downstream::Rete(_), Some(s)) => self.manda(
                true,
                msg(Body::Step(pb::Step {
                    group_id: g as u32,
                    group_round: round,
                    rows: rp,
                    n_embd: self.m.n_embd as u32,
                    format: self.o.state_format as i32,
                    state: encode_state(&s, self.m.n_embd, self.m.stream_count, self.o.state_format),
                    stream_count: crate::network::declared_stream_count(self.o.state_format, self.m.stream_count),
                })),
                |head| membri.iter().any(|&c| head.convs[c].done.is_none()),
            ),
            (Downstream::Locale, None) => {
                let mut logit: Vec<&[f32]> = Vec::with_capacity(rp.len());
                for i in 0..rp.len() {
                    logit.push(self.m.logits(i)?);
                }
                let chiavi: Vec<(u64, u32)> = rp.iter().map(|r| (r.conversation_id, r.conversation_step)).collect();
                let scelte = scegli_righe(&self.convs, self.o, &chiavi, &logit, &mut self.ctl_seme)?;
                if self.o.top.is_some() && self.serv.is_none() {
                    for (r, l) in rp.iter().zip(&logit) {
                        self.top.push(crate::sampling::RecTop { c: r.conversation_id, s: r.conversation_step, t: crate::sampling::top_k(l, crate::stage::TOP_K) });
                    }
                }
                let out: Vec<pb::TokenRow> = rp
                    .iter()
                    .zip(scelte)
                    .map(|(r, (t, lp))| pb::TokenRow { conversation_id: r.conversation_id, attempt_id: r.attempt_id, conversation_step: r.conversation_step, token: t as i32, logprob: lp, draft: Vec::new() })
                    .collect();
                self.pronti.push((pb::Token { group_id: g as u32, group_round: round, prefill: false, rows: out, stage: 0 }, Instant::now()));
                Ok(())
            }
            _ => Err("expected network state from the head or logits from the reference; check layer_end".into()),
        }
    }

    /// Keep each stage's latest timing snapshot, since it includes all previous timings.
    fn record_timings(&mut self, c: &pb::Confirmation) {
        match serde_json::from_str::<serde_json::Value>(&c.timings_json) {
            Ok(v) => {
                self.tempi_stadi.insert(c.stage, v);
            }
            Err(e) => eprintln!("[head] invalid timings from stage {}: {e}", c.stage),
        }
    }

    /// Collect final timing snapshots for `durata`. Stages send them on idle, within half a second of
    /// the last work. Drop everything else because the run is over.
    fn collect_timings(&mut self, durata: Duration) {
        let Downstream::Rete(v) = &self.valle else { return };
        let done = Instant::now() + durata;
        loop {
            let resto = done.saturating_duration_since(Instant::now());
            if resto.is_zero() {
                break;
            }
            match v.rx.recv_timeout(resto) {
                Ok(HeadEvent::Confirmation(c)) if c.kind == pb::AckKind::Timings as i32 => {
                    if let Ok(j) = serde_json::from_str::<serde_json::Value>(&c.timings_json) {
                        self.tempi_stadi.insert(c.stage, j);
                    }
                }
                Ok(_) => {}
                Err(_) => break,
            }
        }
    }

    /// Drop `Token` messages from `pronti` that nobody awaits: generation rounds already consumed by their group
    /// or from an older episode, and messages older than four times `T_lavoro`. If a prefill token has waited
    /// that long, no caller will claim it.
    fn pulisci_pronti(&mut self) {
        let limite = Duration::from_secs(self.o.t_lavoro_s.saturating_mul(4));
        let (round, avviata) = (&self.round, self.gen_avviata);
        self.pronti.retain(|(t, q)| !stale_token(t, round, avviata, q.elapsed(), limite));
    }

    fn current_fault(&self, f: Fonte) -> bool {
        match (&self.valle, f) {
            (Downstream::Rete(v), Fonte::Control(e)) => e == v.col.epoch,
            (Downstream::Rete(v), Fonte::Token(n)) => n >= v.tok_min,
            _ => false,
        }
    }

    /// Wait for a `Token` that matches `pred`. Leave the others in `pronti`. An error `Confirmation`, dropped
    /// link or expired `T_lavoro` triggers recovery.
    fn attendi(&mut self, pred: impl Fn(&pb::Token) -> bool, obsolete: impl Fn(&Self) -> bool, cosa: &str) -> R<Option<(pb::Token, Instant)>> {
        let limite = Instant::now() + Duration::from_secs(self.o.t_lavoro_s);
        loop {
            self.poll_service();
            if self.serv.as_ref().is_some_and(Service::pipe_closed) {
                return Err(Fail::Fatale("agent pipe closed".into()));
            }
            if obsolete(self) {
                return Ok(None);
            }
            if let Some(i) = self.pronti.iter().position(|(t, _)| pred(t)) {
                return Ok(Some(self.pronti.remove(i)));
            }
            let Downstream::Rete(v) = &self.valle else {
                return Err(Fail::Fatale(format!("{cosa}: no token ready in the reference run")));
            };
            let resto = limite.saturating_duration_since(Instant::now());
            let t_att = Instant::now();
            let intervallo = if self.serv.is_some() { Duration::from_millis(50) } else { Duration::from_secs(5) };
            let r = v.rx.recv_timeout(resto.min(intervallo));
            crate::timings::aggiungi(crate::timings::Fase::Attesa, t_att.elapsed());
            match r {
                Ok(HeadEvent::Token(t, q)) => {
                    self.pronti.push((t, q));
                    self.pulisci_pronti();
                }
                Ok(HeadEvent::Confirmation(c)) if c.kind == pb::AckKind::Timings as i32 => self.record_timings(&c),
                Ok(HeadEvent::Confirmation(c)) => {
                    let m = format!("error confirmation from stage {} for conversation {}: {}", c.stage, c.conversation_id, c.error);
                    // A chain-wide Confirmation from a superseded link describes a fault already handled by the recovery
                    // that reconnected the chain, so it must not start another recovery.
                    let corrente = if let Downstream::Rete(v) = &self.valle { v.col.epoch } else { 0 };
                    if c.conversation_id == 0 && conferma_catena_vecchia(c.epoch, u64::from(corrente)) {
                        self.prot.conferma_vecchia(c.stage);
                        eprintln!("[head] ignoring a chain error from stale epoch {} < {corrente}: {m}", c.epoch);
                        continue;
                    }
                    match classify_confirmation(&self.convs, &c) {
                        ConfirmationClass::Ignora => {}
                        ConfirmationClass::Conv(i) => return Err(Fail::GuastoConv(i, m)),
                        ConfirmationClass::Chain => {
                            self.rotta = true;
                            return Err(Fail::Guasto(m));
                        }
                    }
                }
                Ok(HeadEvent::Guasto(f, e)) => {
                    if self.current_fault(f) {
                        self.rotta = true;
                        return Err(Fail::Guasto(format!("connection lost while waiting for {cosa}: {e}")));
                    }
                }
                Err(RecvTimeoutError::Timeout) => {
                    if Instant::now() >= limite {
                        return Err(Fail::Guasto(format!("work deadline ({} s) exceeded while waiting for {cosa}", self.o.t_lavoro_s)));
                    }
                }
                Err(RecvTimeoutError::Disconnected) => return Err(Fail::Fatale(format!("token stream closed while waiting for {cosa}"))),
            }
        }
    }

    /// Commit a token only for the active attempt and expected step (`Conversation::token`).
    fn registra(&mut self, r: &pb::TokenRow, at: Instant, da_stadio: u32) -> R {
        // Count a token with no matching conversation by its source stage and drop it.
        let Some(c) = self.convs.iter().position(|c| c.conv_id == r.conversation_id) else {
            self.prot.token_sconosciuto(da_stadio);
            let n = self.prot.token_sconosciuti_totale();
            if n <= 5 || n % 100 == 0 {
                eprintln!("[head] ignoring token for unknown conversation {} from stage {da_stadio} ({n} discarded)", r.conversation_id);
            }
            return Ok(());
        };
        let t_ms = self.ms(at);
        let adesso = at.saturating_duration_since(self.t0);
        let cv = &mut self.convs[c];
        if cv.done.is_some() {
            return Ok(()); // Conversation already closed: drop
        }
        match cv.c.token(r.attempt_id, r.conversation_step, r.token as u32, adesso) {
            TokenOutcome::Registrato { token_seq } => debug_assert_eq!(token_seq, cv.token.len() as u64 + 1),
            // Keep duplicates and inactive-attempt tokens out of the transcript.
            TokenOutcome::Duplicato | TokenOutcome::Scartato => return Ok(()),
            TokenOutcome::Inatteso { expected } => {
                return Err(Fail::Guasto(format!(
                    "token out of sequence for {}: attempt {}, step {} (expected {expected})",
                    cv.prompt_id, r.attempt_id, r.conversation_step
                )))
            }
        }
        cv.token.push(r.token);
        cv.logprob.push(r.logprob);
        cv.tempi_ms.push(t_ms);
        if cv.ext_id != 0 && cv.token.len() == 1 {
            let ttft_ms = (t_ms - cv.t_arrivo_ms.unwrap_or(t_ms)).max(0.0);
            eprintln!("[head] first token chat={} head_request={} ttft_ms={ttft_ms:.1}", cv.chat_id, cv.ext_id);
        }
        // The proposal is valid at this transcript length and expires when it advances.
        cv.bz.mtp = if self.o.draft_source == DraftSource::Mtp && !r.draft.is_empty() { Some((cv.token.len(), r.draft.clone())) } else { None };
        self.registrati += 1;
        if self.registrati % 16 == 0 {
            eprintln!("[head] progress: {} tokens recorded (episode {})", self.registrati, self.episodio);
        }
        let eog = self.m.e_eog(r.token);
        let mut pezzo = Vec::new();
        if !eog {
            pezzo = self.m.pezzo(r.token);
            self.convs[c].pezzi.extend_from_slice(&pezzo);
        }
        let stop = if self.convs[c].ext_id != 0 {
            match self.service_token(c, r, eog, &pezzo) {
                Ok(stop) => stop,
                Err(()) => {
                    self.service_cancel(c);
                    self.convs[c].done = Some("output_stalled");
                    return Ok(());
                }
            }
        } else {
            false
        };
        let cv = &mut self.convs[c];
        let limite = cv.forzati.as_ref().map_or(cv.max_tokens.unwrap_or(self.o.max_tokens) as usize, |f| f.len());
        if eog || stop || cv.token.len() >= limite {
            cv.done = Some(if eog {
                "eog"
            } else if stop {
                "stop"
            } else {
                "max_token"
            });
            // Close the request once its response is complete so its slot can be reused.
            let mut az = cv.c.finish_response(adesso).map_err(|e| format!("end of response rejected: {e:?}"))?;
            az.extend(cv.c.close_by_user(adesso).map_err(|e| format!("close rejected: {e:?}"))?);
            let slot = cv.slot;
            self.applica(c, az);
            self.m.svuota_posto(slot);
            self.amm.libera(c);
        }
        Ok(())
    }

    /// One run: open conversations and prefill prompts or confirmed prefixes on recovery,
    /// collect first tokens, then generate in tight rounds. End when all conversations are terminal.
    ///
    /// After recovery, prefill only conversations that need to open or rebuild.
    /// Other groups stay in flight and their tokens wait in `pronti`.
    fn giro(&mut self) -> R {
        let n = self.convs.len();
        let da_aprire: Vec<usize> = (0..n).filter(|&c| self.convs[c].done.is_none() && self.convs[c].ammessa && self.convs[c].da_aprire.is_some()).collect();
        for &c in &da_aprire {
            self.poll_service();
            if self.convs[c].done.is_none() {
                self.open(c)?;
            }
        }
        // Conversations awaiting prefill or a first token (already opened for the first time or for recovery).
        let in_ricostruzione = |s: &Self| -> Vec<usize> {
            (0..n)
                .filter(|&c| {
                    let cv = &s.convs[c];
                    cv.done.is_none()
                        && cv.ammessa
                        && !s.in_ammissione(c)
                        && cv.da_aprire.is_none()
                        && !cv.da_inviare.is_empty()
                        && matches!(cv.c.state(), Status::Opening | Status::Rebuilding)
                })
                .collect()
        };
        let da_fare = in_ricostruzione(self);
        // 1. Round-robin prefill, one block per conversation.
        loop {
            let mut fatto = false;
            for &c in &da_fare {
                self.poll_service();
                if self.convs[c].done.is_none() && self.convs[c].inviati < self.convs[c].da_inviare.len() {
                    self.block(c)?;
                    fatto = true;
                }
            }
            if !fatto {
                break;
            }
        }
        // 2. First tokens.
        for &c in &da_fare {
            if !matches!(self.convs[c].c.state(), Status::Opening | Status::Rebuilding) || self.in_ammissione(c) {
                continue; // Already recorded (back after recovering another conversation).
            }
            let (id, tent) = (self.convs[c].conv_id, self.convs[c].c.tentativo());
            let Some((t, q)) = self.attendi(
                |t| t.prefill && t.rows.first().is_some_and(|r| r.conversation_id == id && r.attempt_id == tent),
                |head| head.convs[c].done.is_some(),
                "the first token",
            )? else { continue };
            for r in &t.rows {
                self.registra(r, q, t.stage)?;
            }
        }
        // 3. Tight-round generation, one step in flight per group.
        // Service conversations arrive later, so size groups using `--groups` regardless of the current count.
        let g_tot = if self.serv.is_some() { self.o.groups.max(1) } else { self.o.groups.max(1).min(n) };
        if !self.gen_avviata {
            self.round = vec![initial_round(self.episodio); g_tot];
            self.in_volo.clear();
            self.gen_avviata = true;
        }
        // On return from single-conversation recovery, restart any group without an in-flight
        // step with the recovered conversation (e.g. when it was alone in its group).
        for g in 0..g_tot {
            if self.in_volo.contains(&g) {
                continue;
            }
            let m = self.membri(g);
            if !m.is_empty() {
                let r = self.round[g];
                self.step(g, r, &m)?;
                self.in_volo.push_back(g);
            }
        }
        loop {
            if let Some(&g) = self.in_volo.front() {
                let r = self.round[g];
                let reply = self.attendi(
                    |t| !t.prefill && t.group_id == g as u32 && t.group_round == r,
                    |head| head.membri_volo[g].iter().all(|&(c, _)| head.convs[c].done.is_some()),
                    &format!("group {g}, round {r}"),
                )?;
                self.in_volo.pop_front();
                let Some((t, q)) = reply else {
                    self.membri_volo[g].clear();
                    self.round[g] += 1;
                    let m = self.membri(g);
                    if !m.is_empty() {
                        self.step(g, self.round[g], &m)?;
                        self.in_volo.push_back(g);
                    }
                    self.admission(false)?;
                    continue;
                };
                self.passi += 1;
                self.mis.token(g, q);
                let now = Instant::now();
                if let Some(previous) = self.ultimo_cons {
                    let dt = now.duration_since(previous).as_secs_f64() * 1000.0;
                    self.mis.intervallo(dt);
                    // Bound the interval history during service. Finite runs retain all intervals.
                    let tetto = if self.serv.is_some() { MAX_INTERVALLI } else { usize::MAX };
                    spingi_limitato(if self.lavoro_amm { &mut self.int_con } else { &mut self.int_senza }, dt, tetto);
                }
                self.ultimo_cons = Some(now);
                self.lavoro_amm = false;
                // Count verified draft rows before recording tokens because the response
                // limit may prevent some of those rows from reaching the transcript.
                let previous: Vec<(usize, usize, usize)> = if self.o.max_draft_tokens > 0 {
                    self.membri_volo[g]
                        .iter()
                        .map(|&(c, tent)| {
                            let cv = &self.convs[c];
                            (c, cv.token.len(), t.rows.iter().filter(|r| r.conversation_id == cv.conv_id && r.attempt_id == tent).count())
                        })
                        .collect()
                } else {
                    Vec::new()
                };
                for row in &t.rows {
                    self.registra(row, q, t.stage)?;
                }
                for (c, len_prima, n_righe) in previous {
                    if n_righe == 0 {
                        continue; // Conversation did not return (fault or dropped row). Nothing to count.
                    }
                    let cv = &mut self.convs[c];
                    let k = std::mem::take(&mut cv.bz.in_volo);
                    let registrati = cv.token.len().saturating_sub(len_prima);
                    cv.bz.cont.passo_completato(registrati);
                    self.bozza_tot.passo_completato(registrati);
                    if k > 0 {
                        // Returned rows = 1 + accepted drafts. Rows beyond the head's limit count as accepted.
                        let acc = n_righe.saturating_sub(1).min(k);
                        cv.bz.cont.esito(k, n_righe);
                        self.bozza_tot.esito(k, n_righe);
                        cv.bz.state.esito(k, acc);
                    }
                }
                // A missing row from the step was dropped by a stage. Wait for its error `Confirmation`.
                for (c, tent) in std::mem::take(&mut self.membri_volo[g]) {
                    let cv = &mut self.convs[c];
                    if cv.done.is_none() && cv.c.state() == Status::Active && cv.c.tentativo() == tent && !t.rows.iter().any(|r| r.conversation_id == cv.conv_id) {
                        cv.orfana = true;
                    }
                }
                self.round[g] += 1;
                let m = self.membri(g);
                if !m.is_empty() {
                    let r = self.round[g];
                    self.step(g, r, &m)?;
                    self.in_volo.push_back(g);
                }
                self.admission(false)?;
            } else if self.amm.pendenti() || self.corrente.is_some() || !self.attesa_primo.is_empty() {
                // No step in flight, so generation stops and admission proceeds on its own.
                self.ultimo_cons = None;
                if !self.admission(true)? {
                    if self.convs.iter().any(|c| c.done.is_none() && c.orfana) {
                        break; // Wait for the error below.
                    }
                    if !self.amm.pendenti() && self.corrente.is_none() && self.attesa_primo.is_empty() {
                        break;
                    }
                    return Err(Fail::Fatale("admission blocked: no work in flight, no free slots and no pending arrivals".into()));
                }
            } else {
                break;
            }
        }
        // No step in flight, but a conversation awaits its error. Wait for that error or expired
        // `T_lavoro` (chain fault). Never return `Ok`.
        if self.convs.iter().any(|c| c.done.is_none() && c.orfana) {
            self.attendi(
                |_| false,
                |head| !head.convs.iter().any(|c| c.done.is_none() && c.orfana),
                "an error from an unresponsive conversation",
            )?;
        }
        Ok(())
    }

    /// `ammissione()` owns this conversation while it is prefilling or waiting for its first token.
    fn in_ammissione(&self, c: usize) -> bool {
        self.corrente == Some(c) || self.attesa_primo.iter().any(|&(x, _)| x == c)
    }

    /// Assign group and slot, create the state machine (`Open` action) and mark the conversation admitted.
    fn admit(&mut self, c: usize, g: usize, locale: u32, in_volo: bool) {
        let t = self.ms(Instant::now());
        let adesso = self.adesso();
        let cfg = self.cfg();
        let cv = &mut self.convs[c];
        cv.gruppo = g;
        cv.slot = crate::engine::slot(g as u32, locale);
        cv.ammessa = true;
        if cv.c.sospesa_volontaria() {
            // Resume a suspended batch with the same conversation, a new attempt and prefill of prompt and output so
            // far. The metrics still use the first admission.
            match cv.c.resume(adesso) {
                Ok(az) => self.applica(c, az),
                Err(e) => eprintln!("[head] conversation {} rejected resumption: {e:?}", self.convs[c].prompt_id),
            }
            return;
        }
        cv.in_volo_amm = in_volo;
        cv.t_ammissione_ms = Some(t);
        let (conv, az) = Conversation::new(cv.conv_id, cv.prompt.len() as u32, cfg, adesso);
        cv.c = conv;
        self.applica(c, az);
    }

    /// One admission step between group waits (see module docs). `fermo` means no step is in flight.
    /// Return whether any work was done (arrival, first token, open or block).
    fn admission(&mut self, fermo: bool) -> R<bool> {
        self.giro_amm += 1;
        let mut fatto = false;
        if self.serv.is_some() {
            // Service mode: new requests, cancellations and closures between group Token messages.
            self.poll_service();
        }
        // 1. Arrivals
        let fermo = fermo && self.corrente.is_none() && self.attesa_primo.is_empty();
        let nuovi = self.amm.arriva(self.passi, fermo);
        let t = self.ms(Instant::now());
        for &c in &nuovi {
            self.convs[c].t_arrivo_ms = Some(t);
        }
        fatto |= !nuovi.is_empty();
        // 2. First tokens for admissions whose last block went out in an earlier round.
        let ritardo = if self.o.ritardo_primo == 0 { (2 * self.o.groups as u64).max(2) } else { self.o.ritardo_primo };
        while let Some(&(c, it)) = self.attesa_primo.front() {
            if it + ritardo > self.giro_amm {
                // Consumption point still ahead. Count this round as progress for the loop with no steps in flight.
                fatto = true;
                break;
            }
            let (id, tent) = (self.convs[c].conv_id, self.convs[c].c.tentativo());
            let Some((tk, q)) = self.attendi(
                |t| t.prefill && t.rows.first().is_some_and(|r| r.conversation_id == id && r.attempt_id == tent),
                |head| head.convs[c].done.is_some(),
                "the first token of an admitted conversation",
            )? else { fatto = true; continue };
            self.attesa_primo.pop_front();
            for r in &tk.rows {
                self.registra(r, q, tk.stage)?;
            }
            // Join the group's next step, or restart the group if it has no step in flight.
            let g = self.convs[c].gruppo;
            if !self.in_volo.contains(&g) {
                let m = self.membri(g);
                if !m.is_empty() {
                    let r = self.round[g];
                    self.step(g, r, &m)?;
                    self.in_volo.push_back(g);
                }
            }
            fatto = true;
            self.lavoro_amm = true;
        }
        // Interrupt batch prefill for queued interactive work. If a slot is
        // still needed, choose the active batch with the shortest prefix to replay.
        if self.amm.in_coda_interattive() > 0 {
            if let Some(b) = self.corrente.filter(|&b| self.convs[b].class == Class::Batch) {
                self.suspend_conversation(b);
                fatto = true;
            }
            if self.corrente.is_none() {
                let da_sospendere = {
                    let costo = |c: usize| -> Option<u64> {
                        let cv = &self.convs[c];
                        let attiva = cv.done.is_none() && cv.ammessa && cv.c.state() == Status::Active && !cv.token.is_empty() && !self.in_ammissione(c);
                        attiva.then(|| (cv.prompt.len() + cv.token.len()) as u64)
                    };
                    self.amm.da_sospendere(&costo)
                };
                if let Some(b) = da_sospendere {
                    self.suspend_conversation(b);
                    fatto = true;
                }
            }
        }
        // 3. If no admission is in progress, open the first queued conversation.
        if self.corrente.is_none() {
            if let Some((c, g, locale)) = self.amm.prossima() {
                self.admit(c, g, locale, true);
                self.open(c)?;
                if self.convs[c].done.is_none() {
                    self.corrente = Some(c);
                }
                fatto = true;
            }
        }
        // 4. At most one block for the current admission.
        if let Some(c) = self.corrente {
            self.block(c)?;
            self.lavoro_amm = true;
            fatto = true;
            if self.corrente == Some(c) && self.convs[c].inviati >= self.convs[c].da_inviare.len() {
                self.corrente = None;
                self.attesa_primo.push_back((c, self.giro_amm));
            }
        }
        Ok(fatto)
    }

    /// Close a batch attempt, release its slot, and requeue the conversation
    /// with its transcript intact. Closing the attempt fences off any tokens
    /// still in flight while other conversations continue.
    fn suspend_conversation(&mut self, c: usize) {
        let adesso = self.adesso();
        let id = self.convs[c].conv_id;
        match self.convs[c].c.suspend(adesso) {
            Ok(az) => self.applica(c, az),
            Err(e) => {
                eprintln!("[head] conversation {} rejected suspension: {e:?}", self.convs[c].prompt_id);
                return;
            }
        }
        let cv = &mut self.convs[c];
        cv.da_aprire = None;
        cv.orfana = false;
        cv.ammessa = false;
        cv.da_inviare.clear();
        cv.inviati = 0;
        cv.sospensioni += 1;
        let slot = cv.slot;
        if self.corrente == Some(c) {
            self.corrente = None;
        }
        self.attesa_primo.retain(|&(x, _)| x != c);
        self.m.svuota_posto(slot);
        for (tk, _) in self.pronti.iter_mut() {
            tk.rows.retain(|r| r.conversation_id != id);
        }
        self.pronti.retain(|(tk, _)| !(tk.prefill && tk.rows.is_empty()));
        self.amm.suspend(c);
    }

    /// Admission metrics for the summary (only with `--arrivals`).
    fn misure_ammissione(&self) -> serde_json::Value {
        let in_volo: Vec<&Conv> = self.convs.iter().filter(|c| c.in_volo_amm).collect();
        let ttft: Vec<f64> = in_volo
            .iter()
            .filter_map(|c| Some(c.tempi_ms.first()? - c.t_arrivo_ms?))
            .collect();
        let attese: Vec<f64> = in_volo.iter().filter_map(|c| Some(c.t_ammissione_ms? - c.t_arrivo_ms?)).collect();
        let med = |v: &[f64]| if v.is_empty() { serde_json::Value::Null } else { serde_json::json!(output::percentile(v, 50.0)) };
        let max = |v: &[f64]| if v.is_empty() { serde_json::Value::Null } else { serde_json::json!(v.iter().cloned().fold(0.0, f64::max)) };
        let rapporto = if self.int_con.is_empty() || self.int_senza.is_empty() {
            serde_json::Value::Null
        } else {
            serde_json::json!(self.int_con.iter().cloned().fold(0.0, f64::max) / output::percentile(&self.int_senza, 50.0).max(1e-9))
        };
        serde_json::json!({
            "in_flight_admissions": in_volo.len(),
            "in_flight_ttft_ms_median": med(&ttft),
            "in_flight_ttft_ms_max": max(&ttft),
            "queue_wait_ms_median": med(&attese),
            "queue_wait_ms_max": max(&attese),
            "group_steps": {
                "intervals_without_prefill": self.int_senza.len(),
                "intervals_with_prefill": self.int_con.len(),
                "median_without_prefill_ms": med(&self.int_senza),
                "median_with_prefill_ms": med(&self.int_con),
                "max_with_prefill_ms": max(&self.int_con),
                "max_with_prefill_over_median_without": rapporto,
            },
            "completed_steps": self.passi,
        })
    }

    /// Reconnect all three streams to the first stage (for a restarted tail), dropping the old links.
    fn ricollega(&mut self) -> Result<(), String> {
        let r = self.network.ok_or("no links to reconnect")?;
        let timeout = Duration::from_secs(self.o.attesa_s);
        let Downstream::Rete(v) = &self.valle else { return Err("no links to reconnect".into()) };
        let mut epoch = v.last_epoch.checked_add(1).ok_or("stage link epoch exhausted")?;
        v.col.prefill.abort();
        v.col.gen.abort();
        let _ = v.col.control.shutdown(std::net::Shutdown::Both);
        let new = if self.serv.is_some() {
            let (next, tx) = (r.next.clone(), v.tx.clone());
            let (result, wait) = channel();
            std::thread::spawn(move || {
                let new = connect_forward(&next, timeout, &tx, &mut epoch);
                let _ = result.send((new, epoch));
            });
            loop {
                self.poll_service();
                if self.serv.as_ref().is_some_and(Service::pipe_closed) {
                    return Err("agent pipe closed".into());
                }
                match wait.recv_timeout(Duration::from_millis(50)) {
                    Ok((result, last_epoch)) => {
                        epoch = last_epoch;
                        break result;
                    }
                    Err(RecvTimeoutError::Timeout) => {}
                    Err(RecvTimeoutError::Disconnected) => return Err("reconnect worker stopped".into()),
                }
            }
        } else {
            connect_forward(&r.next, timeout, &v.tx, &mut epoch)
        };
        let Downstream::Rete(v) = &mut self.valle else { unreachable!() };
        v.last_epoch = epoch;
        // Drop the old writers without waiting. Their threads exit on their own.
        v.col = new?;
        v.tok_min = v.tok_serial.load(Ordering::SeqCst);
        self.rotta = false;
        Ok(())
    }

    /// Retire all live attempts after a chain fault. Reconnect if needed, then
    /// give each conversation a new attempt within its own recovery limits.
    fn recovery(&mut self, reason: &str) {
        self.poll_service();
        let adesso = self.adesso();
        self.episodio += 1;
        self.gen_avviata = false;
        self.in_volo.clear();
        self.membri_volo.clear();
        for cv in self.convs.iter_mut() {
            cv.orfana = false;
        }
        // Recover ongoing admissions like other admitted conversations.
        self.corrente = None;
        self.attesa_primo.clear();
        self.ultimo_cons = None;
        eprintln!("[head] chain fault ({reason}); recovery {} at {:.1} s", self.episodio, adesso.as_secs_f64());
        let n = self.convs.len();
        for c in 0..n {
            if self.convs[c].done.is_some() || !self.convs[c].ammessa {
                continue;
            }
            let t = self.convs[c].c.tentativo();
            if let Ok(az) = self.convs[c].c.fault(t, adesso) {
                self.applica(c, az);
            }
            self.convs[c].da_aprire = None;
            let slot = self.convs[c].slot;
            self.m.svuota_posto(slot);
        }
        let mut ricollegata = true;
        let reconnect = self.rotta;
        if reconnect {
            if let Err(e) = self.ricollega() {
                eprintln!("[head] cannot rebuild the chain: {e}");
                ricollegata = false;
            }
        }
        // Drop everything in flight or already arrived because it belongs to closed attempts.
        self.pronti.clear();
        if !reconnect {
            if let Downstream::Rete(v) = &self.valle {
                while v.rx.try_recv().is_ok() {}
            }
        }
        self.poll_service();
        let now = self.adesso();
        for c in 0..n {
            if self.convs[c].done.is_some() || self.convs[c].c.state() != Status::Suspended {
                continue;
            }
            // Without a chain, wait up to T_ripresa (state machine decides). With a chain, resume.
            let az = if ricollegata {
                self.convs[c].c.tempo(now).and_then(|a| if a.is_empty() { self.convs[c].c.chain_ready(now) } else { Ok(a) })
            } else {
                self.convs[c].c.tempo(now + Duration::from_secs(self.o.t_ripresa_s))
            };
            match az {
                Ok(a) => self.applica(c, a),
                Err(e) => eprintln!("[head] conversation {} rejected recovery: {e:?}", self.convs[c].prompt_id),
            }
        }
    }

    /// Retire only the faulty conversation. Close its attempt at all stages,
    /// clear its local slot, and open a new attempt to replay its prefix.
    /// Remove queued tokens from that attempt while other groups stay in flight.
    fn recover_conversation(&mut self, c: usize, reason: &str) {
        let adesso = self.adesso();
        let (id, name, t) = (self.convs[c].conv_id, self.convs[c].prompt_id.clone(), self.convs[c].c.tentativo());
        eprintln!("[head] conversation fault ({reason}); recovering {name}, attempt {t}, at {:.1} s", adesso.as_secs_f64());
        match self.convs[c].c.fault(t, adesso) {
            Ok(az) => self.applica(c, az),
            Err(e) => {
                eprintln!("[head] conversation {name} is already recovering; ignoring fault ({e:?})");
                return;
            }
        }
        self.convs[c].da_aprire = None;
        self.convs[c].orfana = false;
        // Remove it from any admission queues so recovery can replay prefill without interleaving.
        if self.corrente == Some(c) {
            self.corrente = None;
        }
        self.attesa_primo.retain(|&(x, _)| x != c);
        let slot = self.convs[c].slot;
        self.m.svuota_posto(slot);
        for (tk, _) in self.pronti.iter_mut() {
            tk.rows.retain(|r| r.conversation_id != id);
        }
        self.pronti.retain(|(tk, _)| !(tk.prefill && tk.rows.is_empty()));
        let now = self.adesso();
        match self.convs[c].c.chain_ready(now) {
            Ok(a) => self.applica(c, a),
            Err(e) => eprintln!("[head] conversation {name} rejected recovery: {e:?}"),
        }
    }

    /// Admit the initial queue immediately in round-robin mode. FIFO mode
    /// leaves requests for the admission loop to open between steps.
    fn admit_initial(&mut self) {
        if self.o.prefill_order == PrefillOrder::Fifo {
            return;
        }
        while let Some((c, g, locale)) = self.amm.prossima() {
            self.admit(c, g, locale, false);
        }
    }

    fn corsa(&mut self) -> R {
        // In round-robin mode, reserve a slot for each eligible startup arrival.
        for c in self.amm.arriva(0, false) {
            self.convs[c].t_arrivo_ms = Some(0.0);
        }
        self.admit_initial();
        loop {
            match self.giro() {
                Ok(()) => break,
                Err(Fail::Guasto(m)) => {
                    if matches!(self.valle, Downstream::Locale) {
                        return Err(Fail::Fatale(format!("reference run failed: {m}")));
                    }
                    self.recovery(&m);
                }
                Err(Fail::GuastoConv(c, m)) => {
                    if matches!(self.valle, Downstream::Locale) {
                        return Err(Fail::Fatale(format!("reference run failed: {m}")));
                    }
                    self.recover_conversation(c, &m);
                }
                Err(f) => return Err(f),
            }
        }
        // An unfinished conversation (e.g. recovered but never resumed) makes the run unsuccessful.
        let aperte = non_concluse(&self.convs);
        if !aperte.is_empty() {
            return Err(Fail::Fatale(format!("conversations still active at the end of the run: {}", aperte.join(", "))));
        }
        // The authoritative transcript has no gaps or duplicates and matches the written output.
        for cv in &self.convs {
            let tr = cv.c.transcript();
            let ok = tr.len() == cv.token.len()
                && tr.iter().enumerate().all(|(j, r)| r.token_seq == j as u64 + 1 && r.step == j as u32 && r.token as i32 == cv.token[j]);
            if !ok {
                return Err(Fail::Fatale(format!("transcript for {} differs from recorded tokens", cv.prompt_id)));
            }
            eprintln!(
                "[head] conv {}: {:?}, {} tokens, final attempt {}, {} recoveries, transcript has no gaps or duplicates",
                cv.prompt_id,
                cv.c.state(),
                tr.len(),
                cv.c.tentativo(),
                cv.c.recoveries()
            );
        }
        let fallite: Vec<&str> = self.convs.iter().filter(|c| c.done == Some("failed")).map(|c| c.prompt_id.as_str()).collect();
        if !fallite.is_empty() {
            return Err(Fail::Fatale(format!("failed conversations: {}", fallite.join(", "))));
        }
        Ok(())
    }

    fn cfg(&self) -> Config {
        Config { k_riprese: self.o.k_riprese, t_ripresa: Duration::from_secs(self.o.t_ripresa_s) }
    }

    // Long-running local service.

    /// Run the head until a fatal error or failed chain recovery. Admit socket
    /// requests and stream their committed output through the local service.
    fn corsa_servizio(&mut self) -> R {
        loop {
            self.poll_service();
            if self.serv.as_ref().is_some_and(Service::pipe_closed) {
                return Err(Fail::Fatale("agent pipe closed".into()));
            }
            let now = self.ms(Instant::now());
            for c in self.amm.arriva(self.passi, false) {
                self.convs[c].t_arrivo_ms = Some(now);
            }
            self.admit_initial();
            let lavoro = self.convs.iter().any(|c| c.done.is_none() && c.ammessa) || self.amm.pendenti();
            if !lavoro {
                self.service_inactive()?;
                continue;
            }
            match self.giro() {
                Ok(()) => {
                    // A completed round should leave all conversations closed. Otherwise skip the empty round.
                    if self.convs.iter().any(|c| c.done.is_none() && c.ammessa) {
                        std::thread::sleep(Duration::from_millis(20));
                    }
                }
                Err(Fail::Guasto(m)) => {
                    self.recovery(&m);
                    if self.rotta && !self.convs.iter().any(|c| c.done.is_none() && c.ammessa) {
                        return Err(Fail::Fatale(format!("cannot rebuild the chain after a fault: {m}")));
                    }
                }
                Err(Fail::GuastoConv(c, m)) => self.recover_conversation(c, &m),
                Err(f) => return Err(f),
            }
        }
    }

    /// With no conversations, wait for a request in 200 ms intervals. Drain the chain channel
    /// and reconnect if a link dropped while idle.
    fn service_inactive(&mut self) -> R {
        self.service_compact();
        let mut timings = Vec::new();
        let mut caduto = false;
        if let Downstream::Rete(v) = &self.valle {
            while let Ok(e) = v.rx.try_recv() {
                match e {
                    HeadEvent::Confirmation(c) if c.kind == pb::AckKind::Timings as i32 => timings.push(c),
                    HeadEvent::Guasto(f, _) if self.current_fault(f) => caduto = true,
                    _ => {}
                }
            }
        }
        for c in timings {
            self.record_timings(&c);
        }
        self.pronti.clear();
        if caduto || self.rotta {
            self.rotta = true;
            eprintln!("[head] lost connection while idle; reconnecting the chain");
            self.ricollega().map_err(|e| Fail::Fatale(format!("cannot rebuild the chain: {e}")))?;
        }
        if let Some(ev) = self.serv.as_ref().and_then(|s| s.attendi_evento(Duration::from_millis(200))) {
            self.service_event(ev);
        }
        Ok(())
    }

    /// Take events already received on the socket, close finished conversations (`Done`) and update `Status`.
    /// Call on every admission round so new requests enter as arrivals between group `Token` messages.
    fn poll_service(&mut self) {
        if self.serv.is_none() {
            return;
        }
        while let Some(ev) = self.serv.as_ref().and_then(|s| s.prova_evento()) {
            self.service_event(ev);
        }
        let chiuse: Vec<usize> = self.serv_vive.iter().copied().filter(|&c| self.convs[c].done.is_some()).collect();
        for c in chiuse {
            self.service_done(c);
        }
        self.serv_vive.retain(|&c| self.convs[c].done.is_none());
        let state = (self.posti_tot.saturating_sub(self.amm.occupati() as u32), self.serv_vive.len() as u32);
        if self.serv_stato != Some(state) {
            if let Some(s) = &self.serv {
                if s.manda(pbs::FromHead { body: Some(pbs::from_head::Body::Status(pbs::Status { free_slots: state.0, pending: state.1 })) }) {
                    self.serv_stato = Some(state);
                }
            }
        }
    }

    /// Compact request state only when idle. Keep assigning increasing
    /// conversation IDs so late replies cannot match a new request.
    fn service_compact(&mut self) {
        if self.serv.is_none() || self.convs.len() <= SOGLIA_COMPATTAZIONE {
            return;
        }
        if !service_idle(&self.convs, &self.serv_vive, &self.in_volo, self.corrente, &self.attesa_primo, &self.amm) {
            return;
        }
        eprintln!("[head] service idle; compacting {} completed conversations", self.convs.len());
        self.convs = Vec::new();
        self.amm.azzera_conversazioni();
        for m in self.membri_volo.iter_mut() {
            m.clear();
        }
        self.pronti.clear();
        self.esp = Vec::new();
        self.top = Vec::new();
        self.ctl_seme = Vec::new();
    }

    fn service_event(&mut self, ev: EvServ) {
        match ev {
            EvServ::Connesso => self.serv_stato = None,
            EvServ::Request(r) => self.service_add(r),
            EvServ::Cancel(id) => {
                if let Some(c) = self.serv_vive.iter().copied().find(|&c| self.convs[c].ext_id == id) {
                    self.service_cancel(c);
                }
            }
            EvServ::Disconnesso => {
                eprintln!("[head] agent disconnected; cancelling {} conversations", self.serv_vive.len());
                for c in self.serv_vive.clone() {
                    self.service_cancel(c);
                }
                self.serv_stato = None;
            }
        }
    }

    fn reject_request(&self, request: &pbs::Request, code: &str, prompt_tokens: Option<usize>) {
        eprintln!("[head] chat chat={} head_request={} outcome=rejected code={code} prompt_tokens={prompt_tokens:?} ctx_per_slot={} max_tokens={} reserved_output=0",
            request.chat_id, request.id, self.o.ctx_per_slot, request.max_tokens);
        if let Some(s) = &self.serv {
            s.manda(pbs::FromHead { body: Some(pbs::from_head::Body::Done(pbs::Done {
                id: request.id, reason: "error".into(), error: code.into(), ..Default::default()
            })) });
        }
    }

    /// A `Request` queues a conversation for admission at threshold 0 (arrival next round).
    fn service_add(&mut self, r: pbs::Request) {
        if r.id == 0 || self.serv_vive.iter().any(|&c| self.convs[c].ext_id == r.id) {
            return self.reject_request(&r, "invalid_request", None);
        }
        let ceiling = if r.max_tokens == 0 { self.o.max_tokens } else { r.max_tokens };
        let context = self.o.ctx_per_slot as usize;
        let (tokens, dropped) = if !r.prompt_token.is_empty() {
            (r.prompt_token.clone(), 0)
        } else if !r.messages.is_empty() {
            if r.messages.iter().all(|m| m.text.trim().is_empty()) {
                return self.reject_request(&r, "invalid_request", None);
            }
            let mut count = None;
            let result = crate::service::fit_chat_context(&r.messages, context, ceiling, |messages| {
                let (tokens, _, _) = self.m.prepara_conversazione(messages)?;
                let len = tokens.len();
                count = Some(len);
                Ok((tokens, len))
            });
            match result {
                Ok(fitted) => fitted,
                Err(e) if e == crate::service::LAST_MESSAGE_TOO_LONG => {
                    return self.reject_request(&r, "context_length_exceeded", count);
                }
                Err(_) => return self.reject_request(&r, "invalid_request", count),
            }
        } else if r.text.trim().is_empty() {
            return self.reject_request(&r, "invalid_request", None);
        } else {
            match self.m.prepara_prompt(&r.text) {
                Ok((tokens, _, _)) => (tokens, 0),
                Err(_) => return self.reject_request(&r, "invalid_request", None),
            }
        };
        if tokens.is_empty() {
            return self.reject_request(&r, "invalid_request", Some(0));
        }
        let Some(max_tokens) = crate::service::output_budget(context, tokens.len(), ceiling) else {
            return self.reject_request(&r, "context_length_exceeded", Some(tokens.len()));
        };
        eprintln!("[head] chat chat={} head_request={} outcome=admitted prompt_tokens={} ctx_per_slot={context} max_tokens={ceiling} reserved_output={max_tokens} dropped_messages={dropped}",
            r.chat_id, r.id, tokens.len());
        let camp = r.sampling.unwrap_or_default();
        let conv_id = self.prossimo_conv;
        self.prossimo_conv += 1;
        let class = if r.class == pbs::Class::Batch as i32 { Class::Batch } else { Class::Interactive };
        let seed = if camp.seed != 0 { camp.seed } else { seme_conversazione(self.o.seed, conv_id) };
        let idx = self.amm.aggiungi(class);
        debug_assert_eq!(idx, self.convs.len());
        let adesso = self.adesso();
        self.convs.push(Conv {
            prompt_id: r.id.to_string(),
            conv_id,
            gruppo: 0,
            slot: 0,
            c: Conversation::new(conv_id, tokens.len() as u32, self.cfg(), adesso).0,
            da_aprire: None,
            da_inviare: Vec::new(),
            passo_primo: 0,
            prompt: tokens,
            inviati: 0,
            token: Vec::new(),
            logprob: Vec::new(),
            tempi_ms: Vec::new(),
            pezzi: Vec::new(),
            done: None,
            orfana: false,
            forzati: None,
            ammessa: false,
            t_arrivo_ms: None,
            t_ammissione_ms: None,
            in_volo_amm: false,
            seme_ctl: (camp.temperature, seed),
            class,
            sospensioni: 0,
            ext_id: r.id,
            chat_id: r.chat_id,
            max_tokens: Some(max_tokens),
            utf8: Utf8Fragments::default(),
            tagli: Tagli { top_p: camp.top_p, top_k: camp.top_k },
            filtro: StopFilter::new(&r.stop),
            bz: BozzaConv::new(self.o.max_draft_tokens as usize),
        });
        self.serv_vive.push(idx);
    }

    /// Retire the attempt and release the admission as soon as Cancel arrives.
    fn service_cancel(&mut self, c: usize) {
        if self.convs[c].done.is_some() {
            return;
        }
        let now = self.adesso();
        let actions = self.convs[c].c.cancel(now).unwrap_or_default();
        if !self.amm.cancel(c) && self.convs[c].ammessa {
            self.m.svuota_posto(self.convs[c].slot);
            self.amm.libera(c);
            self.applica(c, actions);
        }
        let cv = &mut self.convs[c];
        cv.done = Some("cancelled");
        cv.da_aprire = None;
        cv.orfana = false;
        if self.corrente == Some(c) {
            self.corrente = None;
        }
        self.attesa_primo.retain(|&(idx, _)| idx != c);
        let id = cv.conv_id;
        self.pronti.retain(|(t, _)| !t.prefill || !t.rows.iter().any(|r| r.conversation_id == id));
    }

    /// Report a committed token. A full output queue cancels its request.
    fn service_token(&mut self, c: usize, r: &pb::TokenRow, eog: bool, pezzo: &[u8]) -> Result<bool, ()> {
        let cv = &mut self.convs[c];
        let completo = if eog { String::new() } else { cv.utf8.aggiungi(pezzo) };
        let (text, stop) = if completo.is_empty() { (String::new(), false) } else { cv.filtro.passa(&completo) };
        if let Some(s) = &self.serv {
            let token = pbs::OutputToken { id: cv.ext_id, token: r.token, text, logprob: r.logprob, step: r.conversation_step, eog };
            if !s.manda(pbs::FromHead { body: Some(pbs::from_head::Body::Token(token)) }) {
                return Err(());
            }
        }
        Ok(stop)
    }

    /// Send the terminal frame, then release allocations that are no longer needed.
    fn service_done(&mut self, c: usize) {
        let now = self.ms(Instant::now());
        let cv = &mut self.convs[c];
        let arrivo = cv.t_arrivo_ms.unwrap_or(now);
        let stalled = cv.done == Some("output_stalled");
        let done = pbs::Done {
            id: cv.ext_id,
            reason: cv.done.unwrap_or("failed").to_string(),
            usage: Some(pbs::Usage { prompt_tokens: cv.prompt.len() as u32, generated_tokens: cv.token.len() as u32 }),
            timings: Some(pbs::Timings {
                wait_ms: cv.t_ammissione_ms.map_or(0.0, |a| (a - arrivo).max(0.0)),
                ttft_ms: cv.tempi_ms.first().map_or(0.0, |t| (t - arrivo).max(0.0)),
                total_ms: (now - arrivo).max(0.0),
            }),
            error: if stalled { "head service output queue is full".into() } else { String::new() },
            // On `stop`, the buffered text was the start of the stop string, so drop it.
            remaining_text: if stalled || cv.done == Some("stop") {
                cv.utf8.clear();
                cv.filtro.clear();
                String::new()
            } else {
                // Flush the filter first because its text precedes the incomplete bytes.
                let mut t = cv.filtro.clear();
                t.push_str(&cv.utf8.clear());
                t
            },
            prompt_token: cv.prompt.clone(),
        };
        eprintln!("[head] chat finished chat={} head_request={} prompt_tokens={} completion_tokens={} total_ms={:.1} reason={}",
            cv.chat_id, cv.ext_id, cv.prompt.len(), cv.token.len(), (now - arrivo).max(0.0), done.reason);
        cv.clear_weights();
        if let Some(s) = &self.serv {
            s.manda(pbs::FromHead { body: Some(pbs::from_head::Body::Done(done)) });
        }
    }
}

/// Recompute each conversation alone so its logits match the verifier's
/// batch shape. Feed the prompt and all but the last forced token. Request
/// logits from the final prompt position onward. Split inputs that exceed
/// the batch limit and keep the usual forced replay output format.
fn forma_verificatore(m: &mut Engine, convs: &mut [Conv], o: &HeadOptions, top: &mut Vec<crate::sampling::RecTop>, ctl: &mut Vec<(u64, u32, PosSeme)>) -> Result<(), String> {
    let slot = crate::engine::slot(0, 0);
    let n_batch = m.n_batch;
    for cv in convs.iter_mut() {
        let t_conv = Instant::now();
        let f = cv.forzati.clone().ok_or("verifier replay requires forced tokens")?;
        let p = cv.prompt.len();
        let mut seq = cv.prompt.clone();
        seq.extend_from_slice(&f[..f.len() - 1]);
        let tot = seq.len();
        m.svuota_posto(slot);
        let mut blocchi = 0;
        let mut inizio = 0;
        while inizio < tot {
            let done = (inizio + n_batch).min(tot);
            // Logits for i >= p - 1: position p-1 gives step 0, position p+k-1 gives step k.
            let rows: Vec<Row> = (inizio..done).map(|i| Row { token: seq[i], pos: i as i32, seq: slot as i32, logits: i + 1 >= p }).collect();
            m.run(&rows, None)?;
            for (j, i) in (inizio..done).enumerate() {
                if i + 1 < p {
                    continue;
                }
                let step = i + 1 - p;
                let l = m.logits(j)?;
                cv.token.push(f[step]);
                cv.logprob.push(logprob(l, f[step] as usize));
                ctl.push((cv.conv_id, step as u32, controlla_seme_tagli(l, cv.seme_ctl.0, cv.seme_ctl.1, step as u32, f[step] as usize, cv.tagli).map_err(|e| format!("{} passo {step}: {e}", cv.prompt_id))?));
                if o.top.is_some() {
                    top.push(crate::sampling::RecTop { c: cv.conv_id, s: step as u32, t: crate::sampling::top_k(l, crate::stage::TOP_K) });
                }
            }
            inizio = done;
            blocchi += 1;
        }
        if blocchi > 1 {
            eprintln!("[head] {}: splitting {tot} tokens across {blocchi} blocks (n_batch {n_batch})", cv.prompt_id);
        }
        m.svuota_posto(slot);
        let ms = t_conv.elapsed().as_secs_f64() * 1000.0;
        cv.tempi_ms = vec![ms; cv.token.len()];
        let ultimo_eog = m.e_eog(*f.last().unwrap());
        for &t in &f {
            if !m.e_eog(t) {
                let pz = m.pezzo(t);
                cv.pezzi.extend_from_slice(&pz);
            }
        }
        cv.done = Some(if ultimo_eog { "eog" } else { "max_token" });
        eprintln!("[head] {}: rescored {} positions in {tot} tokens ({blocchi} blocks, {ms:.0} ms)", cv.prompt_id, cv.token.len());
    }
    Ok(())
}

/// Run a finite chain or single-process reference and write its summary.
/// In service mode, bind the local socket after the chain has acknowledged
/// all links, so a connection indicates readiness. Service produces
/// per-request output through that socket.
fn run_service(mut engine: Engine, o: &HeadOptions, network: Option<&HeadLinks>, slots: u32) -> Result<(), String> {
    let network = network.ok_or("--service requires a chain head")?;
    if let Some(addr) = &o.service {
        crate::service::validate_address(addr)?;
    }
    if o.forzati.is_some() || o.forma_verificatore || o.arrivi.is_some() || o.classi.is_some() {
        return Err("--service cannot be combined with --tokens, --verifier-mode, --arrivals or --classes".into());
    }
    if slots as usize > engine.n_batch {
        return Err("slots exceed n_batch; a group step must fit in one batch".into());
    }
    let g = engine.per_ctx.len();
    let amm = Admission::con_gruppi(Vec::new(), Vec::new(), &engine.per_ctx, slots as usize, o.riserva_interattive);
    let valle = connect_downstream(network, Duration::from_secs(o.attesa_s), o.draft_source == DraftSource::Mtp)?;
    let service = if o.service_pipe { Service::pipes()? } else { Service::listen(o.service.as_deref().unwrap_or_default())? };
    if let Some(addr) = service.address() {
        eprintln!("[head] chain ready with {slots} slots and {g} groups; service listening on {addr}");
    } else {
        eprintln!("[head] chain ready with {slots} slots and {g} groups; service over inherited pipes");
    }
    let mut t = Head::new(&mut engine, o, Some(network), Vec::new(), valle, amm, slots);
    t.serv = Some(service);
    let esito = t.corsa_servizio().map_err(|f| match f {
        Fail::Guasto(m) | Fail::GuastoConv(_, m) | Fail::Fatale(m) => m,
    });
    if let Some(s) = &t.serv {
        s.manda(pbs::FromHead { body: Some(pbs::from_head::Body::Status(pbs::Status { free_slots: 0, pending: 0 })) });
    }
    if let Downstream::Rete(v) = t.valle {
        v.col.prefill.abort();
        v.col.gen.abort();
        let _ = v.col.control.shutdown(std::net::Shutdown::Both);
    }
    esito
}

pub fn run(mut engine: Engine, mut options: HeadOptions, network: Option<&HeadLinks>, slots: u32) -> Result<(), String> {
    if options.block == 0 {
        let target = engine.n_batch.min(512);
        options.block = if network.is_some() {
            target.min(crate::network::max_block_tokens_with_ids(engine.n_embd, engine.stream_count, options.state_format, options.draft_source == DraftSource::Mtp))
        } else {
            target
        };
    }
    // With fewer slots than groups the engine creates fewer contexts: schedule only those.
    options.groups = engine.per_ctx.len();
    let o = &options;
    if !engine.head {
        return Err("head must start at layer 0".into());
    }
    if network.is_none() && !engine.tail {
        return Err("reference run requires the full model ([0, n_layer))".into());
    }
    if network.is_some() && engine.tail {
        return Err("chain head must exclude the final layer; use reference for the full model".into());
    }
    if !o.forma_verificatore && (o.block == 0 || o.block > engine.n_batch) {
        return Err(format!("--block {} is outside 1..=n_batch ({})", o.block, engine.n_batch));
    }
    if network.is_some() {
        // Derive the limit from state width and format (see `rete::controlla_blocco`).
        crate::network::check_prefill_block(o.block, engine.n_embd, engine.stream_count, o.state_format)?;
    }
    if o.draft_source == DraftSource::Mtp {
        // Validate draft settings before opening any links.
        if network.is_none() {
            return Err("--draft-source mtp requires a chain head; the tail produces drafts".into());
        }
        if o.max_draft_tokens == 0 {
            return Err("--draft-source mtp needs the same --draft-max-tokens (at least 1) on all stages to rewind rejected rows".into());
        }
        if o.draft_oracle.is_some() && o.mtp_k_prova == 0 {
            // Fixture proposals exercise accepts, rejections, and cache rewind while
            // the tail also builds its own draft.
            return Err("using --test-draft-oracle with --draft-source mtp requires --test-mtp-depth to compute and discard the MTP draft".into());
        }
        crate::network::check_prefill_block_with_tokens(o.block, engine.n_embd, engine.stream_count, o.state_format)?;
    }
    if o.forzati.is_some() && network.is_some() {
        return Err("forced reference replay runs on one node".into());
    }
    if o.forma_verificatore && o.forzati.is_none() {
        return Err("--verifier-mode requires replay with --tokens".into());
    }
    if o.prova_stato_int8 && network.is_none() {
        return Err("--test-state-int8 requires a chain head".into());
    }
    if o.service.is_some() || o.service_pipe {
        return run_service(engine, o, network, slots);
    }
    let prompt = leggi_prompt(&o.prompt, o.salta, o.n_prompt)?;
    let forzati_da: Vec<RecordConv> = match &o.forzati {
        Some(p) => output::leggi_jsonl(p)?,
        None => Vec::new(),
    };
    let soglie = match &o.arrivi {
        Some(a) => parse_arrivals(a, prompt.len())?,
        None => vec![0; prompt.len()],
    };
    if o.arrivi.is_some() && o.forma_verificatore {
        return Err("--arrivals cannot be used with --verifier-mode; conversations are replayed separately".into());
    }
    let iniziali = soglie.iter().filter(|&&s| s == 0).count();
    if iniziali > slots as usize {
        return Err(format!(
            "{iniziali} initial conversations for {slots} slots (--slots); delay the rest with --arrivals"
        ));
    }
    if prompt.len() > engine.n_batch {
        return Err("conversations exceed n_batch; a group step must fit in one batch".into());
    }
    let g_eff = o.groups.max(1).min(prompt.len());
    let classi = match &o.classi {
        Some(t) => parse_classes(t, prompt.len())?,
        None => vec![Class::Interactive; prompt.len()],
    };
    if o.classi.is_some() && o.forma_verificatore {
        return Err("--classes cannot be used with --verifier-mode".into());
    }
    let amm = Admission::con_gruppi(soglie, classi.clone(), &engine.per_ctx[..g_eff], slots as usize, o.riserva_interattive);
    let mut convs = Vec::new();
    for (i, (id, text)) in prompt.iter().enumerate() {
        let (tok, testo_tmpl, usato) = engine.prepara_prompt(text)?;
        if i == 0 {
            eprintln!(
                "[head] chat template: {}\n[head] first prompt: {:?}",
                if usato { "API C" } else { "built-in Gemma 4 template" },
                testo_tmpl
            );
        }
        let mut seme_ctl = (o.temperature, seme_conversazione(o.seed, (i + 1) as u64));
        let mut tagli = Tagli::default();
        let forzati = if o.forzati.is_some() {
            let r = forzati_da.iter().find(|r| &r.prompt_id == id).ok_or_else(|| format!("prompt {id} is missing from the reference transcript"))?;
            if r.token.is_empty() {
                return Err(format!("prompt {id} has no reference tokens"));
            }
            // Conversation seed used to choose the tokens: take it from the record when declared, or derive it from the
            // options and record conversation ID, if present. In the chain, that ID is the prompt index plus 1.
            let cid = if r.conversation_id != 0 { r.conversation_id } else { (i + 1) as u64 };
            seme_ctl = (r.temperature.unwrap_or(o.temperature), r.root_seed.unwrap_or_else(|| seme_conversazione(o.seed, cid)));
            tagli = Tagli { top_p: r.top_p.unwrap_or(0.0), top_k: r.top_k.unwrap_or(0) };
            Some(r.token.clone())
        } else {
            None
        };
        let da_generare = forzati.as_ref().map_or(o.max_tokens as usize, |f| f.len());
        if tok.len() + da_generare > o.ctx_per_slot as usize {
            return Err(format!("prompt {id}: {} prompt tokens + {da_generare} new tokens exceed the slot context ({})", tok.len(), o.ctx_per_slot));
        }
        convs.push(Conv {
            prompt_id: id.clone(),
            conv_id: (i + 1) as u64,
            // Assign group and slot on admission (`Ammissione::prossima`).
            gruppo: 0,
            slot: 0,
            c: Conversation::new((i + 1) as u64, tok.len() as u32, Config::default(), Duration::ZERO).0,
            da_aprire: None,
            da_inviare: Vec::new(),
            passo_primo: 0,
            prompt: tok,
            inviati: 0,
            token: Vec::new(),
            logprob: Vec::new(),
            tempi_ms: Vec::new(),
            pezzi: Vec::new(),
            done: None,
            orfana: false,
            forzati,
            ammessa: false,
            t_arrivo_ms: None,
            t_ammissione_ms: None,
            in_volo_amm: false,
            seme_ctl,
            class: classi[i],
            sospensioni: 0,
            ext_id: 0,
            chat_id: String::new(),
            max_tokens: None,
            utf8: Utf8Fragments::default(),
            tagli,
            filtro: StopFilter::default(),
            bz: BozzaConv::new(if network.is_some() { o.max_draft_tokens as usize } else { 0 }),
        });
    }
    if let Some(p) = &o.draft_oracle {
        let rec = output::leggi_jsonl(p)?;
        for cv in convs.iter_mut() {
            cv.bz.oracolo = rec.iter().find(|r| r.prompt_id == cv.prompt_id).map(|r| r.token.clone());
        }
        eprintln!("[head] using oracle drafts from {p} (change every {}): {} conversations", o.draft_oracle_corrupt_every, convs.iter().filter(|c| c.bz.oracolo.is_some()).count());
    }
    let modo = if network.is_some() {
        "chain"
    } else if o.forzati.is_some() {
        "replay"
    } else {
        "reference"
    };
    let t0 = Instant::now();
    let mut misure_amm: Option<serde_json::Value> = None;
    let mut misure_gruppi: Option<serde_json::Value> = None;
    let mut tempi_stadi: std::collections::BTreeMap<u32, serde_json::Value> = Default::default();
    let mut safeguards: Option<serde_json::Value> = None;
    let mut bozza_riepilogo: Option<serde_json::Value> = None;
    let mut ctl_seme: Vec<(u64, u32, PosSeme)> = Vec::new();
    let (convs, esp, top, durata) = if o.forma_verificatore {
        eprintln!("[head] verifier replay scores each conversation separately (n_batch {})", engine.n_batch);
        let mut convs = convs;
        let mut top = Vec::new();
        forma_verificatore(&mut engine, &mut convs, o, &mut top, &mut ctl_seme)?;
        (convs, Vec::new(), top, t0.elapsed().as_secs_f64())
    } else {
        let valle = match network {
            Some(r) => connect_downstream(r, Duration::from_secs(o.attesa_s), o.draft_source == DraftSource::Mtp)?,
            None => Downstream::Locale,
        };
        eprintln!("[head] chain ready with {} conversations and {} groups", convs.len(), o.groups.max(1).min(convs.len()));
        let t0 = Instant::now();
        let mut t = Head::new(&mut engine, o, network, convs, valle, amm, slots);
        t.t0 = t0;
        let esito = t.corsa().map_err(|f| match f {
            Fail::Guasto(m) | Fail::GuastoConv(_, m) | Fail::Fatale(m) => m,
        });
        let durata = t0.elapsed().as_secs_f64();
        misure_amm = if o.arrivi.is_some() { Some(t.misure_ammissione()) } else { None };
        misure_gruppi = Some(t.mis.json(o.groups.max(1).min(t.convs.len())));
        safeguards = Some(t.prot.json());
        if o.max_draft_tokens > 0 && network.is_some() {
            let mut m = t.bozza_tot.json();
            m["k_max"] = serde_json::json!(o.max_draft_tokens);
            m["max_rows"] = serde_json::json!(o.max_draft_rows);
            if o.draft_source == DraftSource::Mtp {
                // Tail draft statistics appear only for that draft source.
                m["source"] = serde_json::json!(o.draft_source.name());
            }
            bozza_riepilogo = Some(m);
        }
        if t.prot != Protezioni::default() {
            eprintln!("[head] rejected {} stale messages", t.prot.json());
        }
        t.collect_timings(Duration::from_millis(1500));
        tempi_stadi = std::mem::take(&mut t.tempi_stadi);
        let Head { convs, valle, esp, top, ctl_seme: ctl, .. } = t;
        ctl_seme = ctl;
        if let Downstream::Rete(v) = valle {
            v.col.prefill.terminate();
            v.col.gen.terminate();
        }
        esito?;
        (convs, esp, top, durata)
    };
    if let Some(p) = &o.experts {
        experts::write(p, &esp)?;
        eprintln!("[head] wrote {} expert trace rows to {p} ({} batches with missing rows)", esp.len(), engine.esperti_persi);
    }
    if let Some(p) = &o.top {
        crate::sampling::scrivi_top(p, &top)?;
        eprintln!("[head] wrote top-{} trace to {p} ({} rows)", crate::stage::TOP_K, top.len());
    }
    // Seeded resampling counts per conversation and overall (forced reference only).
    let per_conv_seme: Vec<Option<crate::seed_check::ConteggiSeme>> = convs
        .iter()
        .map(|c| {
            if c.forzati.is_none() {
                return None;
            }
            let mut v: Vec<(u32, PosSeme)> = ctl_seme.iter().filter(|x| x.0 == c.conv_id).map(|x| (x.1, x.2)).collect();
            v.sort_by_key(|x| x.0);
            let pos: Vec<PosSeme> = v.into_iter().map(|x| x.1).collect();
            Some(crate::seed_check::conta(&pos, o.tol_seme))
        })
        .collect();
    let seme_totale = per_conv_seme.iter().flatten().fold(None, |acc: Option<crate::seed_check::ConteggiSeme>, c| Some(match acc {
        None => c.clone(),
        Some(a) => crate::seed_check::somma(&a, c),
    }));
    let recs: Vec<RecordConv> = convs
        .iter()
        .zip(per_conv_seme)
        .map(|(c, seed)| RecordConv {
            temperature: Some(c.seme_ctl.0),
            root_seed: Some(c.seme_ctl.1),
            top_p: if c.tagli.top_p != 0.0 { Some(c.tagli.top_p) } else { None },
            top_k: if c.tagli.top_k != 0 { Some(c.tagli.top_k) } else { None },
            seed,
            prompt_id: c.prompt_id.clone(),
            conversation_id: c.conv_id,
            n_token_prompt: c.prompt.len() as u32,
            token: c.token.clone(),
            logprob: c.logprob.clone(),
            logprob_bit: c.logprob.iter().map(|x| x.to_bits()).collect(),
            text: String::from_utf8_lossy(&c.pezzi).into_owned(),
            done: c.done.unwrap_or("?").to_string(),
            ttft_ms: c.tempi_ms.first().copied().unwrap_or(0.0),
            tempi_ms: c.tempi_ms.clone(),
            arrivo_ms: if o.arrivi.is_some() { c.t_arrivo_ms } else { None },
            attesa_coda_ms: if o.arrivi.is_some() { c.t_arrivo_ms.zip(c.t_ammissione_ms).map(|(a, b)| b - a) } else { None },
            ttft_arrivo_ms: if o.arrivi.is_some() { c.t_arrivo_ms.zip(c.tempi_ms.first().copied()).map(|(a, b)| b - a) } else { None },
            class: if o.classi.is_some() { Some(if c.class == Class::Batch { "batch" } else { "interactive" }.to_string()) } else { None },
            sospensioni: if o.classi.is_some() { Some(c.sospensioni) } else { None },
        })
        .collect();
    let riep = output::summary(modo, &recs, durata);
    let mut jv = serde_json::to_value(&riep).map_err(|e| e.to_string())?;
    jv["timings"] = crate::timings::istantanea();
    if !tempi_stadi.is_empty() {
        // Key = stage index (--index, from 1). Values use `timings::StageTimings::json` format.
        jv["stages"] = serde_json::Value::Object(tempi_stadi.iter().map(|(i, v)| (i.to_string(), v.clone())).collect());
    }
    if let Some(m) = seme_totale {
        // Report counts with their margin tolerance. The coordinator applies the decision threshold.
        jv["seed_check"] = serde_json::to_value(&m).map_err(|e| e.to_string())?;
    }
    if o.classi.is_some() {
        jv["classes"] = output::riepilogo_classi(&recs, durata);
    }
    if let Some(m) = misure_amm {
        jv["admission"] = m;
    }
    if let Some(m) = misure_gruppi {
        jv["groups"] = m;
    }
    if let Some(m) = safeguards {
        jv["safeguards"] = m;
    }
    if o.draft_source == DraftSource::Mtp && bozza_riepilogo.is_some() {
        // Include the conversation's draft counters in the log.
        for c in convs.iter().filter(|c| c.bz.cont.passi > 0) {
            let b = &c.bz.cont;
            eprintln!(
                "[head] MTP draft {}: accepted {} of {} proposals ({} steps, {} without drafts, {} discarded rows)",
                c.prompt_id, b.token_accettati, b.token_proposti, b.passi, b.passi_senza_bozza, b.righe_buttate
            );
        }
    }
    if let Some(mut m) = bozza_riepilogo {
        // Per conversation (tests keep all of them): proposals, proposed and accepted tokens, discarded rows.
        m["per_conv"] = serde_json::Value::Array(convs.iter().filter(|c| c.bz.cont.passi > 0).map(|c| {
            let mut j = c.bz.cont.json();
            j["prompt_id"] = serde_json::json!(c.prompt_id);
            j
        }).collect());
        jv["draft"] = m;
    }
    let json = serde_json::to_string_pretty(&jv).map_err(|e| e.to_string())?;
    if let Some(p) = &o.output {
        output::scrivi_jsonl(&format!("{p}.jsonl"), &recs)?;
        std::fs::write(format!("{p}.summary.json"), &json).map_err(|e| e.to_string())?;
    }
    println!("{json}");
    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn prefill_defaults_to_fifo_and_accepts_round_robin() {
        assert_eq!(PrefillOrder::default(), PrefillOrder::Fifo);
        assert_eq!(PrefillOrder::da_testo("fifo"), Ok(PrefillOrder::Fifo));
        assert_eq!(PrefillOrder::da_testo("round-robin"), Ok(PrefillOrder::Turno));
        assert!(PrefillOrder::da_testo("corto").is_err());
    }

    #[test]
    fn reference_prefill_token_carries_resume_step() {
        // Step 0 on first open. After suspension with 6 tokens recorded, use step 6 so the machine accepts it.
        assert_eq!(token_prefill_locale(0, 7, 1, 0, 5, -1.0).rows[0].conversation_step, 0);
        let (mut c, _) = Conversation::new(7, 10, Config::default(), Duration::ZERO);
        for p in 0..6u32 {
            assert!(matches!(c.token(1, p, 1, Duration::ZERO), TokenOutcome::Registrato { .. }));
        }
        c.suspend(Duration::ZERO).unwrap();
        c.resume(Duration::ZERO).unwrap();
        let tk = token_prefill_locale(0, 7, c.tentativo(), c.expected_step(), 9, -1.0);
        let r = &tk.rows[0];
        assert_eq!(r.conversation_step, 6);
        assert!(matches!(c.token(r.attempt_id, r.conversation_step, r.token as u32, Duration::ZERO), TokenOutcome::Registrato { .. }));
        // A resumed conversation must reject step zero because that position is already committed.
        let (mut d, _) = Conversation::new(8, 10, Config::default(), Duration::ZERO);
        d.token(1, 0, 1, Duration::ZERO);
        d.suspend(Duration::ZERO).unwrap();
        d.resume(Duration::ZERO).unwrap();
        assert_eq!(d.token(2, 0, 1, Duration::ZERO), TokenOutcome::Duplicato);
        assert_eq!(d.state(), Status::Rebuilding);
    }

    #[test]
    fn reports_unfinished_conversations() {
        let mut a = conv(1);
        let b = conv(2);
        a.done = Some("eog");
        let v = non_concluse(&[a, b]);
        assert_eq!(v.len(), 1);
        assert!(v[0].starts_with("p2 "), "{v:?}");
    }

    #[test]
    fn episode_rounds_do_not_overlap() {
        // The last round of an episode must still compare below the next episode.
        assert!(initial_round(1) > initial_round(0) + (1u64 << 31));
        assert_eq!(initial_round(0), 1);
        for e in [0u32, 1, 2, 4095, 4096, 1 << 20, u32::MAX - 1] {
            assert!(initial_round(e + 1) - initial_round(e) == 1 << 32, "{e}");
        }
        assert!(initial_round(5) + u64::from(u32::MAX) - 1 < initial_round(6));
    }

    #[test]
    fn stale_tokens_are_removed_from_ready_queue() {
        let tk = |g: u32, r: u64, prefill: bool| pb::Token { group_id: g, group_round: r, prefill, ..Default::default() };
        let round = [initial_round(1) + 5, initial_round(1) + 2];
        let lim = Duration::from_secs(1200);
        let s0 = Duration::ZERO;
        // Drop a consumed round or one from an older episode.
        assert!(stale_token(&tk(0, initial_round(1) + 4, false), &round, true, s0, lim));
        assert!(stale_token(&tk(1, 300, false), &round, true, s0, lim));
        // Keep the expected round, future rounds and unknown groups.
        assert!(!stale_token(&tk(0, initial_round(1) + 5, false), &round, true, s0, lim));
        assert!(!stale_token(&tk(1, initial_round(1) + 3, false), &round, true, s0, lim));
        assert!(!stale_token(&tk(7, 1, false), &round, true, s0, lim));
        // Keep prefill until it is too old. Before generation starts there are no rounds to compare.
        assert!(!stale_token(&tk(0, 0, true), &round, true, Duration::from_secs(100), lim));
        assert!(stale_token(&tk(0, 0, true), &round, true, Duration::from_secs(1300), lim));
        assert!(!stale_token(&tk(0, 1, false), &round, false, s0, lim));
    }

    fn conv(id: u64) -> Conv {
        let (c, az) = Conversation::new(id, 10, Config::default(), Duration::ZERO);
        let Action::Open { tentativo, start_position } = az[0] else { panic!() };
        Conv {
            prompt_id: format!("p{id}"),
            conv_id: id,
            gruppo: 0,
            slot: id as u32 - 1,
            prompt: vec![1; 10],
            c,
            da_aprire: Some((tentativo, start_position)),
            da_inviare: Vec::new(),
            passo_primo: 0,
            inviati: 0,
            token: Vec::new(),
            logprob: Vec::new(),
            tempi_ms: Vec::new(),
            pezzi: Vec::new(),
            done: None,
            orfana: false,
            forzati: None,
            ammessa: true,
            t_arrivo_ms: None,
            t_ammissione_ms: None,
            in_volo_amm: false,
            seme_ctl: (0.0, 1),
            class: Class::Interactive,
            sospensioni: 0,
            ext_id: 0,
            chat_id: String::new(),
            max_tokens: None,
            utf8: Utf8Fragments::default(),
            tagli: Tagli::default(),
            filtro: StopFilter::default(),
            bz: BozzaConv::new(0),
        }
    }

    fn error(conv_id: u64, att: u32) -> pb::Confirmation {
        pb::Confirmation { kind: pb::AckKind::Error as i32, conversation_id: conv_id, attempt_id: att, ok: false, error: "x".into(), ..Default::default() }
    }

    #[test]
    fn error_with_valid_conversation_is_a_conversation_fault() {
        let convs = vec![conv(1), conv(2), conv(3)];
        assert_eq!(classify_confirmation(&convs, &error(2, 1)), ConfirmationClass::Conv(1));
        assert_eq!(classify_confirmation(&convs, &error(3, 1)), ConfirmationClass::Conv(2));
    }

    #[test]
    fn error_without_valid_conversation_is_a_chain_fault() {
        let convs = vec![conv(1), conv(2)];
        assert_eq!(classify_confirmation(&convs, &error(0, 1)), ConfirmationClass::Chain);
        assert_eq!(classify_confirmation(&convs, &error(99, 1)), ConfirmationClass::Chain);
    }

    #[test]
    fn ignores_errors_from_stale_attempts_and_finished_conversations() {
        let mut convs = vec![conv(1), conv(2)];
        assert_eq!(classify_confirmation(&convs, &error(1, 7)), ConfirmationClass::Ignora);
        convs[1].done = Some("eog");
        assert_eq!(classify_confirmation(&convs, &error(2, 1)), ConfirmationClass::Ignora);
        convs[0].c.cancel(Duration::ZERO).unwrap();
        assert_eq!(classify_confirmation(&convs, &error(1, 1)), ConfirmationClass::Ignora);
    }

    #[test]
    fn ignores_successful_confirmation() {
        let convs = vec![conv(1)];
        let ok = pb::Confirmation { kind: pb::AckKind::Open as i32, conversation_id: 1, attempt_id: 1, ok: true, error: String::new(), ..Default::default() };
        assert_eq!(classify_confirmation(&convs, &ok), ConfirmationClass::Ignora);
    }

    /// Only the faulty conversation changes attempt. Tokens from its old attempt are kept out of
    /// its transcript (attempt watermark), while other conversations keep running.
    #[test]
    fn conversation_fault_does_not_change_other_attempts() {
        let mut convs = vec![conv(1), conv(2)];
        let t = Duration::ZERO;
        for cv in convs.iter_mut() {
            assert!(matches!(cv.c.token(1, 0, 5, t), TokenOutcome::Registrato { .. }));
        }
        // Fault and recovery for conversation 0 only, as in `recover_conversation`.
        let t0 = convs[0].c.tentativo();
        convs[0].c.fault(t0, t).unwrap();
        let az = convs[0].c.chain_ready(t).unwrap();
        assert_eq!(az, vec![Action::Open { tentativo: 2, start_position: 0 }]);
        assert_eq!((convs[0].c.tentativo(), convs[0].c.state()), (2, Status::Rebuilding));
        assert_eq!((convs[1].c.tentativo(), convs[1].c.state()), (1, Status::Active));
        // An in-flight token from the failed attempt is dropped. The other conversation's token is recorded.
        assert_eq!(convs[0].c.token(1, 1, 6, t), TokenOutcome::Scartato);
        assert!(matches!(convs[1].c.token(1, 1, 6, t), TokenOutcome::Registrato { .. }));
        // The failed conversation resumes at step = recorded tokens, with the new attempt.
        assert_eq!(convs[0].c.expected_step(), 1);
        assert!(matches!(convs[0].c.token(2, 1, 6, t), TokenOutcome::Registrato { .. }));
    }

    #[test]
    fn per_row_int8_state_has_small_error_and_row_scales() {
        let riga0: Vec<f32> = (0..64).map(|i| ((i as f32) * 0.37).sin() * 3.0).collect();
        let riga1: Vec<f32> = (0..64).map(|i| ((i as f32) * 0.11).cos() * 300.0).collect();
        let mut s = riga0.clone();
        s.extend_from_slice(&riga1);
        let d = degrada_int8(&s, 64);
        assert_eq!(d.len(), s.len());
        assert_ne!(d, s);
        // Maximum error per row <= half the row scale (max/254)
        for (r, orig) in [&riga0, &riga1].iter().enumerate() {
            let m = orig.iter().fold(0f32, |a, x| a.max(x.abs()));
            for j in 0..64 {
                assert!((d[r * 64 + j] - orig[j]).abs() <= m / 254.0 * 1.0001, "row {r}, column {j}");
            }
        }
        // At most 255 distinct values per row.
        let mut v: Vec<i64> = d[..64].iter().map(|x| (*x * 1e6) as i64).collect();
        v.sort_unstable();
        v.dedup();
        assert!(v.len() <= 255);
        // An all-zero row stays zero. Represent the largest absolute value almost exactly.
        assert_eq!(degrada_int8(&[0.0; 8], 4), vec![0.0; 8]);
        assert!((degrada_int8(&[1.0, -0.5, 0.25, 0.0], 4)[0] - 1.0).abs() < 1e-6);
    }

    // Request memory lifetime during service.

    /// Completed service conversation: admitted, `n` tokens recorded, closed as in `registra`.
    fn service_lifetime(amm: &mut Admission, convs: &mut Vec<Conv>, ext: u64, n: u32) {
        let idx = amm.aggiungi(Class::Interactive);
        assert_eq!(idx, convs.len());
        let mut cv = conv(ext + 1000);
        cv.ext_id = ext;
        cv.prompt = vec![7; 20];
        cv.ammessa = false;
        convs.push(cv);
        assert_eq!(amm.arriva(0, false), vec![idx]);
        let (c, _, _) = amm.prossima().unwrap();
        assert_eq!(c, idx);
        let cv = &mut convs[c];
        for p in 0..n {
            let t = cv.c.tentativo();
            assert!(matches!(cv.c.token(t, p, 5, Duration::ZERO), TokenOutcome::Registrato { .. }));
            cv.token.push(5);
            cv.logprob.push(-1.0);
            cv.tempi_ms.push(1.0);
            cv.pezzi.extend_from_slice(b"ab");
        }
        cv.done = Some("max_token");
        cv.c.finish_response(Duration::ZERO).unwrap();
        cv.c.close_by_user(Duration::ZERO).unwrap();
        amm.libera(c);
        cv.clear_weights();
    }

    #[test]
    fn two_thousand_completed_requests_are_compacted() {
        let mut amm = Admission::new(Vec::new(), 2, 2, 4);
        let mut convs: Vec<Conv> = Vec::new();
        for i in 0..2000u64 {
            service_lifetime(&mut amm, &mut convs, i + 1, 4);
            // Completed shells keep no prompt, tokens or transcript.
            let cv = convs.last().unwrap();
            assert!(cv.prompt.is_empty() && cv.token.is_empty() && cv.logprob.is_empty() && cv.pezzi.is_empty() && cv.tempi_ms.is_empty());
            assert!(cv.c.transcript().is_empty());
            assert_eq!(cv.prompt.capacity() + cv.token.capacity() + cv.pezzi.capacity(), 0);
        }
        assert_eq!(convs.len(), 2000);
        let vuote = VecDeque::new();
        assert!(service_idle(&convs, &[], &vuote, None, &VecDeque::new(), &amm));
        // Compaction: reset admission with the same slots.
        convs = Vec::new();
        amm.azzera_conversazioni();
        assert_eq!(amm.occupati(), 0);
        service_lifetime(&mut amm, &mut convs, 2001, 3);
        assert_eq!(convs.len(), 1);
        assert_eq!(convs[0].done, Some("max_token"));
    }

    #[test]
    fn offline_conversations_keep_their_records() {
        let mut cv = conv(1);
        cv.token = vec![1, 2, 3];
        cv.prompt = vec![9; 4];
        cv.clear_weights(); // Zero identifies a finite-run conversation.
        assert_eq!((cv.token.len(), cv.prompt.len()), (3, 4));
    }

    #[test]
    fn active_conversation_prevents_compaction() {
        let mut amm = Admission::new(Vec::new(), 2, 2, 4);
        let mut convs: Vec<Conv> = Vec::new();
        // The live conversation: admitted, with 3 tokens.
        let idx = amm.aggiungi(Class::Interactive);
        let mut viva = conv(5000);
        viva.ext_id = 99;
        viva.ammessa = false;
        convs.push(viva);
        amm.arriva(0, false);
        amm.prossima().unwrap();
        for p in 0..3u32 {
            let t = convs[idx].c.tentativo();
            convs[idx].c.token(t, p, 5, Duration::ZERO);
            convs[idx].token.push(5);
        }
        for i in 0..50u64 {
            service_lifetime(&mut amm, &mut convs, i + 1, 2);
            assert!(!service_idle(&convs, &[idx], &VecDeque::new(), None, &VecDeque::new(), &amm));
        }
        // The live conversation retains tokens, transcript and state.
        assert_eq!(convs[idx].token.len(), 3);
        assert_eq!(convs[idx].c.transcript().len(), 3);
        assert!(convs[idx].done.is_none());
        // `clear_transcript` has no effect on a nonterminal machine.
        convs[idx].c.clear_transcript();
        assert_eq!(convs[idx].c.transcript().len(), 3);
        // Even one conversation in flight or in admission keeps the service from being idle.
        let mut finita = Vec::new();
        service_lifetime(&mut Admission::new(Vec::new(), 1, 1, 1), &mut finita, 1, 1);
        let amm2 = Admission::new(Vec::new(), 1, 1, 1);
        assert!(service_idle(&finita, &[], &VecDeque::new(), None, &VecDeque::new(), &amm2));
        assert!(!service_idle(&finita, &[], &VecDeque::from([0usize]), None, &VecDeque::new(), &amm2));
        assert!(!service_idle(&finita, &[], &VecDeque::new(), Some(0), &VecDeque::new(), &amm2));
        assert!(!service_idle(&finita, &[], &VecDeque::new(), None, &VecDeque::from([(0usize, 1u64)]), &amm2));
    }

    #[test]
    fn late_token_from_compacted_conversation_is_unknown() {
        let mut amm = Admission::new(Vec::new(), 1, 2, 2);
        let mut convs: Vec<Conv> = Vec::new();
        service_lifetime(&mut amm, &mut convs, 1, 2);
        let id_vecchio = convs[0].conv_id;
        convs = Vec::new();
        amm.azzera_conversazioni();
        // `registra` looks up the conversation by ID. After compaction it is missing, so count it in `Protezioni`.
        let mut prot = Protezioni::default();
        if convs.iter().position(|c| c.conv_id == id_vecchio).is_none() {
            prot.token_sconosciuto(2);
        }
        assert_eq!(prot.token_sconosciuti_totale(), 1);
        // IDs remain monotonic across compaction so a late token cannot match a new conversation.
        let new = id_vecchio + 1;
        service_lifetime(&mut amm, &mut convs, 5, 1);
        assert!(convs.iter().all(|c| c.conv_id != new - 1 || c.ext_id != 0));
    }

    #[test]
    fn service_interval_history_is_bounded() {
        let mut v = Vec::new();
        for i in 0..10_000 {
            spingi_limitato(&mut v, i as f64, MAX_INTERVALLI);
        }
        assert_eq!(v.len(), MAX_INTERVALLI);
        assert_eq!(v[0], (10_000 - MAX_INTERVALLI) as f64);
        let mut w = Vec::new();
        for i in 0..5000 {
            spingi_limitato(&mut w, i as f64, usize::MAX);
        }
        assert_eq!(w.len(), 5000);
    }

    // Stop matching across token boundaries.

    /// Pass chunks (token bytes) through `Utf8Fragments` and the filter, as in `service_token`. Return the text sent for
    /// each token, whether stop triggered and the final buffered text.
    fn passa_pezzi(stop: &[&str], pezzi: &[&[u8]]) -> (Vec<String>, bool, String) {
        let stop: Vec<String> = stop.iter().map(|s| s.to_string()).collect();
        let (mut f, mut u) = (StopFilter::new(&stop), Utf8Fragments::default());
        let mut fuori = Vec::new();
        for p in pezzi {
            let completo = u.aggiungi(p);
            let (t, fermo) = if completo.is_empty() { (String::new(), false) } else { f.passa(&completo) };
            fuori.push(t);
            if fermo {
                return (fuori, true, String::new());
            }
        }
        let mut residuo = f.clear();
        residuo.push_str(&u.clear());
        (fuori, false, residuo)
    }

    #[test]
    fn stop_string_inside_one_token() {
        let (fuori, fermo, _) = passa_pezzi(&["FINE"], &[b"Ciao ", b"a tutti FINE e oltre", b"mai"]);
        assert!(fermo);
        assert_eq!(fuori.concat(), "Ciao a tutti ");
        assert_eq!(fuori.len(), 2, "do not consume any more text after the stop string");
    }

    #[test]
    fn stop_string_spans_two_tokens() {
        let (fuori, fermo, _) = passa_pezzi(&["Mario!"], &[b"Ciao, Ma", b"rio", b"! resto"]);
        assert!(fermo);
        assert_eq!(fuori.concat(), "Ciao, ");
        // The "Ma" suffix was held back from the first token.
        assert_eq!(fuori[0], "Ciao, ");
        assert_eq!(fuori[1], "");
    }

    #[test]
    fn stop_string_split_inside_multibyte_character() {
        // Stop "è!" (two bytes for "è"), with the character split across tokens.
        let e = "è".as_bytes();
        let (fuori, fermo, _) = passa_pezzi(&["è!"], &[b"ciao ", &e[..1], &e[1..], b"! fine"]);
        assert!(fermo);
        assert_eq!(fuori.concat(), "ciao ");
        // The stop string stays incomplete, so the buffered prefix "è" comes out whole.
        let (fuori, fermo, residuo) = passa_pezzi(&["è!"], &[b"ciao ", &e[..1], &e[1..], b"?"]);
        assert!(!fermo);
        assert_eq!(format!("{}{}", fuori.concat(), residuo), "ciao è?");
        // Text with a multibyte character before the stop.
        let (fuori, fermo, _) = passa_pezzi(&["STOP"], &["caffè ".as_bytes(), b"STO", b"P"]);
        assert!(fermo);
        assert_eq!(fuori.concat(), "caffè ");
    }

    #[test]
    fn partial_stop_match_is_released() {
        let (fuori, fermo, residuo) = passa_pezzi(&["Mario!"], &[b"Ciao, Mar", b"ia", b" bella"]);
        assert!(!fermo);
        assert_eq!(fuori[0], "Ciao, ");
        assert_eq!(format!("{}{}", fuori.concat(), residuo), "Ciao, Maria bella");
        // A false start at end of response comes out with the buffered text.
        let (fuori, fermo, residuo) = passa_pezzi(&["Mario!"], &[b"Ciao, Mar"]);
        assert!(!fermo);
        assert_eq!(fuori.concat(), "Ciao, ");
        assert_eq!(residuo, "Mar");
    }

    #[test]
    fn earliest_stop_match_wins_and_empty_strings_are_ignored() {
        let (fuori, fermo, _) = passa_pezzi(&["", "zz", "bb"], &[b"aa bb cc zz"]);
        assert!(fermo);
        assert_eq!(fuori.concat(), "aa ");
        let mut f = StopFilter::new(&["".to_string()]);
        assert!(!f.attivo());
        assert_eq!(f.passa("tutto"), ("tutto".to_string(), false));
    }

    #[test]
    fn no_stop_strings_leaves_text_unbuffered() {
        let (fuori, fermo, residuo) = passa_pezzi(&[], &[b"ab", b"cd"]);
        assert!(!fermo);
        assert_eq!(fuori, vec!["ab", "cd"]);
        assert!(residuo.is_empty());
    }
}
