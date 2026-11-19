//! The coordinator assigns one stage to the agent. The manager launches
//! the child for that role and publishes its lifecycle. If it exits, the
//! next assignment comes from the coordinator.
//!
//! An intermediate or tail stage announces readiness on stderr after it
//! opens its listener. The head signals readiness on its inherited output
//! pipe after connecting to the chain.
//!
//! One task serializes assignments, releases and stops. The session can
//! continue processing control messages while the stage is shutting down.

use std::{
    collections::{HashMap, HashSet, VecDeque},
    net::SocketAddr,
    path::{Path, PathBuf},
    process::Stdio,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::{Duration, Instant},
};

use anyhow::{bail, Context, Result};
use bytes::BytesMut;
use publicompute_proto::{
    node::{stage_status::State, StageAssign, StageRelease, StageStatus},
    service as pb,
    wire,
};
use tokio::{
    io::{AsyncBufReadExt, AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, BufReader},
    process::{ChildStdin, ChildStdout, Command},
    sync::{mpsc, oneshot, watch, Mutex as AsyncMutex},
};
use tracing::{info, warn};

use crate::engine::{
    managed,
    output::{self, OutputSender, OutputSize, Queued},
    supervisor::{EngineState, ParkedEngine, Supervisor},
};

/// Configuration shared by stage assignments on this agent.
#[derive(Clone, Debug)]
pub struct StagesConfig {
    /// Stage process executable.
    pub stage_bin: PathBuf,
    /// Path to the full GGUF file. The child loads its assigned layers.
    pub model_path: PathBuf,
    /// Required weights fingerprint. Reject assignments that name different weights.
    pub weights_sha256: Vec<u8>,
    /// Directory for the per-chain and per-epoch stderr logs.
    pub log_dir: PathBuf,
    pub threads: u32,
    /// Number of assigned layers to offload. Use 999 to offload all of them.
    pub ngl: i32,
    /// Additional arguments forwarded unchanged to the stage.
    pub extra_args: Vec<String>,
    /// Environment overrides for the stage child.
    pub env: Vec<(String, String)>,
    /// Startup deadline covering weight loading and connection setup.
    pub load_timeout: Duration,
    /// Grace period between SIGTERM and SIGKILL.
    pub stop_grace: Duration,
    /// Maximum time to wait for exit after forced termination.
    pub kill_wait: Duration,
    /// PEM trust anchor passed to the stage for mutual TLS.
    pub ca_path: Option<PathBuf>,
    /// Directory for the stage identity, verification key and signed assignments.
    pub tls_dir: PathBuf,
    /// Unsigned assignments are rejected by default. Tests can explicitly allow
    /// them to start over plaintext.
    pub in_chiaro: bool,
}

/// Resolve a one-based position in the chain to head, intermediate or tail.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    Head,
    Middle,
    Tail,
}

impl Role {
    pub fn comando(self) -> &'static str {
        match self {
            Role::Head => "head",
            Role::Middle => "middle",
            Role::Tail => "tail",
        }
    }
}

/// Estimate coordinator time minus local time in whole seconds. Use zero
/// when the handshake has no server time. The estimate includes transit delay.
pub fn clock_offset_from_welcome(server_time: Option<&prost_types::Timestamp>, ora_locale_unix: i64) -> i64 {
    server_time.map(|t| t.seconds - ora_locale_unix).unwrap_or(0)
}

fn ruolo_di(a: &StageAssign) -> Result<Role> {
    if a.n_stages < 2 {
        bail!("a chain needs at least two stages (n_stages = {})", a.n_stages);
    }
    if a.stage_index < 1 || a.stage_index > a.n_stages {
        bail!("stage_index {} is outside 1..={}", a.stage_index, a.n_stages);
    }
    Ok(if a.stage_index == 1 {
        Role::Head
    } else if a.stage_index == a.n_stages {
        Role::Tail
    } else {
        Role::Middle
    })
}

fn porta_di(addr: &str, cosa: &str) -> Result<u16> {
    let p: u16 = addr.rsplit_once(':').and_then(|(_, p)| p.parse().ok()).with_context(|| format!("{cosa} \"{addr}\": serve HOST:PORTA"))?;
    if p == 0 {
        bail!("{cosa} \"{addr}\": a fixed port is required");
    }
    Ok(p)
}

/// Validate assignment parameters before reading files or starting a child.
fn validate(a: &StageAssign) -> Result<Role> {
    let r = ruolo_di(a)?;
    if a.slots == 0 || a.ctx_per_slot == 0 || a.groups == 0 || a.n_batch == 0 {
        bail!("slots, ctx_per_slot, groups and n_batch must be greater than zero");
    }
    if a.layer_end >= 0 && (a.layer_end as u32) <= a.layer_start {
        bail!("empty layer range: [{}, {})", a.layer_start, a.layer_end);
    }
    match a.state_format.as_str() {
        "" | "fp32" | "fp16" => {}
        f => bail!("unsupported state_format \"{f}\" (fp32 | fp16)"),
    }
    match r {
        Role::Head => {
            if a.next_addr.is_empty() || a.head_token_addr.is_empty() {
                bail!("head requires next_addr and head_token_addr");
            }
            porta_di(&a.head_token_addr, "head_token_addr")?;
        }
        Role::Middle => {
            if a.listen_addr.is_empty() || a.next_addr.is_empty() {
                bail!("middle stage requires listen_addr and next_addr");
            }
            porta_di(&a.listen_addr, "listen_addr")?;
        }
        Role::Tail => {
            if a.listen_addr.is_empty() || a.head_token_addr.is_empty() {
                bail!("tail requires listen_addr and head_token_addr");
            }
            porta_di(&a.listen_addr, "listen_addr")?;
        }
    }
    Ok(r)
}

/// Build stage command arguments for the local child.
pub fn arguments(cfg: &StagesConfig, a: &StageAssign) -> Result<Vec<String>> {
    let role = validate(a)?;
    let format = if a.state_format.is_empty() { "fp32" } else { a.state_format.as_str() };
    let timeout = cfg.load_timeout.as_secs().max(1);
    let mut v: Vec<String> = vec![role.comando().into()];
    let mut op = |k: &str, val: String| {
        v.push(k.to_string());
        v.push(val);
    };
    op("--model", cfg.model_path.to_string_lossy().into_owned());
    op("--layer-start", a.layer_start.to_string());
    op("--layer-end", a.layer_end.to_string());
    op("--ngl", cfg.ngl.to_string());
    op("--threads", cfg.threads.to_string());
    op("--slots", a.slots.to_string());
    op("--ctx-per-slot", a.ctx_per_slot.to_string());
    op("--groups", a.groups.to_string());
    op("--n-batch", a.n_batch.to_string());
    op("--timeout-s", timeout.to_string());
    match role {
        Role::Head => {
            op("--state-format", format.into());
            op("--next", a.next_addr.clone());
            op("--listen-token", a.head_token_addr.clone());
            // Bound the wait for a response so a dead stage becomes a recoverable chain fault.
            op("--work-timeout-s", "120".into());
        }
        Role::Middle => {
            op("--state-format", format.into());
            op("--listen", a.listen_addr.clone());
            op("--next", a.next_addr.clone());
            op("--index", a.stage_index.to_string());
        }
        Role::Tail => {
            op("--listen", a.listen_addr.clone());
            op("--head-token", a.head_token_addr.clone());
            op("--index", a.stage_index.to_string());
        }
    }
    if role == Role::Head {
        v.push("--service-pipe".into());
    }
    v.extend(cfg.extra_args.iter().cloned());
    Ok(v)
}

/// Construct a lifecycle update for the coordinator.
fn status_message(a: &StageAssign, state: State, error: &str, port: u32, load_seconds: f64) -> StageStatus {
    StageStatus { chain_id: a.chain_id.clone(), assign_epoch: a.assign_epoch, state: state as i32, error: error.to_string(), port, load_seconds }
}

// Head service connection and request routing.

/// Responses the head emits for an individual request.
#[derive(Debug)]
pub enum OutgoingLink {
    Token(pb::OutputToken),
    Done(pb::Done),
}

impl OutputSize for OutgoingLink {
    fn bytes(&self) -> usize {
        match self {
            Self::Token(token) => token.text.capacity(),
            Self::Done(done) => done.remaining_text.capacity() + done.error.capacity() + done.reason.capacity() + done.prompt_token.capacity() * size_of::<i32>(),
        }
    }
}

pub struct HeadResponses {
    messages: mpsc::Receiver<Queued<OutgoingLink>>,
    status: watch::Receiver<bool>,
}

impl HeadResponses {
    pub async fn recv(&mut self) -> Option<OutgoingLink> {
        tokio::select! {
            biased;
            _ = output::wait_for_saturation(&mut self.status) => None,
            message = self.messages.recv() => message.map(|message| message.item),
        }
    }

    pub fn saturated(&self) -> bool {
        *self.status.borrow()
    }
}

/// Multiplex concurrent requests through one head connection. Route each
/// response to its request ID because token messages can be interleaved.
pub struct HeadConnection {
    scrittore: AsyncMutex<Box<dyn AsyncWrite + Send + Unpin>>,
    in_volo: Mutex<HashMap<u64, OutputSender<OutgoingLink>>>,
    prossimo: AtomicU64,
    state: Mutex<pb::Status>,
    chiuso: watch::Sender<bool>,
}

impl HeadConnection {
    fn from_io(mut lettura: impl AsyncRead + Send + Unpin + 'static, scrittura: impl AsyncWrite + Send + Unpin + 'static, mut buf: BytesMut) -> Arc<Self> {
        let (chiuso, _) = watch::channel(false);
        let link = Arc::new(HeadConnection {
            scrittore: AsyncMutex::new(Box::new(scrittura)),
            in_volo: Mutex::new(HashMap::new()),
            prossimo: AtomicU64::new(1),
            state: Mutex::new(pb::Status::default()),
            chiuso,
        });
        let l2 = link.clone();
        tokio::spawn(async move {
            let mut tmp = vec![0u8; 64 << 10];
            'lettura: loop {
                loop {
                    match wire::try_decode_frame::<pb::FromHead>(&mut buf) {
                        Ok(Some(m)) => l2.smista(m),
                        Ok(None) => break,
                        Err(e) => {
                            warn!("head service returned an unreadable frame: {e}");
                            break 'lettura;
                        }
                    }
                }
                match lettura.read(&mut tmp).await {
                    Ok(0) | Err(_) => break,
                    Ok(n) => buf.extend_from_slice(&tmp[..n]),
                }
            }
            // Close outstanding response channels so callers can detect disconnect.
            l2.close();
        });
        link
    }

    fn close(&self) {
        let mut pending = self.in_volo.lock().unwrap();
        self.chiuso.send_replace(true);
        pending.clear();
    }

    fn smista(&self, m: pb::FromHead) {
        match m.body {
            Some(pb::from_head::Body::Token(t)) => {
                if let Some(tx) = self.in_volo.lock().unwrap().get(&t.id) {
                    let _ = tx.send(OutgoingLink::Token(t));
                }
            }
            Some(pb::from_head::Body::Done(f)) => {
                if let Some(tx) = self.in_volo.lock().unwrap().remove(&f.id) {
                    let _ = tx.send(OutgoingLink::Done(f));
                }
            }
            Some(pb::from_head::Body::Status(s)) => *self.state.lock().unwrap() = s,
            Some(pb::from_head::Body::Ready(_)) | None => {}
        }
    }

    /// Subscribe to loss of the head connection.
    pub fn osserva_chiusura(&self) -> watch::Receiver<bool> {
        self.chiuso.subscribe()
    }

    pub fn aperto(&self) -> bool {
        !*self.chiuso.borrow()
    }

    pub fn nuovo_id(&self) -> u64 {
        self.prossimo.fetch_add(1, Ordering::SeqCst)
    }

    /// Return the latest head capacity and active request count for local use.
    #[allow(dead_code)]
    pub fn state(&self) -> pb::Status {
        *self.state.lock().unwrap()
    }

    async fn write(&self, m: pb::FromAgent) -> Result<()> {
        let mut v = Vec::with_capacity(prost::Message::encoded_len(&m) + 8);
        wire::encode_frame(&m, &mut v)?;
        let result = tokio::time::timeout(Duration::from_secs(30), async {
            let mut writer = self.scrittore.lock().await;
            if !self.aperto() {
                return Err(std::io::Error::new(std::io::ErrorKind::BrokenPipe, "head pipe closed"));
            }
            writer.write_all(&v).await
        }).await.context("head pipe write timed out").and_then(|result| result.context("write to the head"));
        if result.is_err() {
            self.close();
        }
        result
    }

    /// Send a request and return its token and completion receiver. Connection
    /// loss closes the receiver without a completion event.
    pub async fn request(&self, r: pb::Request) -> Result<HeadResponses> {
        let id = r.id;
        let (tx, messages, status) = output::channel(256, 4 << 20);
        {
            let mut pending = self.in_volo.lock().unwrap();
            if !self.aperto() {
                bail!("connection to the head closed");
            }
            pending.insert(id, tx);
        }
        if let Err(e) = self.write(pb::FromAgent { body: Some(pb::from_agent::Body::Request(r)) }).await {
            self.in_volo.lock().unwrap().remove(&id);
            return Err(e);
        }
        Ok(HeadResponses { messages, status })
    }

    /// Ask the head to finish this request with a cancellation result.
    pub async fn cancel(&self, id: u64) {
        let _ = self.write(pb::FromAgent { body: Some(pb::from_agent::Body::Cancel(pb::Cancel { id })) }).await;
    }

    /// Drop the response route when the request completes or is abandoned.
    pub fn dimentica(&self, id: u64) {
        self.in_volo.lock().unwrap().remove(&id);
    }
}

// Assignment lifecycle and process supervision.

enum Cmd {
    Assegna(StageAssign, Option<Arc<Supervisor>>),
    Renew(StageAssign),
    Rilascia(StageRelease),
    /// Stop the active stage and close its reporting task at session end.
    FermaTutto(oneshot::Sender<()>),
}

struct Attivo {
    chain_id: String,
    /// Maximum concurrent conversations served by the head of this chain.
    slots: u32,
    epoch: u64,
    /// Ask the stage task to stop and pass along the reason.
    ferma: Option<oneshot::Sender<String>>,
    /// Completion means the child has exited and its final state was reported.
    finito: watch::Receiver<bool>,
    last: StageStatus,
    closing: bool,
}

struct EngineSwitch {
    chain_id: String,
    epoch: u64,
    // The session keeps the routing state through the terminal status even after the worker unparks Llama.
    parked: Option<ParkedEngine>,
    closing: bool,
}

impl EngineSwitch {
    fn matches(&self, a: &StageAssign) -> bool {
        self.chain_id == a.chain_id && self.epoch == a.assign_epoch
    }

    fn close(&mut self) {
        self.closing = true;
        if let Some(parked) = &self.parked {
            parked.returning();
        }
    }

    fn can_start_stage(&self) -> bool {
        self.parked.as_ref().is_some_and(|parked| parked.can_start_stage())
    }
}

/// Mutual TLS uses the enrolled node identity. Assignment authorization uses
/// the coordinator's public signing key from the session handshake.
#[derive(Default, Clone, Debug)]
pub struct TlsFiles {
    pub cert: Option<PathBuf>,
    pub key: Option<PathBuf>,
    pub cap_pub: Option<PathBuf>,
}

#[derive(Default)]
struct Interno {
    tx: Option<mpsc::Sender<StageStatus>>,
    attivo: Option<Attivo>,
    switching: Option<EngineSwitch>,
    // Renewals can arrive after the stage exits and Llama is ready again.
    closed: HashSet<(String, u64)>,
    link: Option<Arc<HeadConnection>>,
    tls: TlsFiles,
    /// Clock correction used when a stage checks signed assignment expiry.
    clock_offset_s: i64,
    /// Local bridge endpoint. With no endpoint, chain links use direct TCP.
    bridge: Option<Arc<crate::bridge::Bridge>>,
}

pub struct Stages {
    cfg: StagesConfig,
    interno: Mutex<Interno>,
    cmd: mpsc::UnboundedSender<Cmd>,
}

impl Stages {
    /// Start the ordered command worker on the current async runtime.
    pub fn new(cfg: StagesConfig) -> Arc<Stages> {
        let (cmd, mut rx) = mpsc::unbounded_channel();
        let s = Arc::new(Stages { cfg, interno: Mutex::new(Interno::default()), cmd });
        let s2 = s.clone();
        tokio::spawn(async move {
            while let Some(c) = rx.recv().await {
                match c {
                    Cmd::Assegna(a, sup) => s2.assign(a, sup).await,
                    Cmd::Renew(a) => s2.renew_switch(&a).await,
                    Cmd::Rilascia(r) => s2.release(r).await,
                    Cmd::FermaTutto(fatto) => {
                        s2.stop_active("session closed").await;
                        let _ = fatto.send(());
                    }
                }
            }
        });
        s
    }

    /// Take the lifecycle receiver used by the session to report stage state.
    pub fn connect_peer(&self, tx: mpsc::Sender<StageStatus>) {
        let mut i = self.interno.lock().unwrap();
        i.closed.clear();
        if i.switching.as_ref().is_some_and(|s| s.parked.is_none()) {
            i.switching = None;
        }
        i.tx = Some(tx);
    }

    /// Inject credential paths for tests. Normal startup writes the enrolled
    /// identity and the verification key received from the coordinator.
    #[cfg(test)]
    pub fn set_tls_files(&self, f: TlsFiles) {
        self.interno.lock().unwrap().tls = f;
    }

    /// Register the bridge endpoint. An assignment that specifies remote
    /// endpoints requires a bridge and fails if none is available.
    pub fn set_bridge(&self, bridge: Arc<crate::bridge::Bridge>) {
        self.interno.lock().unwrap().bridge = Some(bridge);
    }

    /// Give each bridged outgoing link a loopback listener and pass its address
    /// to the stage. Register incoming streams against the stage's own listeners.
    /// An assignment with direct links needs no listeners or routes. Keep the
    /// guard alive until the stage exits.
    async fn connect_bridge(&self, a: &StageAssign, role: Role) -> Result<(StageAssign, Option<crate::bridge::LinkGuard>)> {
        use crate::bridge::Direction;
        let bridge = self.interno.lock().unwrap().bridge.clone();
        let (forward, token) = (role != Role::Tail && !a.next_node_id.is_empty(), role == Role::Tail && !a.head_token_node_id.is_empty());
        let Some(bridge) = bridge else {
            if forward || token {
                bail!("this chain needs an iroh bridge; start the agent with --relay");
            }
            return Ok((a.clone(), None));
        };
        let mut g = bridge.guard();
        let mut b = a.clone();
        if forward {
            b.next_addr = bridge.output(&mut g, &a.chain_id, a.assign_epoch, Direction::Forward, &a.next_node_id, &a.relay_url).await?.to_string();
        }
        if token {
            b.head_token_addr = bridge.output(&mut g, &a.chain_id, a.assign_epoch, Direction::Token, &a.head_token_node_id, &a.relay_url).await?.to_string();
        }
        let locale = |porta: u16| SocketAddr::from(([127, 0, 0, 1], porta));
        if role != Role::Head {
            bridge.incoming_with(&mut g, &a.chain_id, a.assign_epoch, Direction::Forward, locale(porta_di(&a.listen_addr, "listen_addr")?), &a.peer_iroh_ids)?;
        } else {
            bridge.incoming_with(&mut g, &a.chain_id, a.assign_epoch, Direction::Token, locale(porta_di(&a.head_token_addr, "head_token_addr")?), &a.peer_iroh_ids)?;
        }
        Ok((b, Some(g)))
    }

    /// Write the certificate chain and a private PKCS #8 key as PEM for the stage.
    pub fn set_identity(&self, id: &crate::identity::Identity) -> Result<()> {
        let (cert, key) = (self.cfg.tls_dir.join("nodo-cert.pem"), self.cfg.tls_dir.join("nodo-key.pem"));
        let chain: String = id.chain.iter().map(|c| pem("CERTIFICATE", c)).collect();
        write_atomic(&cert, chain.as_bytes(), 0o644)?;
        write_atomic(&key, pem("PRIVATE KEY", id.key_der()).as_bytes(), 0o600)?;
        let mut i = self.interno.lock().unwrap();
        i.tls.cert = Some(cert);
        i.tls.key = Some(key);
        Ok(())
    }

    /// Store the assignment verification key from the handshake. An empty
    /// value leaves verification material unavailable.
    pub fn set_capability_key(&self, chiave: &[u8]) -> Result<()> {
        if chiave.is_empty() {
            self.interno.lock().unwrap().tls.cap_pub = None;
            return Ok(());
        }
        if chiave.len() != 32 {
            bail!("signing key is {} bytes, expected 32", chiave.len());
        }
        let path = self.cfg.tls_dir.join("capability.pub");
        write_atomic(&path, hex::encode(chiave).as_bytes(), 0o644)?;
        self.interno.lock().unwrap().tls.cap_pub = Some(path);
        Ok(())
    }

    /// Pass the signed difference between coordinator and local time to stages
    /// when their assignments carry an expiry to verify.
    pub fn set_clock_offset(&self, clock_offset_s: i64) {
        self.interno.lock().unwrap().clock_offset_s = clock_offset_s;
    }

    pub fn clock_offset(&self) -> i64 {
        self.interno.lock().unwrap().clock_offset_s
    }

    /// Write the signed assignment bytes to the path the stage reloads.
    fn write_capability(&self, a: &StageAssign) -> Result<PathBuf> {
        let path = self.cfg.tls_dir.join(format!("capability-{}-{}.bin", nome_file(&a.chain_id), a.assign_epoch));
        write_atomic(&path, &a.capability, 0o600)?;
        Ok(path)
    }

    /// An unsigned assignment has no TLS arguments. A signed assignment
    /// requires its trust anchor, node identity and verification key. Missing
    /// material fails the assignment before any process starts.
    fn tls_arguments(&self, a: &StageAssign) -> Result<Vec<String>> {
        if a.capability.is_empty() {
            return Ok(vec![]);
        }
        let f = self.interno.lock().unwrap().tls.clone();
        let ca = self.cfg.ca_path.clone().context("signed chains require the coordinator CA (--ca)")?;
        let cert = f.cert.context("signed chains require a node certificate")?;
        let key = f.key.context("signed chains require a node key")?;
        let cap_pub = f.cap_pub.context("signed chains require the signing key from Welcome")?;
        let cap = self.write_capability(a)?;
        let scarto = self.clock_offset();
        let p = |x: PathBuf| x.to_string_lossy().into_owned();
        Ok(vec![
            "--tls-cert".into(),
            p(cert),
            "--tls-key".into(),
            p(key),
            "--tls-ca".into(),
            p(ca),
            "--capability".into(),
            p(cap),
            "--capability-pub".into(),
            p(cap_pub),
            "--clock-offset-s".into(),
            scarto.to_string(),
        ])
    }

    pub fn queue_assignment(&self, a: StageAssign) {
        let _ = self.cmd.send(Cmd::Assegna(a, None));
    }

    /// Reserve the single engine before returning to the session's offer loop.
    pub fn queue_switch(&self, a: StageAssign, sup: &Arc<Supervisor>, allow_new: bool) -> std::result::Result<(), StageStatus> {
        let key = (a.chain_id.clone(), a.assign_epoch);
        let mut i = self.interno.lock().unwrap();
        if i.closed.contains(&key) {
            return Ok(());
        }
        if let Err(e) = self.controlla(&a) {
            i.closed.insert(key);
            return Err(status_message(&a, State::Failed, &format!("{e:#}"), 0, 0.0));
        }
        if let Some(s) = &i.switching {
            if s.matches(&a) {
                if !s.closing {
                    let _ = self.cmd.send(Cmd::Renew(a));
                }
                return Ok(());
            }
        }
        let parked = if allow_new && i.switching.is_none() { sup.try_park() } else { None };
        let Some(parked) = parked else {
            i.closed.insert(key);
            return Err(status_message(&a, State::Stopped, "busy", 0, 0.0));
        };
        i.switching = Some(EngineSwitch { chain_id: key.0, epoch: key.1, parked: Some(parked), closing: false });
        let _ = self.cmd.send(Cmd::Assegna(a, Some(sup.clone())));
        Ok(())
    }

    pub fn serving_chain(&self) -> Option<String> {
        self.interno.lock().unwrap().switching.as_ref().filter(|s| !s.closing).map(|s| s.chain_id.clone())
    }

    pub fn switch_state(&self) -> Option<EngineState> {
        self.interno.lock().unwrap().switching.as_ref().map(|s| {
            if s.closing { EngineState::Recovering } else { EngineState::Parked }
        })
    }

    /// Keep routing and availability on the stage until its terminal report is handled.
    pub fn stage_status_current(&self, status: &StageStatus) -> bool {
        let mut i = self.interno.lock().unwrap();
        let Some(s) = i.switching.as_ref().filter(|s| s.chain_id == status.chain_id && s.epoch == status.assign_epoch) else {
            return false;
        };
        let terminal = matches!(status.state(), State::Stopped | State::Failed);
        let current = terminal || !s.closing;
        if terminal && s.parked.is_none() {
            i.switching = None;
        }
        current
    }

    /// Close admission before enqueueing the stop. A following offer may
    /// already be in flight on the control stream.
    pub fn queue_release(&self, r: StageRelease) -> bool {
        let matched = self.close_assignment(&r.chain_id, r.assign_epoch);
        let _ = self.cmd.send(Cmd::Rilascia(r));
        matched
    }

    fn close_assignment(&self, chain_id: &str, epoch: u64) -> bool {
        let mut i = self.interno.lock().unwrap();
        i.closed.insert((chain_id.to_owned(), epoch));
        let mut matched = false;
        if let Some(s) = i.switching.as_mut().filter(|s| s.chain_id == chain_id && s.epoch <= epoch) {
            s.close();
            matched = true;
        }
        if let Some(a) = i.attivo.as_mut().filter(|a| a.chain_id == chain_id && a.epoch <= epoch) {
            a.closing = true;
            matched = true;
        }
        matched
    }

    pub fn release_current(&self, reason: &str) {
        let key = {
            let i = self.interno.lock().unwrap();
            i.switching.as_ref().filter(|s| !s.closing).map(|s| (s.chain_id.clone(), s.epoch))
                .or_else(|| i.attivo.as_ref().filter(|a| !a.closing).map(|a| (a.chain_id.clone(), a.epoch)))
        };
        if let Some((chain_id, assign_epoch)) = key {
            self.queue_release(StageRelease { chain_id, assign_epoch, reason: reason.into() });
        }
    }

    /// On session close, terminate the child and end stage reporting.
    pub async fn stop_all(&self) {
        self.release_current("session closed");
        let (tx, rx) = oneshot::channel();
        if self.cmd.send(Cmd::FermaTutto(tx)).is_ok() {
            let _ = tokio::time::timeout(self.cfg.stop_grace + self.cfg.kill_wait + Duration::from_secs(5), rx).await;
        }
    }

    /// Return the service connection only when this node's head is ready.
    pub fn head(&self) -> Option<Arc<HeadConnection>> {
        let i = self.interno.lock().unwrap();
        if i.attivo.as_ref().is_some_and(|a| a.closing)
            || i.switching.as_ref().is_some_and(|s| !s.can_start_stage()) {
            return None;
        }
        i.link.clone().filter(|l| l.aperto())
    }

    /// Active chain capacity for concurrent chat leases, with a minimum of one.
    pub fn slots(&self) -> usize {
        self.interno.lock().unwrap().attivo.as_ref().map_or(1, |a| a.slots.max(1) as usize)
    }

    /// Publish RUNNING or READY when activity crosses zero. Ignore transitions
    /// until the child is ready to serve.
    pub fn attivita(&self, attive: bool) {
        let (tx, st) = {
            let mut i = self.interno.lock().unwrap();
            let Some(a) = i.attivo.as_mut() else { return };
            if a.closing || !matches!(a.last.state(), State::Ready | State::Running) {
                return;
            }
            let new = if attive { State::Running } else { State::Ready };
            if a.last.state() == new {
                return;
            }
            a.last.state = new as i32;
            (i.tx.clone(), i.attivo.as_ref().unwrap().last.clone())
        };
        if let Some(tx) = tx {
            let _ = tx.try_send(st);
        }
    }

    /// Read the active stage state for lifecycle assertions.
    #[cfg(test)]
    pub fn ultimo_stato(&self) -> Option<StageStatus> {
        self.interno.lock().unwrap().attivo.as_ref().map(|a| a.last.clone())
    }

    async fn report(&self, s: StageStatus) {
        let tx = {
            let mut i = self.interno.lock().unwrap();
            if let Some(a) = i.attivo.as_mut().filter(|a| a.chain_id == s.chain_id && a.epoch == s.assign_epoch) {
                a.last = s.clone();
            }
            i.tx.clone()
        };
        info!(chain = %s.chain_id, epoch = s.assign_epoch, state = ?s.state(), error = %s.error, "stage status");
        if let Some(tx) = tx {
            let _ = tx.send(s).await;
        }
    }

    async fn assign(self: &Arc<Self>, a: StageAssign, sup: Option<Arc<Supervisor>>) {
        let status = self.start_assignment(&a, sup.as_ref()).await;
        if let Some(status) = status {
            self.finish_assignment(&a);
            self.report(status).await;
        }
    }

    /// A failed launch returns a terminal status. A running child owns completion.
    async fn start_assignment(self: &Arc<Self>, a: &StageAssign, sup: Option<&Arc<Supervisor>>) -> Option<StageStatus> {
        let key = (a.chain_id.clone(), a.assign_epoch);
        if sup.is_none() && self.interno.lock().unwrap().closed.contains(&key) {
            return None;
        }
        let failed = |e: anyhow::Error| status_message(a, State::Failed, &format!("{e:#}"), 0, 0.0);
        let stopped = || status_message(a, State::Stopped, "assignment cancelled", 0, 0.0);
        // Validate the assignment before allocating child resources.
        let role = match self.controlla(a) {
            Ok(r) => r,
            Err(e) => return Some(failed(e)),
        };
        let gpu = if let Some(sup) = sup {
            if let Err(e) = sup.stop_child().await {
                return Some(failed(e.context("could not stop llama-server before starting a stage")));
            }
            let i = self.interno.lock().unwrap();
            let s = i.switching.as_ref().unwrap();
            if !s.can_start_stage() {
                return Some(stopped());
            }
            s.parked.as_ref().unwrap().gpu.clone()
        } else { None };
        // An agent owns one active stage, so classify duplicates before launching another.
        let previous = {
            let i = self.interno.lock().unwrap();
            i.attivo.as_ref().map(|p| (p.chain_id.clone(), p.epoch, p.last.clone()))
        };
        if let Some((chain, epoch, last)) = previous {
            if chain != a.chain_id {
                let m = format!("node already hosts a stage for chain {chain}");
                return Some(status_message(a, State::Failed, &m, 0, 0.0));
            }
            if a.assign_epoch < epoch {
                warn!(chain = %chain, new = a.assign_epoch, current = epoch, "ignored stale assignment");
                return None;
            }
            if a.assign_epoch == epoch {
                self.renew(a, last).await;
                return None;
            }
            self.stop_active("stage reassigned").await;
        }
        // Use the translated addresses so the child reaches bridged peers through loopback.
        let (per_stadio, guard) = match self.connect_bridge(a, role).await {
            Ok(x) => x,
            Err(e) => return Some(failed(e)),
        };
        let args = match arguments(&self.cfg, &per_stadio).and_then(|mut v| {
            v.extend(self.tls_arguments(a)?);
            Ok(v)
        }) {
            Ok(v) => v,
            Err(e) => return Some(failed(e)),
        };
        let (ferma_tx, ferma_rx) = oneshot::channel::<String>();
        let (fin_tx, fin_rx) = watch::channel(false);
        {
            let mut i = self.interno.lock().unwrap();
            if i.closed.contains(&key) || i.switching.as_ref().is_some_and(|s| !s.can_start_stage()) {
                return Some(stopped());
            }
            i.attivo = Some(Attivo {
                chain_id: a.chain_id.clone(),
                slots: a.slots,
                epoch: a.assign_epoch,
                ferma: Some(ferma_tx),
                finito: fin_rx,
                last: status_message(a, State::Loading, "", 0, 0.0),
                closing: false,
            });
        }
        let s2 = self.clone();
        let a = a.clone();
        tokio::spawn(async move {
            let status = s2.vita(&a, role, args, gpu.as_deref(), ferma_rx).await;
            drop(guard);
            s2.pulisci(&a);
            s2.finish_assignment(&a);
            s2.report(status).await;
            let _ = fin_tx.send(true);
        });
        None
    }

    fn finish_assignment(&self, a: &StageAssign) {
        let mut i = self.interno.lock().unwrap();
        i.closed.insert((a.chain_id.clone(), a.assign_epoch));
        if let Some(s) = i.switching.as_mut().filter(|s| s.matches(a)) {
            if let Some(parked) = s.parked.take() {
                parked.unpark();
            }
        }
    }

    async fn renew_switch(&self, a: &StageAssign) {
        let last = {
            let i = self.interno.lock().unwrap();
            i.attivo.as_ref().filter(|s| s.chain_id == a.chain_id && s.epoch == a.assign_epoch && !s.closing)
                .map(|s| s.last.clone())
        };
        // A renewal queued during teardown must never resurrect the stage.
        if let Some(last) = last {
            self.renew(a, last).await;
        }
    }

    async fn renew(&self, a: &StageAssign, last: StageStatus) {
        // The stage reloads the capability file while continuing to serve.
        if !a.capability.is_empty() {
            if let Err(e) = self.write_capability(a) {
                warn!(chain = %a.chain_id, "could not write the renewed capability: {e:#}");
            }
        }
        self.report(last).await;
    }

    fn controlla(&self, a: &StageAssign) -> Result<Role> {
        let role = validate(a)?;
        if a.capability.is_empty() && !self.cfg.in_chiaro {
            bail!("Rejecting StageAssign without a signed capability; stage links must use TLS");
        }
        let impronta = a.model.as_ref().map(|m| m.weights_sha256.as_slice()).unwrap_or_default();
        if impronta != self.cfg.weights_sha256.as_slice() {
            bail!("local GGUF fingerprint does not match the assignment");
        }
        if !self.cfg.stage_bin.exists() {
            bail!("publicompute-stage not found: {}", self.cfg.stage_bin.display());
        }
        if !self.cfg.model_path.exists() {
            bail!("GGUF not found: {}", self.cfg.model_path.display());
        }
        Ok(role)
    }

    async fn release(&self, r: StageRelease) {
        let combacia = {
            let i = self.interno.lock().unwrap();
            i.attivo.as_ref().is_some_and(|a| a.chain_id == r.chain_id && r.assign_epoch >= a.epoch)
        };
        if combacia {
            let reason = if r.reason.is_empty() { "released by the coordinator" } else { r.reason.as_str() };
            self.stop_active(reason).await;
        } else {
            info!(chain = %r.chain_id, epoch = r.assign_epoch, "ignoring StageRelease without a matching stage");
        }
    }

    /// Stop the active stage and wait for its task to report completion.
    async fn stop_active(&self, reason: &str) {
        let (ferma, mut finito) = {
            let mut i = self.interno.lock().unwrap();
            match i.attivo.as_mut() {
                Some(a) => (a.ferma.take(), a.finito.clone()),
                None => return,
            }
        };
        if let Some(f) = ferma {
            let _ = f.send(reason.to_string());
        }
        let _ = finito.wait_for(|f| *f).await;
    }

    /// Drive one child from startup through its final lifecycle update.
    async fn vita(&self, a: &StageAssign, role: Role, args: Vec<String>, gpu: Option<&str>, mut ferma: oneshot::Receiver<String>) -> StageStatus {
        let t0 = Instant::now();
        let porta = if role == Role::Head { 0 } else { porta_di(&a.listen_addr, "listen_addr").unwrap_or(0) as u32 };
        self.report(status_message(a, State::Loading, "", 0, 0.0)).await;
        let log_path = self.cfg.log_dir.join(format!("stage-{}-{}.log", nome_file(&a.chain_id), a.assign_epoch));
        let started = tokio::select! {
            biased;
            reason = &mut ferma => {
                let reason = reason.unwrap_or_else(|_| "agent stopped".into());
                return status_message(a, State::Stopped, &reason, 0, 0.0);
            }
            result = start(&self.cfg, &args, &log_path, role == Role::Head, gpu) => result,
        };
        let (mut figlio, pronto_rx, ultime) = match started {
            Ok(x) => x,
            Err(e) => {
                return status_message(a, State::Failed, &format!("could not start publicompute-stage: {e:#}"), 0, 0.0);
            }
        };
        info!(chain = %a.chain_id, role = role.comando(), log = %log_path.display(), "stage started");
        let scadenza = tokio::time::sleep(self.cfg.load_timeout);
        tokio::pin!(scadenza);
        let esito: Outcome = {
            let pipes = figlio.child.stdout.take().zip(figlio.child.stdin.take());
            let attesa_pronto = async {
                match pipes {
                    Some((reader, writer)) => head_ready(reader, writer).await.map(Some),
                    None if role == Role::Head => bail!("head pipes unavailable"),
                    None => pronto_rx.await.context("stage stderr closed before readiness").map(|_| None),
                }
            };
            tokio::pin!(attesa_pronto);
            tokio::select! {
                result = &mut attesa_pronto => match result {
                    Ok(stream) => Outcome::Pronto(stream),
                    Err(error) => Outcome::Invalid(error),
                },
                st = figlio.child.wait() => Outcome::Uscito(st.map_err(|e| e.to_string())),
                m = &mut ferma => Outcome::Fermato(m.unwrap_or_else(|_| "agent stopped".into())),
                _ = &mut scadenza => Outcome::Scaduto,
            }
        };
        match esito {
            Outcome::Pronto(stream) => {
                let carico = t0.elapsed().as_secs_f64();
                let link = stream.map(|(reader, writer, buf)| HeadConnection::from_io(reader, writer, buf));
                if let Some(l) = &link {
                    self.interno.lock().unwrap().link = Some(l.clone());
                }
                self.report(status_message(a, State::Ready, "", porta, carico)).await;
                // Once ready, watch the process and stop request. For a head, loss of
                // the service connection also ends the assignment.
                let mut caduto = link.as_ref().map(|l| l.osserva_chiusura());
                let done = async {
                    match caduto.as_mut() {
                        Some(c) => {
                            let _ = c.wait_for(|x| *x).await;
                        }
                        None => std::future::pending::<()>().await,
                    }
                };
                tokio::select! {
                    st = figlio.child.wait() => {
                        if let Some(link) = &link {
                            link.close();
                        }
                        self.terminate(a, &mut figlio, "stage launcher exited").await;
                        let m = descrivi_uscita(st.map_err(|e| e.to_string()), &ultime);
                        status_message(a, State::Failed, &m, porta, carico)
                    }
                    m = &mut ferma => {
                        let reason = m.unwrap_or_else(|_| "agent stopped".into());
                        if let Some(link) = &link {
                            link.close();
                        }
                        self.terminate(a, &mut figlio, &reason).await;
                        status_message(a, State::Stopped, &reason, porta, carico)
                    }
                    _ = done => {
                        let m = "lost the connection to the head service".to_string();
                        self.terminate(a, &mut figlio, &m).await;
                        status_message(a, State::Failed, &m, porta, carico)
                    }
                }
            }
            Outcome::Invalid(error) => {
                self.terminate(a, &mut figlio, "invalid stage readiness signal").await;
                let message = format!("stage readiness failed: {error:#}; {}", ultime_righe(&ultime));
                status_message(a, State::Failed, &message, 0, t0.elapsed().as_secs_f64())
            }
            Outcome::Uscito(st) => {
                self.terminate(a, &mut figlio, "stage launcher exited during load").await;
                let m = descrivi_uscita(st, &ultime);
                status_message(a, State::Failed, &format!("stage exited during load: {m}"), 0, t0.elapsed().as_secs_f64())
            }
            Outcome::Fermato(reason) => {
                self.terminate(a, &mut figlio, &reason).await;
                status_message(a, State::Stopped, &reason, 0, t0.elapsed().as_secs_f64())
            }
            Outcome::Scaduto => {
                let m = format!("loading exceeded {} s: {}", self.cfg.load_timeout.as_secs(), ultime_righe(&ultime));
                self.terminate(a, &mut figlio, "stage load timed out").await;
                status_message(a, State::Failed, &m, 0, t0.elapsed().as_secs_f64())
            }
        }
    }

    fn pulisci(&self, a: &StageAssign) {
        let mut i = self.interno.lock().unwrap();
        if i.attivo.as_ref().is_some_and(|x| x.chain_id == a.chain_id && x.epoch == a.assign_epoch) {
            i.attivo = None;
            i.link = None;
        }
    }

    /// Escalate to SIGKILL after the grace period and reap before reporting exit.
    async fn terminate(&self, a: &StageAssign, figlio: &mut managed::Process, reason: &str) {
        info!(reason, "stopping the stage");
        if let Err(error) = managed::stop(figlio, self.cfg.stop_grace, self.cfg.kill_wait, "publicompute-stage").await {
            self.close_assignment(&a.chain_id, a.assign_epoch);
            let switching = self.interno.lock().unwrap().switching.as_ref().is_some_and(|s| s.matches(a));
            warn!(%error, "could not confirm stage group exit; waiting for it to empty");
            if switching {
                self.report(status_message(a, State::Failed, &format!("{error:#}"), 0, 0.0)).await;
            }
            // Keep the switch guard until the entire group has exited.
            while figlio.wait_for_exit("publicompute-stage").await.is_err() {
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        }
    }
}

enum Outcome {
    Pronto(Option<(ChildStdout, ChildStdin, BytesMut)>),
    Invalid(anyhow::Error),
    Uscito(std::result::Result<std::process::ExitStatus, String>),
    Fermato(String),
    Scaduto,
}

async fn head_ready(mut reader: ChildStdout, writer: ChildStdin) -> Result<(ChildStdout, ChildStdin, BytesMut)> {
    let mut buf = BytesMut::with_capacity(4096);
    loop {
        if let Some(message) = wire::try_decode_frame::<pb::FromHead>(&mut buf)? {
            if matches!(message.body, Some(pb::from_head::Body::Ready(_))) {
                return Ok((reader, writer, buf));
            }
            bail!("expected readiness frame from the head");
        }
        if reader.read_buf(&mut buf).await? == 0 {
            bail!("head pipe closed before readiness");
        }
    }
}

fn descrivi_uscita(st: std::result::Result<std::process::ExitStatus, String>, ultime: &Arc<Mutex<VecDeque<String>>>) -> String {
    let state = match st {
        Ok(s) => format!("exited with {s}"),
        Err(e) => format!("could not wait for the process: {e}"),
    };
    format!("{state}; last log lines: {}", ultime_righe(ultime))
}

fn ultime_righe(ultime: &Arc<Mutex<VecDeque<String>>>) -> String {
    let v = ultime.lock().unwrap();
    if v.is_empty() {
        return "(no output)".into();
    }
    v.iter().rev().take(5).rev().cloned().collect::<Vec<_>>().join(" | ")
}

fn nome_file(s: &str) -> String {
    s.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' }).collect()
}

/// Encode a DER object as PEM with lines of at most 64 Base64 characters.
fn pem(etichetta: &str, der: &[u8]) -> String {
    const A: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut b64 = String::with_capacity(der.len() * 4 / 3 + 4);
    for c in der.chunks(3) {
        let n = (u32::from(c[0]) << 16) | (u32::from(*c.get(1).unwrap_or(&0)) << 8) | u32::from(*c.get(2).unwrap_or(&0));
        for i in 0..4 {
            if i <= c.len() {
                b64.push(A[((n >> (18 - 6 * i)) & 63) as usize] as char);
            } else {
                b64.push('=');
            }
        }
    }
    let mut o = format!("-----BEGIN {etichetta}-----\n");
    for r in b64.as_bytes().chunks(64) {
        o.push_str(std::str::from_utf8(r).unwrap());
        o.push('\n');
    }
    o.push_str(&format!("-----END {etichetta}-----\n"));
    o
}

/// Finish writing the temporary file before renaming it so readers always
/// see a complete credential or capability.
fn write_atomic(path: &Path, dati: &[u8], _modo: u32) -> Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
    }
    let tmp = path.with_extension("tmp");
    {
        use std::io::Write;
        #[cfg(unix)]
        let mut f = {
            use std::os::unix::fs::OpenOptionsExt;
            std::fs::OpenOptions::new().write(true).create(true).truncate(true).mode(_modo).open(&tmp)
        }
        .with_context(|| format!("write {}", tmp.display()))?;
        #[cfg(not(unix))]
        let mut f = std::fs::File::create(&tmp).with_context(|| format!("write {}", tmp.display()))?;
        // Set the owner-only Windows ACL before writing private material.
        #[cfg(windows)]
        if _modo & 0o077 == 0 {
            crate::identity::acl::riserva_all_utente(&tmp)?;
        }
        f.write_all(dati)?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, path).with_context(|| format!("rename {}", tmp.display()))?;
    Ok(())
}

#[cfg(test)]
pub(crate) fn porta_libera() -> Result<u16> {
    let l = std::net::TcpListener::bind("127.0.0.1:0")?;
    Ok(l.local_addr()?.port())
}

/// Spawn the stage, retain recent stderr for diagnostics and watch for its readiness line.
async fn start(cfg: &StagesConfig, args: &[String], log_path: &Path, use_pipes: bool, gpu: Option<&str>) -> Result<(managed::Process, oneshot::Receiver<()>, Arc<Mutex<VecDeque<String>>>)> {
    tokio::fs::create_dir_all(&cfg.log_dir).await.with_context(|| format!("create {}", cfg.log_dir.display()))?;
    let mut log = tokio::fs::OpenOptions::new().create(true).append(true).open(log_path).await.with_context(|| format!("open {}", log_path.display()))?;
    let mut cmd = Command::new(&cfg.stage_bin);
    cmd.args(args).envs(cfg.env.iter().cloned()).stderr(Stdio::piped());
    if let Some(uuid) = gpu {
        cmd.env("CUDA_VISIBLE_DEVICES", uuid);
    }
    if use_pipes {
        cmd.stdin(Stdio::piped()).stdout(Stdio::piped());
    } else {
        cmd.stdin(Stdio::null()).stdout(Stdio::null());
    }
    // Make the stage exit with its parent so an orphan cannot hold the GPU.
    #[cfg(target_os = "linux")]
    unsafe {
        cmd.pre_exec(|| {
            libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL);
            Ok(())
        });
    }
    let mut figlio = managed::Process::spawn(cmd).with_context(|| format!("start {}", cfg.stage_bin.display()))?;
    let stderr = figlio.child.stderr.take().expect("stage stderr was piped");
    let (pronto_tx, pronto_rx) = oneshot::channel();
    let ultime: Arc<Mutex<VecDeque<String>>> = Arc::new(Mutex::new(VecDeque::new()));
    let u2 = ultime.clone();
    tokio::spawn(async move {
        let mut pronto = Some(pronto_tx);
        let mut rows = BufReader::new(stderr).lines();
        while let Ok(Some(r)) = rows.next_line().await {
            let _ = log.write_all(format!("{r}\n").as_bytes()).await;
            if r.starts_with("[stage]") && r.contains(" ready on ") {
                if let Some(p) = pronto.take() {
                    let _ = p.send(());
                }
            }
            let mut v = u2.lock().unwrap();
            if v.len() >= 20 {
                v.pop_front();
            }
            v.push_back(r);
        }
    });
    Ok((figlio, pronto_rx, ultime))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use publicompute_proto::common::ModelRef;

    /// Use the fixture binary built alongside the tests.
    pub fn fake_stage() -> PathBuf {
        let exe = std::env::current_exe().unwrap();
        let p = exe.parent().unwrap().parent().unwrap().join("fake-stage");
        assert!(p.exists(), "missing {} (cargo test builds package binaries)", p.display());
        p
    }

    pub fn cfg_test(dir: &Path) -> StagesConfig {
        let model = dir.join("finto.gguf");
        std::fs::write(&model, b"gguf finto").unwrap();
        StagesConfig {
            stage_bin: fake_stage(),
            model_path: model,
            weights_sha256: vec![7; 32],
            log_dir: dir.join("log"),
            threads: 2,
            ngl: 0,
            extra_args: vec![],
            env: vec![],
            load_timeout: Duration::from_secs(20),
            stop_grace: Duration::from_millis(600),
            kill_wait: Duration::from_secs(3),
            ca_path: None,
            tls_dir: dir.join("tls"),
            in_chiaro: true,
        }
    }

    pub fn assignment(chain: &str, epoch: u64, index: u32, n: u32, porta: u16) -> StageAssign {
        StageAssign {
            chain_id: chain.into(),
            stage_index: index,
            n_stages: n,
            model: Some(ModelRef { model_id: "m".into(), quantization: "q".into(), weights_sha256: vec![7; 32] }),
            layer_start: (index - 1) * 10,
            layer_end: if index == n { -1 } else { (index * 10) as i32 },
            slots: 4,
            ctx_per_slot: 512,
            groups: 2,
            n_batch: 64,
            state_format: "fp16".into(),
            listen_addr: if index == 1 { String::new() } else { format!("127.0.0.1:{porta}") },
            next_addr: if index == n { String::new() } else { "127.0.0.1:9".into() },
            head_token_addr: "127.0.0.1:19999".into(),
            capability: vec![],
            assign_epoch: epoch,
            ..Default::default()
        }
    }

    pub async fn prossimo(rx: &mut mpsc::Receiver<StageStatus>) -> StageStatus {
        tokio::time::timeout(Duration::from_secs(20), rx.recv()).await.expect("expected stage status within 20 s").expect("channel closed")
    }

    pub fn tempdir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("publicompute-stadi-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn arguments_for_role() {
        let d = tempdir("argomenti");
        let cfg = StagesConfig { extra_args: vec!["--log-top".into(), "x".into(), "--ingress-max-mib".into(), "96".into()], ..cfg_test(&d) };
        let head = arguments(&cfg, &assignment("c", 1, 1, 3, 0)).unwrap();
        let s = head.join(" ");
        assert_eq!(head[0], "head");
        for expected in [
            "--layer-start 0",
            "--layer-end 10",
            "--slots 4",
            "--ctx-per-slot 512",
            "--groups 2",
            "--n-batch 64",
            "--state-format fp16",
            "--next 127.0.0.1:9",
            "--listen-token 127.0.0.1:19999",
            "--service-pipe",
            "--log-top x",
        ] {
            assert!(s.contains(expected), "missing \"{expected}\" in: {s}");
        }
        assert!(!s.contains("--index"), "head has an unexpected --index: {s}");
        assert!(!s.contains("--block"), "head should calculate the default block: {s}");
        let medio = arguments(&cfg, &assignment("c", 1, 2, 3, 7001)).unwrap().join(" ");
        assert!(medio.starts_with("middle ") && medio.contains("--listen 127.0.0.1:7001") && medio.contains("--next 127.0.0.1:9") && medio.contains("--index 2"), "{medio}");
        assert!(medio.contains("--ingress-max-mib 96"));
        let tail = arguments(&cfg, &assignment("c", 1, 3, 3, 7002)).unwrap().join(" ");
        assert!(tail.starts_with("tail ") && tail.contains("--head-token 127.0.0.1:19999") && tail.contains("--index 3") && tail.contains("--layer-end -1"), "{tail}");
        assert!(!tail.contains("--state-format"));
        assert!(arguments(&cfg, &assignment("c", 1, 1, 1, 0)).is_err(), "reject a single-stage chain");
        assert!(arguments(&cfg, &assignment("c", 1, 4, 3, 1)).is_err(), "index exceeds n_stages");
        let mut a = assignment("c", 1, 3, 3, 7002);
        a.state_format = "int8".into();
        assert!(arguments(&cfg, &a).is_err());
        let mut a = assignment("c", 1, 2, 3, 0);
        a.listen_addr = "127.0.0.1:0".into();
        assert!(arguments(&cfg, &a).is_err(), "port 0");
    }

    fn righe_log(d: &Path, chain: &str, epoch: u64) -> String {
        std::fs::read_to_string(d.join("log").join(format!("stage-{chain}-{epoch}.log"))).unwrap_or_default()
    }

    #[test]
    fn pem_round_trip() {
        use rustls::pki_types::{pem::PemObject, CertificateDer};
        for n in [0usize, 1, 2, 3, 4, 5, 63, 64, 65, 300, 301, 302] {
            let der: Vec<u8> = (0..n).map(|i| (i * 7 + 3) as u8).collect();
            let p = pem("CERTIFICATE", &der);
            let v: Vec<CertificateDer<'static>> = CertificateDer::pem_slice_iter(p.as_bytes()).collect::<Result<_, _>>().unwrap();
            if n == 0 {
                continue;
            }
            assert_eq!(v.len(), 1);
            assert_eq!(v[0].as_ref(), der.as_slice(), "n = {n}");
        }
    }

    /// An unsigned assignment is rejected before spawning a stage unless
    /// plaintext operation is explicitly enabled.
    #[tokio::test]
    async fn unsigned_assignment_is_rejected() {
        let d = tempdir("plaintext-assignment");
        let stages = Stages::new(StagesConfig { in_chiaro: false, ..cfg_test(&d) });
        let (tx, mut rx) = mpsc::channel(16);
        stages.connect_peer(tx);
        let porta = porta_libera().unwrap();
        stages.queue_assignment(assignment("senza-cap", 1, 3, 3, porta));
        let s = prossimo(&mut rx).await;
        assert_eq!(s.state(), State::Failed, "{s:?}");
        assert!(s.error.contains("capability"), "expected a capability error: {}", s.error);
    }

    /// A signed assignment passes all credential paths to the child. An allowed unsigned assignment passes none.
    #[tokio::test]
    async fn capability_enables_tls_and_unsigned_assignment_uses_plaintext() {
        let d = tempdir("tls-args");
        let cfg = StagesConfig { ca_path: Some(d.join("ca.pem")), ..cfg_test(&d) };
        let stages = Stages::new(cfg.clone());
        let (tx, mut rx) = mpsc::channel(16);
        stages.connect_peer(tx);
        // The signed capability determines whether configured TLS inputs are used.
        stages.set_tls_files(TlsFiles { cert: Some(d.join("c.pem")), key: Some(d.join("k.pem")), cap_pub: Some(d.join("p")) });
        let porta = porta_libera().unwrap();
        stages.queue_assignment(assignment("chiaro", 1, 3, 3, porta));
        assert_eq!(prossimo(&mut rx).await.state(), State::Loading);
        assert_eq!(prossimo(&mut rx).await.state(), State::Ready);
        let log = righe_log(&d, "chiaro", 1);
        assert!(log.contains("[fake-stage] tail args:") && !log.contains("--tls-cert") && !log.contains("--capability"), "{log}");
        stages.queue_release(StageRelease { chain_id: "chiaro".into(), assign_epoch: 1, reason: "fine".into() });
        assert_eq!(prossimo(&mut rx).await.state(), State::Stopped);

        // Check that the capability bytes survive the write without re-encoding.
        let porta = porta_libera().unwrap();
        let mut a = assignment("firmata", 2, 3, 3, porta);
        a.capability = vec![1, 2, 3, 4, 5];
        stages.queue_assignment(a.clone());
        assert_eq!(prossimo(&mut rx).await.state(), State::Loading);
        assert_eq!(prossimo(&mut rx).await.state(), State::Ready);
        let log = righe_log(&d, "firmata", 2);
        let cap = d.join("tls/capability-firmata-2.bin");
        for expected in [
            format!("--tls-cert {}", d.join("c.pem").display()),
            format!("--tls-key {}", d.join("k.pem").display()),
            format!("--tls-ca {}", d.join("ca.pem").display()),
            format!("--capability {}", cap.display()),
            format!("--capability-pub {}", d.join("p").display()),
            "--index 3".to_string(),
        ] {
            assert!(log.contains(&expected), "missing \"{expected}\" in: {log}");
        }
        assert_eq!(std::fs::read(&cap).unwrap(), vec![1, 2, 3, 4, 5]);
        // A renewal at the same epoch replaces the file and reuses the running process.
        a.capability = vec![9, 9, 9];
        stages.queue_assignment(a);
        assert_eq!(prossimo(&mut rx).await.state(), State::Ready);
        assert_eq!(std::fs::read(&cap).unwrap(), vec![9, 9, 9]);
        assert_eq!(righe_log(&d, "firmata", 2).matches("tail args:").count(), 1, "renewing the capability restarted the stage");
        // Without server time there is no clock correction.
        assert!(log.contains("--clock-offset-s 0"), "{log}");
        stages.stop_all().await;
    }

    /// Pass the handshake's clock correction only to signed assignments.
    #[tokio::test]
    async fn clock_offset_passed_only_with_capability() {
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() as i64;
        let t = prost_types::Timestamp { seconds: now + 120, nanos: 0 };
        assert_eq!(clock_offset_from_welcome(Some(&t), now), 120);
        assert_eq!(clock_offset_from_welcome(None, now), 0);
        let d = tempdir("tls-scarto");
        let cfg = StagesConfig { ca_path: Some(d.join("ca.pem")), ..cfg_test(&d) };
        let stages = Stages::new(cfg);
        let (tx, mut rx) = mpsc::channel(16);
        stages.connect_peer(tx);
        stages.set_tls_files(TlsFiles { cert: Some(d.join("c.pem")), key: Some(d.join("k.pem")), cap_pub: Some(d.join("p")) });
        stages.set_clock_offset(clock_offset_from_welcome(Some(&t), now));
        stages.queue_assignment(assignment("unsigned", 1, 3, 3, porta_libera().unwrap()));
        assert_eq!(prossimo(&mut rx).await.state(), State::Loading);
        assert_eq!(prossimo(&mut rx).await.state(), State::Ready);
        assert!(!righe_log(&d, "unsigned", 1).contains("--clock-offset-s"));
        stages.queue_release(StageRelease { chain_id: "unsigned".into(), assign_epoch: 1, reason: "fine".into() });
        assert_eq!(prossimo(&mut rx).await.state(), State::Stopped);
        let mut a = assignment("con", 1, 3, 3, porta_libera().unwrap());
        a.capability = vec![1, 2, 3];
        stages.queue_assignment(a);
        assert_eq!(prossimo(&mut rx).await.state(), State::Loading);
        assert_eq!(prossimo(&mut rx).await.state(), State::Ready);
        assert!(righe_log(&d, "con", 1).contains("--clock-offset-s 120"));
        stages.stop_all().await;
    }

    /// Missing TLS material must fail a signed assignment before it starts.
    #[tokio::test]
    async fn signed_chain_fails_without_tls_credentials() {
        let d = tempdir("tls-manca");
        let stages = Stages::new(StagesConfig { ca_path: Some(d.join("ca.pem")), ..cfg_test(&d) });
        let (tx, mut rx) = mpsc::channel(16);
        stages.connect_peer(tx);
        let mut a = assignment("firmata", 1, 3, 3, porta_libera().unwrap());
        a.capability = vec![1];
        stages.queue_assignment(a.clone());
        let s = prossimo(&mut rx).await;
        assert_eq!(s.state(), State::Failed, "{s:?}");
        assert!(s.error.contains("node certificate"), "{}", s.error);
        assert!(righe_log(&d, "firmata", 1).is_empty(), "no stage process should start");
        let d2 = tempdir("tls-manca-ca");
        let st2 = Stages::new(cfg_test(&d2));
        let (tx2, mut rx2) = mpsc::channel(16);
        st2.connect_peer(tx2);
        st2.queue_assignment(a);
        let s = prossimo(&mut rx2).await;
        assert_eq!(s.state(), State::Failed);
        assert!(s.error.contains("coordinator CA"), "{}", s.error);
    }

    #[tokio::test]
    async fn tail_reports_loading_ready_and_stopped_on_release() {
        let d = tempdir("tail");
        let stages = Stages::new(cfg_test(&d));
        let (tx, mut rx) = mpsc::channel(16);
        stages.connect_peer(tx);
        let porta = porta_libera().unwrap();
        stages.queue_assignment(assignment("catena-1", 3, 3, 3, porta));
        let s = prossimo(&mut rx).await;
        assert_eq!((s.state(), s.assign_epoch, s.chain_id.as_str()), (State::Loading, 3, "catena-1"));
        let s = prossimo(&mut rx).await;
        assert_eq!(s.state(), State::Ready, "{s:?}");
        assert_eq!(s.port, porta as u32);
        assert!(s.load_seconds > 0.0);
        // A duplicate announces the existing stage without spawning another child.
        stages.queue_assignment(assignment("catena-1", 3, 3, 3, porta));
        assert_eq!(prossimo(&mut rx).await.state(), State::Ready);
        // A conflicting assignment must leave the active chain alone.
        stages.queue_assignment(assignment("altra", 1, 3, 3, porta_libera().unwrap()));
        let s = prossimo(&mut rx).await;
        assert_eq!((s.state(), s.chain_id.as_str()), (State::Failed, "altra"));
        assert!(s.error.contains("catena-1"), "{}", s.error);
        // A release must match the active epoch before it can stop the stage.
        stages.queue_release(StageRelease { chain_id: "catena-1".into(), assign_epoch: 2, reason: "vecchio".into() });
        stages.queue_release(StageRelease { chain_id: "catena-1".into(), assign_epoch: 3, reason: "chain dissolved".into() });
        let s = prossimo(&mut rx).await;
        assert_eq!(s.state(), State::Stopped, "{s:?}");
        assert!(s.error.contains("chain dissolved"));
        let log = std::fs::read_to_string(d.join("log/stage-catena-1-3.log")).unwrap();
        assert!(log.contains("ready on"), "{log}");
        // A later epoch can be assigned after the current stage exits.
        stages.queue_assignment(assignment("catena-1", 4, 3, 3, porta_libera().unwrap()));
        assert_eq!(prossimo(&mut rx).await.state(), State::Loading);
        assert_eq!(prossimo(&mut rx).await.state(), State::Ready);
        stages.stop_all().await;
        assert_eq!(prossimo(&mut rx).await.state(), State::Stopped);
    }

    #[tokio::test]
    async fn new_epoch_stops_previous_assignment() {
        let d = tempdir("riass");
        let stages = Stages::new(cfg_test(&d));
        let (tx, mut rx) = mpsc::channel(16);
        stages.connect_peer(tx);
        stages.queue_assignment(assignment("c", 1, 2, 3, porta_libera().unwrap()));
        assert_eq!(prossimo(&mut rx).await.state(), State::Loading);
        assert_eq!(prossimo(&mut rx).await.state(), State::Ready);
        stages.queue_assignment(assignment("c", 2, 2, 3, porta_libera().unwrap()));
        let s = prossimo(&mut rx).await;
        assert_eq!((s.state(), s.assign_epoch), (State::Stopped, 1));
        let s = prossimo(&mut rx).await;
        assert_eq!((s.state(), s.assign_epoch), (State::Loading, 2));
        assert_eq!(prossimo(&mut rx).await.state(), State::Ready);
        stages.stop_all().await;
    }

    #[tokio::test]
    async fn bad_fingerprint_and_stage_exit_report_failure() {
        let d = tempdir("failed");
        let stages = Stages::new(cfg_test(&d));
        let (tx, mut rx) = mpsc::channel(16);
        stages.connect_peer(tx);
        let mut a = assignment("c", 1, 3, 3, porta_libera().unwrap());
        a.model.as_mut().unwrap().weights_sha256 = vec![9; 32];
        stages.queue_assignment(a);
        let s = prossimo(&mut rx).await;
        assert_eq!(s.state(), State::Failed);
        assert!(s.error.contains("fingerprint"), "{}", s.error);
        let stadi2 = Stages::new(StagesConfig { stage_bin: d.join("non-esiste"), ..cfg_test(&d) });
        let (tx2, mut rx2) = mpsc::channel(16);
        stadi2.connect_peer(tx2);
        stadi2.queue_assignment(assignment("c", 1, 3, 3, porta_libera().unwrap()));
        let s = prossimo(&mut rx2).await;
        assert_eq!(s.state(), State::Failed);
        assert!(s.error.contains("not found"), "{}", s.error);
    }

    #[tokio::test]
    async fn running_stage_exit_reports_failure_without_restart() {
        let d = tempdir("muore");
        let cfg = StagesConfig { env: vec![("FAKE_STAGE_EXIT_AFTER_MS".into(), "1500".into())], ..cfg_test(&d) };
        let stages = Stages::new(cfg);
        let (tx, mut rx) = mpsc::channel(16);
        stages.connect_peer(tx);
        stages.queue_assignment(assignment("c", 1, 3, 3, porta_libera().unwrap()));
        assert_eq!(prossimo(&mut rx).await.state(), State::Loading);
        assert_eq!(prossimo(&mut rx).await.state(), State::Ready);
        let s = prossimo(&mut rx).await;
        assert_eq!(s.state(), State::Failed, "{s:?}");
        assert!(s.error.contains("exited") && s.error.contains("exiting early"), "{}", s.error);
        // A failed child remains stopped until a coordinator assignment.
        assert!(tokio::time::timeout(Duration::from_millis(1500), rx.recv()).await.is_err());
        assert!(stages.ultimo_stato().is_none(), "exited stage is still active");
    }

    #[tokio::test]
    async fn stage_ignoring_sigterm_is_killed() {
        let d = tempdir("kill");
        let stages = Stages::new(StagesConfig { env: vec![("FAKE_STAGE_IGNORE_TERM".into(), "1".into())], ..cfg_test(&d) });
        let (tx, mut rx) = mpsc::channel(16);
        stages.connect_peer(tx);
        stages.queue_assignment(assignment("c", 1, 3, 3, porta_libera().unwrap()));
        assert_eq!(prossimo(&mut rx).await.state(), State::Loading);
        assert_eq!(prossimo(&mut rx).await.state(), State::Ready);
        let t = Instant::now();
        stages.stop_all().await;
        assert!(t.elapsed() >= Duration::from_millis(500), "waited too long for SIGTERM: {:?}", t.elapsed());
        assert!(t.elapsed() < Duration::from_secs(5), "{:?}", t.elapsed());
        assert_eq!(prossimo(&mut rx).await.state(), State::Stopped);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn forced_stop_reaps_stage_before_returning() {
        let d = tempdir("kill-reap");
        let cfg = StagesConfig {
            env: vec![("FAKE_STAGE_IGNORE_TERM".into(), "1".into())],
            kill_wait: Duration::ZERO,
            ..cfg_test(&d)
        };
        let assign = assignment("c", 1, 3, 3, porta_libera().unwrap());
        let args = arguments(&cfg, &assign).unwrap();
        let (mut child, ready, _) = start(&cfg, &args, &cfg.log_dir.join("reap.log"), false, None).await.unwrap();
        let pid = child.child.id().unwrap();
        tokio::time::timeout(Duration::from_secs(5), ready).await.unwrap().unwrap();
        let stages = Stages::new(cfg);
        stages.terminate(&assign, &mut child, "test").await;
        assert!(child.child.id().is_none());
        assert_eq!(unsafe { libc::waitpid(pid as libc::pid_t, std::ptr::null_mut(), libc::WNOHANG) }, -1);
        assert_eq!(std::io::Error::last_os_error().raw_os_error(), Some(libc::ECHILD));
    }

    #[tokio::test]
    async fn head_is_ready_when_service_accepts_requests() {
        let d = tempdir("head");
        let stages = Stages::new(StagesConfig { env: vec![("FAKE_STAGE_START_DELAY_MS".into(), "700".into())], ..cfg_test(&d) });
        let (tx, mut rx) = mpsc::channel(16);
        stages.connect_peer(tx);
        stages.queue_assignment(assignment("c", 1, 1, 2, 0));
        let loading = prossimo(&mut rx).await;
        assert_eq!(loading.state(), State::Loading, "{loading:?}");
        assert!(stages.head().is_none(), "still loading");
        let s = prossimo(&mut rx).await;
        assert_eq!(s.state(), State::Ready, "{s:?}");
        assert!(s.load_seconds >= 0.6, "{}", s.load_seconds);
        let link = stages.head().expect("head should be ready");
        let id = link.nuovo_id();
        let mut u = link.request(pb::Request { id, text: "ciao".into(), max_tokens: 100, ..Default::default() }).await.unwrap();
        let mut text = String::new();
        let done = loop {
            match u.recv().await.expect("response stream ended") {
                OutgoingLink::Token(t) => text.push_str(&t.text),
                OutgoingLink::Done(f) => break f,
            }
        };
        assert_eq!(text, "Ciao è un test");
        assert_eq!((done.reason.as_str(), done.usage.unwrap().generated_tokens), ("eog", 5));
        // A lost head connection clears readiness.
        stages.stop_all().await;
        assert!(stages.head().is_none());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn head_eof_before_subscription_while_child_is_alive() {
        let dir = tempdir("head-eof");
        let mut config = cfg_test(&dir);
        config.env.push(("FAKE_STAGE_CLOSE_STDOUT".into(), "1".into()));
        let args = arguments(&config, &assignment("c", 1, 1, 2, 0)).unwrap();
        let (mut child, _, _) = start(&config, &args, &dir.join("head.log"), true, None).await.unwrap();
        let stdout = child.child.stdout.take().unwrap();
        let stdin = child.child.stdin.take().unwrap();
        let (reader, writer, buf) = head_ready(stdout, stdin).await.unwrap();
        let link = HeadConnection::from_io(reader, writer, buf);
        tokio::time::timeout(Duration::from_secs(5), async {
            while link.aperto() {
                tokio::task::yield_now().await;
            }
        }).await.unwrap();
        assert!(child.child.try_wait().unwrap().is_none());
        assert!(*link.osserva_chiusura().borrow());
        child.child.kill().await.unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn invalid_or_missing_readiness_does_not_report_ready() {
        for setting in ["FAKE_STAGE_NO_READY", "FAKE_STAGE_INVALID_READY"] {
            let dir = tempdir("head-invalid");
            let mut config = cfg_test(&dir);
            config.env.push((setting.into(), "1".into()));
            let stages = Stages::new(config);
            let (tx, mut rx) = mpsc::channel(16);
            stages.connect_peer(tx);
            stages.queue_assignment(assignment("c", 1, 1, 2, 0));
            assert_eq!(prossimo(&mut rx).await.state(), State::Loading);
            let failed = prossimo(&mut rx).await;
            assert_eq!(failed.state(), State::Failed);
            assert!(failed.error.contains("stage readiness failed"), "{}", failed.error);
            assert!(stages.head().is_none());
            stages.stop_all().await;
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn head_pipe_loss_closes_requests_and_allows_reassignment() {
        let dir = tempdir("head-reassignment");
        let stages = Stages::new(cfg_test(&dir));
        let (tx, mut rx) = mpsc::channel(16);
        stages.connect_peer(tx);
        stages.queue_assignment(assignment("c", 1, 1, 2, 0));
        assert_eq!(prossimo(&mut rx).await.state(), State::Loading);
        assert_eq!(prossimo(&mut rx).await.state(), State::Ready);
        let link = stages.head().unwrap();
        let mut pending = link.request(pb::Request { id: link.nuovo_id(), text: "close-head-pipe".into(), ..Default::default() }).await.unwrap();
        assert!(tokio::time::timeout(Duration::from_secs(5), pending.recv()).await.unwrap().is_none());
        assert_eq!(prossimo(&mut rx).await.state(), State::Failed);
        assert!(!link.aperto());
        stages.queue_assignment(assignment("c", 2, 1, 2, 0));
        assert_eq!(prossimo(&mut rx).await.state(), State::Loading);
        assert_eq!(prossimo(&mut rx).await.state(), State::Ready);
        let link = stages.head().unwrap();
        let mut pending = link.request(pb::Request { id: link.nuovo_id(), text: "ciao".into(), max_tokens: 10, ..Default::default() }).await.unwrap();
        let done = tokio::time::timeout(Duration::from_secs(5), async {
            while let Some(response) = pending.recv().await {
                if let OutgoingLink::Done(done) = response {
                    return done;
                }
            }
            panic!("replacement head closed before completion");
        }).await.unwrap();
        assert_eq!(done.reason, "eog");
        stages.stop_all().await;
    }
}

/// These integration tests launch real stages and require an executable
/// and fixture weights supplied through PUBLICOMPUTE_STAGE_BIN and
/// PUBLICOMPUTE_STAGE_MODEL. They run only when ignored tests are selected.
#[cfg(test)]
#[path = "bridge_stage_tests.rs"]
mod bridge_stage_tests;

#[cfg(test)]
mod real_tests {
    use super::tests::{prossimo, tempdir};
    use super::*;
    use crate::engine::{progress::Progress, stages::StadiEngine, Limits};
    use publicompute_proto::{
        common::ModelRef,
        node::{ChatMessage, ChatRole, InferenceRequest, SamplingParams},
    };
    use sha2::{Digest, Sha256};

    fn ambiente() -> Option<(PathBuf, PathBuf)> {
        Some((PathBuf::from(std::env::var("PUBLICOMPUTE_STAGE_BIN").ok()?), PathBuf::from(std::env::var("PUBLICOMPUTE_STAGE_MODEL").ok()?)))
    }

    fn assign(chain: &str, index: u32, n: u32, taglio: u32, p_listen: u16, p_token: u16, sha: &[u8]) -> StageAssign {
        StageAssign {
            chain_id: chain.into(),
            stage_index: index,
            n_stages: n,
            model: Some(ModelRef { model_id: "qwen-mini".into(), quantization: "f".into(), weights_sha256: sha.to_vec() }),
            layer_start: if index == 1 { 0 } else { taglio },
            layer_end: if index == 1 { taglio as i32 } else { -1 },
            slots: 4,
            ctx_per_slot: 512,
            groups: 2,
            n_batch: 64,
            state_format: "fp32".into(),
            listen_addr: if index == 1 { String::new() } else { format!("127.0.0.1:{p_listen}") },
            next_addr: if index == n { String::new() } else { format!("127.0.0.1:{p_listen}") },
            head_token_addr: format!("127.0.0.1:{p_token}"),
            capability: vec![],
            assign_epoch: 1,
            ..Default::default()
        }
    }

    fn request(text: &str, max_tokens: u32) -> InferenceRequest {
        InferenceRequest {
            request_id: "r".into(),
            messages: vec![ChatMessage { role: ChatRole::User as i32, content: text.into() }],
            sampling: Some(SamplingParams { max_tokens, temperature: 0.0, ..Default::default() }),
        }
    }

    /// Compare a two-stage chain with single-stage output using the same batch
    /// shape. Check that overlapping requests can run and cancelling one
    /// request leaves the others intact.
    #[tokio::test]
    #[ignore]
    async fn real_two_stage_chain_matches_reference() {
        let Some((bin, model)) = ambiente() else { return };
        let d = tempdir("real");
        let sha = Sha256::digest(std::fs::read(&model).unwrap()).to_vec();
        let mk = |name: &str| {
            Stages::new(StagesConfig {
                stage_bin: bin.clone(),
                model_path: model.clone(),
                weights_sha256: sha.clone(),
                log_dir: d.join(name),
                threads: 2,
                ngl: 0,
                extra_args: vec![],
                env: vec![],
                load_timeout: Duration::from_secs(120),
                stop_grace: Duration::from_secs(10),
                kill_wait: Duration::from_secs(15),
                ca_path: None,
                tls_dir: d.join(format!("tls-{name}")),
                in_chiaro: true,
            })
        };
        let (tail, head) = (mk("tail"), mk("head"));
        let (tx_c, mut rx_c) = mpsc::channel(16);
        let (tx_t, mut rx_t) = mpsc::channel(16);
        tail.connect_peer(tx_c);
        head.connect_peer(tx_t);
        let (p_listen, p_token) = (porta_libera().unwrap(), porta_libera().unwrap());
        tail.queue_assignment(assign("real", 2, 2, 6, p_listen, p_token, &sha));
        head.queue_assignment(assign("real", 1, 2, 6, p_listen, p_token, &sha));
        assert_eq!(prossimo(&mut rx_c).await.state(), State::Loading);
        assert_eq!(prossimo(&mut rx_t).await.state(), State::Loading);
        let s = prossimo(&mut rx_c).await;
        assert_eq!(s.state(), State::Ready, "{s:?}");
        let s = prossimo(&mut rx_t).await;
        assert_eq!(s.state(), State::Ready, "{s:?}");
        println!("chain ready: head started in {:.1} s", s.load_seconds);

        let progress = Arc::new(Progress::default());
        let limits = Limits {
            prefill_stall: Duration::from_secs(60),
            decode_stall: Duration::from_secs(60),
            request_max: Duration::from_secs(300),
            call_timeout: Duration::from_secs(5),
            step_timeout: Duration::from_secs(5),
            probe_timeout: Duration::from_secs(5),
        };
        let eng = Arc::new(StadiEngine { stages: head.clone(), attive: Default::default(), limits, progress });
        let text = "The quick brown fox jumps over the lazy dog";
        let g1 = eng.chat(&request(text, 16), None, None).await.unwrap();
        println!("chain: {} tokens {:?}, text {:?}", g1.token_ids.len(), g1.token_ids, g1.text);
        assert!(!g1.token_ids.is_empty());
        assert_eq!(g1.prompt_tokens_sha256.len(), 32);

        // Use the same prompt and zero temperature for the single-stage reference.
        let prompt = d.join("prompt.txt");
        std::fs::write(&prompt, format!("{text}\n")).unwrap();
        let output = d.join("rif");
        let st = std::process::Command::new(&bin)
            .args(["reference", "--model", model.to_str().unwrap(), "--prompt", prompt.to_str().unwrap(), "--n-prompt", "1", "--max-tokens", "16"])
            .args(["--slots", "4", "--ctx-per-slot", "512", "--groups", "2", "--n-batch", "64", "--block", "64", "--temperature", "0", "--threads", "2", "--output", output.to_str().unwrap()])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .unwrap();
        assert!(st.success());
        let row = std::fs::read_to_string(format!("{}.jsonl", output.display())).unwrap();
        let j: serde_json::Value = serde_json::from_str(row.lines().next().unwrap()).unwrap();
        let rif: Vec<u32> = j["token"].as_array().unwrap().iter().map(|t| t.as_i64().unwrap() as u32).collect();
        assert_eq!(g1.token_ids, rif, "chain service tokens should match the reference");

        // Start a second request during the first and cancel a third after partial output.
        let (e1, e2, e3) = (eng.clone(), eng.clone(), eng.clone());
        let (dtx, mut drx, _) = crate::engine::stages::delta_channel();
        let a = tokio::spawn(async move { e1.chat(&request(text, 48), Some(dtx), None).await });
        let primo = tokio::time::timeout(Duration::from_secs(60), drx.recv()).await.expect("expected the first fragment").expect("token channel closed");
        println!("first fragment of the first response: {primo:?}");
        let b = tokio::spawn(async move { e2.chat(&request("Hello world", 24), None, None).await });
        let (ctx, crx) = oneshot::channel();
        let c = tokio::spawn(async move { e3.chat(&request("Another prompt for the third", 200), None, Some(crx)).await });
        tokio::time::sleep(Duration::from_millis(300)).await;
        let _ = ctx.send(());
        let (a, b, c) = (a.await.unwrap().unwrap(), b.await.unwrap().unwrap(), c.await.unwrap().unwrap());
        println!("first: {} tokens, second: {} tokens, third: {} tokens ({:?})", a.token_ids.len(), b.token_ids.len(), c.token_ids.len(), c.finish_reason);
        // A longer generation under concurrent load should keep the deterministic prefix.
        assert_eq!(&a.token_ids[..g1.token_ids.len().min(a.token_ids.len())], &g1.token_ids[..g1.token_ids.len().min(a.token_ids.len())]);
        assert!(!b.token_ids.is_empty());
        assert_eq!(c.finish_reason, publicompute_proto::node::FinishReason::Cancelled);
        assert!(c.token_ids.len() < 200);
        // Exercise the chain again after cancellation clears its slot.
        let g2 = eng.chat(&request(text, 16), None, None).await.unwrap();
        assert_eq!(g2.token_ids, g1.token_ids);

        head.stop_all().await;
        tail.stop_all().await;
        // Activity can report RUNNING and READY before STOPPED, so wait for the terminal state.
        loop {
            let st = prossimo(&mut rx_t).await.state();
            if st == State::Stopped {
                break;
            }
            assert!(matches!(st, State::Running | State::Ready), "{st:?}");
        }
        assert_eq!(prossimo(&mut rx_c).await.state(), State::Stopped);
    }

    /// Compare four concurrent chats with the same requests run independently.
    /// The fixture has no chat template, so the final prompt is selected from
    /// the last user message in the conversation.
    #[tokio::test]
    #[ignore]
    async fn real_four_batched_requests_match_individual_requests() {
        let Some((bin, model)) = ambiente() else { return };
        let d = tempdir("real4");
        let sha = Sha256::digest(std::fs::read(&model).unwrap()).to_vec();
        let mk = |name: &str| {
            Stages::new(StagesConfig {
                stage_bin: bin.clone(),
                model_path: model.clone(),
                weights_sha256: sha.clone(),
                log_dir: d.join(name),
                threads: 2,
                ngl: 0,
                extra_args: vec![],
                env: vec![],
                load_timeout: Duration::from_secs(120),
                stop_grace: Duration::from_secs(10),
                kill_wait: Duration::from_secs(15),
                ca_path: None,
                tls_dir: d.join(format!("tls-{name}")),
                in_chiaro: true,
            })
        };
        let (tail, head) = (mk("tail"), mk("head"));
        let (tx_c, mut rx_c) = mpsc::channel(16);
        let (tx_t, mut rx_t) = mpsc::channel(32);
        tail.connect_peer(tx_c);
        head.connect_peer(tx_t);
        let (p_listen, p_token) = (porta_libera().unwrap(), porta_libera().unwrap());
        tail.queue_assignment(assign("real4", 2, 2, 6, p_listen, p_token, &sha));
        head.queue_assignment(assign("real4", 1, 2, 6, p_listen, p_token, &sha));
        assert_eq!(prossimo(&mut rx_c).await.state(), State::Loading);
        assert_eq!(prossimo(&mut rx_t).await.state(), State::Loading);
        assert_eq!(prossimo(&mut rx_c).await.state(), State::Ready);
        assert_eq!(prossimo(&mut rx_t).await.state(), State::Ready);
        assert_eq!(head.slots(), 4);
        let limits = Limits {
            prefill_stall: Duration::from_secs(60),
            decode_stall: Duration::from_secs(60),
            request_max: Duration::from_secs(300),
            call_timeout: Duration::from_secs(5),
            step_timeout: Duration::from_secs(5),
            probe_timeout: Duration::from_secs(5),
        };
        let eng = Arc::new(StadiEngine { stages: head.clone(), attive: Default::default(), limits, progress: Arc::new(Progress::default()) });
        let prompt = ["The quick brown fox jumps over the lazy dog", "Hello world", "Another prompt for the third request", "Rust is a systems programming language"];
        let lunghezze = [16u32, 24, 20, 32];
        // Capture independent outputs before filling all slots.
        let mut da_sole = Vec::new();
        for (p, n) in prompt.iter().zip(lunghezze) {
            da_sole.push(eng.chat(&request(p, n), None, None).await.unwrap().token_ids);
        }
        let mut task = Vec::new();
        for (i, (p, n)) in prompt.iter().zip(lunghezze).enumerate() {
            let e = eng.clone();
            let mut r = request(p, n);
            if i == 3 {
                let m = |role: ChatRole, c: &str| ChatMessage { role: role as i32, content: c.into() };
                r.messages = vec![m(ChatRole::System, "be brief"), m(ChatRole::User, "hi"), m(ChatRole::Assistant, "hello"), m(ChatRole::User, p)];
            }
            task.push(tokio::spawn(async move { e.chat(&r, None, None).await }));
        }
        let mut insieme = Vec::new();
        for t in task {
            insieme.push(t.await.unwrap().unwrap().token_ids);
        }
        for i in 0..4 {
            println!("request {i}: {} tokens alone, {} in the batch", da_sole[i].len(), insieme[i].len());
            assert!(!da_sole[i].is_empty());
            assert_eq!(insieme[i], da_sole[i], "request {i} should return the same tokens in a batch and on its own");
        }
        head.stop_all().await;
        tail.stop_all().await;
    }

    /// A mutually authenticated chain should match deterministic reference
    /// output. A certificate from the same CA does not authorize a peer
    /// outside the capability, and rejecting that peer must leave the chain alive.
    #[tokio::test]
    #[ignore]
    async fn real_tls_chain_matches_reference_and_rejects_unauthorized_peer() {
        use rcgen::{BasicConstraints, CertificateParams, DnType, ExtendedKeyUsagePurpose, IsCa, Issuer, KeyPair, KeyUsagePurpose, SanType};
        use ring::signature::{Ed25519KeyPair, KeyPair as _};
        use publicompute_proto::node::{CapStage, ChainCapability, SignedCapability};
        use prost::Message;
        let Some((bin, model)) = ambiente() else { return };
        let d = tempdir("real-tls");
        let sha = Sha256::digest(std::fs::read(&model).unwrap()).to_vec();
        // Issue test certificates with node URI identities and a separate key for signing capabilities.
        let ca_key = KeyPair::generate().unwrap();
        let mut p = CertificateParams::new(Vec::<String>::new()).unwrap();
        p.is_ca = IsCa::Ca(BasicConstraints::Constrained(0));
        p.key_usages = vec![KeyUsagePurpose::KeyCertSign, KeyUsagePurpose::CrlSign];
        p.distinguished_name.push(DnType::CommonName, "Test CA");
        let ca = p.self_signed(&ca_key).unwrap();
        let ca_pem = d.join("ca.pem");
        std::fs::write(&ca_pem, ca.pem()).unwrap();
        let issuer = Issuer::new(p, ca_key);
        let nodo = |id: &str| -> (PathBuf, PathBuf) {
            let k = KeyPair::generate().unwrap();
            let mut np = CertificateParams::new(Vec::<String>::new()).unwrap();
            np.subject_alt_names = vec![SanType::URI(format!("publicompute:node:{id}").try_into().unwrap())];
            np.extended_key_usages = vec![ExtendedKeyUsagePurpose::ClientAuth, ExtendedKeyUsagePurpose::ServerAuth];
            np.key_usages = vec![KeyUsagePurpose::DigitalSignature];
            let c = np.signed_by(&k, &issuer).unwrap();
            let (cp, kp) = (d.join(format!("{id}.cert.pem")), d.join(format!("{id}.key.pem")));
            std::fs::write(&cp, c.pem()).unwrap();
            std::fs::write(&kp, k.serialize_pem()).unwrap();
            (cp, kp)
        };
        let (cert_testa, key_testa) = nodo("n_testa");
        let (cert_coda, key_coda) = nodo("n_coda");
        let (cert_x, key_x) = nodo("n_intruso");
        let rng = ring::rand::SystemRandom::new();
        let pk8 = Ed25519KeyPair::generate_pkcs8(&rng).unwrap();
        let firma = Ed25519KeyPair::from_pkcs8(pk8.as_ref()).unwrap();
        let pub_file = d.join("capability.pub");
        std::fs::write(&pub_file, hex::encode(firma.public_key().as_ref())).unwrap();
        let scade = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() as i64 + 3600;
        let c = ChainCapability {
            chain_id: "real-tls".into(),
            assign_epoch: 1,
            expires_unix: scade,
            stages: vec![
                CapStage { index: 1, node_id: "n_testa".into(), model_sha256: sha.clone(), layer_start: 0, layer_end: 6 },
                CapStage { index: 2, node_id: "n_coda".into(), model_sha256: sha.clone(), layer_start: 6, layer_end: -1 },
            ],
        };
        let body = c.encode_to_vec();
        let capability = SignedCapability { signature: firma.sign(&body).as_ref().to_vec(), body, key_id: "test".into() }.encode_to_vec();

        let mk = |name: &str, cert: &PathBuf, key: &PathBuf| {
            let s = Stages::new(StagesConfig {
                stage_bin: bin.clone(),
                model_path: model.clone(),
                weights_sha256: sha.clone(),
                log_dir: d.join(name),
                threads: 2,
                ngl: 0,
                extra_args: vec![],
                env: vec![],
                load_timeout: Duration::from_secs(120),
                stop_grace: Duration::from_secs(10),
                kill_wait: Duration::from_secs(15),
                ca_path: Some(ca_pem.clone()),
                tls_dir: d.join(format!("tls-{name}")),
                in_chiaro: true,
            });
            s.set_tls_files(TlsFiles { cert: Some(cert.clone()), key: Some(key.clone()), cap_pub: Some(pub_file.clone()) });
            s
        };
        let (tail, head) = (mk("tail", &cert_coda, &key_coda), mk("head", &cert_testa, &key_testa));
        let (tx_c, mut rx_c) = mpsc::channel(16);
        let (tx_t, mut rx_t) = mpsc::channel(32);
        tail.connect_peer(tx_c);
        head.connect_peer(tx_t);
        let (p_listen, p_token) = (porta_libera().unwrap(), porta_libera().unwrap());
        let mut a_coda = assign("real-tls", 2, 2, 6, p_listen, p_token, &sha);
        let mut a_testa = assign("real-tls", 1, 2, 6, p_listen, p_token, &sha);
        a_coda.capability = capability.clone();
        a_testa.capability = capability.clone();
        tail.queue_assignment(a_coda);
        head.queue_assignment(a_testa);
        assert_eq!(prossimo(&mut rx_c).await.state(), State::Loading);
        assert_eq!(prossimo(&mut rx_t).await.state(), State::Loading);
        let s = prossimo(&mut rx_c).await;
        assert_eq!(s.state(), State::Ready, "{s:?}");
        let s = prossimo(&mut rx_t).await;
        assert_eq!(s.state(), State::Ready, "{s:?}");
        for name in ["tail", "head"] {
            let log = std::fs::read_to_string(d.join(name).join("stage-real-tls-1.log")).unwrap();
            assert!(log.contains("mutual TLS 1.3 enabled"), "{name}: TLS inactive: {log}");
            println!("{name}: {}", log.lines().find(|r| r.contains("mutual TLS 1.3 enabled")).unwrap());
        }

        let limits = Limits {
            prefill_stall: Duration::from_secs(60),
            decode_stall: Duration::from_secs(60),
            request_max: Duration::from_secs(300),
            call_timeout: Duration::from_secs(5),
            step_timeout: Duration::from_secs(5),
            probe_timeout: Duration::from_secs(5),
        };
        let eng = Arc::new(StadiEngine { stages: head.clone(), attive: Default::default(), limits, progress: Arc::new(Progress::default()) });
        let text = "The quick brown fox jumps over the lazy dog";
        let g1 = eng.chat(&request(text, 16), None, None).await.unwrap();
        println!("TLS chain: {} tokens {:?}, text {:?}", g1.token_ids.len(), g1.token_ids, g1.text);
        assert!(!g1.token_ids.is_empty());

        let prompt = d.join("prompt.txt");
        std::fs::write(&prompt, format!("{text}\n")).unwrap();
        let output = d.join("rif");
        let st = std::process::Command::new(&bin)
            .args(["reference", "--model", model.to_str().unwrap(), "--prompt", prompt.to_str().unwrap(), "--n-prompt", "1", "--max-tokens", "16"])
            .args(["--slots", "4", "--ctx-per-slot", "512", "--groups", "2", "--n-batch", "64", "--block", "64", "--temperature", "0", "--threads", "2", "--output", output.to_str().unwrap()])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .unwrap();
        assert!(st.success());
        let row = std::fs::read_to_string(format!("{}.jsonl", output.display())).unwrap();
        let j: serde_json::Value = serde_json::from_str(row.lines().next().unwrap()).unwrap();
        let rif: Vec<u32> = j["token"].as_array().unwrap().iter().map(|t| t.as_i64().unwrap() as u32).collect();
        assert_eq!(g1.token_ids, rif, "tokens over TLS with a signed capability should match the reference");

        // The certificate establishes a valid TLS identity. The prefill handshake
        // then tests whether that identity is authorized by the capability.
        let (cx, kx) = (cert_x.clone(), key_x.clone());
        let esito = tokio::task::spawn_blocking(move || intruso(&cx, &kx, p_listen)).await.unwrap();
        println!("unauthorized peer: {esito}");
        assert!(esito.contains("is missing from chain"), "expected the unauthorized peer to be rejected: {esito}");
        let log_coda = std::fs::read_to_string(d.join("tail").join("stage-real-tls-1.log")).unwrap();
        assert!(log_coda.contains("n_intruso") && log_coda.contains("rejected"), "{log_coda}");

        // Check output again to make sure rejecting a peer did not interrupt service.
        let g2 = eng.chat(&request(text, 16), None, None).await.unwrap();
        assert_eq!(g2.token_ids, g1.token_ids);

        head.stop_all().await;
        tail.stop_all().await;
    }

    /// Open a fixture TLS connection and send a prefill handshake. Return the
    /// peer's acceptance or refusal, or a transport error.
    fn intruso(cert: &Path, key: &Path, porta: u16) -> String {
        use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
        use rustls::pki_types::{pem::PemObject, CertificateDer, PrivateKeyDer, ServerName, UnixTime};
        use std::io::{Read, Write};
        #[derive(Debug)]
        struct Tutti(Arc<rustls::crypto::CryptoProvider>);
        impl ServerCertVerifier for Tutti {
            fn verify_server_cert(&self, _: &CertificateDer<'_>, _: &[CertificateDer<'_>], _: &ServerName<'_>, _: &[u8], _: UnixTime) -> Result<ServerCertVerified, rustls::Error> {
                Ok(ServerCertVerified::assertion())
            }
            fn verify_tls12_signature(&self, m: &[u8], c: &CertificateDer<'_>, d: &rustls::DigitallySignedStruct) -> Result<HandshakeSignatureValid, rustls::Error> {
                rustls::crypto::verify_tls12_signature(m, c, d, &self.0.signature_verification_algorithms)
            }
            fn verify_tls13_signature(&self, m: &[u8], c: &CertificateDer<'_>, d: &rustls::DigitallySignedStruct) -> Result<HandshakeSignatureValid, rustls::Error> {
                rustls::crypto::verify_tls13_signature(m, c, d, &self.0.signature_verification_algorithms)
            }
            fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
                self.0.signature_verification_algorithms.supported_schemes()
            }
        }
        let provider = Arc::new(rustls::crypto::ring::default_provider());
        let chain: Vec<CertificateDer<'static>> = CertificateDer::pem_file_iter(cert).unwrap().collect::<Result<_, _>>().unwrap();
        let k = PrivateKeyDer::from_pem_file(key).unwrap();
        let mut cfg = rustls::ClientConfig::builder_with_provider(provider.clone())
            .with_protocol_versions(&[&rustls::version::TLS13])
            .unwrap()
            .dangerous()
            .with_custom_certificate_verifier(Arc::new(Tutti(provider)))
            .with_client_auth_cert(chain, k)
            .unwrap();
        cfg.alpn_protocols = vec![b"publicompute-stage/0".to_vec()];
        let tcp = match std::net::TcpStream::connect(("127.0.0.1", porta)) {
            Ok(t) => t,
            Err(e) => return format!("connection: {e}"),
        };
        let _ = tcp.set_read_timeout(Some(Duration::from_secs(10)));
        let conn = rustls::ClientConnection::new(Arc::new(cfg), ServerName::try_from("stage.publicompute").unwrap()).unwrap();
        let mut tls = rustls::StreamOwned::new(conn, tcp);
        let hello = publicompute_proto::stage::Message {
            body: Some(publicompute_proto::stage::message::Body::Hello(publicompute_proto::stage::Hello {
                protocol_major: 0,
                protocol_minor: 1,
                stream: publicompute_proto::stage::Stream::Prefill as i32,
                epoch: 1,
                mtp: false,
            })),
        };
        let mut v = Vec::new();
        wire::encode_frame(&hello, &mut v).unwrap();
        if let Err(e) = tls.write_all(&v).and_then(|_| tls.flush()) {
            return format!("write: {e}");
        }
        let (mut buf, mut tmp) = (BytesMut::new(), vec![0u8; 4096]);
        loop {
            match wire::try_decode_frame::<publicompute_proto::stage::Message>(&mut buf) {
                Ok(Some(m)) => {
                    return match m.body {
                        Some(publicompute_proto::stage::message::Body::Ack(a)) if a.ok => "accepted".into(),
                        Some(publicompute_proto::stage::message::Body::Ack(a)) => format!("rejected: {}", a.error),
                        altro => format!("unexpected reply: {altro:?}"),
                    }
                }
                Ok(None) => {}
                Err(e) => return format!("frame: {e}"),
            }
            match tls.read(&mut tmp) {
                Ok(0) => return "closed without a reply".into(),
                Ok(n) => buf.extend_from_slice(&tmp[..n]),
                Err(e) => return format!("read: {e}"),
            }
        }
    }
}
