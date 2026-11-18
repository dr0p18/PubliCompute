//! The bridge carries stage TCP connections through a peer endpoint so
//! stages can communicate across NAT. On the sending side, each loopback
//! connection opens one bidirectional stream. On the receiving side, the
//! chain ID, epoch and direction select a local stage listener. The stage
//! still performs its own TLS and capability checks on the forwarded bytes.
//!
//! The agent connects to its configured relay and rejects an assignment
//! that names another relay. A half-close is forwarded without ending
//! the other direction. On error or assignment teardown, close both sides.
//! If the destination is not ready, closing the local connection lets
//! the stage retry on its normal schedule.

use std::{
    collections::HashMap,
    net::SocketAddr,
    path::Path,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::{Duration, Instant},
};

use anyhow::{anyhow, bail, Context, Result};
use iroh::{
    endpoint::{presets, Connection, RecvStream, SendStream, VarInt},
    protocol::{AcceptError, ProtocolHandler, Router},
    Endpoint, EndpointAddr, EndpointId, RelayMode, RelayUrl, SecretKey, TransportAddr,
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    sync::{watch, Mutex as AsyncMutex},
};
use tracing::{debug, info, warn};

pub const ALPN: &[u8] = b"publicompute/catena/0";

const WIRE_VERSION: u8 = 1;
/// Stream reset reasons reported to the peer that opened the stream.
const REJECT_HEADER: u32 = 1;
const REJECT_CHAIN: u32 = 2;
const REJECT_STAGE: u32 = 3;
const CLOSED: u32 = 4;
const REJECT_SENDER: u32 = 5;
const REJECT_LIMIT: u32 = 6;
/// Deadline for dialing a peer and opening a stream.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// Deadline for reading an incoming routing header.
const HEADER_TIMEOUT: Duration = Duration::from_secs(10);
/// Concurrency budget shared by incoming streams across all peer connections.
const STREAM_MAX: usize = 64;
/// Concurrent connection limit per authenticated endpoint ID.
const MAX_CONNECTIONS_PER_PEER: usize = 4;

/// Admission limits for incoming connections and streams.
#[derive(Clone, Copy, Debug)]
pub struct BridgeLimits {
    pub stream_max: usize,
    pub connections_per_peer: usize,
}

impl Default for BridgeLimits {
    fn default() -> Self {
        BridgeLimits { stream_max: STREAM_MAX, connections_per_peer: MAX_CONNECTIONS_PER_PEER }
    }
}
const BUF: usize = 64 * 1024;

/// Routing direction selects which stage listener receives the stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Direction {
    /// Route prefill, generation and control traffic toward the next stage.
    Forward,
    /// Route generated tokens from the tail to the head.
    Token,
}

impl Direction {
    fn code(self) -> u8 {
        match self {
            Direction::Forward => 1,
            Direction::Token => 2,
        }
    }

    fn from_code(c: u8) -> Option<Direction> {
        match c {
            1 => Some(Direction::Forward),
            2 => Some(Direction::Token),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Direction::Forward => "forward",
            Direction::Token => "token",
        }
    }
}

/// Counters and route observations for one directed stage link.
#[derive(Clone, Debug, Default)]
pub struct LinkMetric {
    pub chain_id: String,
    pub epoch: u64,
    pub verso: &'static str,
    /// Whether this agent initiated the peer connection.
    pub in_uscita: bool,
    /// Last observed route type. Empty until a stream starts.
    pub percorso: String,
    /// Seconds from connection establishment to the first direct path observation.
    pub primo_diretto_s: Option<f64>,
    /// Bytes forwarded toward the remote stage.
    pub byte_inviati: u64,
    /// Bytes delivered from the peer to the local stage.
    pub byte_ricevuti: u64,
    pub stream_aperti: u64,
}

#[derive(Default)]
struct AdmissionStats {
    inviati: AtomicU64,
    ricevuti: AtomicU64,
    stream: AtomicU64,
    state: Mutex<RouteState>,
}

#[derive(Default)]
struct RouteState {
    percorso: String,
    inizio: Option<Instant>,
    primo_diretto_s: Option<f64>,
}

impl AdmissionStats {
    /// Record the latest route and measure the first transition to a direct path.
    fn subscribe(&self, conn: &Connection, chi: &LinkKey, in_uscita: bool) {
        let p = percorso(conn);
        let mut s = self.state.lock().unwrap();
        let inizio = *s.inizio.get_or_insert_with(Instant::now);
        if s.percorso != p {
            info!(chain = %chi.chain_id, epoch = chi.epoch, direction = chi.verso.name(), outgoing = in_uscita, path = p, "bridge: path changed");
            s.percorso = p.to_string();
        }
        if p == "direct" && s.primo_diretto_s.is_none() {
            let t = inizio.elapsed().as_secs_f64();
            s.primo_diretto_s = Some(t);
            info!(chain = %chi.chain_id, epoch = chi.epoch, direction = chi.verso.name(), outgoing = in_uscita, seconds = format!("{t:.3}"), "bridge: first direct path");
        }
    }

    fn metric(&self, chi: &LinkKey, in_uscita: bool) -> LinkMetric {
        let s = self.state.lock().unwrap();
        LinkMetric {
            chain_id: chi.chain_id.clone(),
            epoch: chi.epoch,
            verso: chi.verso.name(),
            in_uscita,
            percorso: s.percorso.clone(),
            primo_diretto_s: s.primo_diretto_s,
            byte_inviati: self.inviati.load(Ordering::Relaxed),
            byte_ricevuti: self.ricevuti.load(Ordering::Relaxed),
            stream_aperti: self.stream.load(Ordering::Relaxed),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct LinkKey {
    chain_id: String,
    epoch: u64,
    verso: Direction,
}

/// Local listener and sender policy for an incoming assignment route.
struct IncomingLink {
    destinazione: SocketAddr,
    conti: Arc<AdmissionStats>,
    /// Closing an assignment interrupts its active forwards.
    chiuso: watch::Receiver<bool>,
    /// Allowed endpoint IDs for this chain. An empty list leaves per-chain filtering disabled.
    attesi: Vec<EndpointId>,
}

struct IncomingState {
    limiti: BridgeLimits,
    stream_vivi: AtomicU64,
    per_mittente: Mutex<HashMap<EndpointId, usize>>,
    /// Count rejections that happen before reading stream data.
    rifiutate: AtomicU64,
    accettate: AtomicU64,
    entrate: Mutex<HashMap<LinkKey, IncomingLink>>,
    /// In tests, retain a snapshot handle for each link in creation order.
    registro: Mutex<Vec<(LinkKey, bool, Arc<AdmissionStats>)>>,
}

/// One bridge endpoint shared by all assignments on an agent.
pub struct Bridge {
    router: Router,
    relay: RelayUrl,
    dentro: Arc<IncomingState>,
}

/// Dropping the assignment guard closes its listeners, interrupts active
/// forwards and removes its incoming routes.
pub struct LinkGuard {
    close: watch::Sender<bool>,
    dentro: Arc<IncomingState>,
    chiavi: Vec<LinkKey>,
    summary: Vec<(LinkKey, bool, Arc<AdmissionStats>)>,
}

impl Drop for LinkGuard {
    fn drop(&mut self) {
        self.close.send_replace(true);
        let mut e = self.dentro.entrate.lock().unwrap();
        for k in &self.chiavi {
            e.remove(k);
        }
        drop(e);
        for (k, in_uscita, c) in &self.summary {
            let m = c.metric(k, *in_uscita);
            if m.stream_aperti > 0 {
                info!(
                    chain = %m.chain_id, epoch = m.epoch, direction = m.verso, outgoing = m.in_uscita, path = %m.percorso,
                    first_direct_s = ?m.primo_diretto_s, bytes_sent = m.byte_inviati, bytes_received = m.byte_ricevuti,
                    stream = m.stream_aperti, "bridge: connection closed"
                );
            }
        }
    }
}

/// Load or create the endpoint secret as 32 hex-encoded bytes stored in an owner-only file.
pub fn load_key(file: &Path) -> Result<SecretKey> {
    if file.exists() {
        let s = std::fs::read_to_string(file).context("read the iroh key")?;
        let b: [u8; 32] = hex::decode(s.trim())
            .ok()
            .and_then(|v| v.try_into().ok())
            .with_context(|| format!("{}: expected 64 hexadecimal characters", file.display()))?;
        return Ok(SecretKey::from_bytes(&b));
    }
    let b: [u8; 32] = rand::random();
    if let Some(d) = file.parent() {
        std::fs::create_dir_all(d)?;
    }
    write_private(file, hex::encode(b).as_bytes())?;
    Ok(SecretKey::from_bytes(&b))
}

#[cfg(unix)]
fn write_private(path: &Path, dati: &[u8]) -> Result<()> {
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;
    let mut f = std::fs::OpenOptions::new().write(true).create_new(true).mode(0o600).open(path).context("create the iroh key")?;
    f.write_all(dati)?;
    Ok(())
}

#[cfg(not(unix))]
fn write_private(path: &Path, dati: &[u8]) -> Result<()> {
    use std::io::Write;
    // On Windows, restrict access before writing the endpoint secret.
    let mut f = std::fs::File::create(path).context("create the iroh key")?;
    crate::identity::acl::riserva_all_utente(path)?;
    f.write_all(dati)?;
    Ok(())
}

fn percorso(conn: &Connection) -> &'static str {
    let (mut ip, mut relay, mut scelto_ip) = (false, false, false);
    for p in conn.paths().iter() {
        ip |= p.is_ip();
        relay |= p.is_relay();
        scelto_ip |= p.is_selected() && p.is_ip();
    }
    if scelto_ip {
        "direct"
    } else if relay && ip {
        "mixed"
    } else {
        "relay"
    }
}

fn read_relay(url: &str) -> Result<RelayUrl> {
    url.parse().map_err(|e| anyhow!("invalid relay \"{url}\": {e}"))
}

impl Bridge {
    /// Start the endpoint with the configured relay. Call in_linea to wait for connectivity.
    pub async fn new(relay: &str, chiave: SecretKey) -> Result<Arc<Bridge>> {
        Self::with_limits(relay, chiave, BridgeLimits::default()).await
    }

    pub async fn with_limits(relay: &str, chiave: SecretKey, limiti: BridgeLimits) -> Result<Arc<Bridge>> {
        let url = read_relay(relay)?;
        let ep = Endpoint::builder(presets::Minimal)
            .secret_key(chiave)
            .relay_mode(RelayMode::custom([url.clone()]))
            .clear_address_lookup()
            .portmapper_config(iroh::endpoint::PortmapperConfig::Disabled)
            .bind()
            .await
            .map_err(|e| anyhow!("open iroh endpoint: {e}"))?;
        let dentro = Arc::new(IncomingState {
            limiti,
            stream_vivi: AtomicU64::new(0),
            per_mittente: Mutex::new(HashMap::new()),
            rifiutate: AtomicU64::new(0),
            accettate: AtomicU64::new(0),
            entrate: Mutex::new(HashMap::new()), registro: Mutex::new(Vec::new()),
        });
        let router = Router::builder(ep).accept(ALPN, Accetta { dentro: dentro.clone() }).spawn();
        Ok(Arc::new(Bridge { router, relay: url, dentro }))
    }

    /// Return the public endpoint ID used in the handshake and stage assignments.
    pub fn id(&self) -> String {
        self.router.endpoint().id().to_string()
    }

    /// Wait up to the given deadline for relay connectivity.
    pub async fn in_linea(&self, entro: Duration) -> bool {
        tokio::time::timeout(entro, self.router.endpoint().online()).await.is_ok()
    }

    /// Close all streams so stage listeners observe connection loss.
    pub async fn close(&self) {
        let _ = self.router.shutdown().await;
    }

    /// Create a guard that will own the new assignment's routes and forwards.
    pub fn guard(&self) -> LinkGuard {
        LinkGuard { close: watch::channel(false).0, dentro: self.dentro.clone(), chiavi: Vec::new(), summary: Vec::new() }
    }

    #[cfg(test)]
    pub fn connessioni_aperte(&self) -> u64 {
        self.dentro.accettate.load(Ordering::Relaxed)
    }

    #[cfg(test)]
    pub fn connessioni_rifiutate(&self) -> u64 {
        self.dentro.rifiutate.load(Ordering::Relaxed)
    }

    #[cfg(test)]
    pub fn metrics(&self) -> Vec<LinkMetric> {
        self.dentro.registro.lock().unwrap().iter().map(|(k, u, c)| c.metric(k, *u)).collect()
    }

    /// Register a fixture route without a sender allowlist.
    #[cfg(test)]
    pub fn incoming(&self, g: &mut LinkGuard, chain_id: &str, epoch: u64, verso: Direction, destinazione: SocketAddr) {
        self.incoming_with(g, chain_id, epoch, verso, destinazione, &[]).unwrap();
    }

    /// Register a local listener for a chain, epoch and direction. Restrict
    /// incoming streams to the listed endpoint IDs when supplied. Log a
    /// warning if the empty list leaves sender admission open.
    pub fn incoming_with(&self, g: &mut LinkGuard, chain_id: &str, epoch: u64, verso: Direction, destinazione: SocketAddr, attesi: &[String]) -> Result<()> {
        let attesi: Vec<EndpointId> =
            attesi.iter().map(|s| s.parse().map_err(|e| anyhow!("invalid expected iroh ID \"{s}\": {e}"))).collect::<Result<_>>()?;
        if attesi.is_empty() {
            warn!(chain = %chain_id, epoch, direction = verso.name(), "bridge: StageAssign has no expected iroh ID; incoming peer identity is unchecked");
        }
        let k = LinkKey { chain_id: chain_id.to_string(), epoch, verso };
        let conti = Arc::new(AdmissionStats::default());
        self.dentro.entrate.lock().unwrap().insert(k.clone(), IncomingLink { destinazione, conti: conti.clone(), chiuso: g.close.subscribe(), attesi });
        if cfg!(test) {
            self.dentro.registro.lock().unwrap().push((k.clone(), false, conti.clone()));
        }
        g.chiavi.push(k.clone());
        g.summary.push((k, false, conti));
        Ok(())
    }

    /// Open a loopback listener that forwards to the specified endpoint.
    /// An empty relay uses the local configuration. Reject a different relay.
    pub async fn output(&self, g: &mut LinkGuard, chain_id: &str, epoch: u64, verso: Direction, nodo: &str, relay: &str) -> Result<SocketAddr> {
        let id: EndpointId = nodo.parse().map_err(|e| anyhow!("invalid iroh ID \"{nodo}\": {e}"))?;
        if !relay.is_empty() && read_relay(relay)? != self.relay {
            bail!("relay_url {relay} does not match the agent relay ({}): rejected", self.relay);
        }
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.context("local bridge port")?;
        let locale = listener.local_addr()?;
        let k = LinkKey { chain_id: chain_id.to_string(), epoch, verso };
        let conti = Arc::new(AdmissionStats::default());
        if cfg!(test) {
            self.dentro.registro.lock().unwrap().push((k.clone(), true, conti.clone()));
        }
        g.summary.push((k.clone(), true, conti.clone()));
        let u = Arc::new(OutgoingLink {
            ep: self.router.endpoint().clone(),
            addr: EndpointAddr::from_parts(id, [TransportAddr::Relay(self.relay.clone())]),
            chiave: k,
            conti,
            conn: AsyncMutex::new(None),
            chiuso: g.close.subscribe(),
        });
        let mut chiuso = g.close.subscribe();
        let per_stream = chiuso.clone();
        info!(chain = %chain_id, epoch, direction = verso.name(), peer = %id, local = %locale, "bridge: outgoing connection");
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = chiusura(&mut chiuso) => break,
                    r = listener.accept() => match r {
                        Ok((tcp, _)) => {
                            let (u, chiuso) = (u.clone(), per_stream.clone());
                            tokio::spawn(async move { u.servi(tcp, chiuso).await });
                        }
                        Err(e) => {
                            warn!("bridge: local accept: {e}");
                            tokio::time::sleep(Duration::from_millis(100)).await;
                        }
                    },
                }
            }
        });
        Ok(locale)
    }
}

/// Cache the outgoing connection and reconnect on the next dial if it drops.
struct OutgoingLink {
    ep: Endpoint,
    addr: EndpointAddr,
    chiave: LinkKey,
    conti: Arc<AdmissionStats>,
    conn: AsyncMutex<Option<Connection>>,
    /// Assignment closure also terminates the cached peer connection.
    chiuso: watch::Receiver<bool>,
}

impl OutgoingLink {
    async fn connessione(&self) -> Result<Connection> {
        let mut c = self.conn.lock().await;
        if let Some(x) = c.as_ref() {
            if x.close_reason().is_none() {
                return Ok(x.clone());
            }
        }
        self.conti.state.lock().unwrap().inizio.get_or_insert_with(Instant::now);
        let new = self.ep.connect(self.addr.clone(), ALPN).await.map_err(|e| anyhow!("iroh connection: {e}"))?;
        // The path watcher holds a connection handle, so it must explicitly close
        // the connection when the assignment ends.
        let (conn, conti, chi, mut chiuso) = (new.clone(), self.conti.clone(), self.chiave.clone(), self.chiuso.clone());
        tokio::spawn(async move {
            loop {
                conti.subscribe(&conn, &chi, true);
                tokio::select! {
                    _ = conn.closed() => break,
                    _ = chiusura(&mut chiuso) => {
                        conn.close(VarInt::from_u32(CLOSED), b"assignment closed");
                        break;
                    }
                    _ = tokio::time::sleep(Duration::from_millis(100)) => {}
                }
            }
        });
        *c = Some(new.clone());
        Ok(new)
    }

    /// Open a peer stream for one accepted TCP connection and send its routing header before copying bytes.
    async fn servi(&self, tcp: TcpStream, chiuso: watch::Receiver<bool>) {
        let k = &self.chiave;
        let aperto = tokio::time::timeout(CONNECT_TIMEOUT, async {
            let conn = self.connessione().await?;
            let (mut send, recv) = conn.open_bi().await.map_err(|e| anyhow!("open stream: {e}"))?;
            send.write_all(&header(k)).await.map_err(|e| anyhow!("stream header: {e}"))?;
            Ok::<_, anyhow::Error>((conn, send, recv))
        })
        .await;
        let (conn, send, recv) = match aperto {
            Ok(Ok(x)) => x,
            Ok(Err(e)) => {
                warn!(chain = %k.chain_id, direction = k.verso.name(), "bridge: destination unreachable: {e:#}");
                return; // Dropping the TCP connection prompts the stage to retry.
            }
            Err(_) => {
                warn!(chain = %k.chain_id, direction = k.verso.name(), "bridge: destination unreachable after {} s", CONNECT_TIMEOUT.as_secs());
                return;
            }
        };
        self.conti.stream.fetch_add(1, Ordering::Relaxed);
        self.conti.subscribe(&conn, k, true);
        if let Err(e) = forward(tcp, send, recv, &self.conti, chiuso).await {
            e.write(k, "outgoing");
        }
    }
}

/// Wait for assignment closure, including a dropped guard.
async fn chiusura(rx: &mut watch::Receiver<bool>) {
    let _ = rx.wait_for(|c| *c).await;
}

fn header(k: &LinkKey) -> Vec<u8> {
    let id = k.chain_id.as_bytes();
    let mut v = Vec::with_capacity(12 + id.len());
    v.push(WIRE_VERSION);
    v.push(k.verso.code());
    v.extend_from_slice(&k.epoch.to_le_bytes());
    v.extend_from_slice(&(id.len() as u16).to_le_bytes());
    v.extend_from_slice(id);
    v
}

async fn read_header(recv: &mut RecvStream) -> Result<LinkKey> {
    let mut fissa = [0u8; 12];
    recv.read_exact(&mut fissa).await.map_err(|e| anyhow!("stream header: {e}"))?;
    if fissa[0] != WIRE_VERSION {
        bail!("unsupported header version {}", fissa[0]);
    }
    let verso = Direction::from_code(fissa[1]).with_context(|| format!("unknown direction {}", fissa[1]))?;
    let epoch = u64::from_le_bytes(fissa[2..10].try_into().unwrap());
    let n = u16::from_le_bytes(fissa[10..12].try_into().unwrap()) as usize;
    if n > 256 {
        bail!("chain_id is {n} bytes");
    }
    let mut id = vec![0u8; n];
    recv.read_exact(&mut id).await.map_err(|e| anyhow!("stream header: {e}"))?;
    Ok(LinkKey { chain_id: String::from_utf8(id).context("chain_id is not UTF-8")?, epoch, verso })
}

/// Copy in both directions until both finish or a failure or assignment
/// closure stops them. Always close the local TCP connection before returning.
async fn forward(tcp: TcpStream, mut send: SendStream, mut recv: RecvStream, conti: &AdmissionStats, mut chiuso: watch::Receiver<bool>) -> Result<(), Caduta> {
    let _ = tcp.set_nodelay(true);
    let avviato = std::sync::atomic::AtomicBool::new(false);
    let (mut rd, mut wr) = tcp.into_split();
    let verso_stream = async {
        let mut buf = vec![0u8; BUF];
        loop {
            let n = rd.read(&mut buf).await.map_err(|e| format!("read local TCP: {e}"))?;
            if n == 0 {
                // Forward this FIN and let the other direction finish independently.
                send.finish().map_err(|e| format!("finish stream: {e}"))?;
                return Ok::<(), String>(());
            }
            send.write_all(&buf[..n]).await.map_err(|e| format!("write stream: {e}"))?;
            conti.inviati.fetch_add(n as u64, Ordering::Relaxed);
        }
    };
    let verso_tcp = async {
        let mut buf = vec![0u8; BUF];
        loop {
            match recv.read(&mut buf).await.map_err(|e| format!("read stream: {e}"))? {
                None => {
                    wr.shutdown().await.map_err(|e| format!("close local TCP: {e}"))?;
                    return Ok::<(), String>(());
                }
                Some(n) => {
                    wr.write_all(&buf[..n]).await.map_err(|e| format!("write local TCP: {e}"))?;
                    avviato.store(true, Ordering::Relaxed);
                    conti.ricevuti.fetch_add(n as u64, Ordering::Relaxed);
                }
            }
        }
    };
    let esito = tokio::select! {
        r = async { tokio::try_join!(verso_stream, verso_tcp).map(|_| ()) } => r,
        _ = chiusura(&mut chiuso) => Err("assignment closed".to_string()),
    };
    if esito.is_err() {
        // Reset the peer's stream immediately so it need not wait for a timeout.
        let _ = send.reset(VarInt::from_u32(CLOSED));
        let _ = recv.stop(VarInt::from_u32(CLOSED));
    }
    esito.map_err(|reason| Caduta { reason, avviato: avviato.load(Ordering::Relaxed) })
}

/// Track whether a failed forward received bytes from the destination.
/// When it never replied, the local stage retries and repeated failures
/// before readiness do not need a log line each.
struct Caduta {
    reason: String,
    avviato: bool,
}

impl Caduta {
    fn write(&self, k: &LinkKey, lato: &str) {
        if self.avviato {
            info!(chain = %k.chain_id, direction = k.verso.name(), "bridge: {lato} stream closed: {}", self.reason);
        } else {
            debug!(chain = %k.chain_id, direction = k.verso.name(), "bridge: {lato} stream did not start: {}", self.reason);
        }
    }
}

/// Each incoming bidirectional stream represents one TCP connection on the peer.
#[derive(Clone)]
struct Accetta {
    dentro: Arc<IncomingState>,
}

impl std::fmt::Debug for Accetta {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Acceptor")
    }
}

impl IncomingState {
    /// Before accepting streams, require that some registered chain allow the
    /// peer. With no registered chains, let stream routing handle retryable startup.
    fn expected_sender(&self, id: &EndpointId) -> bool {
        let e = self.entrate.lock().unwrap();
        e.is_empty() || e.values().any(|x| x.attesi.is_empty() || x.attesi.contains(id))
    }
}

/// Release the sender's connection slot when its reservation is dropped.
struct SenderGuard {
    dentro: Arc<IncomingState>,
    id: EndpointId,
}

impl SenderGuard {
    fn take(dentro: &Arc<IncomingState>, id: EndpointId) -> Option<SenderGuard> {
        let mut m = dentro.per_mittente.lock().unwrap();
        let n = m.entry(id).or_insert(0);
        if *n >= dentro.limiti.connections_per_peer {
            return None;
        }
        *n += 1;
        Some(SenderGuard { dentro: dentro.clone(), id })
    }
}

impl Drop for SenderGuard {
    fn drop(&mut self) {
        let mut m = self.dentro.per_mittente.lock().unwrap();
        if let Some(n) = m.get_mut(&self.id) {
            *n -= 1;
            if *n == 0 {
                m.remove(&self.id);
            }
        }
    }
}

/// Release a global stream slot when its reservation is dropped.
struct StreamGuard(Arc<IncomingState>);

impl StreamGuard {
    fn take(dentro: &Arc<IncomingState>) -> Option<StreamGuard> {
        let previous = dentro.stream_vivi.fetch_add(1, Ordering::SeqCst);
        if previous >= dentro.limiti.stream_max as u64 {
            dentro.stream_vivi.fetch_sub(1, Ordering::SeqCst);
            return None;
        }
        Some(StreamGuard(dentro.clone()))
    }
}

impl Drop for StreamGuard {
    fn drop(&mut self) {
        self.0.stream_vivi.fetch_sub(1, Ordering::SeqCst);
    }
}

impl ProtocolHandler for Accetta {
    async fn accept(&self, conn: Connection) -> Result<(), AcceptError> {
        // Admit the authenticated peer and reserve its connection slot before reading any headers.
        let da = conn.remote_id();
        if !self.dentro.expected_sender(&da) {
            self.dentro.rifiutate.fetch_add(1, Ordering::Relaxed);
            warn!(peer = %da, "bridge: closed connection from an unexpected peer");
            conn.close(VarInt::from_u32(REJECT_SENDER), b"unexpected peer");
            return Ok(());
        }
        let Some(_posto) = SenderGuard::take(&self.dentro, da) else {
            self.dentro.rifiutate.fetch_add(1, Ordering::Relaxed);
            warn!(peer = %da, "bridge: closed excess connection from peer");
            conn.close(VarInt::from_u32(REJECT_LIMIT), b"too many connections");
            return Ok(());
        };
        self.dentro.accettate.fetch_add(1, Ordering::Relaxed);
        // A disconnected peer ends stream acceptance and releases its connection reservation.
        while let Ok((send, recv)) = conn.accept_bi().await {
            let (dentro, conn) = (self.dentro.clone(), conn.clone());
            tokio::spawn(async move { serve_incoming(dentro, conn, send, recv).await });
        }
        self.dentro.accettate.fetch_sub(1, Ordering::Relaxed);
        Ok(())
    }
}

async fn serve_incoming(dentro: Arc<IncomingState>, conn: Connection, send: SendStream, mut recv: RecvStream) {
    let rifiuta = |mut send: SendStream, mut recv: RecvStream, code: u32| {
        let _ = send.reset(VarInt::from_u32(code));
        let _ = recv.stop(VarInt::from_u32(code));
    };
    // Hold the global stream reservation for the full forwarding lifetime.
    let Some(_posto) = StreamGuard::take(&dentro) else {
        warn!(peer = %conn.remote_id(), "bridge: rejected excess concurrent stream");
        return rifiuta(send, recv, REJECT_LIMIT);
    };
    let k = match tokio::time::timeout(HEADER_TIMEOUT, read_header(&mut recv)).await {
        Ok(Ok(k)) => k,
        Ok(Err(e)) => {
            warn!(peer = %conn.remote_id(), "bridge: rejected stream: {e:#}");
            return rifiuta(send, recv, REJECT_HEADER);
        }
        Err(_) => return rifiuta(send, recv, REJECT_HEADER),
    };
    let da = conn.remote_id();
    let trovata = dentro.entrate.lock().unwrap().get(&k).map(|e| (e.destinazione, e.conti.clone(), e.chiuso.clone(), e.attesi.clone()));
    let Some((destinazione, conti, chiuso, attesi)) = trovata else {
        // Reject the stream so the dialing stage can retry after the route is registered.
        debug!(chain = %k.chain_id, epoch = k.epoch, direction = k.verso.name(), "bridge: stream for an unknown chain");
        return rifiuta(send, recv, REJECT_CHAIN);
    };
    if !attesi.is_empty() && !attesi.contains(&da) {
        warn!(peer = %da, chain = %k.chain_id, "bridge: rejected stream from a peer outside the chain");
        return rifiuta(send, recv, REJECT_SENDER);
    }
    let tcp = match TcpStream::connect(destinazione).await {
        Ok(t) => t,
        Err(_) => return rifiuta(send, recv, REJECT_STAGE), // If the stage listener is unavailable, reset the stream so the caller can retry.
    };
    conti.stream.fetch_add(1, Ordering::Relaxed);
    // Monitor the route for the lifetime of this forward to capture the first direct path.
    let guarda = async {
        loop {
            conti.subscribe(&conn, &k, false);
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    };
    let r = tokio::select! {
        r = forward(tcp, send, recv, &conti, chiuso) => r,
        _ = guarda => unreachable!(),
    };
    if let Err(e) = r {
        e.write(&k, "incoming");
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use iroh_relay::server::{RelayConfig, Server, ServerConfig};
    use std::net::Ipv4Addr;

    /// Start a fixture relay on loopback so the bridge tests need no external service.
    pub async fn relay_di_prova() -> (Server, String) {
        let mut config = ServerConfig::default();
        config.relay = Some(RelayConfig::new((Ipv4Addr::LOCALHOST, 0)));
        let server = Server::spawn(config).await.expect("relay");
        let url = format!("http://{}", server.http_addr().expect("HTTP port"));
        (server, url)
    }

    pub async fn ponte_di_prova(url: &str) -> Arc<Bridge> {
        let p = Bridge::new(url, SecretKey::from_bytes(&rand::random())).await.unwrap();
        assert!(p.in_linea(Duration::from_secs(20)).await, "test relay unreachable");
        // The handshake advertises the endpoint ID in this textual form.
        let id = p.id();
        assert!(id.len() == 64 && id.bytes().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()), "unexpected iroh ID: {id}");
        p
    }

    /// Start a TCP echo fixture and return its listen address.
    async fn eco() -> SocketAddr {
        let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let a = l.local_addr().unwrap();
        tokio::spawn(async move {
            while let Ok((mut s, _)) = l.accept().await {
                tokio::spawn(async move {
                    let (mut r, mut w) = s.split();
                    let _ = tokio::io::copy(&mut r, &mut w).await;
                    let _ = w.shutdown().await;
                });
            }
        });
        a
    }

    #[test]
    fn key_is_created_and_read_back() {
        let f = std::env::temp_dir().join(format!("publicompute-ponte-chiave-{}", std::process::id()));
        let _ = std::fs::remove_file(&f);
        let (a, b) = (load_key(&f).unwrap(), load_key(&f).unwrap());
        assert_eq!(a.public(), b.public());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(std::fs::metadata(&f).unwrap().permissions().mode() & 0o777, 0o600);
        }
        std::fs::write(&f, "xx").unwrap();
        assert!(load_key(&f).is_err());
        std::fs::remove_file(&f).unwrap();
    }

    #[tokio::test]
    async fn header_round_trip() {
        let k = LinkKey { chain_id: "catena-è".into(), epoch: 7, verso: Direction::Token };
        let b = header(&k);
        assert_eq!(b[0], WIRE_VERSION);
        assert_eq!(b.len(), 12 + k.chain_id.len());
    }

    /// Verify unchanged bytes in both directions, correct byte counts and propagation of a half-close.
    #[tokio::test]
    async fn bridge_echo_and_clean_shutdown() {
        let (server, url) = relay_di_prova().await;
        let (a, b) = (ponte_di_prova(&url).await, ponte_di_prova(&url).await);
        let dest = eco().await;
        let mut gb = b.guard();
        b.incoming(&mut gb, "c", 1, Direction::Forward, dest);
        let mut ga = a.guard();
        let locale = a.output(&mut ga, "c", 1, Direction::Forward, &b.id(), &url).await.unwrap();

        // Concurrent TCP connections on the same link need separate streams.
        let mut s1 = TcpStream::connect(locale).await.unwrap();
        let mut s2 = TcpStream::connect(locale).await.unwrap();
        let dati: Vec<u8> = (0..300_000u32).map(|i| (i * 31 % 251) as u8).collect();
        s1.write_all(&dati).await.unwrap();
        s2.write_all(b"ciao").await.unwrap();
        let mut piccolo = [0u8; 4];
        s2.read_exact(&mut piccolo).await.unwrap();
        assert_eq!(&piccolo, b"ciao");
        let mut indietro = vec![0u8; dati.len()];
        s1.read_exact(&mut indietro).await.unwrap();
        assert_eq!(indietro, dati);
        // The echo peer closes on EOF. Read its EOF to verify that the FIN propagated.
        s1.shutdown().await.unwrap();
        let mut resto = Vec::new();
        tokio::time::timeout(Duration::from_secs(5), s1.read_to_end(&mut resto)).await.expect("expected EOF within 5 s").unwrap();
        assert!(resto.is_empty());

        let m = a.metrics();
        assert_eq!(m.len(), 1);
        assert!(m[0].in_uscita && m[0].stream_aperti == 2, "{m:?}");
        assert_eq!(m[0].byte_inviati, dati.len() as u64 + 4);
        assert_eq!(m[0].byte_ricevuti, dati.len() as u64 + 4);
        assert!(["direct", "mixed", "relay"].contains(&m[0].percorso.as_str()), "{m:?}");
        let mb = b.metrics();
        assert_eq!(mb[0].byte_ricevuti, dati.len() as u64 + 4, "{mb:?}");
        // Dropping the outgoing guard must release its peer connection.
        assert_eq!(b.connessioni_aperte(), 1);
        drop(ga);
        tokio::time::timeout(Duration::from_secs(5), async {
            while b.connessioni_aperte() != 0 {
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("dropping the guard should close the iroh connection");
        drop(gb);
        a.close().await;
        b.close().await;
        server.shutdown().await.ok();
    }

    /// Send four bytes to the loopback listener and report whether they echo back.
    async fn eco_ok(locale: SocketAddr) -> bool {
        let Ok(mut s) = TcpStream::connect(locale).await else { return false };
        let _ = s.write_all(b"ping").await;
        let mut buf = [0u8; 4];
        matches!(tokio::time::timeout(Duration::from_secs(8), s.read_exact(&mut buf)).await, Ok(Ok(_)))
    }

    async fn ponte_con_limiti(url: &str, l: BridgeLimits) -> Arc<Bridge> {
        let p = Bridge::with_limits(url, SecretKey::from_bytes(&rand::random()), l).await.unwrap();
        assert!(p.in_linea(Duration::from_secs(20)).await, "test relay unreachable");
        p
    }

    /// Reject unexpected peers before reading headers when all registered
    /// chains use allowlists. A chain with an empty list permits admission.
    #[tokio::test]
    async fn unexpected_sender_rejected_before_header() {
        let (server, url) = relay_di_prova().await;
        let (a, b, c) = (ponte_di_prova(&url).await, ponte_di_prova(&url).await, ponte_di_prova(&url).await);
        let dest = eco().await;
        let mut gb = b.guard();
        b.incoming_with(&mut gb, "c", 1, Direction::Forward, dest, &[a.id()]).unwrap();
        let (mut ga, mut gc) = (a.guard(), c.guard());
        let da_a = a.output(&mut ga, "c", 1, Direction::Forward, &b.id(), &url).await.unwrap();
        let da_c = c.output(&mut gc, "c", 1, Direction::Forward, &b.id(), &url).await.unwrap();
        assert!(eco_ok(da_a).await, "expected peer should pass");
        assert!(!eco_ok(da_c).await, "unexpected peer should be rejected");
        assert_eq!(b.connessioni_rifiutate(), 1);
        // An empty allowlist permits the connection.
        let mut gb2 = b.guard();
        b.incoming_with(&mut gb2, "d", 1, Direction::Forward, dest, &[]).unwrap();
        let da_c2 = c.output(&mut gc, "d", 1, Direction::Forward, &b.id(), &url).await.unwrap();
        assert!(eco_ok(da_c2).await, "an empty peer list allows any peer");
        // The same peer can still fail authorization for a route with an allowlist.
        assert!(!eco_ok(da_c).await, "a chain with a peer list still rejects unexpected peers");
        assert!(b.incoming_with(&mut gb2, "e", 1, Direction::Forward, dest, &["non-un-id".into()]).is_err());
        a.close().await;
        b.close().await;
        c.close().await;
        server.shutdown().await.ok();
    }

    /// Enforce the connection limit separately for each sender.
    #[tokio::test]
    async fn per_sender_connection_limit() {
        let (server, url) = relay_di_prova().await;
        let a = ponte_di_prova(&url).await;
        let b = ponte_con_limiti(&url, BridgeLimits { stream_max: 64, connections_per_peer: 2 }).await;
        let dest = eco().await;
        let mut gb = b.guard();
        let mut ga = a.guard();
        let mut locali = Vec::new();
        for name in ["c1", "c2", "c3"] {
            b.incoming_with(&mut gb, name, 1, Direction::Forward, dest, &[a.id()]).unwrap();
            locali.push(a.output(&mut ga, name, 1, Direction::Forward, &b.id(), &url).await.unwrap());
        }
        // Keep both TCP connections open to hold their corresponding peer reservations.
        let mut aperti = Vec::new();
        for l in &locali[..2] {
            let mut s = TcpStream::connect(*l).await.unwrap();
            s.write_all(b"ping").await.unwrap();
            let mut buf = [0u8; 4];
            s.read_exact(&mut buf).await.unwrap();
            aperti.push(s);
        }
        assert!(!eco_ok(locali[2]).await, "reject a third connection from the same peer");
        assert_eq!(b.connessioni_rifiutate(), 1);
        drop(aperti);
        a.close().await;
        b.close().await;
        server.shutdown().await.ok();
    }

    /// Enforce the global stream limit across connections.
    #[tokio::test]
    async fn global_stream_limit() {
        let (server, url) = relay_di_prova().await;
        let a = ponte_di_prova(&url).await;
        let b = ponte_con_limiti(&url, BridgeLimits { stream_max: 2, connections_per_peer: 4 }).await;
        let dest = eco().await;
        let mut gb = b.guard();
        b.incoming_with(&mut gb, "c", 1, Direction::Forward, dest, &[a.id()]).unwrap();
        let mut ga = a.guard();
        let locale = a.output(&mut ga, "c", 1, Direction::Forward, &b.id(), &url).await.unwrap();
        let mut aperti = Vec::new();
        for _ in 0..2 {
            let mut s = TcpStream::connect(locale).await.unwrap();
            s.write_all(b"ping").await.unwrap();
            let mut buf = [0u8; 4];
            s.read_exact(&mut buf).await.unwrap();
            aperti.push(s);
        }
        assert!(!eco_ok(locale).await, "reject a third concurrent stream");
        // Releasing a reservation makes capacity available again.
        drop(aperti.pop());
        tokio::time::sleep(Duration::from_millis(500)).await;
        assert!(eco_ok(locale).await, "accept a new stream when a slot is free");
        a.close().await;
        b.close().await;
        server.shutdown().await.ok();
    }

    /// Close the local connection promptly when the route or listener is not
    /// ready. A retry through the same port can succeed once both are ready.
    #[tokio::test]
    async fn destination_unavailable_then_recovers() {
        let (server, url) = relay_di_prova().await;
        let (a, b) = (ponte_di_prova(&url).await, ponte_di_prova(&url).await);
        let mut ga = a.guard();
        let locale = a.output(&mut ga, "c", 2, Direction::Forward, &b.id(), "").await.unwrap();
        let chiusa = |locale| async move {
            let mut s = TcpStream::connect(locale).await.unwrap();
            let _ = s.write_all(b"x").await;
            let mut buf = [0u8; 8];
            // EOF and reset both tell the stage to retry.
            matches!(tokio::time::timeout(Duration::from_secs(8), s.read(&mut buf)).await.expect("close within 8 s"), Ok(0) | Err(_))
        };
        assert!(chiusa(locale).await, "unknown chain");
        let mut gb = b.guard();
        let morta = {
            let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            l.local_addr().unwrap()
        };
        b.incoming(&mut gb, "c", 1, Direction::Forward, eco().await);
        assert!(chiusa(locale).await, "wrong epoch");
        b.incoming(&mut gb, "c", 2, Direction::Token, eco().await);
        assert!(chiusa(locale).await, "wrong direction");
        b.incoming(&mut gb, "c", 2, Direction::Forward, morta);
        assert!(chiusa(locale).await, "stage is not listening");
        b.incoming(&mut gb, "c", 2, Direction::Forward, eco().await);
        let mut s = TcpStream::connect(locale).await.unwrap();
        s.write_all(b"ora si").await.unwrap();
        let mut buf = [0u8; 6];
        s.read_exact(&mut buf).await.unwrap();
        assert_eq!(&buf, b"ora si");
        // Reject a relay mismatch before opening the local listener.
        assert!(a.output(&mut ga, "c", 2, Direction::Token, &b.id(), "http://127.0.0.1:1").await.is_err());
        assert!(a.output(&mut ga, "c", 2, Direction::Token, "non-un-id", "").await.is_err());
        a.close().await;
        b.close().await;
        server.shutdown().await.ok();
    }

    /// Closing either endpoint or dropping the assignment guard must tear down
    /// both local TCP connections.
    #[tokio::test]
    async fn bridge_disconnect_closes_both_ends() {
        let (server, url) = relay_di_prova().await;
        // Keep the peer socket open so the fixture can observe teardown.
        async fn muto() -> (SocketAddr, tokio::sync::mpsc::UnboundedReceiver<()>) {
            let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let a = l.local_addr().unwrap();
            let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
            tokio::spawn(async move {
                while let Ok((mut s, _)) = l.accept().await {
                    let tx = tx.clone();
                    tokio::spawn(async move {
                        let mut buf = [0u8; 64];
                        loop {
                            match s.read(&mut buf).await {
                                Ok(0) | Err(_) => break,
                                Ok(n) => {
                                    let _ = s.write_all(&buf[..n]).await;
                                }
                            }
                        }
                        let _ = tx.send(());
                    });
                }
            });
            (a, rx)
        }
        for caso in ["initiator endpoint", "acceptor endpoint", "initiator guard", "acceptor guard"] {
            let (a, b) = (ponte_di_prova(&url).await, ponte_di_prova(&url).await);
            let (dest, mut visto) = muto().await;
            let mut gb = b.guard();
            b.incoming(&mut gb, "c", 1, Direction::Forward, dest);
            let mut ga = a.guard();
            let locale = a.output(&mut ga, "c", 1, Direction::Forward, &b.id(), &url).await.unwrap();
            let mut s = TcpStream::connect(locale).await.unwrap();
            s.write_all(b"ping").await.unwrap();
            let mut buf = [0u8; 4];
            s.read_exact(&mut buf).await.unwrap();
            let t0 = Instant::now();
            match caso {
                "initiator endpoint" => a.close().await,
                "acceptor endpoint" => b.close().await,
                "initiator guard" => drop(ga),
                _ => drop(gb),
            }
            let mut resto = [0u8; 8];
            let r = tokio::time::timeout(Duration::from_secs(10), s.read(&mut resto)).await.unwrap_or_else(|_| panic!("{caso}: near side is still open"));
            assert!(matches!(r, Ok(0) | Err(_)), "{caso}: {r:?}");
            tokio::time::timeout(Duration::from_secs(10), visto.recv()).await.unwrap_or_else(|_| panic!("{caso}: far side is still open"));
            println!("{caso}: both sides closed after {:.3} s", t0.elapsed().as_secs_f64());
            a.close().await;
            b.close().await;
        }
        server.shutdown().await.ok();
    }
}
