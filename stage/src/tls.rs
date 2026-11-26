//! Stage links with credentials use mutual TLS 1.3. A signed chain
//! capability binds each certificate identity to a stage position and an
//! expiry. Check the immediate predecessor on forward and control streams,
//! and the tail on the token stream to the head. The connector also checks
//! the server at the destination position. Expiry closes active links,
//! and an updated capability can renew them.
//!
//! Pump threads expose a local TCP socket to the stage's existing stream
//! code. Serialize access to TLS state without holding its lock during
//! network writes so traffic can flow in both directions.

use prost::Message;
use publicompute_proto::node::{CapStage, ChainCapability, SignedCapability};
use publicompute_proto::stage::Stream;
use ring::signature::{UnparsedPublicKey, ED25519};
use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::client::verify_server_cert_signed_by_trust_anchor;
use rustls::crypto::{verify_tls12_signature, verify_tls13_signature, CryptoProvider, WebPkiSupportedAlgorithms};
use rustls::pki_types::pem::PemObject;
use rustls::pki_types::{CertificateDer, PrivateKeyDer, ServerName, UnixTime};
use rustls::server::WebPkiClientVerifier;
use rustls::{ClientConfig, ClientConnection, Connection, DigitallySignedStruct, RootCertStore, ServerConfig, ServerConnection, SignatureScheme};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock, Weak};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Keep the URI SAN prefix in sync with the coordinator's certificate issuer.
const NODE_URI_PREFIX: &str = "publicompute:node:";
/// ALPN for inter-stage streams, to distinguish them from other TLS services using the same certificate.
const ALPN: &[u8] = b"publicompute-stage/0";
/// How often the stage checks the capability file and expiration.
const CAPABILITY_POLL_INTERVAL: Duration = Duration::from_millis(250);

// Signed chain capabilities.

/// Verified capability, mapping nodes to slots.
#[derive(Clone, Debug)]
pub struct Capability {
    pub chain_id: String,
    pub epoch: u64,
    pub expires_unix: i64,
    /// Stages in order. `stadi[0]` is the head.
    pub stages: Vec<CapStage>,
}

fn unix_now() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0)
}

impl Capability {
    /// Check the Ed25519 signature, expiry and slot consistency. `pubblica` is the coordinator's 32-byte key.
    pub fn verifica(pubblica: &[u8], firmata: &[u8], now: i64) -> Result<Capability, String> {
        let sc = SignedCapability::decode(firmata).map_err(|e| format!("capability illeggibile: {e}"))?;
        UnparsedPublicKey::new(&ED25519, pubblica)
            .verify(&sc.body, &sc.signature)
            .map_err(|_| "invalid capability signature".to_string())?;
        let c = ChainCapability::decode(sc.body.as_slice()).map_err(|e| format!("capability illeggibile: {e}"))?;
        if now >= c.expires_unix {
            return Err(format!("capability expired ({} s ago)", now - c.expires_unix));
        }
        if c.stages.len() < 2 {
            return Err("capability needs at least two stages".into());
        }
        let mut visti = std::collections::HashSet::new();
        for (i, s) in c.stages.iter().enumerate() {
            if s.index as usize != i + 1 {
                return Err(format!("capability slots out of order (slot {} at index {})", s.index, i + 1));
            }
            if s.node_id.is_empty() || !visti.insert(s.node_id.clone()) {
                return Err("capability has an empty or duplicate node_id".into());
            }
        }
        Ok(Capability { chain_id: c.chain_id, epoch: c.assign_epoch, expires_unix: c.expires_unix, stages: c.stages })
    }

    fn posto_di(&self, node_id: &str) -> Option<usize> {
        self.stages.iter().position(|s| s.node_id == node_id)
    }
}

/// Public key from a file: 32 raw bytes or 64 hex digits.
pub fn read_public_key(path: &Path) -> Result<Vec<u8>, String> {
    let b = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if b.len() == 32 {
        return Ok(b);
    }
    let t = String::from_utf8_lossy(&b);
    let t = t.trim();
    if t.len() == 64 && t.chars().all(|c| c.is_ascii_hexdigit()) {
        return Ok((0..32).map(|i| u8::from_str_radix(&t[2 * i..2 * i + 2], 16).unwrap()).collect());
    }
    Err(format!("{}: expected a 32-byte Ed25519 key (raw or hex)", path.display()))
}

// Extract the node identity from a certificate.

/// One DER element: (tag, content, remainder). Only one-byte tags, enough for the X.509 fields used here.
fn tlv(b: &[u8]) -> Option<(u8, &[u8], &[u8])> {
    let tag = *b.first()?;
    let l0 = *b.get(1)? as usize;
    let (len, off) = if l0 < 0x80 {
        (l0, 2)
    } else {
        let n = l0 & 0x7f;
        if n == 0 || n > 4 {
            return None;
        }
        let mut v = 0usize;
        for i in 0..n {
            v = (v << 8) | *b.get(2 + i)? as usize;
        }
        (v, 2 + n)
    };
    let done = off.checked_add(len)?;
    Some((tag, b.get(off..done)?, &b[done..]))
}

/// node_id (`publicompute:node:<id>`) from the URI SAN of a DER certificate. Verify the chain first.
pub fn node_id_from_cert(der: &[u8]) -> Result<String, String> {
    const OID_SAN: &[u8] = &[0x55, 0x1d, 0x11];
    let err = || "certificate URI SAN has no node_id".to_string();
    let (_, cert, _) = tlv(der).ok_or_else(err)?;
    let (_, tbs, _) = tlv(cert).ok_or_else(err)?;
    let mut resto = tbs;
    while let Some((tag, cont, r)) = tlv(resto) {
        resto = r;
        if tag != 0xa3 {
            continue;
        }
        let (_, estensioni, _) = tlv(cont).ok_or_else(err)?;
        let mut es = estensioni;
        while let Some((_, ext, r)) = tlv(es) {
            es = r;
            let (t, oid, mut rest) = tlv(ext).ok_or_else(err)?;
            if t != 0x06 || oid != OID_SAN {
                continue;
            }
            // [BOOLEAN critical] OCTET STRING { SEQUENCE { GeneralName... } }
            if let Some((0x01, _, r2)) = tlv(rest) {
                rest = r2;
            }
            let (_, ottetti, _) = tlv(rest).ok_or_else(err)?;
            let (_, nomi, _) = tlv(ottetti).ok_or_else(err)?;
            let mut n = nomi;
            while let Some((tn, val, r3)) = tlv(n) {
                n = r3;
                if tn == 0x86 {
                    if let Some(id) = std::str::from_utf8(val).ok().and_then(|s| s.strip_prefix(NODE_URI_PREFIX)) {
                        if !id.is_empty() {
                            return Ok(id.to_string());
                        }
                    }
                }
            }
        }
    }
    Err(err())
}

// Link authentication and authorization.

/// Stage files and position (from `--tls-cert`, `--tls-key`, `--tls-ca`, `--capability`, `--capability-pub`).
#[derive(Clone, Debug)]
pub struct TlsOptions {
    pub cert: PathBuf,
    pub key: PathBuf,
    pub ca: PathBuf,
    pub capability: PathBuf,
    pub capability_pub: PathBuf,
    /// This stage's position in the chain, starting at 1 (the head).
    pub index: u32,
    /// Coordinator time minus local time, in seconds. Apply the offset only
    /// to capability expiry. Certificate verification uses local time.
    pub clock_offset_s: i64,
}

/// TLS connection error. `Identity`: peer differs from the capability, so retrying is pointless.
/// `Rete`: link failed, so retry because a proxy may still be starting up.
#[derive(Debug)]
pub enum TlsError {
    Identity(String),
    Rete(String),
}

impl std::fmt::Display for TlsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TlsError::Identity(s) | TlsError::Rete(s) => f.write_str(s),
        }
    }
}

pub struct StageSecurity {
    mio_id: String,
    index: u32,
    cap: RwLock<Capability>,
    cap_path: PathBuf,
    cap_pub: Vec<u8>,
    /// Clock offset from the coordinator in seconds. "Coordinator" time is `ora_unix() + scarto_s`.
    clock_offset_s: i64,
    cap_visto: Mutex<Option<(SystemTime, u64)>>,
    chain: Vec<CertificateDer<'static>>,
    key: PrivateKeyDer<'static>,
    roots: Arc<RootCertStore>,
    provider: Arc<CryptoProvider>,
    server: Arc<ServerConfig>,
    prossimo: AtomicU64,
    /// Open connections (TLS socket, local side), tracked so they can close at expiry.
    registro: Mutex<HashMap<u64, (TcpStream, TcpStream)>>,
    /// Expiry already reported and connections closed.
    scaduta_segnalata: Mutex<bool>,
}

impl StageSecurity {
    /// Load certificate, key, CA and capability, and check that this node occupies slot `index` in the capability.
    pub fn new(o: &TlsOptions) -> Result<Arc<StageSecurity>, String> {
        let provider = Arc::new(rustls::crypto::ring::default_provider());
        let chain: Vec<CertificateDer<'static>> = CertificateDer::pem_file_iter(&o.cert)
            .map_err(|e| format!("--tls-cert {}: {e}", o.cert.display()))?
            .collect::<Result<_, _>>()
            .map_err(|e| format!("--tls-cert {}: {e}", o.cert.display()))?;
        if chain.is_empty() {
            return Err(format!("--tls-cert {}: no certificate", o.cert.display()));
        }
        let key = PrivateKeyDer::from_pem_file(&o.key).map_err(|e| format!("--tls-key {}: {e}", o.key.display()))?;
        let mut roots = RootCertStore::empty();
        for c in CertificateDer::pem_file_iter(&o.ca).map_err(|e| format!("--tls-ca {}: {e}", o.ca.display()))? {
            roots.add(c.map_err(|e| format!("--tls-ca {}: {e}", o.ca.display()))?).map_err(|e| format!("--tls-ca: {e}"))?;
        }
        if roots.is_empty() {
            return Err(format!("--tls-ca {}: no CA", o.ca.display()));
        }
        let roots = Arc::new(roots);
        let mio_id = node_id_from_cert(&chain[0])?;
        let cap_pub = read_public_key(&o.capability_pub)?;
        let (cap, visto) = Self::read_capability(&o.capability, &cap_pub, o.clock_offset_s)?;
        if o.index == 0 || o.index as usize > cap.stages.len() {
            return Err(format!("position {} is outside the chain of {} stages", o.index, cap.stages.len()));
        }
        let al_posto = &cap.stages[o.index as usize - 1].node_id;
        if *al_posto != mio_id {
            return Err(format!("capability assigns {al_posto} to slot {}, but this node is {mio_id}", o.index));
        }
        let verificatore = WebPkiClientVerifier::builder_with_provider(roots.clone(), provider.clone())
            .build()
            .map_err(|e| format!("client verifier: {e}"))?;
        let mut server = ServerConfig::builder_with_provider(provider.clone())
            .with_protocol_versions(&[&rustls::version::TLS13])
            .map_err(|e| e.to_string())?
            .with_client_cert_verifier(verificatore)
            .with_single_cert(chain.clone(), key.clone_key())
            .map_err(|e| format!("certificate or key: {e}"))?;
        server.alpn_protocols = vec![ALPN.to_vec()];
        server.send_tls13_tickets = 0;
        let s = Arc::new(StageSecurity {
            mio_id,
            index: o.index,
            cap: RwLock::new(cap),
            cap_path: o.capability.clone(),
            cap_pub,
            clock_offset_s: o.clock_offset_s,
            cap_visto: Mutex::new(Some(visto)),
            chain,
            key,
            roots,
            provider,
            server: Arc::new(server),
            prossimo: AtomicU64::new(1),
            registro: Mutex::new(HashMap::new()),
            scaduta_segnalata: Mutex::new(false),
        });
        let debole = Arc::downgrade(&s);
        std::thread::Builder::new().name("tls-expiry-monitor".into()).spawn(move || monitor_expiry(debole)).map_err(|e| e.to_string())?;
        Ok(s)
    }

    fn read_capability(path: &Path, cap_pub: &[u8], clock_offset_s: i64) -> Result<(Capability, (SystemTime, u64)), String> {
        let meta = std::fs::metadata(path).map_err(|e| format!("--capability {}: {e}", path.display()))?;
        let b = std::fs::read(path).map_err(|e| format!("--capability {}: {e}", path.display()))?;
        let c = Capability::verifica(cap_pub, &b, unix_now() + clock_offset_s)?;
        Ok((c, (meta.modified().unwrap_or(UNIX_EPOCH), meta.len())))
    }

    /// Coordinator Unix time, estimated as local time plus the offset.
    pub fn now(&self) -> i64 {
        unix_now() + self.clock_offset_s
    }

    pub fn mio_id(&self) -> &str {
        &self.mio_id
    }

    pub fn capability(&self) -> Capability {
        self.cap.read().unwrap().clone()
    }

    pub fn scaduta(&self) -> bool {
        self.now() >= self.cap.read().unwrap().expires_unix
    }

    /// Can a peer with this identity open this stream to us?
    /// - `Token`: only the head receives these, and only from the tail.
    /// - `Prefill`, `Generation`, `Control`: only from our immediate predecessor in the capability.
    pub fn admit(&self, stream: Stream, peer: Option<&str>) -> Result<(), String> {
        let peer = peer.ok_or("peer has no identity")?;
        let cap = self.cap.read().unwrap();
        if self.now() >= cap.expires_unix {
            return Err("capability expired".into());
        }
        let n = cap.stages.len();
        let i = self.index as usize;
        let Some(slot) = cap.posto_di(peer) else {
            return Err(format!("{peer} is missing from chain {} capability", cap.chain_id));
        };
        match stream {
            Stream::Token => {
                if i != 1 {
                    return Err("only the head accepts Token streams".into());
                }
                if slot != n - 1 {
                    return Err(format!("Token stream from {peer} (slot {}), which is not the tail (slot {n})", slot + 1));
                }
            }
            Stream::Prefill | Stream::Generation | Stream::Control => {
                if i == 1 {
                    return Err(format!("head accepts only Token streams, got {}", crate::network::stream_name(stream)));
                }
                if slot != i - 2 {
                    return Err(format!(
                        "{} stream from {peer} (slot {}), which is not the preceding stage (slot {})",
                        crate::network::stream_name(stream),
                        slot + 1,
                        i - 1
                    ));
                }
            }
            Stream::Unspecified => return Err("unspecified stream".into()),
        }
        Ok(())
    }

    /// Expected server for `stream`: the next stage, or the head for the tail's `Token` stream.
    pub fn expected_outgoing_peer(&self, stream: Stream) -> Result<String, String> {
        let cap = self.cap.read().unwrap();
        let (n, i) = (cap.stages.len(), self.index as usize);
        match stream {
            Stream::Token if i == n => Ok(cap.stages[0].node_id.clone()),
            Stream::Prefill | Stream::Generation | Stream::Control if i < n => Ok(cap.stages[i].node_id.clone()),
            altro => Err(format!("stage at slot {i} of {n} does not open {} streams", crate::network::stream_name(altro))),
        }
    }

    /// On accept, perform TLS handshake and return the local socket and the peer node_id. TLS checks the peer's
    /// certificate against the CA. When `Hello` arrives, `admit` checks that the capability puts the
    /// peer in the right slot for the stream.
    pub fn wrap_server(self: &Arc<Self>, tcp: TcpStream, scadenza: Duration) -> Result<(TcpStream, String), TlsError> {
        if self.scaduta() {
            return Err(TlsError::Identity("capability expired".into()));
        }
        let conn = ServerConnection::new(self.server.clone()).map_err(|e| TlsError::Rete(e.to_string()))?;
        let (app, peer) = self.pompa(Connection::Server(conn), tcp, scadenza)?;
        Ok((app, peer))
    }

    /// On connect, the server node_id must match `expected` or the handshake fails.
    pub fn wrap_client(self: &Arc<Self>, tcp: TcpStream, expected: &str, scadenza: Duration) -> Result<TcpStream, TlsError> {
        if self.scaduta() {
            return Err(TlsError::Identity("capability expired".into()));
        }
        let verifica = VerifyNode { roots: self.roots.clone(), expected: expected.to_string(), algs: self.provider.signature_verification_algorithms };
        let mut cfg = ClientConfig::builder_with_provider(self.provider.clone())
            .with_protocol_versions(&[&rustls::version::TLS13])
            .map_err(|e| TlsError::Rete(e.to_string()))?
            .dangerous()
            .with_custom_certificate_verifier(Arc::new(verifica))
            .with_client_auth_cert(self.chain.clone(), self.key.clone_key())
            .map_err(|e| TlsError::Rete(e.to_string()))?;
        cfg.alpn_protocols = vec![ALPN.to_vec()];
        cfg.enable_sni = false;
        cfg.resumption = rustls::client::Resumption::disabled();
        let name = ServerName::try_from("stage.publicompute").map_err(|e| TlsError::Rete(e.to_string()))?;
        let conn = ClientConnection::new(Arc::new(cfg), name).map_err(|e| TlsError::Rete(e.to_string()))?;
        let (app, peer) = self.pompa(Connection::Client(conn), tcp, scadenza)?;
        if peer != expected {
            // The verifier should catch this, but verify that the identity is present.
            return Err(TlsError::Identity(format!("server is {peer}, expected {expected}")));
        }
        Ok(app)
    }

    /// Blocking handshake (sockets use `scadenza`) and pump thread startup. Return the local socket and the
    /// peer's node_id.
    fn pompa(self: &Arc<Self>, mut conn: Connection, mut raw: TcpStream, scadenza: Duration) -> Result<(TcpStream, String), TlsError> {
        let network = |e: std::io::Error| {
            // A certificate that fails verification is an identity error, so a retry will not help.
            if e.kind() == std::io::ErrorKind::InvalidData {
                TlsError::Identity(format!("handshake TLS: {e}"))
            } else {
                TlsError::Rete(format!("handshake TLS: {e}"))
            }
        };
        let _ = raw.set_nodelay(true);
        // Use a deadline for the whole handshake so partial records cannot keep extending it.
        let done = std::time::Instant::now() + scadenza;
        let _ = raw.set_write_timeout(Some(scadenza));
        while conn.wants_write() || conn.is_handshaking() {
            while conn.wants_write() {
                conn.write_tls(&mut raw).map_err(network)?;
            }
            if !conn.is_handshaking() {
                break;
            }
            let resto = done.saturating_duration_since(std::time::Instant::now());
            if resto.is_zero() {
                return Err(network(std::io::Error::new(std::io::ErrorKind::TimedOut, "deadline exceeded")));
            }
            let _ = raw.set_read_timeout(Some(resto));
            if conn.wants_read() {
                match conn.read_tls(&mut raw).map_err(network)? {
                    0 => return Err(network(std::io::ErrorKind::UnexpectedEof.into())),
                    _ => {
                        conn.process_new_packets()
                            .map_err(|e| network(std::io::Error::new(std::io::ErrorKind::InvalidData, e)))?;
                    }
                }
            }
        }
        let _ = raw.set_read_timeout(None);
        let peer = conn
            .peer_certificates()
            .and_then(|c| c.first())
            .ok_or_else(|| TlsError::Identity("peer did not send a certificate".into()))
            .and_then(|c| node_id_from_cert(c).map_err(TlsError::Identity))?;

        let (app, inner) = loopback_pair().map_err(|e| TlsError::Rete(format!("socket locale: {e}")))?;
        let id = self.prossimo.fetch_add(1, Ordering::SeqCst);
        let clone = |s: &TcpStream| s.try_clone().map_err(|e| TlsError::Rete(e.to_string()));
        self.registro.lock().unwrap().insert(id, (clone(&raw)?, clone(&inner)?));
        let state = Arc::new(Mutex::new(conn));
        let vivi = Arc::new(AtomicUsize::new(2));
        // network → app
        {
            let (state, vivi, me) = (state.clone(), vivi.clone(), self.clone());
            let (mut r, mut w) = (clone(&raw)?, clone(&inner)?);
            let (raw_k, inner_k) = (clone(&raw)?, clone(&inner)?);
            std::thread::Builder::new()
                .name("tls-rete-app".into())
                .spawn(move || {
                    pump_reads(&state, &mut r, &mut w, &raw_k, &inner_k);
                    me.finito(id, &vivi);
                })
                .map_err(|e| TlsError::Rete(e.to_string()))?;
        }
        // app → network
        {
            let (state, vivi, me) = (state, vivi, self.clone());
            let (mut r, mut w) = (clone(&inner)?, raw);
            let (raw_k, inner_k) = (clone(&w)?, clone(&inner)?);
            std::thread::Builder::new()
                .name("tls-app-rete".into())
                .spawn(move || {
                    pump_writes(&state, &mut r, &mut w, &raw_k, &inner_k);
                    me.finito(id, &vivi);
                })
                .map_err(|e| TlsError::Rete(e.to_string()))?;
        }
        Ok((app, peer))
    }

    fn finito(&self, id: u64, vivi: &AtomicUsize) {
        if vivi.fetch_sub(1, Ordering::SeqCst) == 1 {
            self.registro.lock().unwrap().remove(&id);
        }
    }

    /// Close all open connections.
    fn close_all(&self) -> usize {
        let tutte: Vec<_> = self.registro.lock().unwrap().drain().collect();
        for (_, (raw, inner)) in &tutte {
            let _ = raw.shutdown(Shutdown::Both);
            let _ = inner.shutdown(Shutdown::Both);
        }
        tutte.len()
    }

    /// Reload a changed capability file. Accept only a valid capability for the same chain, with an equal or
    /// newer epoch and this node in the same slot, so a new assignment cannot replace a renewal.
    fn ricarica(&self) {
        let Ok(meta) = std::fs::metadata(&self.cap_path) else { return };
        let visto = (meta.modified().unwrap_or(UNIX_EPOCH), meta.len());
        {
            let mut v = self.cap_visto.lock().unwrap();
            if *v == Some(visto) {
                return;
            }
            *v = Some(visto);
        }
        match Self::read_capability(&self.cap_path, &self.cap_pub, self.clock_offset_s) {
            Ok((new, _)) => {
                let mut cap = self.cap.write().unwrap();
                let al_posto = new.stages.get(self.index as usize - 1).map(|s| s.node_id.as_str());
                if new.chain_id == cap.chain_id && new.epoch >= cap.epoch && al_posto == Some(self.mio_id.as_str()) {
                    eprintln!("[tls] renewed capability for chain {} epoch {}; expires in {} s", new.chain_id, new.epoch, new.expires_unix - self.now());
                    *cap = new;
                    *self.scaduta_segnalata.lock().unwrap() = false;
                } else {
                    eprintln!("[tls] ignored replacement capability with a different chain, epoch or slot");
                }
            }
            Err(e) => eprintln!("[tls] rejected replacement capability: {e}"),
        }
    }
}

/// Check the capability file and expiry every `CAPABILITY_POLL_INTERVAL`. Close all connections at expiry.
fn monitor_expiry(s: Weak<StageSecurity>) {
    loop {
        std::thread::sleep(CAPABILITY_POLL_INTERVAL);
        let Some(s) = s.upgrade() else { return };
        s.ricarica();
        if s.scaduta() {
            let mut seg = s.scaduta_segnalata.lock().unwrap();
            if !*seg {
                *seg = true;
                let n = s.close_all();
                eprintln!("[tls] capability expired; closed {n} connections and rejecting new ones until renewal");
            } else {
                // Connections opened between expiry and this check
                s.close_all();
            }
        }
    }
}

/// Connected local socket pair (application side, pump side). Drop any other local
/// connection that reaches the ephemeral port before ours.
fn loopback_pair() -> std::io::Result<(TcpStream, TcpStream)> {
    let l = TcpListener::bind(("127.0.0.1", 0))?;
    let addr = l.local_addr()?;
    let app = TcpStream::connect(addr)?;
    let mio = app.local_addr()?;
    loop {
        let (inner, da) = l.accept()?;
        if da == mio {
            let _ = app.set_nodelay(true);
            let _ = inner.set_nodelay(true);
            return Ok((app, inner));
        }
    }
}

/// Tear down both ends on error or completion.
fn shut_down(raw: &TcpStream, inner: &TcpStream) {
    let _ = raw.shutdown(Shutdown::Both);
    let _ = inner.shutdown(Shutdown::Both);
}

fn pump_reads(state: &Mutex<Connection>, raw: &mut TcpStream, a_app: &mut TcpStream, raw_k: &TcpStream, inner_k: &TcpStream) {
    let mut buf = vec![0u8; 64 << 10];
    let mut chiaro = vec![0u8; 64 << 10];
    loop {
        let n = match raw.read(&mut buf) {
            Ok(0) => {
                // Peer shutdown, with or without close_notify, looks like end of stream to the app.
                let _ = a_app.shutdown(Shutdown::Write);
                return;
            }
            Ok(n) => n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => {
                shut_down(raw_k, inner_k);
                return;
            }
        };
        let mut da_app = Vec::new();
        let mut allerta = Vec::new();
        let (mut error, mut done) = (None::<String>, false);
        {
            let mut g = state.lock().unwrap();
            let mut cur = &buf[..n];
            while !cur.is_empty() && error.is_none() && !done {
                match g.read_tls(&mut cur) {
                    Ok(0) => break,
                    Ok(_) => {}
                    Err(e) => {
                        error = Some(e.to_string());
                        break;
                    }
                }
                if let Err(e) = g.process_new_packets() {
                    error = Some(e.to_string());
                    break;
                }
                loop {
                    match g.reader().read(&mut chiaro) {
                        Ok(0) => {
                            done = true; // close_notify
                            break;
                        }
                        Ok(k) => da_app.extend_from_slice(&chiaro[..k]),
                        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                        Err(e) => {
                            error = Some(e.to_string());
                            break;
                        }
                    }
                }
            }
            if error.is_some() {
                // Send any rustls alert to the peer so it has a reason for the close.
                while g.wants_write() {
                    if g.write_tls(&mut allerta).is_err() {
                        break;
                    }
                }
            }
        }
        if !da_app.is_empty() && a_app.write_all(&da_app).is_err() {
            shut_down(raw_k, inner_k);
            return;
        }
        if let Some(e) = error {
            eprintln!("[tls] connection closed: {e}");
            let mut r = raw_k;
            let _ = r.write_all(&allerta);
            shut_down(raw_k, inner_k);
            return;
        }
        if done {
            let _ = a_app.shutdown(Shutdown::Write);
            return;
        }
    }
}

fn pump_writes(state: &Mutex<Connection>, da_app: &mut TcpStream, raw: &mut TcpStream, raw_k: &TcpStream, inner_k: &TcpStream) {
    let mut buf = vec![0u8; 64 << 10];
    loop {
        let n = match da_app.read(&mut buf) {
            Ok(n) => n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => 0,
        };
        let mut v = Vec::with_capacity(n + 64);
        {
            let mut g = state.lock().unwrap();
            if n == 0 {
                g.send_close_notify();
            } else if g.writer().write_all(&buf[..n]).is_err() {
                drop(g);
                shut_down(raw_k, inner_k);
                return;
            }
            while g.wants_write() {
                if g.write_tls(&mut v).is_err() {
                    break;
                }
            }
        }
        // Write to the network without holding the lock, so the reader can keep delivering data.
        if raw.write_all(&v).is_err() {
            shut_down(raw_k, inner_k);
            return;
        }
        if n == 0 {
            let _ = raw.shutdown(Shutdown::Write);
            return;
        }
    }
}

// Verify the remote node when opening a link.

/// Verify the server certificate chain to the coordinator CA **and** check the expected node_id.
/// Node certificates have only a URI SAN, so the placeholder DNS name is irrelevant.
#[derive(Debug)]
struct VerifyNode {
    roots: Arc<RootCertStore>,
    expected: String,
    algs: WebPkiSupportedAlgorithms,
}

impl ServerCertVerifier for VerifyNode {
    fn verify_server_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp: &[u8],
        now: UnixTime,
    ) -> Result<ServerCertVerified, rustls::Error> {
        let parsed = rustls::server::ParsedCertificate::try_from(end_entity)?;
        verify_server_cert_signed_by_trust_anchor(&parsed, &self.roots, intermediates, now, self.algs.all)?;
        let id = node_id_from_cert(end_entity).map_err(rustls::Error::General)?;
        if id != self.expected {
            return Err(rustls::Error::General(format!("server is {id}, expected {} from the capability", self.expected)));
        }
        Ok(ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(&self, m: &[u8], cert: &CertificateDer<'_>, dss: &DigitallySignedStruct) -> Result<HandshakeSignatureValid, rustls::Error> {
        verify_tls12_signature(m, cert, dss, &self.algs)
    }

    fn verify_tls13_signature(&self, m: &[u8], cert: &CertificateDer<'_>, dss: &DigitallySignedStruct) -> Result<HandshakeSignatureValid, rustls::Error> {
        verify_tls13_signature(m, cert, dss, &self.algs)
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.algs.supported_schemes()
    }
}

#[cfg(test)]
pub(crate) mod fixtures {
    //! Generate a test CA, node credentials, and signed assignments.
    use super::*;
    use rcgen::{BasicConstraints, CertificateParams, DnType, ExtendedKeyUsagePurpose, IsCa, Issuer, KeyPair, KeyUsagePurpose, SanType};
    use ring::signature::{Ed25519KeyPair, KeyPair as _};

    static N: AtomicU64 = AtomicU64::new(0);

    pub struct Mondo {
        pub dir: PathBuf,
        pub ca_pem: PathBuf,
        issuer: Issuer<'static, KeyPair>,
        firma: Ed25519KeyPair,
        pub chiave_pubblica: PathBuf,
        pub cap_pub: Vec<u8>,
    }

    impl Mondo {
        pub fn new() -> Mondo {
            let dir = std::env::temp_dir().join(format!("publicompute-tls-{}-{}", std::process::id(), N.fetch_add(1, Ordering::SeqCst)));
            std::fs::create_dir_all(&dir).unwrap();
            let ca_key = KeyPair::generate().unwrap();
            let mut p = CertificateParams::new(Vec::<String>::new()).unwrap();
            p.is_ca = IsCa::Ca(BasicConstraints::Constrained(0));
            p.key_usages = vec![KeyUsagePurpose::KeyCertSign, KeyUsagePurpose::CrlSign];
            p.distinguished_name.push(DnType::CommonName, "Test CA");
            let ca = p.self_signed(&ca_key).unwrap();
            let ca_pem = dir.join("ca.pem");
            std::fs::write(&ca_pem, ca.pem()).unwrap();
            let rng = ring::rand::SystemRandom::new();
            let pk8 = Ed25519KeyPair::generate_pkcs8(&rng).unwrap();
            let firma = Ed25519KeyPair::from_pkcs8(pk8.as_ref()).unwrap();
            let cap_pub = firma.public_key().as_ref().to_vec();
            let chiave_pubblica = dir.join("capability.pub");
            std::fs::write(&chiave_pubblica, cap_pub.iter().map(|b| format!("{b:02x}")).collect::<String>()).unwrap();
            Mondo { dir, ca_pem, issuer: Issuer::new(p, ca_key), firma, chiave_pubblica, cap_pub }
        }

        /// Write credentials with the same URI identity format used by the verifier.
        pub fn nodo(&self, id: &str) -> (PathBuf, PathBuf) {
            let k = KeyPair::generate().unwrap();
            let mut p = CertificateParams::new(Vec::<String>::new()).unwrap();
            p.subject_alt_names = vec![SanType::URI(format!("{NODE_URI_PREFIX}{id}").try_into().unwrap())];
            p.extended_key_usages = vec![ExtendedKeyUsagePurpose::ClientAuth, ExtendedKeyUsagePurpose::ServerAuth];
            p.key_usages = vec![KeyUsagePurpose::DigitalSignature];
            let c = p.signed_by(&k, &self.issuer).unwrap();
            let (cp, kp) = (self.dir.join(format!("{id}.cert.pem")), self.dir.join(format!("{id}.key.pem")));
            std::fs::write(&cp, c.pem()).unwrap();
            std::fs::write(&kp, k.serialize_pem()).unwrap();
            (cp, kp)
        }

        pub fn firma(&self, chain: &str, epoch: u64, expires_unix: i64, nodi: &[&str]) -> Vec<u8> {
            let c = ChainCapability {
                chain_id: chain.into(),
                assign_epoch: epoch,
                expires_unix: expires_unix,
                stages: nodi
                    .iter()
                    .enumerate()
                    .map(|(i, n)| CapStage { index: i as u32 + 1, node_id: (*n).into(), model_sha256: vec![9; 32], layer_start: i as u32 * 10, layer_end: if i + 1 == nodi.len() { -1 } else { (i as i32 + 1) * 10 } })
                    .collect(),
            };
            let body = c.encode_to_vec();
            SignedCapability { signature: self.firma.sign(&body).as_ref().to_vec(), body, key_id: "test".into() }.encode_to_vec()
        }

        /// Stage security: write the capability file and load all credentials.
        pub fn stage(&self, id: &str, index: u32, capability: &[u8]) -> Result<(Arc<StageSecurity>, PathBuf), String> {
            self.stadio_con_scarto(id, index, capability, 0)
        }

        /// Like `stage`, with a clock offset from the coordinator.
        pub fn stadio_con_scarto(&self, id: &str, index: u32, capability: &[u8], clock_offset_s: i64) -> Result<(Arc<StageSecurity>, PathBuf), String> {
            let (cert, key) = self.nodo(id);
            let cap = self.dir.join(format!("{id}-{index}.cap"));
            std::fs::write(&cap, capability).unwrap();
            let s = StageSecurity::new(&TlsOptions { cert, key, ca: self.ca_pem.clone(), capability: cap.clone(), capability_pub: self.chiave_pubblica.clone(), index, clock_offset_s })?;
            Ok((s, cap))
        }
    }

    impl Drop for Mondo {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    pub fn fra(seconds: i64) -> i64 {
        unix_now() + seconds
    }
}

#[cfg(test)]
mod test {
    use super::fixtures::*;
    use super::*;
    use crate::network::{self, accept_with, ack_ok, connect_with, msg, write_sync};
    use publicompute_proto::stage::{message::Body, Confirmation};
    use std::sync::mpsc::{channel, Receiver};
    use std::time::Instant;

    const NODI: [&str; 3] = ["n_a", "n_b", "n_c"];

    /// Listen like a stage: ack, then forward each received message with its stream and peer to
    /// the channel. Reply to each control message with `Confirmation`.
    fn listen(sic: &Arc<StageSecurity>) -> (String, Receiver<(Stream, Option<String>, publicompute_proto::stage::Message)>) {
        let l = network::listen("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap().to_string();
        let (tx, rx) = channel();
        let s = sic.clone();
        std::thread::spawn(move || {
            accept_with(Some(s), l, move |f, mut lett, mut w| {
                if ack_ok(&mut w).is_err() {
                    return;
                }
                let peer = lett.peer.clone();
                while let Ok(Some(m)) = lett.read() {
                    if f == Stream::Control {
                        let _ = write_sync(&mut w, &msg(Body::Confirmation(Confirmation { conversation_id: 99, ..Default::default() })));
                    }
                    let _ = tx.send((f, peer.clone(), m));
                }
            });
        });
        (addr, rx)
    }

    fn conf(id: u64, error: String) -> publicompute_proto::stage::Message {
        msg(Body::Confirmation(Confirmation { conversation_id: id, error, ..Default::default() }))
    }

    const ATTESA: Duration = Duration::from_secs(4);

    /// Test chain n_a (head) → n_b → n_c, valid for an hour.
    struct Chain {
        mondo: Mondo,
        a: Arc<StageSecurity>,
        b: Arc<StageSecurity>,
        c: Arc<StageSecurity>,
    }

    fn catena() -> Chain {
        let mondo = Mondo::new();
        let cap = mondo.firma("c_1", 1, fra(3600), &NODI);
        let a = mondo.stage("n_a", 1, &cap).unwrap().0;
        let b = mondo.stage("n_b", 2, &cap).unwrap().0;
        let c = mondo.stage("n_c", 3, &cap).unwrap().0;
        Chain { mondo, a, b, c }
    }

    /// Partial TLS records must still time out at the handshake deadline.
    #[test]
    fn trickled_tls_handshake_hits_deadline() {
        use std::io::Write;
        let k = catena();
        let l = network::listen("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        let goccia = std::thread::spawn(move || {
            let mut c = std::net::TcpStream::connect(addr).unwrap();
            // TLS handshake record header with a 512-byte body, then one byte every 200 ms.
            let _ = c.write_all(&[0x16, 0x03, 0x01, 0x02, 0x00]);
            for _ in 0..15 {
                std::thread::sleep(Duration::from_millis(200));
                if c.write_all(&[0]).is_err() {
                    break;
                }
            }
        });
        let (s, _) = l.accept().unwrap();
        let t0 = Instant::now();
        let r = k.b.wrap_server(s, Duration::from_millis(700));
        assert!(r.is_err(), "doveva fallire");
        assert!(t0.elapsed() < Duration::from_millis(1600), "expired after {:?}", t0.elapsed());
        goccia.join().unwrap();
    }

    #[test]
    fn reads_node_id_from_uri_san() {
        let m = Mondo::new();
        let (cert, _) = m.nodo("n_xyz");
        let der = CertificateDer::pem_file_iter(&cert).unwrap().next().unwrap().unwrap();
        assert_eq!(node_id_from_cert(&der).unwrap(), "n_xyz");
        let ca = CertificateDer::pem_file_iter(&m.ca_pem).unwrap().next().unwrap().unwrap();
        assert!(node_id_from_cert(&ca).is_err(), "CA must have no node_id");
        assert!(node_id_from_cert(&[1, 2, 3]).is_err());
    }

    #[test]
    fn tls_carries_traffic_in_both_directions() {
        let k = catena();
        // Large frames exercise both TLS pumps across multiple records.
        let (addr_b, rx_b) = listen(&k.b);
        let (mut l_ctl, mut w_ctl) = connect_with(Some(&k.a), &addr_b, Stream::Control, ATTESA, 1).unwrap();
        let (_l_pre, mut w_pre) = connect_with(Some(&k.a), &addr_b, Stream::Prefill, ATTESA, 1).unwrap();
        let grande = "x".repeat(5 << 20);
        write_sync(&mut w_pre, &conf(1, grande.clone())).unwrap();
        write_sync(&mut w_ctl, &conf(2, "ciao".into())).unwrap();
        let mut visti = HashMap::new();
        for _ in 0..2 {
            let (f, peer, m) = rx_b.recv_timeout(Duration::from_secs(10)).unwrap();
            visti.insert(f, (peer, m));
        }
        let (peer, m) = &visti[&Stream::Prefill];
        assert_eq!(peer.as_deref(), Some("n_a"), "peer should match the certificate node_id");
        assert!(matches!(&m.body, Some(Body::Confirmation(c)) if c.error == grande));
        assert!(visti.contains_key(&Stream::Control));
        match l_ctl.read().unwrap().unwrap().body {
            Some(Body::Confirmation(c)) => assert_eq!(c.conversation_id, 99),
            altro => panic!("unexpected reply: {altro:?}"),
        }
        let (addr_a, rx_a) = listen(&k.a);
        let (_l, mut w_tok) = connect_with(Some(&k.c), &addr_a, Stream::Token, ATTESA, 1).unwrap();
        write_sync(&mut w_tok, &conf(3, "token".into())).unwrap();
        let (f, peer, _) = rx_a.recv_timeout(Duration::from_secs(5)).unwrap();
        assert_eq!((f, peer.as_deref()), (Stream::Token, Some("n_c")));
    }

    #[test]
    fn valid_certificate_outside_capability_is_rejected() {
        let k = catena();
        let (addr_b, rx_b) = listen(&k.b);
        // The intruder has a certificate from the same CA and a capability for another chain that names n_b.
        let altra = k.mondo.firma("c_altra", 1, fra(3600), &["n_x", "n_b"]);
        let (x, _) = k.mondo.stage("n_x", 1, &altra).unwrap();
        let e = connect_with(Some(&x), &addr_b, Stream::Prefill, ATTESA, 1).err().expect("unauthorized peer was accepted");
        assert!(e.contains("rejected") && e.contains("is missing from chain"), "{e}");
        assert!(rx_b.recv_timeout(Duration::from_millis(300)).is_err(), "stage accepted data from an unauthorized peer");
        assert!(connect_with(Some(&k.a), &addr_b, Stream::Prefill, ATTESA, 1).is_ok());
    }

    #[test]
    fn control_allowed_only_from_direct_predecessor() {
        let k = catena();
        let (addr_b, _rx) = listen(&k.b);
        // The successor is in the chain but cannot replace the predecessor's control stream.
        let altra = k.mondo.firma("c_altra", 1, fra(3600), &["n_c", "n_b"]);
        let (c2, _) = k.mondo.stage("n_c", 1, &altra).unwrap();
        for f in [Stream::Control, Stream::Prefill, Stream::Generation] {
            let e = connect_with(Some(&c2), &addr_b, f, ATTESA, 1).err().expect("should reject the connection");
            assert!(e.contains("preceding stage"), "{f:?}: {e}");
        }
        // Only the head accepts Token, even when sent by the tail.
        let (_, e) = (0, connect_with(Some(&k.c), &addr_b, Stream::Token, ATTESA, 1).err().expect("should reject the connection"));
        assert!(e.contains("server is n_b"), "tail should expect the head: {e}");
    }

    #[test]
    fn tokens_allowed_only_from_tail() {
        let k = catena();
        let (addr_a, rx_a) = listen(&k.a);
        // Membership alone does not let an intermediate send tokens to the head.
        let altra = k.mondo.firma("c_altra", 1, fra(3600), &["n_a", "n_b"]);
        let (b2, _) = k.mondo.stage("n_b", 2, &altra).unwrap();
        let e = connect_with(Some(&b2), &addr_a, Stream::Token, ATTESA, 1).err().expect("should reject the connection");
        assert!(e.contains("is not the tail"), "{e}");
        let e = connect_with(Some(&b2), &addr_a, Stream::Control, ATTESA, 1).err().expect("should reject the connection");
        assert!(e.contains("does not open") || e.contains("only Token"), "{e}");
        assert!(rx_a.recv_timeout(Duration::from_millis(300)).is_err());
        assert!(connect_with(Some(&k.c), &addr_a, Stream::Token, ATTESA, 1).is_ok());
    }

    #[test]
    fn verifies_server_identity_against_capability() {
        let k = catena();
        // Head expects n_b but gets n_c at the address with a valid cert. The client rejects the handshake.
        let (addr_c, _rx) = listen(&k.c);
        let e = connect_with(Some(&k.a), &addr_c, Stream::Prefill, ATTESA, 1).err().expect("should reject the connection");
        assert!(e.contains("server is n_c"), "{e}");
    }

    #[test]
    fn rejects_plaintext_client() {
        let k = catena();
        let (addr_b, rx_b) = listen(&k.b);
        let mut s = TcpStream::connect(&addr_b).unwrap();
        let _ = write_sync(&mut s, &msg(Body::Hello(publicompute_proto::stage::Hello {
            protocol_major: network::WIRE_VERSION.0,
            protocol_minor: network::WIRE_VERSION.1,
            stream: Stream::Control as i32,
            epoch: 1,
            mtp: false,
        })));
        s.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        let mut b = [0u8; 64];
        // Server closes or sends a TLS alert. A plaintext HelloAck would be invalid.
        let n = s.read(&mut b).unwrap_or(0);
        assert!(n == 0 || b[0] == 0x15, "plaintext reply: {:?}", &b[..n]);
        assert!(rx_b.recv_timeout(Duration::from_millis(300)).is_err());
    }

    #[test]
    fn rejects_tampered_capability_and_wrong_slot() {
        let m = Mondo::new();
        let cap = m.firma("c_1", 1, fra(3600), &NODI);
        assert!(Capability::verifica(&m.cap_pub, &cap, unix_now()).is_ok());
        let mut brutta = cap.clone();
        brutta[8] ^= 1;
        assert!(Capability::verifica(&m.cap_pub, &brutta, unix_now()).is_err());
        let altro = Mondo::new();
        assert!(Capability::verifica(&altro.cap_pub, &cap, unix_now()).is_err());
        assert!(Capability::verifica(&m.cap_pub, &cap, fra(7200)).unwrap_err().contains("expired"));
        // The node refuses to start with a modified capability or a slot it does not own.
        assert!(m.stage("n_a", 1, &brutta).is_err());
        assert!(m.stage("n_a", 2, &cap).err().unwrap().contains("assigns n_b to slot 2"));
        let doppia = m.firma("c_1", 1, fra(3600), &["n_a", "n_a"]);
        assert!(Capability::verifica(&m.cap_pub, &doppia, unix_now()).is_err());
    }

    /// Capability expiry uses coordinator time, with a signed offset from local time.
    #[test]
    fn expiry_checks_use_clock_offset() {
        let m = Mondo::new();
        // With a positive offset of 900 seconds, the coordinator has already reached expiry.
        let cap = m.firma("c_1", 1, fra(600), &NODI);
        let e = m.stadio_con_scarto("n_a", 1, &cap, 900).err().expect("accepted an expired capability");
        assert!(e.contains("expired"), "{e}");
        let (s, _) = m.stadio_con_scarto("n_a", 1, &cap, 0).unwrap();
        assert!(!s.scaduta());
        // A negative offset moves the coordinator's clock before expiry.
        assert!(m.stadio_con_scarto("n_a", 1, &cap, -900).is_ok());
        let vecchia = m.firma("c_1", 1, fra(-120), &NODI);
        let (s, _) = m.stadio_con_scarto("n_a", 1, &vecchia, -300).unwrap();
        assert!(!s.scaduta());
        assert!(m.stadio_con_scarto("n_a", 1, &vecchia, 0).is_err());
    }

    /// Check that the watcher applies the offset to an open connection's deadline.
    #[test]
    fn expiry_monitor_uses_clock_offset() {
        let m = Mondo::new();
        // Expires in 1 h. Offset +3598 s means 2 s until expiry for the coordinator.
        let cap = m.firma("c_1", 1, fra(3600), &NODI);
        let a = m.stadio_con_scarto("n_a", 1, &cap, 3597).unwrap().0;
        let b = m.stadio_con_scarto("n_b", 2, &cap, 3597).unwrap().0;
        let (addr_b, _rx) = listen(&b);
        let (mut l, _w) = connect_with(Some(&a), &addr_b, Stream::Control, ATTESA, 1).unwrap();
        let t0 = Instant::now();
        let r = l.read();
        assert!(matches!(r, Ok(None) | Err(_)), "{r:?}");
        assert!(a.scaduta() && t0.elapsed() < Duration::from_secs(8), "{:?}", t0.elapsed());
    }

    #[test]
    fn expiry_closes_connections() {
        let m = Mondo::new();
        let cap = m.firma("c_1", 1, fra(2), &NODI);
        let a = m.stage("n_a", 1, &cap).unwrap().0;
        let b = m.stage("n_b", 2, &cap).unwrap().0;
        let (addr_b, _rx) = listen(&b);
        let (mut l, _w) = connect_with(Some(&a), &addr_b, Stream::Control, ATTESA, 1).unwrap();
        std::thread::sleep(Duration::from_millis(300));
        assert!(!a.scaduta());
        let t0 = Instant::now();
        let r = l.read();
        assert!(matches!(r, Ok(None) | Err(_)), "{r:?}");
        assert!(t0.elapsed() < Duration::from_secs(5), "connection closed late: {:?}", t0.elapsed());
        std::thread::sleep(Duration::from_millis(500));
        let e = connect_with(Some(&a), &addr_b, Stream::Prefill, ATTESA, 1).err().expect("accepted a connection with an expired capability");
        assert!(e.contains("expired") || e.contains("rejected"), "{e}");
    }

    #[test]
    fn renewal_in_same_epoch_preserves_connections() {
        let m = Mondo::new();
        let cap = m.firma("c_1", 1, fra(2), &NODI);
        let (a, _) = m.stage("n_a", 1, &cap).unwrap();
        let (b, file_b) = m.stage("n_b", 2, &cap).unwrap();
        let (addr_b, rx_b) = listen(&b);
        let (_l, mut w) = connect_with(Some(&a), &addr_b, Stream::Prefill, ATTESA, 1).unwrap();
        // A renewal keeps the same chain and epoch and extends its expiry.
        let new = m.firma("c_1", 1, fra(3600), &NODI);
        std::fs::write(&file_b, &new).unwrap();
        // Renew both endpoints because either can close the connection at expiry.
        let file_a = m.dir.join("n_a-1.cap");
        std::fs::write(&file_a, &new).unwrap();
        std::thread::sleep(Duration::from_millis(3200));
        assert!(!a.scaduta() && !b.scaduta());
        write_sync(&mut w, &conf(5, "still connected".into())).unwrap();
        let (_, _, msg) = rx_b.recv_timeout(Duration::from_secs(5)).expect("connection closed despite capability renewal");
        assert!(matches!(msg.body, Some(Body::Confirmation(c)) if c.conversation_id == 5));
        // Reject a renewal for a different chain or with a lower epoch.
        let altra = m.firma("c_altra", 9, fra(7200), &NODI);
        std::fs::write(&file_b, &altra).unwrap();
        std::thread::sleep(Duration::from_millis(700));
        assert_eq!(b.capability().chain_id, "c_1");
    }
}
