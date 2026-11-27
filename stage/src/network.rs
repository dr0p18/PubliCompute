//! Each stage stream has a TCP connection and uses a varint length prefix
//! for one protobuf message. Reader and writer threads feed a stage's
//! compute loop. On startup, an intermediate sends HelloAck only when its
//! successor is ready, so the acknowledgement propagates readiness upstream.

use crate::tls::{TlsError, StageSecurity};
use prost::{encoding, Message};
use publicompute_proto::stage::{self as pb, message::Body};
use publicompute_proto::wire::{encode_frame, FrameError, MAX_FRAME_SIZE};
use std::io::{BufReader, Read, Write};
use std::net::{TcpListener, TcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

const INGRESS_EVENT_OVERHEAD: usize = 256;
const STEP_ROW_OVERHEAD: usize = 64;
pub const CONTROL_INGRESS_BYTES: usize = 3 * MAX_FRAME_SIZE + INGRESS_EVENT_OVERHEAD;

// Peers must agree on which slot IDs exist in each group.
pub const WIRE_VERSION: (u32, u32) = (2, 0);

// Time limits cover connection setup, stalled writes, and optional idle
// reads. A silent stream may be healthy while no work is active, so idle
// reads have no deadline by default. The writer byte cap bounds queued
// output and uses the write timeout to bound waiting for room.
static T_HELLO_S: AtomicU64 = AtomicU64::new(10);
static T_LETTURA_S: AtomicU64 = AtomicU64::new(0);
static T_SCRITTURA_S: AtomicU64 = AtomicU64::new(120);
static CODA_MAX_MIB: AtomicUsize = AtomicUsize::new(256);

/// When configured, every link uses mutual TLS and checks the signed
/// chain assignment. Otherwise connections use plaintext.
static SICUREZZA: std::sync::OnceLock<Arc<StageSecurity>> = std::sync::OnceLock::new();

/// Enable TLS and capabilities for all process streams. Call once at startup.
pub fn set_security(s: Arc<StageSecurity>) {
    let _ = SICUREZZA.set(s);
}

fn security() -> Option<Arc<StageSecurity>> {
    SICUREZZA.get().cloned()
}

/// Maximum accepted connections still waiting for `Hello`. Close any excess immediately.
pub const HELLO_PENDENTI_MAX: usize = 64;

/// Reject an oversized hello from its length header, before allocating its payload.
pub const HELLO_FRAME_MAX: usize = 64 << 10;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinkTimeouts {
    pub hello_s: u64,
    pub lettura_s: u64,
    pub scrittura_s: u64,
    pub coda_max_mib: usize,
}

impl Default for LinkTimeouts {
    fn default() -> Self {
        Self { hello_s: 10, lettura_s: 0, scrittura_s: 120, coda_max_mib: 256 }
    }
}

/// Set timeouts for sockets created hereafter. Call once at startup.
pub fn set_link_timeouts(s: LinkTimeouts) {
    T_HELLO_S.store(s.hello_s.max(1), Ordering::Relaxed);
    T_LETTURA_S.store(s.lettura_s, Ordering::Relaxed);
    T_SCRITTURA_S.store(s.scrittura_s, Ordering::Relaxed);
    CODA_MAX_MIB.store(s.coda_max_mib.max(1), Ordering::Relaxed);
}

fn seconds(a: &AtomicU64) -> Option<Duration> {
    match a.load(Ordering::Relaxed) {
        0 => None,
        n => Some(Duration::from_secs(n)),
    }
}

pub fn stream_name(f: pb::Stream) -> &'static str {
    match f {
        pb::Stream::Prefill => "prefill",
        pb::Stream::Generation => "generation",
        pb::Stream::Control => "control",
        pb::Stream::Token => "token",
        pb::Stream::Unspecified => "?",
    }
}

pub fn msg(c: Body) -> pb::Message {
    pb::Message { body: Some(c) }
}

/// Keep resets, opens, prefill blocks, and closes on one stream. TCP preserves
/// order only within one connection, so a close sent elsewhere could arrive
/// after the open that reuses its slot.
pub fn on_generation_stream(c: &Body) -> bool {
    matches!(c, Body::Step(_))
}

/// Frame one message. An oversized message must be rejected because it is
/// sent in a single frame.
pub fn encode(m: &pb::Message) -> Result<Vec<u8>, String> {
    let mut v = Vec::with_capacity(m.encoded_len_hint());
    match encode_frame(m, &mut v) {
        Ok(()) => Ok(v),
        Err(FrameError::TooLarge(n)) => Err(format!(
            "message is {n} bytes; frame limit is {MAX_FRAME_SIZE} bytes (16 MiB). Reduce the prefill block size (--block)."
        )),
        Err(e) => Err(e.to_string()),
    }
}

trait Hint {
    fn encoded_len_hint(&self) -> usize;
}
impl Hint for pb::Message {
    fn encoded_len_hint(&self) -> usize {
        use prost::Message;
        self.encoded_len() + 8
    }
}

pub struct Reader {
    stream: BufReader<TcpStream>,
    /// Epoch declared in `Hello` (only readers created by `read_hello`, otherwise 0).
    pub epoca_hello: u32,
    /// The tail's token stream advertises draft capability in its hello.
    pub mtp_hello: bool,
    /// Peer node_id from its TLS certificate (accepted TLS connections only, `None` in plaintext).
    pub peer: Option<String>,
}

impl Reader {
    pub fn new(stream: TcpStream) -> Self {
        let _ = stream.set_read_timeout(seconds(&T_LETTURA_S));
        Self { stream: BufReader::new(stream), epoca_hello: 0, mtp_hello: false, peer: None }
    }

    fn set_read_deadline(&self, d: Option<Duration>) {
        let _ = self.stream.get_ref().set_read_timeout(d);
    }

    /// Read one message within a whole-frame deadline. Check its declared
    /// length before reading or allocating the payload.
    fn leggi_limitato(&mut self, max: usize, done: Instant) -> Result<Option<pb::Message>, String> {
        let frame = self.read_frame(max, Some(done), None, |_| Ok(()))?;
        frame.map(|(bytes, ())| pb::Message::decode(bytes.as_slice())).transpose().map_err(|e| e.to_string())
    }

    /// Next message. `Ok(None)` if the peer closed between messages.
    pub fn read(&mut self) -> Result<Option<pb::Message>, String> {
        let frame = self.read_frame(MAX_FRAME_SIZE, None, None, |_| Ok(()))?;
        frame.map(|(bytes, ())| pb::Message::decode(bytes.as_slice())).transpose().map_err(|e| e.to_string())
    }

    pub fn read_admitted(&mut self, budget: &Arc<IngressBudget>, wait: Duration, stream: pb::Stream, n_batch: usize) -> Result<Option<(pb::Message, IngressPermit)>, String> {
        let frame = self.read_frame(MAX_FRAME_SIZE, None, Some(Duration::from_secs(30)), |len| {
            // The payload, decoded byte fields (including Vec growth), and repeated fields
            // can coexist. Each StepRow uses at least two wire bytes, each token at least one.
            let repeated = match stream {
                pb::Stream::Generation => (len / 2).min(n_batch).max(4) * (2 * size_of::<pb::StepRow>() + STEP_ROW_OVERHEAD),
                pb::Stream::Prefill => len.min(n_batch).max(4) * 2 * size_of::<i32>(),
                _ => 0,
            };
            budget.reserve(3 * len + repeated + INGRESS_EVENT_OVERHEAD, wait)
        });
        self.set_read_deadline(seconds(&T_LETTURA_S));
        let Some((payload, mut permit)) = frame? else { return Ok(None) };
        check_ingress(&payload, stream, n_batch).map_err(|e| e.to_string())?;
        let message = pb::Message::decode(bytes::Bytes::from(payload)).map_err(|e| e.to_string())?;
        let resident = INGRESS_EVENT_OVERHEAD + match message.body.as_ref() {
            Some(Body::Step(step)) => step.state.capacity() + step.rows.capacity() * size_of::<pb::StepRow>() + STEP_ROW_OVERHEAD * step.rows.len(),
            Some(Body::Block(block)) => block.state.capacity() + block.token.capacity() * size_of::<i32>(),
            Some(Body::Confirmation(c)) => c.error.capacity() + c.timings_json.capacity(),
            _ => 0,
        };
        permit.release(permit.bytes - resident);
        Ok(Some((message, permit)))
    }

    fn read_chunk(&mut self, bytes: &mut [u8], deadline: Option<Instant>) -> Result<usize, String> {
        if let Some(deadline) = deadline {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err("deadline exceeded".into());
            }
            self.set_read_deadline(Some(remaining));
        }
        self.stream.read(bytes).map_err(|error| error.to_string())
    }

    fn read_frame<P>(&mut self, max: usize, deadline: Option<Instant>, body_timeout: Option<Duration>, reserve: impl FnOnce(usize) -> Result<P, String>) -> Result<Option<(Vec<u8>, P)>, String> {
        let mut prefix = [0u8; 10];
        for index in 0..prefix.len() {
            if self.read_chunk(&mut prefix[index..=index], deadline)? == 0 {
                return if index == 0 { Ok(None) } else { Err("connection closed in frame length".into()) };
            }
            if prefix[index] & 0x80 != 0 {
                continue;
            }
            let len = prost::encoding::decode_varint(&mut &prefix[..=index]).map_err(|_| "invalid frame length")?;
            if len > max as u64 {
                return Err(format!("frame is {len} bytes; limit for this message is {max} bytes"));
            }
            let permit = reserve(len as usize)?;
            let mut payload = vec![0; len as usize];
            let deadline = deadline.or_else(|| body_timeout.map(|timeout| Instant::now() + timeout));
            let mut offset = 0;
            while offset < payload.len() {
                let count = self.read_chunk(&mut payload[offset..], deadline)?;
                if count == 0 {
                    return Err("connection closed mid-frame".into());
                }
                offset += count;
            }
            return Ok(Some((payload, permit)));
        }
        Err("invalid frame length".into())
    }
}

/// Inspect repeated field counts before prost allocates their vectors.
fn check_ingress(mut payload: &[u8], stream: pb::Stream, n_batch: usize) -> Result<(), prost::DecodeError> {
    use encoding::{decode_key, decode_varint, skip_field, DecodeContext, WireType};

    fn delimited<'a>(wire: WireType, bytes: &mut &'a [u8]) -> Result<&'a [u8], prost::DecodeError> {
        encoding::check_wire_type(WireType::LengthDelimited, wire)?;
        let len = decode_varint(bytes)?;
        if len > bytes.len() as u64 {
            return Err(prost::DecodeError::new("truncated stage field"));
        }
        let (field, rest) = bytes.split_at(len as usize);
        *bytes = rest;
        Ok(field)
    }

    let mut count = 0;
    let mut seen = false;
    while !payload.is_empty() {
        let (tag, wire) = decode_key(&mut payload)?;
        match (stream, tag) {
            (pb::Stream::Prefill, 3 | 4 | 8 | 9) |
            (pb::Stream::Generation, 5) |
            (pb::Stream::Control, 7) => seen = true,
            (_, 1..=9) => return Err(prost::DecodeError::new("unexpected message on stage stream")),
            _ => {}
        }
        if !matches!((stream, tag), (pb::Stream::Generation, 5) | (pb::Stream::Prefill, 4)) {
            skip_field(wire, tag, &mut payload, DecodeContext::default())?;
            continue;
        }
        let mut work = delimited(wire, &mut payload)?;
        while !work.is_empty() {
            let (field, wire) = decode_key(&mut work)?;
            match (stream, field, wire) {
                (pb::Stream::Generation, 3, _) => {
                    delimited(wire, &mut work)?;
                    count += 1;
                }
                (pb::Stream::Prefill, 12, WireType::Varint) => {
                    decode_varint(&mut work)?;
                    count += 1;
                }
                (pb::Stream::Prefill, 12, _) => {
                    let mut tokens = delimited(wire, &mut work)?;
                    while !tokens.is_empty() {
                        decode_varint(&mut tokens)?;
                        count += 1;
                        if count > n_batch {
                            return Err(prost::DecodeError::new("too many prefill token IDs"));
                        }
                    }
                }
                _ => skip_field(wire, field, &mut work, DecodeContext::default())?,
            }
            if count > n_batch {
                return Err(prost::DecodeError::new("too many stage rows or prefill token IDs"));
            }
        }
    }
    if !seen {
        return Err(prost::DecodeError::new("missing stage message body"));
    }
    Ok(())
}

pub struct IngressBudget {
    used: Mutex<(usize, usize)>,
    available: Condvar,
    bytes: usize,
    events: usize,
}

impl IngressBudget {
    pub fn new(bytes: usize, events: usize) -> Arc<Self> {
        Arc::new(Self { used: Mutex::new((0, 0)), available: Condvar::new(), bytes, events })
    }

    pub fn reserve(self: &Arc<Self>, bytes: usize, wait: Duration) -> Result<IngressPermit, String> {
        if bytes > self.bytes {
            return Err(format!("frame of {bytes} bytes exceeds the ingress budget of {} bytes", self.bytes));
        }
        let usage = self.used.lock().unwrap();
        let (mut usage, _) = self.available.wait_timeout_while(usage, wait, |usage| {
            bytes > self.bytes - usage.0 || usage.1 >= self.events
        }).unwrap();
        if bytes > self.bytes - usage.0 || usage.1 >= self.events {
            return Err("ingress queue remained full".into());
        }
        usage.0 += bytes;
        usage.1 += 1;
        Ok(IngressPermit { budget: self.clone(), bytes })
    }
}

pub struct IngressPermit {
    budget: Arc<IngressBudget>,
    bytes: usize,
}

impl IngressPermit {
    fn release(&mut self, bytes: usize) {
        let mut usage = self.budget.used.lock().unwrap();
        self.bytes -= bytes;
        usage.0 -= bytes;
        self.budget.available.notify_all();
    }
}

impl Drop for IngressPermit {
    fn drop(&mut self) {
        let mut usage = self.budget.used.lock().unwrap();
        usage.0 -= self.bytes;
        usage.1 -= 1;
        self.budget.available.notify_all();
    }
}

/// Bytes queued by a writer. The cap applies backpressure to compute without blocking it indefinitely.
struct Budget {
    in_coda: Mutex<usize>,
    cv: Condvar,
    tetto: usize,
}

/// A writer drains a bounded byte queue off the compute thread. A send
/// that would exceed the cap waits up to the write timeout for room.
pub struct Writer {
    tx: Option<Sender<Vec<u8>>>,
    fault: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
    name: String,
    budget: Arc<Budget>,
    socket: Arc<TcpStream>,
}

impl Writer {
    pub fn new(stream: TcpStream, name: &str) -> Self {
        Self::con_tetto(stream, name, CODA_MAX_MIB.load(Ordering::Relaxed) << 20)
    }

    fn con_tetto(stream: TcpStream, name: &str, tetto: usize) -> Self {
        let _ = stream.set_write_timeout(seconds(&T_SCRITTURA_S));
        let socket = Arc::new(stream);
        let writer = socket.clone();
        let (tx, rx) = channel::<Vec<u8>>();
        let fault = Arc::new(AtomicBool::new(false));
        let budget = Arc::new(Budget { in_coda: Mutex::new(0), cv: Condvar::new(), tetto });
        let (g, b) = (fault.clone(), budget.clone());
        let n = name.to_string();
        let handle = std::thread::spawn(move || {
            let mut stream = &*writer;
            for dati in rx {
                let r = stream.write_all(&dati);
                let mut queued = b.in_coda.lock().unwrap();
                *queued -= dati.len();
                if let Err(e) = r {
                    eprintln!("[network] write to {n}: {e}");
                    g.store(true, Ordering::SeqCst);
                    b.cv.notify_all();
                    break;
                }
                b.cv.notify_all();
            }
            let _ = stream.flush();
            let _ = stream.shutdown(std::net::Shutdown::Write);
        });
        Self { tx: Some(tx), fault, handle: Some(handle), name: name.to_string(), budget, socket }
    }

    pub fn queue_timeout() -> Duration {
        seconds(&T_SCRITTURA_S).unwrap_or(Duration::from_secs(600))
    }

    pub fn send(&self, m: &pb::Message) -> Result<(), String> {
        self.send_with_wait(m, Self::queue_timeout())
    }

    pub fn send_with_wait(&self, m: &pb::Message, limite: Duration) -> Result<(), String> {
        if self.send_frame_with_wait(encode(m)?, limite)?.is_some() {
            return Err(format!("connection {} too slow: queue limit {} bytes for more than {limite:?}", self.name, self.budget.tetto));
        }
        Ok(())
    }

    /// Return the encoded frame if the queue is still full when the wait ends.
    pub fn send_frame_with_wait(&self, bytes: Vec<u8>, limite: Duration) -> Result<Option<Vec<u8>>, String> {
        let n = bytes.len();
        {
            let done = Instant::now() + limite;
            let mut p = self.budget.in_coda.lock().unwrap();
            loop {
                if self.fault.load(Ordering::SeqCst) {
                    return Err(format!("connection {} lost", self.name));
                }
                // Admit a single message to an empty queue even if it exceeds the cap.
                if *p == 0 || *p + n <= self.budget.tetto {
                    *p += n;
                    break;
                }
                let resto = done.saturating_duration_since(Instant::now());
                if resto.is_zero() {
                    return Ok(Some(bytes));
                }
                p = self.budget.cv.wait_timeout(p, resto).unwrap().0;
            }
        }
        let r = self.tx.as_ref().ok_or("writer closed").map_err(String::from);
        let r = r.and_then(|t| t.send(bytes).map_err(|_| format!("connection {} lost", self.name)));
        if r.is_err() {
            *self.budget.in_coda.lock().unwrap() -= n;
            self.budget.cv.notify_all();
        }
        r.map(|()| None)
    }

    pub fn failed(&self) -> bool {
        self.fault.load(Ordering::SeqCst)
    }

    pub fn abort(&self) {
        let _ = self.socket.shutdown(std::net::Shutdown::Both);
    }

    /// Drain the queue, close the write side and wait for the thread.
    pub fn terminate(mut self) {
        self.tx.take();
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

fn imposta(stream: &TcpStream) {
    let _ = stream.set_nodelay(true);
    let _ = stream.set_write_timeout(seconds(&T_SCRITTURA_S));
}

pub fn write_sync(stream: &mut TcpStream, m: &pb::Message) -> Result<(), String> {
    let b = encode(m)?;
    stream.write_all(&b).map_err(|e| e.to_string())
}

/// Connect to `address` for `stream` (retry for up to `timeout`) and send `Hello` declaring
/// link `epoch`. After `HelloAck`, return the reader and write stream.
pub fn connect(address: &str, stream: pb::Stream, timeout: Duration, epoch: u32) -> Result<(Reader, TcpStream), String> {
    connect_with(security().as_ref(), address, stream, timeout, epoch)
}

/// Open the token stream and advertise whether the tail can draft continuations.
pub fn connect_tokens(address: &str, timeout: Duration, epoch: u32, mtp: bool) -> Result<(Reader, TcpStream), String> {
    connect_advertising(security().as_ref(), address, pb::Stream::Token, timeout, epoch, mtp)
}

/// Like `connect`, with explicit security so tests can use several configurations in one process.
pub fn connect_with(sic: Option<&Arc<StageSecurity>>, address: &str, stream: pb::Stream, timeout: Duration, epoch: u32) -> Result<(Reader, TcpStream), String> {
    connect_advertising(sic, address, stream, timeout, epoch, false)
}

pub fn connect_forward(address: &str, timeout: Duration, epoch: &mut u32) -> Result<[(Reader, TcpStream); 3], String> {
    let sic = security();
    let done = Instant::now() + timeout;
    loop {
        let attempt = || {
            let prefill = try_connect(sic.as_ref(), address, pb::Stream::Prefill, done, *epoch, false)?;
            let gen = try_connect(sic.as_ref(), address, pb::Stream::Generation, done, *epoch, false)?;
            let ctl = try_connect(sic.as_ref(), address, pb::Stream::Control, done, *epoch, false)?;
            Ok([prefill, gen, ctl])
        };
        match attempt() {
            Ok(streams) => return Ok(streams),
            Err(ConnectAttempt::Riprova(e)) => {
                if Instant::now() >= done {
                    return Err(e);
                }
                // A partial handshake can retire the whole epoch at the peer.
                *epoch = epoch.checked_add(1).ok_or("stage link epoch exhausted")?;
                std::thread::sleep(Duration::from_millis(200).min(done.saturating_duration_since(Instant::now())));
            }
            Err(ConnectAttempt::Fatale(e)) => return Err(e),
        }
    }
}

fn connect_advertising(sic: Option<&Arc<StageSecurity>>, address: &str, stream: pb::Stream, timeout: Duration, epoch: u32, mtp: bool) -> Result<(Reader, TcpStream), String> {
    let done = Instant::now() + timeout;
    loop {
        match try_connect(sic, address, stream, done, epoch, mtp) {
            Ok(r) => return Ok(r),
            // A proxy accepts immediately and closes if its destination is not ready, so retry.
            Err(ConnectAttempt::Riprova(e)) => {
                if Instant::now() >= done {
                    return Err(e);
                }
                std::thread::sleep(Duration::from_millis(200).min(done.saturating_duration_since(Instant::now())));
            }
            Err(ConnectAttempt::Fatale(e)) => return Err(e),
        }
    }
}

enum ConnectAttempt {
    Riprova(String),
    Fatale(String),
}

fn try_connect(sic: Option<&Arc<StageSecurity>>, address: &str, stream_kind: pb::Stream, done: Instant, epoch: u32, mtp: bool) -> Result<(Reader, TcpStream), ConnectAttempt> {
    use ConnectAttempt::{Fatale, Riprova};
    let retry = |e| Riprova(format!("cannot connect to {address} ({}): {e}", stream_name(stream_kind)));
    let remaining = || {
        let rest = done.saturating_duration_since(Instant::now());
        if rest.is_zero() { Err(retry("deadline exceeded".into())) } else { Ok(rest) }
    };
    remaining()?;
    let risolti: Vec<_> = address.to_socket_addrs().map_err(|e| Fatale(format!("indirizzo {address}: {e}")))?.collect();
    // Without a deadline, `connect` may hang for minutes on an unresponsive host.
    let t_conn = seconds(&T_HELLO_S).unwrap_or(Duration::from_secs(10));
    let mut last = String::from("no address");
    let mut stream = None;
    for a in &risolti {
        match TcpStream::connect_timeout(a, t_conn.min(remaining()?)) {
            Ok(s) => {
                stream = Some(s);
                break;
            }
            Err(e) => last = e.to_string(),
        }
    }
    let stream = stream.ok_or_else(|| retry(last))?;
    let stream = match sic {
        // TLS: server must be our successor in the capability (or the head for Token).
        Some(sic) => {
            let expected = sic.expected_outgoing_peer(stream_kind).map_err(Fatale)?;
            sic.wrap_client(stream, &expected, t_conn.min(remaining()?)).map_err(|e| match e {
                TlsError::Identity(m) => Fatale(format!("{address} ({}): {m}", stream_name(stream_kind))),
                TlsError::Rete(m) => retry(m),
            })?
        }
        None => stream,
    };
    imposta(&stream);
    let mut w = stream.try_clone().map_err(|e| Fatale(e.to_string()))?;
    w.set_write_timeout(Some(remaining()?)).map_err(|e| retry(e.to_string()))?;
    write_sync(
        &mut w,
        &msg(Body::Hello(pb::Hello { protocol_major: WIRE_VERSION.0, protocol_minor: WIRE_VERSION.1, stream: stream_kind as i32, epoch, mtp })),
    )
    .map_err(&retry)?;
    let _ = w.set_write_timeout(seconds(&T_SCRITTURA_S));
    let mut l = Reader::new(stream);
    let ack = l.leggi_limitato(HELLO_FRAME_MAX, done);
    l.set_read_deadline(seconds(&T_LETTURA_S));
    match ack {
        Ok(Some(pb::Message { body: Some(Body::Ack(a)) })) if a.ok => Ok((l, w)),
        Ok(Some(pb::Message { body: Some(Body::Ack(a)) })) => {
            Err(Fatale(format!("{address} rejected {}: {}", stream_name(stream_kind), a.error)))
        }
        Ok(None) => Err(retry("connection closed before HelloAck".into())),
        Ok(altro) => Err(Fatale(format!("{address}: unexpected reply to Hello: {altro:?}"))),
        Err(e) => Err(retry(e)),
    }
}

/// On accept, wait up to `t_hello_s` seconds for `Hello`. A silent client cannot block
/// the caller longer than that.
pub fn read_hello(stream: TcpStream) -> Result<(pb::Stream, Reader, TcpStream), String> {
    read_hello_within(stream, seconds(&T_HELLO_S).unwrap_or(Duration::from_secs(10)))
}

/// Like `read_hello` for an accepted connection. With security on, do the TLS handshake first and check on
/// `Hello` that the peer may open that stream. Otherwise send a rejecting `HelloAck` and close.
pub fn read_hello_secured(sic: Option<&Arc<StageSecurity>>, stream: TcpStream) -> Result<(pb::Stream, Reader, TcpStream), String> {
    let entro = seconds(&T_HELLO_S).unwrap_or(Duration::from_secs(10));
    let done = Instant::now() + entro;
    let Some(sic) = sic else { return read_hello_until(stream, done) };
    // TLS and the stage hello share a single deadline.
    let (locale, peer) = sic.wrap_server(stream, entro).map_err(|e| format!("TLS: {e}"))?;
    let (f, mut l, mut w) = read_hello_until(locale, done)?;
    l.peer = Some(peer.clone());
    if let Err(reason) = sic.admit(f, Some(&peer)) {
        let _ = write_sync(&mut w, &msg(Body::Ack(pb::HelloAck { ok: false, error: reason.clone() })));
        let _ = w.shutdown(std::net::Shutdown::Both);
        return Err(format!("rejected {} stream from {peer}: {reason}", stream_name(f)));
    }
    Ok((f, l, w))
}

fn read_hello_within(stream: TcpStream, entro: Duration) -> Result<(pb::Stream, Reader, TcpStream), String> {
    read_hello_until(stream, Instant::now() + entro)
}

fn read_hello_until(stream: TcpStream, done: Instant) -> Result<(pb::Stream, Reader, TcpStream), String> {
    imposta(&stream);
    let w = stream.try_clone().map_err(|e| e.to_string())?;
    let mut l = Reader::new(stream);
    let primo = l
        .leggi_limitato(HELLO_FRAME_MAX, done)
        .map_err(|e| format!("timed out waiting for Hello: {e}"))?;
    l.set_read_deadline(seconds(&T_LETTURA_S));
    match primo {
        Some(pb::Message { body: Some(Body::Hello(h)) }) => {
            if h.protocol_major != WIRE_VERSION.0 {
                let mut w2 = w;
                let _ = write_sync(
                    &mut w2,
                    &msg(Body::Ack(pb::HelloAck { ok: false, error: format!("unsupported version {}", h.protocol_major) })),
                );
                return Err(format!("unsupported protocol version {}", h.protocol_major));
            }
            let f = pb::Stream::try_from(h.stream).unwrap_or(pb::Stream::Unspecified);
            l.epoca_hello = h.epoch;
            l.mtp_hello = h.mtp;
            Ok((f, l, w))
        }
        altro => Err(format!("unexpected first message (expected Hello): {altro:?}")),
    }
}

/// Accept connections from `listener`. Each gets a thread to read `Hello` with a deadline, then call
/// `gestore(flusso, lettore, scrittura)` on the same thread, so it can run the read loop. A silent or slow
/// client ties up only its own thread for at most `t_hello_s`. Close new connections above `HELLO_PENDENTI_MAX`
/// pending `Hello` messages.
pub fn accept<F>(listener: TcpListener, gestore: F)
where
    F: Fn(pb::Stream, Reader, TcpStream) + Send + Sync + 'static,
{
    accept_with(security(), listener, gestore)
}

/// Like `accept`, with explicit security.
pub fn accept_with<F>(sic: Option<Arc<StageSecurity>>, listener: TcpListener, gestore: F)
where
    F: Fn(pb::Stream, Reader, TcpStream) + Send + Sync + 'static,
{
    let gestore = Arc::new(gestore);
    let pendenti = Arc::new(AtomicUsize::new(0));
    for s in listener.incoming().flatten() {
        if pendenti.fetch_add(1, Ordering::SeqCst) >= HELLO_PENDENTI_MAX {
            pendenti.fetch_sub(1, Ordering::SeqCst);
            eprintln!("[network] rejected connection: too many peers have yet to send Hello");
            continue;
        }
        let (g, p, sic) = (gestore.clone(), pendenti.clone(), sic.clone());
        std::thread::spawn(move || {
            let r = read_hello_secured(sic.as_ref(), s);
            p.fetch_sub(1, Ordering::SeqCst);
            match r {
                Ok((f, l, w)) => g(f, l, w),
                Err(e) => eprintln!("[network] hello: {e}"),
            }
        });
    }
}

pub fn ack_ok(w: &mut TcpStream) -> Result<(), String> {
    write_sync(w, &msg(Body::Ack(pb::HelloAck { ok: true, error: String::new() })))
}

pub fn listen(address: &str) -> Result<TcpListener, String> {
    TcpListener::bind(address).map_err(|e| format!("bind {address}: {e}"))
}

// Wire encodings for residual rows.

/// Format chosen by the sender for state sent downstream (`--state-format`).
pub fn parse_state_format(t: &str) -> Result<pb::StateFormat, String> {
    match t {
        "fp32" => Ok(pb::StateFormat::Fp32Le),
        "fp16" => Ok(pb::StateFormat::Fp16Le),
        "int8" => Ok(pb::StateFormat::Int8RowLe),
        "int8-stream" => Ok(pb::StateFormat::Int8StreamLe),
        altro => Err(format!("--state-format: invalid value '{altro}' (fp32, fp16, int8 or int8-stream)")),
    }
}

/// Return the byte width of one row. Per-stream int8 adds one float scale
/// for each segment of the full residual width.
pub fn bytes_per_row(width: usize, stream_count: usize, format: pb::StateFormat) -> usize {
    match format {
        pb::StateFormat::Fp16Le => 2 * width,
        pb::StateFormat::Int8RowLe => 4 + width,
        pb::StateFormat::Int8StreamLe => 4 * stream_count.max(1) + width,
        _ => 4 * width,
    }
}

/// Leave room in the frame for metadata alongside the residual bytes.
pub const MARGINE_FRAME: usize = 1024;

/// Bound prefill rows by the wire frame limit and the encoded row width.
pub fn max_block_tokens(width: usize, stream_count: usize, format: pb::StateFormat) -> usize {
    max_block_tokens_with_ids(width, stream_count, format, false)
}

pub fn max_block_tokens_with_ids(width: usize, stream_count: usize, format: pb::StateFormat, token_ids: bool) -> usize {
    let bytes = bytes_per_row(width, stream_count, format) + if token_ids { 5 } else { 0 };
    (MAX_FRAME_SIZE - MARGINE_FRAME) / bytes.max(1)
}

pub fn with_prefill_state(mut block: pb::PrefillBlock, state: &[f32], width: usize, streams: usize, format: pb::StateFormat) -> pb::PrefillBlock {
    block.state = encode_state(state, width, streams, format);
    block.n_embd = width as u32;
    block.stream_count = declared_stream_count(format, streams);
    block.format = format as i32;
    block
}

/// Reject a prefill block that will not fit in one frame. Report the maximum
/// without changing the batch shape, which can affect numerical results.
pub fn check_prefill_block(block: usize, width: usize, stream_count: usize, format: pb::StateFormat) -> Result<(), String> {
    if format == pb::StateFormat::Int8StreamLe && (stream_count == 0 || width % stream_count != 0) {
        return Err(format!("per-stream INT8: width {width} is not divisible by {stream_count} streams"));
    }
    let massimo = max_block_tokens(width, stream_count, format);
    if block <= massimo {
        return Ok(());
    }
    Err(format!(
        "{block} tokens use {} state bytes ({width} values per token, format {}), exceeding the 16 MiB frame limit. Set --block to at most {massimo} and keep downstream formats no wider.",
        block * bytes_per_row(width, stream_count, format),
        format.as_str_name()
    ))
}

/// Include up to five bytes per input token when the block carries token IDs.
pub fn check_prefill_block_with_tokens(block: usize, width: usize, stream_count: usize, format: pb::StateFormat) -> Result<(), String> {
    check_prefill_block(block, width, stream_count, format)?;
    let byte = block * (bytes_per_row(width, stream_count, format) + 5);
    if byte + MARGINE_FRAME <= MAX_FRAME_SIZE {
        return Ok(());
    }
    Err(format!("{block} tokens with input IDs (--draft-source mtp) use {byte} bytes, exceeding the 16 MiB frame limit. Reduce --block."))
}

/// Encode uncompressed state using little-endian float32 values.
pub fn state_to_bytes(v: &[f32]) -> Vec<u8> {
    let mut o = Vec::with_capacity(v.len() * 4);
    for x in v {
        o.extend_from_slice(&x.to_le_bytes());
    }
    o
}

/// Convert to binary16 with round-to-nearest, ties-to-even. Preserve
/// infinities and NaNs and handle subnormal inputs and results.
pub fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let segno = ((bits >> 16) & 0x8000) as u16;
    let esp = ((bits >> 23) & 0xff) as i32;
    let man = bits & 0x7f_ffff;
    if esp == 0xff {
        return segno | 0x7c00 | if man != 0 { 0x200 | (man >> 13) as u16 } else { 0 };
    }
    let e = esp - 127 + 15;
    if e >= 0x1f {
        return segno | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return segno;
        }
        let m = man | 0x80_0000;
        let sh = (14 - e) as u32;
        let mut h = m >> sh;
        let resto = m & ((1 << sh) - 1);
        let meta = 1u32 << (sh - 1);
        if resto > meta || (resto == meta && h & 1 == 1) {
            h += 1;
        }
        return segno | h as u16;
    }
    let mut h = ((e as u32) << 10) | (man >> 13);
    let resto = man & 0x1fff;
    if resto > 0x1000 || (resto == 0x1000 && h & 1 == 1) {
        h += 1;
    }
    segno | h as u16
}

pub fn f16_to_f32(h: u16) -> f32 {
    let segno = ((h as u32) & 0x8000) << 16;
    let esp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = if esp == 0 {
        if man == 0 {
            segno
        } else {
            return if segno != 0 { -(man as f32) * (1.0 / 16_777_216.0) } else { man as f32 * (1.0 / 16_777_216.0) };
        }
    } else if esp == 0x1f {
        segno | 0x7f80_0000 | (man << 13)
    } else {
        segno | ((esp + 112) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

/// Encode an int8 segment with a float32 scale derived from the maximum
/// finite magnitude. Per-row encoding uses one segment. Per-stream encoding
/// places one segment at each stream boundary.
fn encode_int8(v: &[f32], seg: usize) -> Vec<u8> {
    let seg = seg.max(1);
    let mut o = Vec::with_capacity(v.len() / seg * (4 + seg) + 4);
    for s in v.chunks(seg) {
        let m = s.iter().fold(0f32, |a, &x| if x.is_finite() { a.max(x.abs()) } else { a });
        let scala = m / 127.0;
        o.extend_from_slice(&scala.to_le_bytes());
        for &x in s {
            // NaN -> 0. ±inf -> ±127. All-zero segment (scale 0) -> 0.
            let q = if scala == 0.0 || x.is_nan() { 0.0 } else { (x / scala).round().clamp(-127.0, 127.0) };
            o.push((q as i8) as u8);
        }
    }
    o
}

/// Encode rows in the selected format and count transmitted bytes. In
/// per-stream int8, n_flussi must divide the row width.
pub fn encode_state(v: &[f32], n_embd: usize, stream_count: usize, format: pb::StateFormat) -> Vec<u8> {
    let b = match format {
        pb::StateFormat::Fp16Le => {
            let mut o = Vec::with_capacity(v.len() * 2);
            for &x in v {
                o.extend_from_slice(&f32_to_f16(x).to_le_bytes());
            }
            o
        }
        pb::StateFormat::Int8RowLe => encode_int8(v, n_embd),
        pb::StateFormat::Int8StreamLe => {
            assert!(stream_count > 0 && n_embd % stream_count == 0, "per-stream INT8: {n_embd} values are not divisible by {stream_count} streams");
            encode_int8(v, n_embd / stream_count)
        }
        _ => state_to_bytes(v),
    };
    crate::timings::conta_stato_inviato(b.len(), if n_embd == 0 { 0 } else { v.len() / n_embd });
    b
}

/// Decode the declared state format. `attesi` is the number of values (rows × `n_embd`). Check lengths, format
/// and, for INT8, that every scale is finite and nonnegative.
pub fn decode_state(b: &[u8], format: i32, attesi: usize, n_embd: usize, stream_count: usize) -> Result<Vec<f32>, String> {
    crate::timings::conta_stato_ricevuto(b.len());
    match pb::StateFormat::try_from(format) {
        Ok(pb::StateFormat::Fp32Le) => {
            if b.len() != attesi * 4 {
                return Err(format!("state has {} bytes, expected {} ({} FP32 values)", b.len(), attesi * 4, attesi));
            }
            let v: Vec<f32> = b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            require_finite(&v, "FP32")?;
            Ok(v)
        }
        Ok(pb::StateFormat::Fp16Le) => {
            if b.len() != attesi * 2 {
                return Err(format!("state has {} bytes, expected {} ({} FP16 values)", b.len(), attesi * 2, attesi));
            }
            let v: Vec<f32> = b.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect();
            require_finite(&v, "FP16")?;
            Ok(v)
        }
        Ok(pb::StateFormat::Int8RowLe) => decode_int8(b, attesi, n_embd, n_embd, "INT8"),
        Ok(pb::StateFormat::Int8StreamLe) => {
            if stream_count == 0 || n_embd % stream_count != 0 {
                return Err(format!("per-stream INT8 state: n_embd = {n_embd} is not divisible by {stream_count} streams"));
            }
            decode_int8(b, attesi, n_embd, n_embd / stream_count, "per-stream INT8")
        }
        _ => Err(format!("unsupported state format {format} (FP32, FP16, per-row INT8 or per-stream INT8)")),
    }
}

/// Reject nonfinite residual state before computation so the error stays
/// attributable to its conversation. The linear count can vectorize.
fn require_finite(v: &[f32], name: &str) -> Result<(), String> {
    let cattivi = v.iter().fold(0usize, |a, x| a + (!x.is_finite()) as usize);
    if cattivi != 0 {
        let i = v.iter().position(|x| !x.is_finite()).unwrap_or(0);
        return Err(format!("{name} state has {cattivi} non-finite values (first at index {i})"));
    }
    Ok(())
}

/// Check that the received format matches the conversation's declared encoding.
pub fn check_state_format(dichiarato: pb::StateFormat, format: i32) -> Result<(), String> {
    if format == dichiarato as i32 {
        return Ok(());
    }
    let name = |f: i32| pb::StateFormat::try_from(f).map_or_else(|_| format!("{f}"), |x| x.as_str_name().to_string());
    Err(format!("state format {} differs from the Open format ({})", name(format), dichiarato.as_str_name()))
}

/// Format declared by `Open`: `FORMATO_STATO_UNSPECIFIED` (field omitted) means FP32.
pub fn declared_format(apri_formato: i32) -> Result<pb::StateFormat, String> {
    match pb::StateFormat::try_from(apri_formato) {
        Ok(pb::StateFormat::Unspecified) => Ok(pb::StateFormat::Fp32Le),
        Ok(f) => Ok(f),
        Err(_) => Err(format!("unsupported state format {apri_formato} in Open")),
    }
}

/// Decode INT8 in segments of `seg` values. `n_embd` is the row width, used to check that `attesi` spans whole
/// rows. Each scale must be finite and nonnegative.
fn decode_int8(b: &[u8], attesi: usize, n_embd: usize, seg: usize, name: &str) -> Result<Vec<f32>, String> {
    if n_embd == 0 || seg == 0 || attesi % n_embd != 0 {
        return Err(format!("{name} state: {attesi} values are not a multiple of n_embd = {n_embd}"));
    }
    let n_seg = attesi / seg;
    let lung = n_seg * (4 + seg);
    if b.len() != lung {
        return Err(format!("state has {} bytes, expected {lung} ({n_seg} {name} segments of 4 + {seg} bytes)", b.len()));
    }
    let mut out = Vec::with_capacity(attesi);
    for (i, s) in b.chunks_exact(4 + seg).enumerate() {
        let scala = f32::from_le_bytes([s[0], s[1], s[2], s[3]]);
        if !scala.is_finite() || scala < 0.0 {
            return Err(format!("{name} state: invalid scale ({scala}) in segment {i}"));
        }
        out.extend(s[4..].iter().map(|&q| (q as i8) as f32 * scala));
    }
    Ok(out)
}

/// Message field `stream_count`: set only for per-stream INT8, 0 in other formats.
pub fn declared_stream_count(format: pb::StateFormat, stream_count: usize) -> u32 {
    if format == pb::StateFormat::Int8StreamLe { stream_count as u32 } else { 0 }
}

/// A per-stream INT8 message declares `stream_count`, which must match the receiving model.
fn check_stream_count(format: i32, dichiarati: u32, stream_count: usize) -> Result<(), String> {
    if format == pb::StateFormat::Int8StreamLe as i32 && dichiarati as usize != stream_count {
        return Err(format!("declared stream_count {dichiarati} differs from the model ({stream_count})"));
    }
    Ok(())
}

/// Check the shape of a `PrefillBlock` without touching its slot. Return input state (n_token × n_embd).
pub fn validate_prefill_block(b: &pb::PrefillBlock, n_embd: usize, stream_count: usize, n_batch: usize, ctx_per_slot: u32) -> Result<Vec<f32>, String> {
    if b.n_token == 0 {
        return Err("empty block (n_token = 0)".into());
    }
    let n = b.n_token as usize;
    if n > n_batch {
        return Err(format!("block of {n} tokens exceeds n_batch = {n_batch}"));
    }
    if b.n_embd as usize != n_embd {
        return Err(format!("n_embd {} differs from the model ({n_embd})", b.n_embd));
    }
    if b.start_position as u64 + b.n_token as u64 > ctx_per_slot as u64 {
        return Err(format!("positions {}..{} exceed the slot context ({ctx_per_slot})", b.start_position, b.start_position as u64 + b.n_token as u64));
    }
    // Draft-enabled prefill supplies exactly one input token per residual row.
    if !b.token.is_empty() && b.token.len() != n {
        return Err(format!("{} input tokens for {n} rows", b.token.len()));
    }
    check_stream_count(b.format, b.stream_count, stream_count)?;
    decode_state(&b.state, b.format, n * n_embd, n_embd, stream_count)
}

/// Check the shape of a `Step`. Return state for all rows (rows × n_embd).
pub fn validate_step(p: &pb::Step, n_embd: usize, stream_count: usize, n_batch: usize) -> Result<Vec<f32>, String> {
    if p.n_embd as usize != n_embd {
        return Err(format!("n_embd {} differs from the model ({n_embd})", p.n_embd));
    }
    if p.rows.len() > n_batch {
        return Err(format!("step of {} rows exceeds n_batch = {n_batch}", p.rows.len()));
    }
    check_stream_count(p.format, p.stream_count, stream_count)?;
    decode_state(&p.state, p.format, p.rows.len() * n_embd, n_embd, stream_count)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn ingress_budget_waits_only_for_bounded_capacity() {
        let data = IngressBudget::new(10, 2);
        let control = IngressBudget::new(10, 2);
        let first = data.reserve(8, Duration::ZERO).unwrap();
        assert!(data.reserve(3, Duration::from_millis(5)).is_err());
        let second = data.reserve(1, Duration::ZERO).unwrap();
        assert!(data.reserve(1, Duration::ZERO).is_err());
        assert!(control.reserve(10, Duration::ZERO).is_ok());
        let waiting = data.clone();
        let reader = std::thread::spawn(move || waiting.reserve(8, Duration::from_secs(2)));
        drop(first);
        assert!(reader.join().unwrap().is_ok());
        assert!(data.reserve(11, Duration::ZERO).is_err());
        drop(second);
        let mut decoded = data.reserve(10, Duration::ZERO).unwrap();
        decoded.release(1);
        assert!(data.reserve(2, Duration::ZERO).is_err());
        drop(decoded);
        assert!(data.reserve(10, Duration::ZERO).is_ok());
    }

    #[test]
    fn prefill_block_budget_keeps_medium_batches_and_fits_deepseek() {
        for include_ids in [false, true] {
            let medium = max_block_tokens_with_ids(2048, 1, pb::StateFormat::Fp16Le, include_ids);
            assert_eq!(512.min(medium), 512);
            let deepseek = max_block_tokens_with_ids(4096 * 4, 4, pb::StateFormat::Fp16Le, include_ids);
            assert_eq!(deepseek, 511);
            assert!(check_prefill_block_with_tokens(deepseek, 4096 * 4, 4, pb::StateFormat::Fp16Le).is_ok());
            assert!(check_prefill_block(512, 4096 * 4, 4, pb::StateFormat::Fp16Le).is_err());
        }
    }

    #[test]
    fn admitted_reader_rejects_overload_before_reading_the_body_and_accepts_replay() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let mut old = TcpStream::connect(address).unwrap();
        let (socket, _) = listener.accept().unwrap();
        let budget = IngressBudget::new(1024, 1);
        let occupied = budget.reserve(1, Duration::ZERO).unwrap();
        old.write_all(&[1]).unwrap();
        let mut reader = Reader::new(socket);
        let rejected = reader.read_admitted(&budget, Duration::from_millis(5), pb::Stream::Prefill, 4);
        assert!(matches!(rejected, Err(error) if error == "ingress queue remained full"));
        drop(reader);
        drop(occupied);
        let mut replay = TcpStream::connect(address).unwrap();
        let (socket, _) = listener.accept().unwrap();
        let mut reader = Reader::new(socket);
        let replayed = msg(Body::Block(block(2, 4, 0)));
        write_sync(&mut replay, &replayed).unwrap();
        assert_eq!(reader.read_admitted(&budget, Duration::ZERO, pb::Stream::Prefill, 4).unwrap().unwrap().0, replayed);
    }

    #[test]
    fn work_is_rejected_on_control_and_step_rows_are_checked_before_decode() {
        let step = msg(Body::Step(pb::Step {
            rows: vec![pb::StepRow::default(); 3],
            ..Default::default()
        })).encode_to_vec();
        assert!(check_ingress(&step, pb::Stream::Control, 0).unwrap_err().to_string().contains("unexpected message"));
        assert!(check_ingress(&step, pb::Stream::Generation, 2).unwrap_err().to_string().contains("too many stage rows"));
        check_ingress(&step, pb::Stream::Generation, 3).unwrap();
        let confirmation = msg(Body::Confirmation(pb::Confirmation::default())).encode_to_vec();
        check_ingress(&confirmation, pb::Stream::Control, 0).unwrap();
        let reset = msg(Body::Reset(pb::Reset { chain_epoch: 7 })).encode_to_vec();
        check_ingress(&reset, pb::Stream::Prefill, 0).unwrap();
        assert!(check_ingress(&reset, pb::Stream::Generation, 0).is_err());
    }

    #[test]
    fn queued_prefill_forwarding_preserves_replay_and_mtp_metadata() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let mut peer = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (socket, _) = listener.accept().unwrap();
        let mut reader = Reader::new(socket);
        let budget = IngressBudget::new(1024, 2);
        let (tx, rx) = channel();
        for last in [false, true] {
            let mut original = block(2, 4, if last { 12 } else { 10 });
            original.group_id = 3;
            original.attempt_id = 7;
            original.last = last;
            original.first_token_step = if last { 5 } else { 0 };
            original.draft_count = if last { 3 } else { 0 };
            original.token = vec![101, 103];
            write_sync(&mut peer, &msg(Body::Block(original.clone()))).unwrap();
            tx.send(reader.read_admitted(&budget, Duration::ZERO, pb::Stream::Prefill, 4).unwrap().unwrap()).unwrap();
            let (message, _permit) = rx.recv().unwrap();
            let Some(Body::Block(received)) = message.body else { panic!("expected a prefill block") };
            let forwarded = with_prefill_state(received, &[1.0; 8], 4, 1, pb::StateFormat::Fp16Le);
            original.state = encode_state(&[1.0; 8], 4, 1, pb::StateFormat::Fp16Le);
            original.format = pb::StateFormat::Fp16Le as i32;
            assert_eq!(forwarded, original);
        }
    }

    fn block(n: u32, n_embd: u32, pos: u32) -> pb::PrefillBlock {
        pb::PrefillBlock {
            conversation_id: 1,
            attempt_id: 1,
            start_position: pos,
            n_token: n,
            n_embd,
            format: pb::StateFormat::Fp32Le as i32,
            state: vec![0u8; (n * n_embd * 4) as usize],
            ..Default::default()
        }
    }

    #[test]
    fn prefill_token_ids_are_optional_or_one_per_row() {
        let mut b = block(3, 4, 0);
        b.token = vec![7, 8, 9];
        assert_eq!(validate_prefill_block(&b, 4, 1, 512, 1024).unwrap().len(), 12);
        b.token = vec![7, 8];
        assert!(validate_prefill_block(&b, 4, 1, 512, 1024).unwrap_err().contains("2 input tokens for 3 rows"));
        // Include varint tokens in the frame budget. On these narrow rows the
        // extra bytes make the state-only maximum too large.
        let n = max_block_tokens(8, 1, pb::StateFormat::Fp32Le);
        assert!(check_prefill_block(n, 8, 1, pb::StateFormat::Fp32Le).is_ok());
        assert!(check_prefill_block_with_tokens(n, 8, 1, pb::StateFormat::Fp32Le).is_err());
        assert!(check_prefill_block_with_tokens(400_000, 8, 1, pb::StateFormat::Fp32Le).is_ok());
        assert!(check_prefill_block_with_tokens(409, 4 * 2560, 1, pb::StateFormat::Fp32Le).is_ok());
    }

    #[test]
    fn accepts_valid_prefill_block() {
        assert_eq!(validate_prefill_block(&block(3, 4, 0), 4, 1, 512, 1024).unwrap().len(), 12);
    }

    #[test]
    fn malformed_prefill_reports_block_failure() {
        assert!(validate_prefill_block(&block(0, 4, 0), 4, 1, 512, 1024).unwrap_err().contains("n_token = 0"));
        assert!(validate_prefill_block(&block(2, 8, 0), 4, 1, 512, 1024).unwrap_err().contains("n_embd"));
        let mut b = block(2, 4, 0);
        b.format = pb::StateFormat::Fp16Le as i32;
        assert!(validate_prefill_block(&b, 4, 1, 512, 1024).is_err());
        let mut b = block(2, 4, 0);
        b.state.pop();
        assert!(validate_prefill_block(&b, 4, 1, 512, 1024).is_err());
        assert!(validate_prefill_block(&block(5, 4, 0), 4, 1, 4, 1024).unwrap_err().contains("n_batch"));
        assert!(validate_prefill_block(&block(2, 4, 1023), 4, 1, 512, 1024).unwrap_err().contains("context"));
        assert!(validate_prefill_block(&block(2, 4, 1022), 4, 1, 512, 1024).is_ok());
    }

    #[test]
    fn rejects_non_finite_state() {
        for (x, name) in [(f32::NAN, "NaN"), (f32::INFINITY, "inf"), (f32::NEG_INFINITY, "-inf")] {
            let v = vec![1.0f32, x, 2.0, 3.0];
            let e = decode_state(&state_to_bytes(&v), pb::StateFormat::Fp32Le as i32, 4, 4, 1).unwrap_err();
            assert!(e.contains("non-finite") && e.contains("FP32"), "{name}: {e}");
            let b16 = encode_state(&v, 4, 1, pb::StateFormat::Fp16Le);
            let e = decode_state(&b16, pb::StateFormat::Fp16Le as i32, 4, 4, 1).unwrap_err();
            assert!(e.contains("non-finite") && e.contains("FP16"), "{name}: {e}");
        }
        // An encoded half-float can overflow even when the original value is finite.
        let b16 = encode_state(&[70000.0f32, 1.0], 2, 1, pb::StateFormat::Fp16Le);
        assert!(decode_state(&b16, pb::StateFormat::Fp16Le as i32, 2, 2, 1).is_err());
        let mut b = block(1, 4, 0);
        b.state = state_to_bytes(&[0.0, f32::NAN, 0.0, 0.0]);
        assert!(validate_prefill_block(&b, 4, 1, 512, 1024).unwrap_err().contains("non-finite"));
    }

    #[test]
    fn enforces_declared_format() {
        assert!(check_state_format(pb::StateFormat::Fp16Le, pb::StateFormat::Fp16Le as i32).is_ok());
        let e = check_state_format(pb::StateFormat::Fp16Le, pb::StateFormat::Fp32Le as i32).unwrap_err();
        assert!(e.contains("differs") && e.contains("FP16"), "{e}");
        assert_eq!(declared_format(0).unwrap(), pb::StateFormat::Fp32Le);
        assert_eq!(declared_format(2).unwrap(), pb::StateFormat::Fp16Le);
        assert!(declared_format(99).is_err());
    }

    /// Time the finiteness check at two residual widths and allow a loose
    /// upper bound for unoptimized test builds.
    #[test]
    fn finiteness_check_benchmark() {
        for (n_embd, rows) in [(4096usize, 16usize), (10240, 16)] {
            let v = casuale(n_embd * rows, 3);
            let b = state_to_bytes(&v);
            let reps = 200;
            let t = std::time::Instant::now();
            for _ in 0..reps {
                std::hint::black_box(require_finite(std::hint::black_box(&v), "FP32").unwrap());
            }
            let controllo = t.elapsed() / reps;
            let t = std::time::Instant::now();
            for _ in 0..reps {
                std::hint::black_box(decode_state(&b, pb::StateFormat::Fp32Le as i32, v.len(), n_embd, 1).unwrap());
            }
            let totale = t.elapsed() / reps;
            eprintln!("finiteness check {n_embd}x{rows}: {controllo:?} of {totale:?} for decode and check");
            assert!(controllo < std::time::Duration::from_millis(5), "{controllo:?}");
        }
    }

    #[test]
    fn rejects_malformed_step_and_accepts_empty_step() {
        let row = pb::StepRow { conversation_id: 1, attempt_id: 1, conversation_step: 1, position: 5, ..Default::default() };
        let ok = pb::Step { n_embd: 4, format: pb::StateFormat::Fp32Le as i32, rows: vec![row, row], state: vec![0u8; 32], ..Default::default() };
        assert_eq!(validate_step(&ok, 4, 1, 512).unwrap().len(), 8);
        assert!(validate_step(&pb::Step { n_embd: 5, ..ok.clone() }, 4, 1, 512).is_err());
        assert!(validate_step(&pb::Step { format: pb::StateFormat::Fp16Le as i32, ..ok.clone() }, 4, 1, 512).is_err());
        assert!(validate_step(&pb::Step { state: vec![0u8; 31], ..ok.clone() }, 4, 1, 512).is_err());
        assert!(validate_step(&ok, 4, 1, 1).is_err());
        let vuoto = pb::Step { n_embd: 4, format: pb::StateFormat::Fp32Le as i32, ..Default::default() };
        assert!(validate_step(&vuoto, 4, 1, 512).unwrap().is_empty());
    }

    #[test]
    fn state_round_trip_and_validation() {
        let v = vec![1.0f32, -2.5, 3.25e-7, f32::MAX];
        let b = state_to_bytes(&v);
        assert_eq!(b.len(), 16);
        let r = decode_state(&b, pb::StateFormat::Fp32Le as i32, 4, 4, 1).unwrap();
        assert_eq!(v.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), r.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
        assert!(decode_state(&b, pb::StateFormat::Fp32Le as i32, 5, 5, 1).is_err());
        assert!(decode_state(&b, pb::StateFormat::Fp16Le as i32, 4, 4, 1).is_err());
        // The generic float32 path must preserve each bit of the raw encoding.
        assert_eq!(encode_state(&v, 4, 1, pb::StateFormat::Fp32Le), b);
    }

    fn casuale(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let u = ((s >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0;
                u * 10f32.powi(((s >> 20) % 5) as i32 - 2)
            })
            .collect()
    }

    #[test]
    fn fp16_conversion_edge_cases() {
        for (x, h) in [(0.0f32, 0x0000u16), (-0.0, 0x8000), (1.0, 0x3c00), (-2.0, 0xc000), (65504.0, 0x7bff), (f32::INFINITY, 0x7c00), (f32::NEG_INFINITY, 0xfc00), (1e9, 0x7c00), (5.960_464_5e-8, 0x0001), (6.103_515_6e-5, 0x0400), (1e-9, 0x0000)] {
            assert_eq!(f32_to_f16(x), h, "{x}");
        }
        assert!(f16_to_f32(f32_to_f16(f32::NAN)).is_nan());
        assert!(f32_to_f16(f32::NAN) & 0x3ff != 0);
        // Halfway inputs round toward the endpoint with an even significand.
        assert_eq!(f32_to_f16(1.0 + 2f32.powi(-11)), 0x3c00);
        assert_eq!(f32_to_f16(1.0 + 3.0 * 2f32.powi(-11)), 0x3c02);
        assert_eq!(f32_to_f16(65520.0), 0x7c00);
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
        assert_eq!(f16_to_f32(0x8001), -(2f32.powi(-24)));
        assert_eq!(f16_to_f32(0x03ff), 1023.0 * 2f32.powi(-24));
        for h in 0u16..=0xffff {
            if (h >> 10) & 0x1f == 0x1f {
                continue;
            }
            assert_eq!(f32_to_f16(f16_to_f32(h)), h, "{h:#06x}");
        }
        for x in casuale(20000, 7) {
            let y = f16_to_f32(f32_to_f16(x));
            if x.abs() >= 6.2e-5 {
                assert!(((y - x) / x).abs() <= 4.9e-4, "{x} -> {y}");
            } else {
                assert!((y - x).abs() <= 2f32.powi(-25) * 1.0001, "{x} -> {y}");
            }
        }
    }

    #[test]
    fn round_trip_for_three_formats() {
        let n_embd = 64;
        let mut v = casuale(5 * n_embd, 42);
        for x in &mut v[n_embd..2 * n_embd] {
            *x = 0.0;
        }
        for x in &mut v[2 * n_embd..3 * n_embd] {
            *x = 3.5;
        }
        for x in &mut v[3 * n_embd..4 * n_embd] {
            *x = *x * 1e4;
            *x = x.clamp(-6e4, 6e4);
        }
        let b = encode_state(&v, n_embd, 1, pb::StateFormat::Fp32Le);
        assert_eq!(b.len(), v.len() * 4);
        let r = decode_state(&b, pb::StateFormat::Fp32Le as i32, v.len(), n_embd, 1).unwrap();
        assert!(v.iter().zip(&r).all(|(a, b)| a.to_bits() == b.to_bits()));
        let b = encode_state(&v, n_embd, 1, pb::StateFormat::Fp16Le);
        assert_eq!(b.len(), v.len() * 2);
        let r = decode_state(&b, pb::StateFormat::Fp16Le as i32, v.len(), n_embd, 1).unwrap();
        for (a, y) in v.iter().zip(&r) {
            assert!((a - y).abs() <= a.abs() * 4.9e-4 + 2f32.powi(-25), "{a} {y}");
        }
        // Int8 reconstruction can differ by up to half of one quantization interval.
        let b = encode_state(&v, n_embd, 1, pb::StateFormat::Int8RowLe);
        assert_eq!(b.len(), 5 * (4 + n_embd));
        let r = decode_state(&b, pb::StateFormat::Int8RowLe as i32, v.len(), n_embd, 1).unwrap();
        for (i, row) in v.chunks(n_embd).enumerate() {
            let m = row.iter().fold(0f32, |a, x| a.max(x.abs()));
            let scala = m / 127.0;
            for (j, x) in row.iter().enumerate() {
                let e = (x - r[i * n_embd + j]).abs();
                assert!(e <= scala / 2.0 * 1.0001 + 1e-30, "row {i}: {x} -> {} (scale {scala})", r[i * n_embd + j]);
            }
        }
        assert!(r[n_embd..2 * n_embd].iter().all(|&x| x == 0.0));
        assert!(r[2 * n_embd..3 * n_embd].iter().all(|&x| (x - 3.5).abs() < 1e-5));
    }

    #[test]
    fn nan_and_infinity_edge_cases() {
        let v = vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1.0];
        // Binary16 carries nonfinite values. Message validation rejects them before decoding a batch.
        let h = f16_to_f32(f32_to_f16(f32::NAN));
        assert!(h.is_nan());
        let e = decode_state(&encode_state(&v, 4, 1, pb::StateFormat::Fp16Le), pb::StateFormat::Fp16Le as i32, 4, 4, 1).unwrap_err();
        assert!(e.contains("non-finite"), "{e}");
        // Int8 scale uses finite values only. Infinities saturate and NaN maps to zero.
        let r = decode_state(&encode_state(&v, 4, 1, pb::StateFormat::Int8RowLe), pb::StateFormat::Int8RowLe as i32, 4, 4, 1).unwrap();
        assert_eq!(r[0], 0.0);
        assert!((r[1] - 1.0).abs() < 1e-6 && (r[2] + 1.0).abs() < 1e-6 && (r[3] - 1.0).abs() < 1e-6);
        assert_eq!(f16_to_f32(f32_to_f16(1e6)), f32::INFINITY);
        assert!(decode_state(&encode_state(&[1e6], 1, 1, pb::StateFormat::Fp16Le), pb::StateFormat::Fp16Le as i32, 1, 1, 1).is_err());
    }

    #[test]
    fn rejects_invalid_lengths_and_formats() {
        let v = casuale(8, 3);
        let fp16 = encode_state(&v, 4, 1, pb::StateFormat::Fp16Le);
        let i8b = encode_state(&v, 4, 1, pb::StateFormat::Int8RowLe);
        let f16i = pb::StateFormat::Fp16Le as i32;
        let i8i = pb::StateFormat::Int8RowLe as i32;
        assert!(decode_state(&fp16[1..], f16i, 8, 4, 1).is_err());
        assert!(decode_state(&fp16, f16i, 9, 4, 1).is_err());
        assert!(decode_state(&i8b[1..], i8i, 8, 4, 1).is_err());
        assert!(decode_state(&i8b, i8i, 12, 4, 1).is_err());
        assert!(decode_state(&i8b, i8i, 8, 3, 1).is_err());
        assert!(decode_state(&i8b, i8i, 8, 0, 1).is_err());
        assert!(decode_state(&[], 0, 0, 4, 1).is_err());
        assert!(decode_state(&[], 99, 0, 4, 1).is_err());
        let mut m = i8b.clone();
        m[0..4].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(decode_state(&m, i8i, 8, 4, 1).unwrap_err().contains("scale"));
        m[0..4].copy_from_slice(&(-1.0f32).to_le_bytes());
        assert!(decode_state(&m, i8i, 8, 4, 1).is_err());
        for f in [pb::StateFormat::Fp32Le, pb::StateFormat::Fp16Le, pb::StateFormat::Int8RowLe] {
            assert!(decode_state(&[], f as i32, 0, 4, 1).unwrap().is_empty());
        }
    }

    #[test]
    fn block_and_step_validate_three_formats() {
        let v = casuale(3 * 4, 11);
        for f in [pb::StateFormat::Fp32Le, pb::StateFormat::Fp16Le, pb::StateFormat::Int8RowLe] {
            let mut b = block(3, 4, 0);
            b.format = f as i32;
            b.state = encode_state(&v, 4, 1, f);
            assert_eq!(validate_prefill_block(&b, 4, 1, 512, 1024).unwrap().len(), 12, "{f:?}");
            let row = pb::StepRow::default();
            let p = pb::Step { n_embd: 4, format: f as i32, rows: vec![row.clone(), row.clone(), row], state: encode_state(&v, 4, 1, f), ..Default::default() };
            assert_eq!(validate_step(&p, 4, 1, 512).unwrap().len(), 12, "{f:?}");
        }
    }

    #[test]
    fn parses_state_format() {
        assert_eq!(parse_state_format("fp32").unwrap(), pb::StateFormat::Fp32Le);
        assert_eq!(parse_state_format("fp16").unwrap(), pb::StateFormat::Fp16Le);
        assert_eq!(parse_state_format("int8").unwrap(), pb::StateFormat::Int8RowLe);
        assert!(parse_state_format("fp8").is_err());
    }

    #[test]
    fn block_limit_depends_on_width_and_format() {
        use pb::StateFormat::{Fp16Le, Fp32Le, Int8RowLe};
        // Four streams of 2560 values each exercise a wider residual row.
        let q = 4 * 2560;
        assert_eq!(bytes_per_row(q, 1, Fp32Le), 40960);
        assert_eq!(bytes_per_row(q, 1, Fp16Le), 20480);
        assert_eq!(bytes_per_row(q, 1, Int8RowLe), 10244);
        assert_eq!(max_block_tokens(q, 1, Fp32Le), 409);
        assert_eq!(max_block_tokens(q, 1, Fp16Le), 819);
        assert_eq!(max_block_tokens(q, 1, Int8RowLe), 1637);
        assert_eq!(max_block_tokens(3840, 1, Fp32Le), 1092);
        assert!(check_prefill_block(512, 3840, 1, Fp32Le).is_ok());
        let e = check_prefill_block(512, q, 1, Fp32Le).unwrap_err();
        assert!(e.contains("16 MiB") && e.contains("--block") && e.contains("409") && e.contains("10240"), "{e}");
        assert!(check_prefill_block(409, q, 1, Fp32Le).is_ok());
        assert!(check_prefill_block(410, q, 1, Fp32Le).is_err());
        assert!(check_prefill_block(512, q, 1, Fp16Le).is_ok());
        assert!(check_prefill_block(820, q, 1, Fp16Le).is_err());
        assert!(check_prefill_block(1638, q, 1, Int8RowLe).is_err());
    }

    #[test]
    fn largest_block_fits_in_one_frame() {
        // Verify the computed limit against serialized messages on both sides of the boundary.
        for (width, k, format) in [
            (4 * 2560, 1, pb::StateFormat::Fp32Le),
            (4 * 2560, 1, pb::StateFormat::Fp16Le),
            (4 * 2560, 1, pb::StateFormat::Int8RowLe),
            (4 * 2560, 4, pb::StateFormat::Int8StreamLe),
            (3840, 1, pb::StateFormat::Fp32Le),
            (7, 1, pb::StateFormat::Int8RowLe),
        ] {
            let n = max_block_tokens(width, k, format);
            let block = |rows: usize| {
                msg(Body::Block(pb::PrefillBlock {
                    conversation_id: u64::MAX,
                    attempt_id: u32::MAX,
                    group_id: u32::MAX,
                    start_position: u32::MAX,
                    n_token: rows as u32,
                    n_embd: width as u32,
                    format: format as i32,
                    state: vec![0u8; rows * bytes_per_row(width, k, format)],
                    last: true,
                    first_token_step: u32::MAX,
                    stream_count: 0,
                    token: Vec::new(),
                    draft_count: u32::MAX,
                    save_checkpoint: true,
                    slot: u32::MAX,
                    slot_generation: u64::MAX,
                }))
            };
            assert!(encode(&block(n)).is_ok(), "width {width}, {format:?}: {n} tokens should fit");
            assert!(check_prefill_block(n, width, k, format).is_ok());
            assert!(check_prefill_block(n + 1, width, k, format).is_err());
            if bytes_per_row(width, k, format) > MARGINE_FRAME {
                assert!(encode(&block(n + 1)).is_err(), "width {width}, {format:?}: {} tokens should exceed the limit", n + 1);
            }
        }
    }

    #[test]
    fn state_formats_at_qwen_width() {
        // Compare encodings on streams with different amplitudes. With one
        // shared scale, the largest stream sets the error bound for every stream.
        let (flussi, n_embd, rows) = (4, 2560, 3);
        let width = flussi * n_embd;
        let mut v = casuale(rows * width, 7);
        for row in v.chunks_mut(width) {
            for (f, stream) in row.chunks_mut(n_embd).enumerate() {
                for x in stream {
                    *x *= [1.0, 8.0, 0.25, 30.0][f];
                }
            }
        }
        let b = encode_state(&v, width, 1, pb::StateFormat::Fp32Le);
        assert_eq!(b.len(), rows * bytes_per_row(width, 1, pb::StateFormat::Fp32Le));
        assert_eq!(decode_state(&b, pb::StateFormat::Fp32Le as i32, v.len(), width, 1).unwrap(), v);
        let b = encode_state(&v, width, 1, pb::StateFormat::Fp16Le);
        assert_eq!(b.len(), rows * bytes_per_row(width, 1, pb::StateFormat::Fp16Le));
        let r = decode_state(&b, pb::StateFormat::Fp16Le as i32, v.len(), width, 1).unwrap();
        assert!(v.iter().zip(&r).all(|(a, b)| (a - b).abs() <= a.abs() * 1e-3 + 1e-6));
        let b = encode_state(&v, width, 1, pb::StateFormat::Int8RowLe);
        assert_eq!(b.len(), rows * bytes_per_row(width, 1, pb::StateFormat::Int8RowLe));
        let r = decode_state(&b, pb::StateFormat::Int8RowLe as i32, v.len(), width, 1).unwrap();
        for (row, resa) in v.chunks(width).zip(r.chunks(width)) {
            let scala = row.iter().fold(0f32, |a, &x| a.max(x.abs())) / 127.0;
            assert!(row.iter().zip(resa).all(|(a, b)| (a - b).abs() <= scala / 2.0 * 1.0001));
        }
        assert!(decode_state(&b, pb::StateFormat::Int8RowLe as i32, rows * n_embd, n_embd, 1).is_err());
        let m = pb::PrefillBlock { n_token: rows as u32, n_embd: n_embd as u32, state: state_to_bytes(&v), ..Default::default() };
        assert!(validate_prefill_block(&m, width, 1, 512, 1024).unwrap_err().contains("n_embd"));
    }

    /// Alternate stream amplitudes to test the error reduction from separate scales.
    fn flussi_di_ampiezza_diversa(rows: usize, n_embd: usize) -> Vec<f32> {
        let mut v = casuale(rows * 4 * n_embd, 21);
        for row in v.chunks_mut(4 * n_embd) {
            for (f, stream) in row.chunks_mut(n_embd).enumerate() {
                for x in stream {
                    *x *= [1.0, 100.0, 1.0, 100.0][f];
                }
            }
        }
        v
    }

    #[test]
    fn per_stream_int8_round_trip_within_half_scale() {
        use pb::StateFormat::Int8StreamLe;
        let (k, n_embd, rows) = (4, 64, 5);
        let width = k * n_embd;
        let mut v = flussi_di_ampiezza_diversa(rows, n_embd);
        v[width..2 * width].fill(0.0);
        let b = encode_state(&v, width, k, Int8StreamLe);
        assert_eq!(b.len(), rows * (4 * k + width));
        assert_eq!(b.len(), rows * bytes_per_row(width, k, Int8StreamLe));
        let r = decode_state(&b, Int8StreamLe as i32, v.len(), width, k).unwrap();
        for (seg, resa) in v.chunks(n_embd).zip(r.chunks(n_embd)) {
            let scala = seg.iter().fold(0f32, |a, x| a.max(x.abs())) / 127.0;
            for (x, y) in seg.iter().zip(resa) {
                assert!((x - y).abs() <= scala / 2.0 * 1.0001 + 1e-30, "{x} -> {y} (scala {scala})");
            }
        }
    }

    #[test]
    fn per_stream_int8_reduces_error_in_small_stream() {
        let (k, n_embd, rows) = (4, 256, 3);
        let width = k * n_embd;
        let v = flussi_di_ampiezza_diversa(rows, n_embd);
        let errore_piccoli = |f: pb::StateFormat| {
            let r = decode_state(&encode_state(&v, width, k, f), f as i32, v.len(), width, k).unwrap();
            let (mut e, mut n) = (0f64, 0f64);
            for (ri, (a, b)) in v.chunks(width).zip(r.chunks(width)).enumerate() {
                for fl in [0, 2] {
                    for j in fl * n_embd..(fl + 1) * n_embd {
                        e += ((a[j] - b[j]) as f64).powi(2);
                        n += (a[j] as f64).powi(2);
                    }
                }
                let _ = ri;
            }
            (e / n).sqrt()
        };
        let per_riga = errore_piccoli(pb::StateFormat::Int8RowLe);
        let per_flusso = errore_piccoli(pb::StateFormat::Int8StreamLe);
        assert!(per_flusso * 20.0 < per_riga, "per stream {per_flusso:e}, per row {per_riga:e}");
    }

    #[test]
    fn single_stream_int8_is_bit_identical_to_per_row() {
        let v = casuale(7 * 40, 5);
        let a = encode_state(&v, 40, 1, pb::StateFormat::Int8RowLe);
        let b = encode_state(&v, 40, 1, pb::StateFormat::Int8StreamLe);
        assert_eq!(a, b);
        let ra = decode_state(&a, pb::StateFormat::Int8RowLe as i32, v.len(), 40, 1).unwrap();
        let rb = decode_state(&b, pb::StateFormat::Int8StreamLe as i32, v.len(), 40, 1).unwrap();
        assert!(ra.iter().zip(&rb).all(|(x, y)| x.to_bits() == y.to_bits()));
    }

    #[test]
    fn per_stream_int8_rejects_invalid_widths_and_stream_counts() {
        use pb::StateFormat::Int8StreamLe;
        let v = casuale(2 * 12, 9);
        let b = encode_state(&v, 12, 4, Int8StreamLe);
        let i = Int8StreamLe as i32;
        assert!(decode_state(&b, i, 24, 12, 4).is_ok());
        assert!(decode_state(&b, i, 24, 12, 5).unwrap_err().contains("not divisible"));
        assert!(decode_state(&b, i, 24, 12, 0).is_err());
        assert!(decode_state(&b, i, 24, 12, 3).is_err());
        assert!(decode_state(&b[1..], i, 24, 12, 4).is_err());
        let mut m = b.clone();
        m[0..4].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(decode_state(&m, i, 24, 12, 4).unwrap_err().contains("scale"));
        assert!(check_prefill_block(8, 10, 4, Int8StreamLe).unwrap_err().contains("not divisible"));
        assert!(check_prefill_block(8, 12, 0, Int8StreamLe).is_err());
        assert!(check_prefill_block(8, 12, 4, Int8StreamLe).is_ok());
        let mut bl = block(2, 12, 0);
        bl.format = i;
        bl.state = b.clone();
        bl.stream_count = 4;
        assert_eq!(validate_prefill_block(&bl, 12, 4, 512, 1024).unwrap().len(), 24);
        bl.stream_count = 2;
        assert!(validate_prefill_block(&bl, 12, 4, 512, 1024).unwrap_err().contains("stream_count"));
        bl.stream_count = 0;
        assert!(validate_prefill_block(&bl, 12, 4, 512, 1024).is_err());
        let row = pb::StepRow::default();
        let p = pb::Step { n_embd: 12, format: i, rows: vec![row.clone(), row], state: b, stream_count: 4, ..Default::default() };
        assert_eq!(validate_step(&p, 12, 4, 512).unwrap().len(), 24);
        assert!(validate_step(&pb::Step { stream_count: 1, ..p.clone() }, 12, 4, 512).is_err());
        assert_eq!(declared_stream_count(Int8StreamLe, 4), 4);
        assert_eq!(declared_stream_count(pb::StateFormat::Int8RowLe, 4), 0);
        assert_eq!(parse_state_format("int8-stream").unwrap(), Int8StreamLe);
    }

    #[test]
    fn frame_over_16_mib_returns_clear_error() {
        let m = msg(Body::Block(pb::PrefillBlock { state: vec![0u8; MAX_FRAME_SIZE + 1], ..Default::default() }));
        let e = encode(&m).unwrap_err();
        assert!(e.contains("16 MiB") && e.contains("--block"), "{e}");
        let ok = msg(Body::Block(pb::PrefillBlock { state: vec![0u8; 1 << 20], ..Default::default() }));
        assert!(encode(&ok).is_ok());
    }

    #[test]
    fn open_and_close_share_a_stream() {
        let ap = Body::Open(pb::Open::default());
        let ch = Body::Close(pb::Close::default());
        let bl = Body::Block(pb::PrefillBlock::default());
        // Slot reuse requires the previous close to reach the stage before the next open.
        assert_eq!(on_generation_stream(&ap), on_generation_stream(&ch));
        assert_eq!(on_generation_stream(&ap), on_generation_stream(&bl));
        assert!(on_generation_stream(&Body::Step(pb::Step::default())));
    }

    #[test]
    fn frame_round_trip_over_loopback_tcp() {
        let l = listen("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap().to_string();
        let t = std::thread::spawn(move || {
            let (s, _) = l.accept().unwrap();
            let (f, mut lett, mut w) = read_hello(s).unwrap();
            assert_eq!(f, pb::Stream::Prefill);
            assert_eq!(lett.epoca_hello, 7, "the acceptor should see the epoch from Hello");
            ack_ok(&mut w).unwrap();
            let m = lett.read().unwrap().unwrap();
            let _ = write_sync(&mut w, &m);
        });
        let (mut lett, w) = connect(&addr, pb::Stream::Prefill, Duration::from_secs(5), 7).unwrap();
        let sc = Writer::new(w, "test");
        let m = msg(Body::Close(pb::Close { conversation_id: 9, attempt_id: 2, slot: 0, slot_generation: 3, keep_parked: true }));
        sc.send(&m).unwrap();
        let eco = lett.read().unwrap().unwrap();
        assert_eq!(eco, m);
        sc.terminate();
        t.join().unwrap();
    }

    #[test]
    fn rejects_previous_protocol_major() {
        let listener = listen("127.0.0.1:0").unwrap();
        let mut peer = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let hello = msg(Body::Hello(pb::Hello {
            protocol_major: 1,
            protocol_minor: 0,
            stream: pb::Stream::Prefill as i32,
            epoch: 1,
            mtp: false,
        }));
        write_sync(&mut peer, &hello).unwrap();
        let (socket, _) = listener.accept().unwrap();
        let error = read_hello(socket).err().expect("accepted an incompatible peer");
        assert!(error.contains("unsupported protocol version"), "{error}");

        let deadline = Instant::now() + Duration::from_secs(2);
        let reply = Reader::new(peer).leggi_limitato(HELLO_FRAME_MAX, deadline).unwrap().unwrap();
        assert!(matches!(reply.body, Some(Body::Ack(ack)) if !ack.ok && ack.error.contains("unsupported version")));
    }

    /// A client waiting to send its hello must leave the listener free for another client.
    #[test]
    fn silent_client_does_not_block_accept() {
        let l = listen("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap().to_string();
        let (tx, rx) = std::sync::mpsc::channel::<pb::Stream>();
        let tx = Mutex::new(tx);
        std::thread::spawn(move || {
            accept(l, move |f, _l, mut w| {
                let _ = ack_ok(&mut w);
                let _ = tx.lock().unwrap().send(f);
                std::thread::sleep(Duration::from_millis(300));
            })
        });
        let _muto = TcpStream::connect(&addr).unwrap();
        let t0 = Instant::now();
        let (_lett, _w) = connect(&addr, pb::Stream::Control, Duration::from_secs(5), 1).unwrap();
        assert!(t0.elapsed() < Duration::from_secs(3), "served after {:?}", t0.elapsed());
        assert_eq!(rx.recv_timeout(Duration::from_secs(2)).unwrap(), pb::Stream::Control);
    }

    #[test]
    fn missing_hello_times_out() {
        let l = listen("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        let _muto = TcpStream::connect(addr).unwrap();
        let (s, _) = l.accept().unwrap();
        let t0 = Instant::now();
        let e = read_hello_within(s, Duration::from_millis(300)).err().expect("should time out");
        assert!(t0.elapsed() < Duration::from_secs(5), "{:?}", t0.elapsed());
        assert!(e.contains("timed out waiting for Hello"), "{e}");
    }

    /// A client that keeps sending bytes must still obey the whole-hello deadline.
    #[test]
    fn trickled_hello_hits_deadline() {
        let l = listen("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        let goccia = std::thread::spawn(move || {
            let mut c = TcpStream::connect(addr).unwrap();
            // One byte arrives every 200 ms, so an individual read timeout would keep being reset.
            let _ = c.write_all(&[100]);
            for _ in 0..15 {
                std::thread::sleep(Duration::from_millis(200));
                if c.write_all(&[0]).is_err() {
                    break;
                }
            }
        });
        let (s, _) = l.accept().unwrap();
        let t0 = Instant::now();
        let e = read_hello_within(s, Duration::from_millis(700)).err().expect("should time out");
        assert!(t0.elapsed() < Duration::from_millis(1600), "timed out after {:?}: {e}", t0.elapsed());
        goccia.join().unwrap();
    }

    /// Reject an oversized hello using only the declared length.
    #[test]
    fn rejects_hello_over_64_kib_immediately() {
        let l = listen("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        let mut c = TcpStream::connect(addr).unwrap();
        c.write_all(&[0x80, 0x80, 0x40]).unwrap();
        let (s, _) = l.accept().unwrap();
        let t0 = Instant::now();
        let e = read_hello_within(s, Duration::from_secs(5)).err().expect("should be rejected");
        assert!(t0.elapsed() < Duration::from_secs(1), "waited too long: {:?}: {e}", t0.elapsed());
        assert!(e.contains("limit for this message"), "{e}");
    }

    /// With no peer reads, queued output must hit backpressure and return a timeout.
    #[test]
    fn send_queue_limit_with_nonreading_peer() {
        let l = listen("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        let client = TcpStream::connect(addr).unwrap();
        let (_peer, _) = l.accept().unwrap();
        T_SCRITTURA_S.store(1, Ordering::Relaxed);
        let sc = Writer::con_tetto(client, "test", 1 << 20);
        let grosso = msg(Body::Block(pb::PrefillBlock { state: vec![0u8; 1 << 20], ..Default::default() }));
        let t0 = Instant::now();
        let mut esito = Ok(());
        // The socket buffer absorbs the first writes before the application queue fills.
        for _ in 0..200 {
            esito = sc.send(&grosso);
            if esito.is_err() {
                break;
            }
        }
        T_SCRITTURA_S.store(120, Ordering::Relaxed);
        let e = esito.expect_err("send queue should reject the frame");
        assert!(e.contains("too slow") || e.contains("lost"), "{e}");
        assert!(t0.elapsed() < Duration::from_secs(20), "{:?}", t0.elapsed());
    }
}
