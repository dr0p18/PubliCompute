//! Local service for the head over inherited pipes or a loopback socket.
//! Incoming frames become request events for the
//! compute thread, and a separate writer sends tokens and final results.
//! Disconnect or invalid input tells the head to cancel open requests.
//! The message types are defined in proto/publicompute/stage/service/v1/service.proto.

use bytes::BytesMut;
use prost::Message;
use publicompute_proto::service as pb;
use publicompute_proto::wire::{encode_frame, try_decode_frame};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub use pb::{FromAgent, FromHead};

#[derive(Debug)]
pub enum Event {
    Connesso,
    Request(pb::Request),
    Cancel(u64),
    /// Abort open requests when the session can no longer carry their responses.
    Disconnesso,
}

/// Listening service. Create it with [`Service::listen`]. The head reads events with [`Service::prova_evento`] or
/// [`Service::attendi_evento`] and writes with [`Service::manda`].
pub struct Service {
    rx: Receiver<Event>,
    output: Arc<Mutex<Option<Output>>>,
    addr: Option<SocketAddr>,
}

const OUTPUT_BYTES: usize = 32 << 20;
const OUTPUT_FRAMES: usize = 256;
const DONE_BYTES: usize = 16 << 20;
const DONE_FRAMES: usize = 16;
const OUTPUT_TIMEOUT: Duration = Duration::from_secs(30);

struct Output {
    tx: Sender<Vec<u8>>,
    queue: Arc<Mutex<OutputQueue>>,
    socket: Option<TcpStream>,
}

#[derive(Default)]
struct OutputQueue {
    bytes: usize,
    frames: usize,
    progress: Option<Instant>,
    closed: bool,
}

impl Output {
    fn new(socket: Option<TcpStream>) -> (Self, Receiver<Vec<u8>>) {
        let (tx, rx) = channel();
        (Self { tx, queue: Arc::default(), socket }, rx)
    }

    fn close(&self) {
        self.queue.lock().unwrap().closed = true;
        if let Some(socket) = &self.socket {
            let _ = socket.shutdown(std::net::Shutdown::Both);
        }
    }

    fn closed(&self) -> bool {
        let q = self.queue.lock().unwrap();
        q.closed || q.progress.is_some_and(|t| t.elapsed() >= OUTPUT_TIMEOUT)
    }

    fn send(&self, message: FromHead) -> bool {
        let done = matches!(message.body, Some(pb::from_head::Body::Done(_)));
        // Keep space for terminal frames in the same FIFO as the tokens.
        let max_bytes = if done { OUTPUT_BYTES + DONE_BYTES } else { OUTPUT_BYTES };
        let max_frames = if done { OUTPUT_FRAMES + DONE_FRAMES } else { OUTPUT_FRAMES };
        let len = message.encoded_len();
        let size = len + prost::encoding::encoded_len_varint(len as u64);
        let mut q = self.queue.lock().unwrap();
        if q.closed || q.frames >= max_frames || size > max_bytes - q.bytes.min(max_bytes)
            || len > publicompute_proto::wire::MAX_FRAME_SIZE
        {
            return false;
        }
        let mut frame = Vec::with_capacity(size);
        if encode_frame(&message, &mut frame).is_err() || self.tx.send(frame).is_err() {
            return false;
        }
        q.bytes += size;
        q.frames += 1;
        q.progress.get_or_insert_with(Instant::now);
        true
    }
}

/// Bind only to a loopback address.
pub fn validate_address(a: &str) -> Result<SocketAddr, String> {
    use std::net::ToSocketAddrs;
    let v: Vec<SocketAddr> = a.to_socket_addrs().map_err(|e| format!("--service {a}: {e}"))?.collect();
    let s = v.first().copied().ok_or_else(|| format!("--service {a}: could not resolve address"))?;
    if !s.ip().is_loopback() || v.iter().any(|x| !x.ip().is_loopback()) {
        return Err(format!("--service {a}: a loopback address is required (127.0.0.1)"));
    }
    Ok(s)
}

impl Service {
    pub fn listen(address: &str) -> Result<Service, String> {
        let s = validate_address(address)?;
        let listener = TcpListener::bind(s).map_err(|e| format!("bind {address}: {e}"))?;
        let addr = listener.local_addr().map_err(|e| e.to_string())?;
        let (tx, rx) = channel::<Event>();
        let output = Arc::new(Mutex::new(None));
        let attivo = Arc::new(AtomicBool::new(false));
        let u2 = output.clone();
        std::thread::spawn(move || {
            for conn in listener.incoming() {
                let Ok(stream) = conn else { continue };
                if attivo.swap(true, Ordering::SeqCst) {
                    let _ = stream.shutdown(std::net::Shutdown::Both);
                    continue;
                }
                start_connection(stream, tx.clone(), u2.clone(), attivo.clone());
            }
        });
        Ok(Service { rx, output, addr: Some(addr) })
    }

    pub fn pipes() -> Result<Service, String> {
        let mut stdout = std::io::stdout();
        let ready = FromHead { body: Some(pb::from_head::Body::Ready(pb::Ready {})) };
        let mut frame = Vec::new();
        encode_frame(&ready, &mut frame).map_err(|e| e.to_string())?;
        stdout.write_all(&frame).and_then(|_| stdout.flush()).map_err(|e| format!("write readiness: {e}"))?;
        let (tx, rx) = channel();
        let (writer, frames) = Output::new(None);
        let queue = writer.queue.clone();
        let output = Arc::new(Mutex::new(Some(writer)));
        let reader_output = output.clone();
        std::thread::spawn(move || {
            read_requests(&mut std::io::stdin().lock(), &tx);
            *reader_output.lock().unwrap() = None;
            let _ = tx.send(Event::Disconnesso);
        });
        std::thread::spawn(move || write_responses(&mut stdout.lock(), frames, queue));
        Ok(Service { rx, output, addr: None })
    }

    /// Actual address, useful when binding to port 0.
    pub fn address(&self) -> Option<SocketAddr> {
        self.addr
    }

    pub fn pipe_closed(&self) -> bool {
        let output = self.output.lock().unwrap();
        if let Some(writer) = output.as_ref().filter(|writer| writer.closed()) {
            writer.close();
            return self.addr.is_none();
        }
        self.addr.is_none() && output.is_none()
    }

    /// Take an event that has already arrived, without waiting.
    pub fn prova_evento(&self) -> Option<Event> {
        self.rx.try_recv().ok()
    }

    /// Wait up to `durata` for an event.
    pub fn attendi_evento(&self, durata: Duration) -> Option<Event> {
        match self.rx.recv_timeout(durata) {
            Ok(e) => Some(e),
            Err(RecvTimeoutError::Timeout) | Err(RecvTimeoutError::Disconnected) => None,
        }
    }

    /// Queue output or fail promptly when the active client falls behind.
    pub fn manda(&self, m: FromHead) -> bool {
        let terminal = matches!(m.body, Some(pb::from_head::Body::Done(_)));
        let output = self.output.lock().unwrap();
        let Some(writer) = output.as_ref() else { return false };
        if writer.send(m) {
            return true;
        }
        if terminal {
            writer.close();
        }
        false
    }
}

fn start_connection(stream: TcpStream, ev: Sender<Event>, output: Arc<Mutex<Option<Output>>>, attivo: Arc<AtomicBool>) {
    let _ = stream.set_nodelay(true);
    let _ = stream.set_write_timeout(Some(OUTPUT_TIMEOUT));
    let (Ok(mut write), Ok(shutdown)) = (stream.try_clone(), stream.try_clone()) else {
        attivo.store(false, Ordering::SeqCst);
        return;
    };
    let (writer, frames) = Output::new(Some(shutdown));
    let queue = writer.queue.clone();
    *output.lock().unwrap() = Some(writer);
    let _ = ev.send(Event::Connesso);
    std::thread::spawn(move || {
        write_responses(&mut write, frames, queue);
        let _ = write.shutdown(std::net::Shutdown::Both);
    });
    std::thread::spawn(move || {
        let mut stream = stream;
        read_requests(&mut stream, &ev);
        if let Some(writer) = output.lock().unwrap().take() {
            writer.close();
        }
        let _ = ev.send(Event::Disconnesso);
        attivo.store(false, Ordering::SeqCst);
    });
}

fn write_responses(writer: &mut impl Write, frames: Receiver<Vec<u8>>, queue: Arc<Mutex<OutputQueue>>) {
    for frame in frames {
        let closed = queue.lock().unwrap().closed;
        if closed || writer.write_all(&frame).and_then(|_| writer.flush()).is_err() {
            break;
        }
        let mut q = queue.lock().unwrap();
        q.bytes -= frame.len();
        q.frames -= 1;
        q.progress = (q.frames > 0).then(Instant::now);
    }
    queue.lock().unwrap().closed = true;
}

fn read_requests(reader: &mut impl Read, events: &Sender<Event>) {
    let mut buf = BytesMut::new();
    let mut tmp = vec![0u8; 64 << 10];
    loop {
        match try_decode_frame::<FromAgent>(&mut buf) {
            Ok(Some(message)) => {
                let event = match message.body {
                    Some(pb::from_agent::Body::Request(request)) => Event::Request(request),
                    Some(pb::from_agent::Body::Cancel(cancel)) => Event::Cancel(cancel.id),
                    None => continue,
                };
                if events.send(event).is_err() {
                    break;
                }
            }
            Ok(None) => match reader.read(&mut tmp) {
                Ok(0) | Err(_) => break,
                Ok(count) => buf.extend_from_slice(&tmp[..count]),
            },
            Err(error) => {
                eprintln!("[service] invalid frame from the agent: {error}");
                break;
            }
        }
    }
}

/// Split a stream of UTF-8 bytes received in chunks. A character can span several tokens, so
/// return only valid text and keep the unfinished character for later.
#[derive(Default, Debug)]
pub struct Utf8Fragments {
    resto: Vec<u8>,
}

impl Utf8Fragments {
    /// Add the bytes of a token and return the complete text so far (empty if the token completes nothing).
    /// Render bytes that cannot be valid UTF-8 as U+FFFD, as `from_utf8_lossy` does.
    pub fn aggiungi(&mut self, pezzo: &[u8]) -> String {
        self.resto.extend_from_slice(pezzo);
        let mut fuori = String::new();
        loop {
            match std::str::from_utf8(&self.resto) {
                Ok(s) => {
                    fuori.push_str(s);
                    self.resto.clear();
                    return fuori;
                }
                Err(e) => {
                    let valido = e.valid_up_to();
                    fuori.push_str(std::str::from_utf8(&self.resto[..valido]).unwrap());
                    match e.error_len() {
                        // Invalid sequence: emit one replacement character and resume after it.
                        Some(n) => {
                            fuori.push('\u{FFFD}');
                            self.resto.drain(..valido + n);
                        }
                        // Truncated sequence at the end: wait for the remaining bytes.
                        None => {
                            self.resto.drain(..valido);
                            return fuori;
                        }
                    }
                }
            }
        }
    }

    /// At the end, replace leftover bytes from an incomplete character.
    pub fn clear(&mut self) -> String {
        let s = String::from_utf8_lossy(&self.resto).into_owned();
        self.resto.clear();
        s
    }
}


// Prompt rendering and history trimming.

/// Render the role markers and trimmed content directly. The native chat
/// formatter cannot evaluate this template. Open the response with an
/// empty reasoning channel.
pub fn gemma4_conversation<'a>(msgs: impl IntoIterator<Item = &'a pb::Message>) -> Result<String, String> {
    let mut t = String::from("<bos>");
    for m in msgs {
        let role = match m.role.as_str() {
            "system" => "system",
            "user" => "user",
            "assistant" => "model",
            altro => return Err(format!("unknown role {altro:?} (expected system, user or assistant)")),
        };
        t.push_str(&format!("<|turn>{role}\n{}<turn|>\n", m.text.trim()));
    }
    t.push_str("<|turn>model\n<|channel>thought\n<channel|>");
    Ok(t)
}

/// Render the chat form for weight files without a usable template. The
/// prompt begins with an explicit BOS marker and the system text. User
/// turns open an assistant segment with reasoning disabled, and assistant
/// content ends with an end-of-sentence marker.
pub fn deepseek4_conversation<'a>(msgs: impl IntoIterator<Item = &'a pb::Message>) -> Result<String, String> {
    let mut t = String::from("<\u{ff5c}begin\u{2581}of\u{2581}sentence\u{ff5c}>");
    for m in msgs {
        match m.role.as_str() {
            "system" => t.push_str(&m.text),
            "user" => {
                t.push_str("<\u{ff5c}User\u{ff5c}>");
                t.push_str(&m.text);
                t.push_str("<\u{ff5c}Assistant\u{ff5c}></think>");
            }
            "assistant" => {
                t.push_str(&m.text);
                t.push_str("<\u{ff5c}end\u{2581}of\u{2581}sentence\u{ff5c}>");
            }
            altro => return Err(format!("unknown role {altro:?} (expected system, user or assistant)")),
        }
    }
    Ok(t)
}

const CONTEXT_ATTEMPTS: usize = 8;
pub const LAST_MESSAGE_TOO_LONG: &str = "the last message alone exceeds the slot context";

/// Keep enough room for a useful reply when history can be shortened.
pub fn fit_chat_context<T>(
    messages: &[pb::Message],
    context: usize,
    ceiling: u32,
    mut prepare: impl FnMut(&[&pb::Message]) -> Result<(T, usize), String>,
) -> Result<(T, usize), String> {
    let preferred = ceiling.min(512) as usize;
    let result = fit_context(messages, context.saturating_sub(preferred), &mut prepare);
    if preferred > 1 && result.as_ref().err().is_some_and(|e| e == LAST_MESSAGE_TOO_LONG) {
        return fit_context(messages, context.saturating_sub(1), prepare);
    }
    result
}

/// Return the available output budget. A full context has no room for a reply.
pub fn output_budget(context: usize, prompt: usize, ceiling: u32) -> Option<u32> {
    let budget = context.saturating_sub(prompt).min(ceiling as usize) as u32;
    (budget > 0).then_some(budget)
}

/// Try a bounded search over whole turns and keep the prepared prompt that fit.
pub fn fit_context<T>(
    messages: &[pb::Message],
    limit: usize,
    mut prepare: impl FnMut(&[&pb::Message]) -> Result<(T, usize), String>,
) -> Result<(T, usize), String> {
    let last = messages.last().ok_or("empty conversation")?;
    let last_index = messages.len() - 1;
    let mut candidate = messages.iter().collect::<Vec<_>>();
    let (full, count) = prepare(&candidate)?;
    if count <= limit {
        return Ok((full, 0));
    }
    drop(full);

    let system = |message: &pb::Message| message.role == "system";
    let mut cuts = vec![0];
    if let Some(first) = messages.iter().position(|message| !system(message)) {
        cuts.extend((first + 1..last_index).filter(|&index| !system(&messages[index]) && messages[index].role != "assistant"));
    }
    cuts.push(last_index);
    let mut prepare_suffix = |start| {
        candidate.clear();
        candidate.extend(messages.iter().enumerate().filter_map(|(index, message)| (index >= start || system(message)).then_some(message)));
        if candidate.len() == messages.len() {
            return Ok(None);
        }
        let (prepared, count) = prepare(&candidate)?;
        Ok::<_, String>(Some((prepared, count, candidate.len())))
    };

    if let Some((smallest, count, size)) = prepare_suffix(last_index)? {
        if count <= limit {
            let mut best = (smallest, messages.len() - size);
            let (mut too_long, mut fits) = (0, cuts.len() - 1);
            for _ in 0..CONTEXT_ATTEMPTS - 2 {
                if fits - too_long <= 1 {
                    break;
                }
                let probe = too_long + (fits - too_long) / 2;
                if let Some((prepared, count, size)) = prepare_suffix(cuts[probe])? {
                    if count <= limit {
                        best = (prepared, messages.len() - size);
                        fits = probe;
                        continue;
                    }
                }
                too_long = probe;
            }
            return Ok(best);
        }
        if size == 1 {
            return Err(LAST_MESSAGE_TOO_LONG.into());
        }
    }
    if last_index > 0 {
        let (prepared, count) = prepare(&[last])?;
        if count <= limit {
            return Ok((prepared, last_index));
        }
    }
    Err(LAST_MESSAGE_TOO_LONG.into())
}

#[cfg(test)]
mod test {
    use super::*;
    use publicompute_proto::service::{from_agent::Body as CA, from_head::Body as CT};


    fn m(r: &str, t: &str) -> pb::Message {
        pb::Message { role: r.into(), text: t.into() }
    }

    #[test]
    fn output_queue_bounds_tokens_and_keeps_terminal_frame_in_order() {
        use pb::from_head::Body;
        let (output, frames) = Output::new(None);
        for step in 0..OUTPUT_FRAMES as u32 {
            let token = pb::OutputToken { id: 1, step, ..Default::default() };
            assert!(output.send(FromHead { body: Some(Body::Token(token)) }));
        }
        assert!(!output.send(FromHead { body: Some(Body::Token(pb::OutputToken { id: 1, ..Default::default() })) }));
        assert!(output.send(FromHead { body: Some(Body::Done(pb::Done { id: 1, ..Default::default() })) }));
        let mut written = Vec::new();
        let queue = output.queue.clone();
        drop(output);
        write_responses(&mut written, frames, queue.clone());
        let mut buf = BytesMut::from(written.as_slice());
        for step in 0..OUTPUT_FRAMES as u32 {
            let message = try_decode_frame::<FromHead>(&mut buf).unwrap().unwrap();
            assert!(matches!(message.body, Some(Body::Token(t)) if t.id == 1 && t.step == step));
        }
        let message = try_decode_frame::<FromHead>(&mut buf).unwrap().unwrap();
        assert!(matches!(message.body, Some(Body::Done(d)) if d.id == 1));
        assert!(buf.is_empty());
        assert_eq!(queue.lock().unwrap().bytes, 0);
    }

    #[test]
    fn output_queue_bounds_bytes_and_detects_stalled_pipe() {
        let (output, _frames) = Output::new(None);
        let large_token = || FromHead {
            body: Some(pb::from_head::Body::Token(pb::OutputToken {
                id: 1,
                text: "x".repeat((OUTPUT_BYTES / 3) + 1),
                ..Default::default()
            })),
        };
        assert!(output.send(large_token()));
        assert!(output.send(large_token()));
        assert!(!output.send(large_token()));
        output.queue.lock().unwrap().progress = Some(Instant::now() - OUTPUT_TIMEOUT);
        let (_tx, rx) = channel();
        let service = Service { rx, output: Arc::new(Mutex::new(Some(output))), addr: None };
        assert!(service.pipe_closed());
    }

    #[test]
    fn gemma4_single_message_matches_existing_format() {
        let t = gemma4_conversation(&[m("user", "  ciao  ")]).unwrap();
        assert_eq!(t, "<bos><|turn>user\nciao<turn|>\n<|turn>model\n<|channel>thought\n<channel|>");
    }

    #[test]
    fn gemma4_multiple_turns_with_system_message() {
        let t = gemma4_conversation(&[m("system", "sii breve"), m("user", "uno"), m("assistant", "due"), m("user", "tre")]).unwrap();
        assert_eq!(
            t,
            "<bos><|turn>system\nsii breve<turn|>\n<|turn>user\nuno<turn|>\n<|turn>model\ndue<turn|>\n<|turn>user\ntre<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
        );
        assert!(gemma4_conversation(&[m("tool", "x")]).is_err());
    }

    #[test]
    fn deepseek4_single_turn() {
        let t = deepseek4_conversation(&[m("user", "ciao")]).unwrap();
        assert_eq!(t, "<｜begin▁of▁sentence｜><｜User｜>ciao<｜Assistant｜></think>");
    }

    #[test]
    fn deepseek4_multiple_turns_with_system_message() {
        let t = deepseek4_conversation(&[m("system", "sii breve"), m("user", "uno"), m("assistant", "due"), m("user", "tre")]).unwrap();
        assert_eq!(
            t,
            "<｜begin▁of▁sentence｜>sii breve<｜User｜>uno<｜Assistant｜></think>due<｜end▁of▁sentence｜><｜User｜>tre<｜Assistant｜></think>"
        );
    }

    #[test]
    fn deepseek4_rejects_unsupported_role() {
        assert!(deepseek4_conversation(&[m("tool", "x")]).is_err());
        assert!(deepseek4_conversation(&[m("user", "a"), m("developer", "x")]).is_err());
    }

    fn prepare(messages: &[&pb::Message]) -> Result<(Vec<String>, usize), String> {
        Ok((messages.iter().map(|message| message.text.clone()).collect(), messages.iter().map(|message| message.text.len()).sum()))
    }

    #[test]
    fn chat_output_uses_free_context_for_every_request_form() {
        let messages = vec![pb::Message { role: "user".into(), text: "short".into() }];
        for ceiling in [4096, 8192, 16384] {
            let (prompt, dropped) = fit_chat_context(&messages, 8192, ceiling, |_| Ok((8, 8))).unwrap();
            assert_eq!((prompt, dropped), (8, 0));
            assert_eq!(output_budget(8192, prompt, ceiling), Some(ceiling.min(8184)));
        }
        assert_eq!(output_budget(8192, 8191, 16384), Some(1));
        assert_eq!(output_budget(8192, 8192, 16384), None);
    }

    #[test]
    fn chat_context_trims_history_then_accepts_a_one_token_reply() {
        let messages = vec![
            pb::Message { role: "user".into(), text: "old".into() },
            pb::Message { role: "assistant".into(), text: "reply".into() },
            pb::Message { role: "user".into(), text: "new".into() },
        ];
        let prepare = |history: &[&pb::Message]| {
            let count = 8191 + 512 * (history.len() - 1);
            Ok((count, count))
        };
        assert_eq!(fit_chat_context(&messages, 8192, 16384, prepare), Ok((8191, 2)));
        assert_eq!(
            fit_chat_context(&messages, 8192, 16384, |_| Ok((8192, 8192))).unwrap_err(),
            LAST_MESSAGE_TOO_LONG
        );
        let mut attempts = 0;
        assert!(fit_chat_context::<()>(&messages, 8192, 16384, |_| {
            attempts += 1;
            Err("tokenizer failed".into())
        }).is_err());
        assert_eq!(attempts, 1);
    }

    #[test]
    fn fit_context_drops_oldest_messages_and_keeps_system_message() {
        let v = vec![m("system", "SS"), m("user", "aaaa"), m("assistant", "bbbb"), m("user", "cccc"), m("assistant", "dddd"), m("user", "ee")];
        let (r, tolti) = fit_context(&v, 100, prepare).unwrap();
        assert_eq!((r.len(), tolti), (6, 0));
        let (r, tolti) = fit_context(&v, 12, prepare).unwrap();
        assert_eq!(tolti, 2);
        assert_eq!(r, ["SS", "cccc", "dddd", "ee"]);
        let (r, tolti) = fit_context(&v, 4, prepare).unwrap();
        assert_eq!(r, ["SS", "ee"]);
        assert_eq!(tolti, 4);
        let (r, tolti) = fit_context(&v, 3, prepare).unwrap();
        assert_eq!((r.len(), tolti, r[0].as_str()), (1, 5, "ee"));
        assert!(fit_context(&v, 1, prepare).is_err());
    }

    #[test]
    fn fit_context_bounds_tokenization_and_returns_the_accepted_tokens() {
        let mut history = vec![m("system", "system")];
        for _ in 0..4000 {
            history.extend([m("user", "ask"), m("assistant", "reply")]);
        }
        history.push(m("user", "last"));
        let mut calls = 0;
        let (tokens, dropped) = fit_context(&history, 2000, |messages| {
            calls += 1;
            assert_ne!(messages[1].role, "assistant");
            let tokens = vec![calls; messages.len()];
            Ok((tokens, messages.len()))
        }).unwrap();
        assert!(calls <= CONTEXT_ATTEMPTS);
        assert!(tokens.len() <= 2000);
        assert_eq!(tokens.len() + dropped, history.len());
        assert!((1..=calls).contains(&tokens[0]));

        let mut calls = 0;
        let (tokens, dropped) = fit_context(&history, 1, |messages| {
            calls += 1;
            Ok((vec![calls; messages.len()], messages.len()))
        }).unwrap();
        assert_eq!((calls, tokens, dropped), (3, vec![3], history.len() - 1));
    }

    fn write(s: &mut TcpStream, m: &FromAgent) {
        let mut v = Vec::new();
        encode_frame(m, &mut v).unwrap();
        s.write_all(&v).unwrap();
    }

    fn read(s: &mut TcpStream, buf: &mut BytesMut) -> FromHead {
        let mut tmp = [0u8; 4096];
        loop {
            if let Some(m) = try_decode_frame::<FromHead>(buf).unwrap() {
                return m;
            }
            let n = s.read(&mut tmp).unwrap();
            assert!(n > 0, "connection closed");
            buf.extend_from_slice(&tmp[..n]);
        }
    }

    fn request(id: u64, text: &str) -> FromAgent {
        FromAgent {
            body: Some(CA::Request(pb::Request {
                id,
                text: text.into(),
                max_tokens: 8,
                sampling: Some(pb::Sampling { temperature: 0.0, seed: 7, ..Default::default() }),
                class: pb::Class::Interactive as i32,
                ..Default::default()
            })),
        }
    }

    const T: Duration = Duration::from_secs(5);

    #[test]
    fn only_loopback_addresses_are_allowed() {
        assert!(validate_address("127.0.0.1:0").is_ok());
        assert!(validate_address("[::1]:0").is_ok());
        assert!(validate_address("0.0.0.0:9000").is_err());
        assert!(validate_address("192.0.2.5:9000").is_err());
        assert!(Service::listen("0.0.0.0:0").is_err());
    }

    #[test]
    fn requests_cancels_and_responses_round_trip() {
        let sv = Service::listen("127.0.0.1:0").unwrap();
        let mut c = TcpStream::connect(sv.address().unwrap()).unwrap();
        let mut buf = BytesMut::new();
        assert!(matches!(sv.attendi_evento(T), Some(Event::Connesso)));
        write(&mut c, &request(41, "ciao"));
        write(&mut c, &FromAgent { body: Some(CA::Cancel(pb::Cancel { id: 41 })) });
        match sv.attendi_evento(T) {
            Some(Event::Request(r)) => {
                assert_eq!((r.id, r.text.as_str(), r.max_tokens), (41, "ciao", 8));
                assert_eq!(r.sampling.unwrap().seed, 7);
            }
            e => panic!("expected Request, got {e:?}"),
        }
        assert!(matches!(sv.attendi_evento(T), Some(Event::Cancel(41))));
        assert!(sv.manda(FromHead { body: Some(CT::Status(pb::Status { free_slots: 3, pending: 1 })) }));
        assert!(sv.manda(FromHead { body: Some(CT::Token(pb::OutputToken { id: 41, token: 9, text: "è".into(), logprob: -0.5, step: 0, eog: false })) }));
        assert!(sv.manda(FromHead {
            body: Some(CT::Done(pb::Done { id: 41, reason: "cancelled".into(), usage: Some(pb::Usage { prompt_tokens: 5, generated_tokens: 1 }), ..Default::default() })),
        }));
        match read(&mut c, &mut buf).body {
            Some(CT::Status(s)) => assert_eq!((s.free_slots, s.pending), (3, 1)),
            x => panic!("{x:?}"),
        }
        match read(&mut c, &mut buf).body {
            Some(CT::Token(t)) => assert_eq!((t.id, t.token, t.text.as_str(), t.logprob), (41, 9, "è", -0.5)),
            x => panic!("{x:?}"),
        }
        match read(&mut c, &mut buf).body {
            Some(CT::Done(f)) => assert_eq!((f.id, f.reason.as_str(), f.usage.unwrap().prompt_tokens), (41, "cancelled", 5)),
            x => panic!("{x:?}"),
        }
    }

    #[test]
    fn single_connection_and_reconnect_after_disconnect() {
        let sv = Service::listen("127.0.0.1:0").unwrap();
        let a = TcpStream::connect(sv.address().unwrap()).unwrap();
        assert!(matches!(sv.attendi_evento(T), Some(Event::Connesso)));
        // Check that refusing the extra client produces no head event.
        let mut b = TcpStream::connect(sv.address().unwrap()).unwrap();
        b.set_read_timeout(Some(T)).unwrap();
        let mut x = [0u8; 8];
        assert!(matches!(b.read(&mut x), Ok(0) | Err(_)));
        assert!(sv.prova_evento().is_none());
        // Once the active client drops, the head must receive the disconnect before a replacement connects.
        drop(a);
        assert!(matches!(sv.attendi_evento(T), Some(Event::Disconnesso)));
        assert!(!sv.manda(FromHead { body: Some(CT::Status(pb::Status::default())) }));
        let mut c = loop {
            let c = TcpStream::connect(sv.address().unwrap()).unwrap();
            if let Some(Event::Connesso) = sv.attendi_evento(Duration::from_millis(300)) {
                break c;
            }
        };
        write(&mut c, &request(1, "x"));
        assert!(matches!(sv.attendi_evento(T), Some(Event::Request(_))));
    }

    #[test]
    fn malformed_frame_closes_connection() {
        let sv = Service::listen("127.0.0.1:0").unwrap();
        let mut c = TcpStream::connect(sv.address().unwrap()).unwrap();
        assert!(matches!(sv.attendi_evento(T), Some(Event::Connesso)));
        c.write_all(&[0xff, 0xff, 0xff, 0xff, 0x0f]).unwrap();
        assert!(matches!(sv.attendi_evento(T), Some(Event::Disconnesso)));
    }

    #[test]
    fn split_utf8_emits_only_valid_text() {
        let mut p = Utf8Fragments::default();
        // Split two- and three-byte characters so no partial code point is emitted.
        assert_eq!(p.aggiungi(b"ab"), "ab");
        assert_eq!(p.aggiungi(&[0xC3]), "");
        assert_eq!(p.aggiungi(&[0xA8, b'!']), "è!");
        assert_eq!(p.aggiungi(&[0xE2]), "");
        assert_eq!(p.aggiungi(&[0x82]), "");
        assert_eq!(p.aggiungi(&[0xAC]), "€");
        assert_eq!(p.aggiungi(&[0xFF, b'x']), "\u{FFFD}x");
        assert_eq!(p.aggiungi(&[0xC3]), "");
        assert_eq!(p.clear(), "\u{FFFD}");
        assert_eq!(p.clear(), "");
        assert_eq!(p.aggiungi(b""), "");
    }
}
