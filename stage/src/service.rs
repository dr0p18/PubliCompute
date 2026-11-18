//! Local service socket for the head. It accepts one loopback client once
//! the chain is ready. Incoming frames become request events for the
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
use std::time::Duration;

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
    output: Arc<Mutex<Option<Sender<Vec<u8>>>>>,
    addr: SocketAddr,
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
        let output: Arc<Mutex<Option<Sender<Vec<u8>>>>> = Arc::new(Mutex::new(None));
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
        Ok(Service { rx, output, addr })
    }

    /// Actual address, useful when binding to port 0.
    pub fn address(&self) -> SocketAddr {
        self.addr
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

    /// Queue output for the active client. Return false if the frame exceeds
    /// the payload limit, there is no session, or its writer has closed.
    pub fn manda(&self, m: FromHead) -> bool {
        let mut v = Vec::with_capacity(m.encoded_len() + 8);
        if encode_frame(&m, &mut v).is_err() {
            return false;
        }
        match self.output.lock().unwrap().as_ref() {
            Some(tx) => tx.send(v).is_ok(),
            None => false,
        }
    }
}

fn start_connection(stream: TcpStream, ev: Sender<Event>, output: Arc<Mutex<Option<Sender<Vec<u8>>>>>, attivo: Arc<AtomicBool>) {
    let _ = stream.set_nodelay(true);
    let Ok(mut write) = stream.try_clone() else {
        attivo.store(false, Ordering::SeqCst);
        return;
    };
    let (tx_out, rx_out) = channel::<Vec<u8>>();
    *output.lock().unwrap() = Some(tx_out);
    let _ = ev.send(Event::Connesso);
    // Removing the sender lets the writer exit after pending output.
    std::thread::spawn(move || {
        for dati in rx_out {
            if write.write_all(&dati).is_err() {
                break;
            }
        }
        let _ = write.shutdown(std::net::Shutdown::Both);
    });
    std::thread::spawn(move || {
        let mut stream = stream;
        let mut buf = BytesMut::new();
        let mut tmp = vec![0u8; 64 << 10];
        'lettura: loop {
            loop {
                match try_decode_frame::<FromAgent>(&mut buf) {
                    Ok(Some(m)) => match m.body {
                        Some(pb::from_agent::Body::Request(r)) => {
                            let _ = ev.send(Event::Request(r));
                        }
                        Some(pb::from_agent::Body::Cancel(a)) => {
                            let _ = ev.send(Event::Cancel(a.id));
                        }
                        None => {}
                    },
                    Ok(None) => break,
                    Err(e) => {
                        eprintln!("[service] invalid frame from the agent: {e}");
                        break 'lettura;
                    }
                }
            }
            match stream.read(&mut tmp) {
                Ok(0) | Err(_) => break,
                Ok(n) => buf.extend_from_slice(&tmp[..n]),
            }
        }
        // Drop the departing client's output sender before notifying the head
        // so pending responses cannot enter the closed session.
        *output.lock().unwrap() = None;
        let _ = stream.shutdown(std::net::Shutdown::Both);
        let _ = ev.send(Event::Disconnesso);
        attivo.store(false, Ordering::SeqCst);
    });
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
pub fn gemma4_conversation(msgs: &[pb::Message]) -> Result<String, String> {
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
pub fn deepseek4_conversation(msgs: &[pb::Message]) -> Result<String, String> {
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

/// Remove the oldest non-system messages until the prompt fits, dropping
/// orphaned assistant replies along the way. Preserve system messages when
/// possible and always keep the last message. Return the trimmed history
/// and number of removals. An empty history, a token counting error, or
/// a final message too large to fit by itself is an error.
pub fn fit_context(
    mut msgs: Vec<pb::Message>,
    limite: usize,
    mut conta: impl FnMut(&[pb::Message]) -> Result<usize, String>,
) -> Result<(Vec<pb::Message>, usize), String> {
    let totale = msgs.len();
    let Some(last) = msgs.last().cloned() else { return Err("empty conversation".into()) };
    let sistema = |m: &pb::Message| m.role == "system";
    loop {
        if conta(&msgs)? <= limite {
            return Ok((msgs.clone(), totale - msgs.len()));
        }
        // Oldest non-system message, as long as it is not the last one.
        let Some(i) = msgs.iter().position(|m| !sistema(m)).filter(|&i| i + 1 < msgs.len()) else { break };
        msgs.remove(i);
        while msgs.iter().position(|m| !sistema(m)).is_some_and(|i| msgs[i].role == "assistant" && i + 1 < msgs.len()) {
            let i = msgs.iter().position(|m| !sistema(m)).unwrap();
            msgs.remove(i);
        }
    }
    // Keep the last message and the system message if present. If they do not fit, try the last message alone.
    if conta(&msgs)? <= limite {
        return Ok((msgs.clone(), totale - msgs.len()));
    }
    let solo = vec![last];
    if conta(&solo)? <= limite {
        return Ok((solo, totale - 1));
    }
    Err("the last message alone exceeds the slot context".into())
}

#[cfg(test)]
mod test {
    use super::*;
    use publicompute_proto::service::{from_agent::Body as CA, from_head::Body as CT};


    fn m(r: &str, t: &str) -> pb::Message {
        pb::Message { role: r.into(), text: t.into() }
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

    fn conta(v: &[pb::Message]) -> Result<usize, String> {
        Ok(v.iter().map(|x| x.text.len()).sum())
    }

    #[test]
    fn fit_context_drops_oldest_messages_and_keeps_system_message() {
        let v = vec![m("system", "SS"), m("user", "aaaa"), m("assistant", "bbbb"), m("user", "cccc"), m("assistant", "dddd"), m("user", "ee")];
        let (r, tolti) = fit_context(v.clone(), 100, conta).unwrap();
        assert_eq!((r.len(), tolti), (6, 0));
        let (r, tolti) = fit_context(v.clone(), 12, conta).unwrap();
        assert_eq!(tolti, 2);
        assert_eq!(r.iter().map(|x| x.text.as_str()).collect::<Vec<_>>(), ["SS", "cccc", "dddd", "ee"]);
        let (r, tolti) = fit_context(v.clone(), 4, conta).unwrap();
        assert_eq!((r.iter().map(|x| x.text.as_str()).collect::<Vec<_>>(), tolti), (vec!["SS", "ee"], 4));
        let (r, tolti) = fit_context(v.clone(), 3, conta).unwrap();
        assert_eq!((r.len(), tolti, r[0].text.as_str()), (1, 5, "ee"));
        assert!(fit_context(v, 1, conta).is_err());
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
        let mut c = TcpStream::connect(sv.address()).unwrap();
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
        let a = TcpStream::connect(sv.address()).unwrap();
        assert!(matches!(sv.attendi_evento(T), Some(Event::Connesso)));
        // Check that refusing the extra client produces no head event.
        let mut b = TcpStream::connect(sv.address()).unwrap();
        b.set_read_timeout(Some(T)).unwrap();
        let mut x = [0u8; 8];
        assert!(matches!(b.read(&mut x), Ok(0) | Err(_)));
        assert!(sv.prova_evento().is_none());
        // Once the active client drops, the head must receive the disconnect before a replacement connects.
        drop(a);
        assert!(matches!(sv.attendi_evento(T), Some(Event::Disconnesso)));
        assert!(!sv.manda(FromHead { body: Some(CT::Status(pb::Status::default())) }));
        let mut c = loop {
            let c = TcpStream::connect(sv.address()).unwrap();
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
        let mut c = TcpStream::connect(sv.address()).unwrap();
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
