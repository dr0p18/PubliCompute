//! Classify UDP reachability with two STUN binding requests from the same
//! socket to different ports. Equal mapped ports indicate an easy mapping,
//! different ports indicate a symmetric mapping, and no replies indicate a
//! blocked path. Only the classification leaves this module. Mapped ports
//! are used for comparison and discarded.

use std::net::SocketAddr;
use std::time::Duration;

use rand::random;
use tokio::net::UdpSocket;

const COOKIE: u32 = 0x2112_A442;
const BINDING_REQUEST: u16 = 0x0001;
const BINDING_SUCCESS: u16 = 0x0101;
const ATTR_MAPPED_ADDRESS: u16 = 0x0001;
const ATTR_XOR_MAPPED_ADDRESS: u16 = 0x0020;

/// Classification sent in the session handshake.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NatClass {
    Facile,
    Simmetrico,
    Bloccato,
    Sconosciuto,
}

impl NatClass {
    pub fn as_str(self) -> &'static str {
        match self {
            NatClass::Facile => "facile",
            NatClass::Simmetrico => "simmetrico",
            NatClass::Bloccato => "bloccato",
            NatClass::Sconosciuto => "sconosciuto",
        }
    }
}

impl std::fmt::Display for NatClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            NatClass::Facile => "easy",
            NatClass::Simmetrico => "symmetric",
            NatClass::Bloccato => "blocked",
            NatClass::Sconosciuto => "unknown",
        })
    }
}

/// Binding probe target with two UDP ports on the same host.
#[derive(Clone, Debug)]
pub struct StunCfg {
    pub host: String,
    pub port_a: u16,
    pub port_b: u16,
    pub timeout: Duration,
    pub attempts: u32,
}

impl StunCfg {
    pub fn new(host: String, port_a: u16, port_b: u16) -> Self {
        StunCfg { host, port_a, port_b, timeout: Duration::from_secs(2), attempts: 2 }
    }
}

/// A probe result carries only the mapped port because it is enough to compare replies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    Porta(u16),
    NessunaRisposta,
    Error,
}

/// Both production requests share one socket so the mapped ports are comparable.
pub trait Trasporto {
    async fn binding(&mut self, porta: u16) -> Outcome;
}

/// One reply is inconclusive. Two replies can be compared and two timeouts indicate a blocked path.
pub async fn classifica_con<T: Trasporto>(t: &mut T, port_a: u16, port_b: u16) -> NatClass {
    let a = t.binding(port_a).await;
    let b = t.binding(port_b).await;
    match (a, b) {
        (Outcome::Error, _) | (_, Outcome::Error) => NatClass::Sconosciuto,
        (Outcome::Porta(x), Outcome::Porta(y)) if x == y => NatClass::Facile,
        (Outcome::Porta(_), Outcome::Porta(_)) => NatClass::Simmetrico,
        (Outcome::NessunaRisposta, Outcome::NessunaRisposta) => NatClass::Bloccato,
        _ => NatClass::Sconosciuto,
    }
}

/// Probe the configured STUN target. DNS or socket errors yield an unknown classification.
pub async fn classifica(cfg: &StunCfg) -> NatClass {
    match UdpTrasporto::new(cfg).await {
        Some(mut t) => classifica_con(&mut t, cfg.port_a, cfg.port_b).await,
        None => NatClass::Sconosciuto,
    }
}

struct UdpTrasporto {
    sock: UdpSocket,
    ip: std::net::IpAddr,
    timeout: Duration,
    attempts: u32,
}

impl UdpTrasporto {
    async fn new(cfg: &StunCfg) -> Option<Self> {
        let ip = tokio::net::lookup_host((cfg.host.as_str(), cfg.port_a)).await.ok()?.find(|a| a.is_ipv4())?.ip();
        let sock = UdpSocket::bind("0.0.0.0:0").await.ok()?;
        Some(UdpTrasporto { sock, ip, timeout: cfg.timeout, attempts: cfg.attempts })
    }
}

impl Trasporto for UdpTrasporto {
    async fn binding(&mut self, porta: u16) -> Outcome {
        let dest = SocketAddr::new(self.ip, porta);
        for _ in 0..self.attempts.max(1) {
            let (req, tid) = codifica_richiesta();
            if self.sock.send_to(&req, dest).await.is_err() {
                return Outcome::Error;
            }
            let done = tokio::time::Instant::now() + self.timeout;
            let mut buf = [0u8; 1500];
            loop {
                match tokio::time::timeout_at(done, self.sock.recv_from(&mut buf)).await {
                    Err(_) => break, // Retry the probe when the read deadline expires.
                    Ok(Err(_)) => return Outcome::Error,
                    Ok(Ok((n, from))) => {
                        if from != dest {
                            continue;
                        }
                        if let Some(p) = decodifica_risposta(&buf[..n], &tid) {
                            return Outcome::Porta(p);
                        }
                    }
                }
            }
        }
        Outcome::NessunaRisposta
    }
}

/// Build the 20-byte STUN Binding Request header with a random transaction ID and no attributes.
pub fn codifica_richiesta() -> (Vec<u8>, [u8; 12]) {
    let tid: [u8; 12] = random();
    (codifica_richiesta_con(&tid), tid)
}

fn codifica_richiesta_con(tid: &[u8; 12]) -> Vec<u8> {
    let mut m = Vec::with_capacity(20);
    m.extend_from_slice(&BINDING_REQUEST.to_be_bytes());
    m.extend_from_slice(&0u16.to_be_bytes());
    m.extend_from_slice(&COOKIE.to_be_bytes());
    m.extend_from_slice(tid);
    m
}

/// Read the mapped port from a success reply with the expected transaction
/// ID. Prefer the XOR-mapped attribute and fall back to the plain mapped one.
pub fn decodifica_risposta(m: &[u8], tid: &[u8; 12]) -> Option<u16> {
    if m.len() < 20 || u16::from_be_bytes([m[0], m[1]]) != BINDING_SUCCESS {
        return None;
    }
    let len = u16::from_be_bytes([m[2], m[3]]) as usize;
    if u32::from_be_bytes([m[4], m[5], m[6], m[7]]) != COOKIE || &m[8..20] != tid || m.len() < 20 + len {
        return None;
    }
    let mut attrs = &m[20..20 + len];
    let mut semplice = None;
    while attrs.len() >= 4 {
        let t = u16::from_be_bytes([attrs[0], attrs[1]]);
        let l = u16::from_be_bytes([attrs[2], attrs[3]]) as usize;
        if attrs.len() < 4 + l {
            return None;
        }
        let v = &attrs[4..4 + l];
        if (t == ATTR_XOR_MAPPED_ADDRESS || t == ATTR_MAPPED_ADDRESS) && l >= 8 && matches!(v[1], 1 | 2) {
            let p = u16::from_be_bytes([v[2], v[3]]);
            if t == ATTR_XOR_MAPPED_ADDRESS {
                return Some(p ^ (COOKIE >> 16) as u16);
            }
            semplice = Some(p);
        }
        let next_offset = 4 + l.div_ceil(4) * 4;
        attrs = if attrs.len() >= next_offset { &attrs[next_offset..] } else { &[] };
    }
    semplice
}

#[cfg(test)]
mod tests {
    use super::*;

    fn risposta(tid: &[u8; 12], porta: u16, famiglia: u8) -> Vec<u8> {
        let xp = porta ^ (COOKIE >> 16) as u16;
        let mut val = vec![0, famiglia];
        val.extend_from_slice(&xp.to_be_bytes());
        let ip = [203u8, 0, 113, 7];
        let cookie = COOKIE.to_be_bytes();
        if famiglia == 1 {
            val.extend(ip.iter().zip(cookie).map(|(a, b)| a ^ b));
        } else {
            val.extend([0u8; 16]);
        }
        let mut attr = Vec::new();
        attr.extend_from_slice(&ATTR_XOR_MAPPED_ADDRESS.to_be_bytes());
        attr.extend_from_slice(&(val.len() as u16).to_be_bytes());
        attr.extend(val);
        let mut m = Vec::new();
        m.extend_from_slice(&BINDING_SUCCESS.to_be_bytes());
        m.extend_from_slice(&(attr.len() as u16).to_be_bytes());
        m.extend_from_slice(&COOKIE.to_be_bytes());
        m.extend_from_slice(tid);
        m.extend(attr);
        m
    }

    #[test]
    fn binding_request_is_well_formed() {
        let tid = [7u8; 12];
        let m = codifica_richiesta_con(&tid);
        assert_eq!(m.len(), 20);
        assert_eq!(&m[0..2], &[0x00, 0x01]);
        assert_eq!(&m[2..4], &[0, 0]);
        assert_eq!(&m[4..8], &[0x21, 0x12, 0xA4, 0x42]);
        assert_eq!(&m[8..], &tid);
        let (r1, t1) = codifica_richiesta();
        let (_, t2) = codifica_richiesta();
        assert_eq!(&r1[8..], &t1);
        assert_ne!(t1, t2);
    }

    #[test]
    fn decodes_xor_mapped_ipv4() {
        let tid = [9u8; 12];
        assert_eq!(decodifica_risposta(&risposta(&tid, 54321, 1), &tid), Some(54321));
        assert_eq!(decodifica_risposta(&risposta(&tid, 54321, 2), &tid), Some(54321));
    }

    #[test]
    fn decoder_rejects_invalid_responses() {
        let tid = [9u8; 12];
        let ok = risposta(&tid, 4000, 1);
        assert_eq!(decodifica_risposta(&ok, &[1u8; 12]), None, "wrong transaction ID");
        assert_eq!(decodifica_risposta(&ok[..10], &tid), None, "truncated response");
        let mut error = ok.clone();
        error[0] = 0x01;
        error[1] = 0x11;
        assert_eq!(decodifica_risposta(&error, &tid), None, "expected a Binding Success response");
        let mut cookie = ok;
        cookie[4] = 0;
        assert_eq!(decodifica_risposta(&cookie, &tid), None);
    }

    struct Finto(Vec<Outcome>);
    impl Trasporto for Finto {
        async fn binding(&mut self, _porta: u16) -> Outcome {
            self.0.remove(0)
        }
    }

    async fn class(e: [Outcome; 2]) -> NatClass {
        classifica_con(&mut Finto(e.to_vec()), 3478, 3479).await
    }

    #[tokio::test]
    async fn classifies_port_mappings() {
        use Outcome::*;
        assert_eq!(class([Porta(5000), Porta(5000)]).await, NatClass::Facile);
        assert_eq!(class([Porta(5000), Porta(5001)]).await, NatClass::Simmetrico);
        assert_eq!(class([NessunaRisposta, NessunaRisposta]).await, NatClass::Bloccato);
        assert_eq!(class([Error, Error]).await, NatClass::Sconosciuto);
        assert_eq!(class([Porta(5000), NessunaRisposta]).await, NatClass::Sconosciuto);
        assert_eq!(NatClass::Bloccato.as_str(), "bloccato");
    }

    /// Exercise both a stable port mapping and a target that never replies.
    #[tokio::test]
    async fn udp_transport_on_loopback() {
        let srv = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let porta = srv.local_addr().unwrap().port();
        tokio::spawn(async move {
            let mut b = [0u8; 64];
            while let Ok((n, from)) = srv.recv_from(&mut b).await {
                if n == 20 {
                    let mut tid = [0u8; 12];
                    tid.copy_from_slice(&b[8..20]);
                    let _ = srv.send_to(&risposta(&tid, from.port(), 1), from).await;
                }
            }
        });
        let mut cfg = StunCfg::new("127.0.0.1".into(), porta, porta);
        assert_eq!(classifica(&cfg).await, NatClass::Facile);
        let muto = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        cfg.port_b = muto.local_addr().unwrap().port();
        cfg.timeout = Duration::from_millis(100);
        cfg.attempts = 1;
        assert_eq!(classifica(&cfg).await, NatClass::Sconosciuto);
        cfg.port_a = cfg.port_b;
        assert_eq!(classifica(&cfg).await, NatClass::Bloccato);
    }

    /// This probe requires network access to the configured STUN service.
    #[tokio::test]
    #[ignore]
    async fn real_stun_probe() {
        let c = classifica(&StunCfg::new("relay.publicompute.com".into(), 3478, 3479)).await;
        println!("NAT type: {c}");
    }
}
