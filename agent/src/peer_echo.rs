//! Echo a bounded sequence on a separate protocol so a probe cannot reach a stage.

use std::{
    collections::{HashMap, HashSet},
    fmt,
    future::Future,
    sync::{Arc, Mutex},
    time::{Duration, SystemTime},
};

use anyhow::{anyhow, ensure, Context, Result};
use futures_util::{FutureExt, StreamExt};
use iroh::{
    endpoint::{Connection, PathEvent, PathId, VarInt},
    protocol::{AcceptError, ProtocolHandler},
    Endpoint, EndpointAddr, EndpointId, RelayUrl, TransportAddr,
};
use publicompute_proto::node::{PeerProbe, PeerProbePath, PeerProbeResult};
use tokio::{
    sync::Semaphore,
    time::{self, Instant},
};

use crate::network::{median_and_p95, micros, RTT_SAMPLES};

pub const ALPN: &[u8] = b"publicompute/echo/0";
const MAX_PROBE: Duration = Duration::from_secs(20);
const PATH_WAIT: Duration = Duration::from_secs(4);
const PERMITS: usize = 8;

type Nonce = [u8; 32];

pub struct Request {
    peer: EndpointId,
    nonce: Nonce,
    deadline: Instant,
}

impl Request {
    pub fn from_proto(probe: &PeerProbe, clock_offset: i64) -> Option<Self> {
        let expiry: SystemTime = probe.expires_at?.try_into().ok()?;
        let skew = Duration::from_secs(clock_offset.unsigned_abs());
        let local = SystemTime::now();
        let now = if clock_offset < 0 {
            local.checked_sub(skew)?
        } else {
            local.checked_add(skew)?
        };
        let remaining = expiry.duration_since(now).ok()?;
        if remaining.is_zero() || remaining > MAX_PROBE {
            return None;
        }
        Some(Self {
            peer: probe.peer_id.parse().ok()?,
            nonce: probe.nonce.as_slice().try_into().ok()?,
            deadline: Instant::now() + remaining,
        })
    }
}

struct Permit {
    peer: EndpointId,
    deadline: Instant,
    used: bool,
}

#[derive(Clone)]
pub struct Echo {
    state: Arc<Mutex<EchoState>>,
    connections: Arc<Semaphore>,
}

#[derive(Default)]
struct EchoState {
    permits: HashMap<Nonce, Permit>,
    waiting: HashSet<EndpointId>,
}

struct Waiting<'a> {
    state: &'a Mutex<EchoState>,
    peer: EndpointId,
    deadline: Instant,
}

impl Drop for Waiting<'_> {
    fn drop(&mut self) {
        self.state.lock().unwrap().waiting.remove(&self.peer);
    }
}

impl fmt::Debug for Echo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Echo").finish_non_exhaustive()
    }
}

impl Default for Echo {
    fn default() -> Self {
        Self {
            state: Default::default(),
            connections: Arc::new(Semaphore::new(PERMITS)),
        }
    }
}

impl Echo {
    pub fn prepare(&self, request: Request) -> bool {
        let mut state = self.state.lock().unwrap();
        let permits = &mut state.permits;
        let now = Instant::now();
        permits.retain(|_, permit| permit.deadline > now);
        if request.deadline <= now
            || permits.len() >= PERMITS
            || permits.contains_key(&request.nonce)
        {
            return false;
        }
        permits.insert(
            request.nonce,
            Permit {
                peer: request.peer,
                deadline: request.deadline,
                used: false,
            },
        );
        true
    }

    pub fn clear(&self) {
        self.state.lock().unwrap().permits.clear();
    }

    fn begin_wait(&self, peer: EndpointId) -> Option<Waiting<'_>> {
        let now = Instant::now();
        let mut state = self.state.lock().unwrap();
        let deadline = state
            .permits
            .values()
            .filter(|p| p.peer == peer && !p.used && now < p.deadline)
            .map(|p| p.deadline)
            .min()?;
        state.waiting.insert(peer).then_some(())?;
        Some(Waiting {
            state: &self.state,
            peer,
            deadline,
        })
    }

    fn claim(&self, peer: EndpointId, nonce: Nonce) -> Option<Instant> {
        let mut state = self.state.lock().unwrap();
        let permit = state.permits.get_mut(&nonce)?;
        if permit.peer != peer || permit.used || Instant::now() >= permit.deadline {
            return None;
        }
        permit.used = true;
        Some(permit.deadline)
    }

    async fn respond(&self, conn: &Connection) -> Result<()> {
        let peer = conn.remote_id();
        let waiting = self
            .begin_wait(peer)
            .context("unexpected echo peer or nonce already pending")?;
        let _slot = self.connections.try_acquire()?;
        // A peer must present its nonce on the first stream before it is echoed.
        let ((mut send, mut recv), deadline) = time::timeout_at(waiting.deadline, async {
            let (send, mut recv) = conn.accept_bi().await?;
            let mut nonce = [0; 32];
            recv.read_exact(&mut nonce).await?;
            let deadline = self
                .claim(peer, nonce)
                .context("unknown or spent echo nonce")?;
            Ok::<_, anyhow::Error>(((send, recv), deadline))
        })
        .await??;
        drop(waiting);
        time::timeout_at(deadline, async {
            let selected = wait_path(conn).await?;
            stable_path(conn, selected, async {
                send.write_all(&[u8::from(selected.direct)]).await?;
                for sequence in 0..RTT_SAMPLES as u8 {
                    let mut frame = [0];
                    recv.read_exact(&mut frame).await?;
                    ensure!(frame == [sequence], "unexpected echo sequence");
                    send.write_all(&frame).await?;
                }
                Ok(())
            })
            .await?;
            // The dialer requires this confirmation that the receiver's path also stayed fixed.
            send.write_all(&[1]).await?;
            send.finish()?;
            send.stopped().await?;
            Ok(())
        })
        .await?
    }
}

impl ProtocolHandler for Echo {
    async fn accept(&self, conn: Connection) -> Result<(), AcceptError> {
        if let Err(err) = self.respond(&conn).await {
            tracing::debug!(%err, "peer echo rejected or interrupted");
            conn.close(VarInt::from_u32(1), b"echo rejected");
        }
        Ok(())
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct Selected {
    id: PathId,
    direct: bool,
}

impl Selected {
    fn replaced_by(self, id: PathId, direct: bool) -> bool {
        self != Self { id, direct }
    }
}

fn selected_path(conn: &Connection) -> Option<Selected> {
    conn.paths()
        .iter()
        .find(|path| path.is_selected() && (path.is_ip() || path.is_relay()))
        .map(|path| Selected {
            id: path.id(),
            direct: path.is_ip(),
        })
}

async fn wait_path(conn: &Connection) -> Result<Selected> {
    // A validated IP path can exist while application data still uses the relay.
    let mut paths = conn.paths_stream();
    let _ = time::timeout(PATH_WAIT, async {
        while let Some(list) = paths.next().await {
            if list.iter().any(|path| path.is_ip() && path.is_selected()) {
                break;
            }
        }
    })
    .await;
    selected_path(conn).context("no selected path")
}

fn path_interrupted(event: Option<PathEvent>, selected: Selected) -> bool {
    match event {
        Some(PathEvent::Selected {
            id, remote_addr, ..
        }) => selected.replaced_by(id, remote_addr.is_ip()),
        Some(PathEvent::Closed { id, .. }) => id == selected.id,
        Some(PathEvent::Lagged { .. }) | None => true,
        _ => false,
    }
}

async fn stable_path<T>(
    conn: &Connection,
    selected: Selected,
    work: impl Future<Output = Result<T>>,
) -> Result<T> {
    let mut events = conn.path_events();
    ensure!(
        selected_path(conn) == Some(selected),
        "echo path changed before sampling"
    );
    tokio::pin!(work);
    loop {
        tokio::select! {
            biased;
            event = events.next() => ensure!(!path_interrupted(event, selected), "echo path changed during sampling"),
            result = &mut work => {
                while let Some(event) = events.next().now_or_never() {
                    ensure!(!path_interrupted(event, selected), "echo path changed during sampling");
                }
                ensure!(selected_path(conn) == Some(selected), "echo path changed during sampling");
                return result;
            }
        }
    }
}

pub async fn probe(endpoint: Endpoint, relay: RelayUrl, request: Request) -> PeerProbeResult {
    let nonce = request.nonce.to_vec();
    match time::timeout_at(request.deadline, dial(endpoint, relay, request)).await {
        Ok(Ok((path, median_micros, p95_micros))) => PeerProbeResult {
            nonce,
            path: path as i32,
            median_micros,
            p95_micros,
        },
        error => {
            tracing::debug!(?error, "peer echo failed");
            PeerProbeResult {
                nonce,
                ..Default::default()
            }
        }
    }
}

async fn dial(
    endpoint: Endpoint,
    relay: RelayUrl,
    request: Request,
) -> Result<(PeerProbePath, u32, u32)> {
    let addr = EndpointAddr::from_parts(request.peer, [TransportAddr::Relay(relay)]);
    let conn = endpoint
        .connect(addr, ALPN)
        .await
        .map_err(|err| anyhow!("connect peer echo: {err}"))?;
    let (mut send, mut recv) = conn.open_bi().await?;
    send.write_all(&request.nonce).await?;
    let mut path = [0];
    recv.read_exact(&mut path).await?;
    ensure!(path[0] <= 1, "unexpected echo path");
    let selected = wait_path(&conn).await?;
    let mut samples = stable_path(&conn, selected, async {
        let mut samples = [0; RTT_SAMPLES];
        for (sequence, sample) in samples.iter_mut().enumerate() {
            let start = Instant::now();
            send.write_all(&[sequence as u8]).await?;
            let mut frame = [0];
            recv.read_exact(&mut frame).await?;
            *sample = micros(start.elapsed());
            ensure!(frame == [sequence as u8], "incorrect echo reply");
        }
        let mut confirmation = [0];
        recv.read_exact(&mut confirmation).await?;
        ensure!(confirmation == [1], "unconfirmed receiver path");
        Ok(samples)
    })
    .await?;
    send.finish()?;
    let (median, p95) = median_and_p95(&mut samples);
    let path = if selected.direct && path == [1] {
        PeerProbePath::Direct
    } else {
        PeerProbePath::Relay
    };
    Ok((path, median, p95))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bridge::tests::{ponte_di_prova, relay_di_prova};

    fn request(peer: EndpointId, nonce: Nonce) -> Request {
        Request {
            peer,
            nonce,
            deadline: Instant::now() + Duration::from_secs(15),
        }
    }

    #[test]
    fn authorization_binds_the_nonce_to_one_peer_and_one_use() {
        let echo = Echo::default();
        let peer = iroh::SecretKey::from_bytes(&[1; 32]).public();
        let stranger = iroh::SecretKey::from_bytes(&[2; 32]).public();
        let nonce = [3; 32];
        assert!(echo.begin_wait(peer).is_none());
        assert!(echo.prepare(request(peer, nonce)));
        assert!(echo.begin_wait(peer).is_some());
        assert!(echo.begin_wait(stranger).is_none());
        assert!(echo.claim(stranger, nonce).is_none());
        assert!(echo.claim(peer, [4; 32]).is_none());
        assert!(echo.claim(peer, nonce).is_some());
        assert!(echo.claim(peer, nonce).is_none());
        assert!(!echo.prepare(request(peer, nonce)));
        assert!(!echo.prepare(Request {
            peer,
            nonce: [5; 32],
            deadline: Instant::now()
        }));
    }

    #[test]
    fn a_relay_selection_does_not_match_the_sampled_direct_path() {
        let direct = Selected {
            id: PathId::ZERO,
            direct: true,
        };
        assert!(!direct.replaced_by(PathId::ZERO, true));
        assert!(direct.replaced_by(PathId::ZERO, false));
        assert!(direct.replaced_by(PathId::MAX, true));
        assert!(path_interrupted(None, direct));
    }

    #[tokio::test]
    async fn only_one_nonce_wait_per_peer_and_expiration_releases_the_slot() {
        use iroh::{
            endpoint::{presets, PortmapperConfig},
            protocol::Router,
            RelayMode,
        };

        let endpoint = || {
            Endpoint::builder(presets::Minimal)
                .relay_mode(RelayMode::Disabled)
                .portmapper_config(PortmapperConfig::Disabled)
                .bind()
        };
        let echo = Echo::default();
        let receiver = Router::builder(endpoint().await.unwrap())
            .accept(ALPN, echo.clone())
            .spawn();
        let sender = endpoint().await.unwrap();
        let deadline = Instant::now() + Duration::from_secs(2);
        assert!(echo.prepare(Request {
            peer: sender.id(),
            nonce: [1; 32],
            deadline
        }));
        let conn = sender
            .connect(receiver.endpoint().addr(), ALPN)
            .await
            .unwrap();
        let (mut send, _recv) = conn.open_bi().await.unwrap();
        send.write_all(&[1]).await.unwrap(); // Only a prefix of the nonce.

        time::timeout(Duration::from_secs(1), async {
            while echo.connections.available_permits() == PERMITS {
                time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap();
        let duplicate = sender
            .connect(receiver.endpoint().addr(), ALPN)
            .await
            .unwrap();
        time::timeout_at(deadline, duplicate.closed())
            .await
            .unwrap();
        assert_eq!(echo.connections.available_permits(), PERMITS - 1);

        time::timeout_at(deadline + Duration::from_secs(1), conn.closed())
            .await
            .unwrap();
        assert!(Instant::now() >= deadline);
        assert_eq!(echo.connections.available_permits(), PERMITS);
        assert!(echo.prepare(request(sender.id(), [2; 32])));
        assert!(echo.begin_wait(sender.id()).is_some());
        sender.close().await;
        receiver.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn echo_protocol_requires_the_prepared_peer_and_handles_ten_frames() {
        let (_relay, url) = relay_di_prova().await;
        let sender = ponte_di_prova(&url).await;
        let receiver = ponte_di_prova(&url).await;
        let stranger = ponte_di_prova(&url).await;
        let (sender_id, receiver_id) =
            (sender.id().parse().unwrap(), receiver.id().parse().unwrap());
        let nonce = [3; 32];
        assert!(receiver.echo.prepare(request(sender_id, nonce)));
        let unwanted = stranger.probe(request(receiver_id, nonce)).await;
        assert_eq!(unwanted.path(), PeerProbePath::Unspecified);
        let measured = sender.probe(request(receiver_id, nonce)).await;
        assert!(matches!(
            measured.path(),
            PeerProbePath::Direct | PeerProbePath::Relay
        ));
        assert!(measured.median_micros > 0 && measured.p95_micros >= measured.median_micros);
        assert_eq!(
            sender.probe(request(receiver_id, nonce)).await.path(),
            PeerProbePath::Unspecified
        );
        sender.close().await;
        receiver.close().await;
        stranger.close().await;
    }
}
