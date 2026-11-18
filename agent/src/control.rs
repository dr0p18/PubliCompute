//! The local control protocol uses one JSON line per request and reply.
//! A follow request keeps the connection open and emits a snapshot whenever
//! its contents change. The default socket is private. A configured socket
//! path permits access by the agent's group. The session and preference
//! store own the states exposed here.

use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, AtomicU32, Ordering},
        Arc, Mutex,
    },
    time::{Duration, Instant},
};

use publicompute_proto::{
    common::ModelRef,
    node::{ContributionStatus, ContributionSummary},
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::watch;

use crate::{
    contribution::{Contribution, PreferenceMode, ModelSelection},
    engine::supervisor::EngineState,
    gpu::Telemetria,
};

/// Minimum interval between accepted preference changes.
const PREFERENCE_GAP: Duration = Duration::from_secs(5);
/// GPU samples have no change notification, so followers poll for them at this interval.
const FOLLOW_POLL: Duration = Duration::from_secs(10);

/// Changes to the shared session state advance the tick counter to wake
/// status followers. The same state holds the local pause switch.
pub struct Live {
    connected: AtomicBool,
    leases: AtomicU32,
    chain: Mutex<Option<String>>,
    paused: watch::Sender<bool>,
    tick: watch::Sender<u64>,
}

impl Live {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            connected: AtomicBool::new(false),
            leases: AtomicU32::new(0),
            chain: Mutex::new(None),
            paused: watch::channel(false).0,
            tick: watch::channel(0).0,
        })
    }

    /// Notify followers when a value stored outside this state changes.
    pub fn notify(&self) {
        self.tick.send_modify(|t| *t += 1);
    }

    pub fn set_connected(&self, v: bool) {
        if self.connected.swap(v, Ordering::Relaxed) != v {
            self.notify();
        }
    }

    pub fn set_leases(&self, n: u32) {
        if self.leases.swap(n, Ordering::Relaxed) != n {
            self.notify();
        }
    }

    pub fn set_chain(&self, id: Option<String>) {
        let mut c = self.chain.lock().unwrap();
        if *c != id {
            *c = id;
            drop(c);
            self.notify();
        }
    }

    pub fn is_paused(&self) -> bool {
        *self.paused.borrow()
    }

    /// Pausing lasts until restart, which makes the card available again.
    pub fn set_paused(&self, v: bool) {
        if self.paused.send_replace(v) != v {
            self.notify();
        }
    }

    /// Subscribe to changes the session must publish as availability updates.
    pub fn pause_rx(&self) -> watch::Receiver<bool> {
        self.paused.subscribe()
    }
}

#[derive(Deserialize)]
#[serde(tag = "op", rename_all = "lowercase")]
enum Request {
    Status,
    Follow,
    Preference { mode: String, model: Option<String>, quant: Option<String> },
    Pause { active: bool },
    Models,
}

#[derive(Serialize, Clone, PartialEq)]
struct PreferenceView {
    mode: &'static str,
    model: Option<String>,
    quant: Option<String>,
    revision: u64,
}

#[derive(Serialize, Clone, PartialEq)]
struct GpuView {
    name: String,
    vram_used_bytes: u64,
    vram_total_bytes: u64,
    temperature_celsius: u32,
}

#[derive(Serialize, Clone, PartialEq)]
struct ContributionView {
    effective_mode: &'static str,
    serving: Option<String>,
    in_chain: bool,
    free_seat: bool,
    reason: String,
    /// Earliest time for a preference change, in Unix seconds.
    stable_until: Option<i64>,
    revision: u64,
}

#[derive(Serialize, Clone, PartialEq)]
struct SummaryView {
    days: u32,
    free_value_mc: i64,
    free_chats: u32,
    earned_mc: i64,
    service_seconds: u64,
    work_seconds: u64,
    balance_mc: i64,
}

#[derive(Serialize, Clone, PartialEq)]
struct Status {
    version: &'static str,
    connected: bool,
    paused: bool,
    preference: PreferenceView,
    engine: &'static str,
    loaded_model: Option<String>,
    active_leases: u32,
    chain_id: Option<String>,
    gpu: Option<GpuView>,
    gpu_lost: bool,
    gpu_lost_reason: Option<&'static str>,
    contribution: Option<ContributionView>,
    summary: Option<SummaryView>,
}

#[derive(Serialize, Clone)]
struct LocalModel {
    id: String,
    quant: String,
    state: &'static str,
    loaded: bool,
    bytes: u64,
}

/// Replies carry a stable error code and a displayable message.
struct Refusal(&'static str, String);

pub struct Control {
    pub contribution: Arc<Contribution>,
    pub gpu: Arc<Telemetria>,
    pub live: Arc<Live>,
    pub engine: watch::Receiver<EngineState>,
    pub model: ModelRef,
    pub model_path: Option<PathBuf>,
    last_change: Mutex<Option<Instant>>,
}

fn mode_name(m: PreferenceMode) -> &'static str {
    match m {
        PreferenceMode::Usage => "use",
        PreferenceMode::Free => "free",
    }
}

fn engine_name(s: EngineState) -> &'static str {
    match s {
        EngineState::Starting => "starting",
        EngineState::Ready => "ready",
        EngineState::Busy => "working",
        EngineState::Settling => "settling",
        EngineState::Recovering => "recovering",
        EngineState::Quarantined => "quarantined",
    }
}

fn contribution_view(s: &ContributionStatus) -> ContributionView {
    ContributionView {
        effective_mode: match s.effective_mode() {
            publicompute_proto::node::ContributionMode::Free => "free",
            _ => "use",
        },
        serving: s.serving.as_ref().map(|m| m.model_id.clone()).filter(|id| !id.is_empty()),
        in_chain: s.in_chain,
        free_seat: s.free_seat,
        reason: s.reason().as_str_name().to_string(),
        stable_until: s.stable_until.as_ref().map(|t| t.seconds),
        revision: s.revision,
    }
}

fn summary_view(r: &ContributionSummary) -> SummaryView {
    SummaryView {
        days: r.days,
        free_value_mc: r.free_value_mc,
        free_chats: r.free_chats,
        earned_mc: r.earned_mc,
        service_seconds: r.service_seconds,
        work_seconds: r.work_seconds,
        balance_mc: r.balance_mc,
    }
}

impl Control {
    pub fn new(
        contribution: Arc<Contribution>,
        gpu: Arc<Telemetria>,
        live: Arc<Live>,
        engine: watch::Receiver<EngineState>,
        model: ModelRef,
        model_path: Option<PathBuf>,
    ) -> Arc<Self> {
        Arc::new(Self { contribution, gpu, live, engine, model, model_path, last_change: Mutex::new(None) })
    }

    /// Selection is limited to the weights the agent has loaded.
    fn models(&self) -> Vec<LocalModel> {
        if self.model.model_id.is_empty() {
            return vec![];
        }
        let bytes = self.model_path.as_ref().and_then(|p| std::fs::metadata(p).ok()).map(|m| m.len()).unwrap_or(0);
        let engine = *self.engine.borrow();
        let loaded = self.gpu.loss().is_none() && matches!(engine, EngineState::Ready | EngineState::Busy | EngineState::Settling);
        vec![LocalModel { id: self.model.model_id.clone(), quant: self.model.quantization.clone(), state: if loaded { "ready" } else { engine_name(engine) }, loaded, bytes }]
    }

    fn status(&self) -> Status {
        let p = self.contribution.preference();
        let gpu = self.gpu.letture().into_iter().next().map(|l| GpuView {
            name: l.name,
            vram_used_bytes: l.vram_usata,
            vram_total_bytes: l.vram_totale,
            temperature_celsius: l.temperature,
        });
        let gpu_loss = self.gpu.loss();
        Status {
            version: env!("CARGO_PKG_VERSION"),
            connected: self.live.connected.load(Ordering::Relaxed),
            paused: self.live.is_paused(),
            preference: PreferenceView {
                mode: mode_name(p.modo),
                model: p.model.as_ref().map(|m| m.id.clone()),
                quant: p.model.as_ref().map(|m| m.quant.clone()),
                revision: p.revision,
            },
            engine: engine_name(*self.engine.borrow()),
            loaded_model: self.models().into_iter().find(|m| m.loaded).map(|m| m.id),
            active_leases: self.live.leases.load(Ordering::Relaxed),
            chain_id: self.live.chain.lock().unwrap().clone(),
            gpu,
            gpu_lost: gpu_loss.is_some(),
            gpu_lost_reason: gpu_loss.map(crate::gpu::ReadFailure::reason),
            contribution: self.contribution.state().as_ref().map(contribution_view),
            summary: self.contribution.summary().as_ref().map(summary_view),
        }
    }

    fn set_preference(&self, mode: &str, model: Option<String>, quant: Option<String>) -> Result<Value, Refusal> {
        let bad = |t: &str| Refusal("bad_request", t.to_string());
        let (modo, scelto) = match mode {
            "free" => (PreferenceMode::Free, None),
            "use" => {
                let id = model.ok_or_else(|| bad("mode use needs a model"))?;
                let local = self.models();
                let found = local.iter().find(|m| m.id == id && quant.as_deref().is_none_or(|q| q == m.quant));
                let Some(found) = found else {
                    return Err(Refusal("not_installed", format!("model {id} is not installed on this machine")));
                };
                (PreferenceMode::Usage, Some(ModelSelection { id: found.id.clone(), quant: found.quant.clone() }))
            }
            other => return Err(Refusal("bad_request", format!("unknown mode \"{other}\": use or free"))),
        };
        // Apply the cooldown after validation so rejected requests do not consume it.
        let mut last = self.last_change.lock().unwrap();
        if last.is_some_and(|t| t.elapsed() < PREFERENCE_GAP) {
            return Err(Refusal("too_soon", "preference changed a moment ago: wait a few seconds".into()));
        }
        let p = self.contribution.imposta(modo, scelto).map_err(|e| Refusal("internal", format!("{e:#}")))?;
        *last = Some(Instant::now());
        self.live.notify();
        Ok(json!({"ok": true, "revision": p.revision}))
    }

    /// Handle requests that complete with a single reply.
    fn handle(&self, req: Request) -> Value {
        let refused = |r: Refusal| json!({"ok": false, "error": r.0, "text": r.1});
        match req {
            Request::Status => json!({"ok": true, "status": self.status()}),
            Request::Models => json!({"ok": true, "models": self.models()}),
            Request::Pause { active } => {
                self.live.set_paused(active);
                json!({"ok": true})
            }
            Request::Preference { mode, model, quant } => self.set_preference(&mode, model, quant).unwrap_or_else(refused),
            Request::Follow => unreachable!("follow is served by `follow`"),
        }
    }
}

#[cfg(unix)]
pub use unix::Listener;

#[cfg(not(unix))]
pub struct Listener;

#[cfg(not(unix))]
impl Listener {
    pub fn bind(_: PathBuf, _: bool) -> anyhow::Result<Self> {
        tracing::warn!("local control socket is unavailable on this platform");
        Ok(Self)
    }

    pub fn serve(self, _: Arc<Control>) -> anyhow::Result<()> {
        Ok(())
    }
}

#[cfg(unix)]
mod unix {
    use super::*;
    use anyhow::{Context, Result};
    use std::{
        io::ErrorKind,
        os::unix::{fs::FileTypeExt, net},
    };
    use tokio::{
        io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
        net::{unix::OwnedWriteHalf, UnixListener, UnixStream},
    };
    use tracing::{debug, info};

    pub struct Listener(pub(super) net::UnixListener);

    impl Listener {
        // Set the process-wide umask before starting the runtime so the bind inherits it.
        pub fn bind(path: PathBuf, group_access: bool) -> Result<Self> {
            let bind = || -> std::io::Result<net::UnixListener> {
                if std::fs::symlink_metadata(&path).is_ok_and(|m| m.file_type().is_socket()) {
                    if net::UnixStream::connect(&path).is_err_and(|e| e.kind() == ErrorKind::ConnectionRefused) {
                        std::fs::remove_file(&path)?;
                    }
                }
                let mode = if group_access { 0o660 } else { 0o600 };
                let old = unsafe { libc::umask(0o777 ^ mode) };
                let listener = net::UnixListener::bind(&path);
                unsafe { libc::umask(old) };
                listener
            };
            let listener = bind().with_context(|| format!("bind control socket {}", path.display()))?;
            listener.set_nonblocking(true)?;
            Ok(Self(listener))
        }

        pub fn serve(self, control: Arc<Control>) -> Result<()> {
            let listener = UnixListener::from_std(self.0)?;
            info!(socket = ?listener.local_addr(), "control socket ready");
            tokio::spawn(async move {
                loop {
                    match listener.accept().await {
                        Ok((stream, _)) => {
                            tokio::spawn(client(control.clone(), stream));
                        }
                        Err(e) => debug!("accept on the control socket: {e}"),
                    }
                }
            });
            Ok(())
        }
    }

    async fn send(w: &mut OwnedWriteHalf, v: &Value) -> std::io::Result<()> {
        let mut line = v.to_string();
        line.push('\n');
        w.write_all(line.as_bytes()).await
    }

    async fn client(control: Arc<Control>, stream: UnixStream) {
        let (r, mut w) = stream.into_split();
        let mut lines = BufReader::new(r).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            let reply = match serde_json::from_str::<Request>(&line) {
                Ok(Request::Follow) => {
                    // A follower owns this connection until it disconnects.
                    follow(&control, &mut w, &mut lines).await;
                    return;
                }
                Ok(req) => control.handle(req),
                Err(e) => json!({"ok": false, "error": "bad_request", "text": e.to_string()}),
            };
            if send(&mut w, &reply).await.is_err() {
                return;
            }
        }
    }

    /// Send an initial snapshot and then send only snapshots that differ from
    /// the last one. Notifications can wake a follower without changing any visible field.
    async fn follow(c: &Control, w: &mut OwnedWriteHalf, lines: &mut tokio::io::Lines<BufReader<tokio::net::unix::OwnedReadHalf>>) {
        let mut tick = c.live.tick.subscribe();
        let mut pref = c.contribution.subscribe();
        let mut engine = c.engine.clone();
        let mut gpu = c.gpu.device_rx();
        let mut poll = tokio::time::interval(FOLLOW_POLL);
        let mut last: Option<Status> = None;
        loop {
            let now = c.status();
            if last.as_ref() != Some(&now) {
                if send(w, &json!({"ok": true, "status": now})).await.is_err() {
                    return;
                }
                last = Some(now);
            }
            tokio::select! {
                _ = tick.changed() => {}
                _ = pref.changed() => {}
                _ = engine.changed() => {}
                _ = gpu.changed() => {}
                _ = poll.tick() => {}
                // After subscription the connection is one-way. Any input, including EOF, ends it.
                _ = lines.next_line() => return,
            }
        }
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::contribution::Preference;
    use tokio::{
        io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
        net::UnixStream,
    };

    struct Rig {
        sock: PathBuf,
        control: Arc<Control>,
        engine: watch::Sender<EngineState>,
    }

    fn rig(name: &str) -> Rig {
        let dir = std::env::temp_dir().join(format!("publicompute-control-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let pref = Preference { modo: PreferenceMode::Usage, model: Some(ModelSelection { id: "qwen".into(), quant: "Q4_K_M".into() }), revision: 0 };
        let contribution = Contribution::open(&dir, pref).unwrap();
        let (engine_tx, engine_rx) = watch::channel(EngineState::Ready);
        let model = ModelRef { model_id: "qwen".into(), quantization: "Q4_K_M".into(), weights_sha256: vec![] };
        let control = Control::new(contribution, Arc::new(Telemetria::default()), Live::new(), engine_rx, model, None);
        let sock = dir.join("agent.sock");
        test_listener(sock.clone()).serve(control.clone()).unwrap();
        Rig { sock, control, engine: engine_tx }
    }

    fn test_listener(sock: PathBuf) -> Listener {
        // Set the process-wide umask only at the production entry point so tests can share a process.
        let socket = std::os::unix::net::UnixListener::bind(&sock).unwrap();
        socket.set_nonblocking(true).unwrap();
        Listener(socket)
    }

    struct Client(tokio::io::Lines<BufReader<tokio::net::unix::OwnedReadHalf>>, tokio::net::unix::OwnedWriteHalf);

    impl Client {
        async fn connect(sock: &PathBuf) -> Client {
            let (r, w) = UnixStream::connect(sock).await.unwrap().into_split();
            Client(BufReader::new(r).lines(), w)
        }
        async fn write(&mut self, req: &str) {
            self.1.write_all(format!("{req}\n").as_bytes()).await.unwrap();
        }
        async fn read(&mut self) -> Value {
            let line = tokio::time::timeout(Duration::from_secs(5), self.0.next_line()).await.expect("timeout").unwrap().expect("eof");
            serde_json::from_str(&line).unwrap()
        }
        async fn ask(&mut self, req: &str) -> Value {
            self.write(req).await;
            self.read().await
        }
    }

    #[tokio::test]
    async fn status_round_trip() {
        let r = rig("status");
        let mut c = Client::connect(&r.sock).await;
        let v = c.ask(r#"{"op":"status"}"#).await;
        assert_eq!(v["ok"], true);
        let s = &v["status"];
        assert_eq!(s["version"], env!("CARGO_PKG_VERSION"));
        assert_eq!(s["connected"], false);
        assert_eq!(s["engine"], "ready");
        assert_eq!(s["preference"]["mode"], "use");
        assert_eq!(s["preference"]["model"], "qwen");
        assert_eq!(s["loaded_model"], "qwen");
        assert_eq!(s["active_leases"], 0);
        assert!(s["summary"].is_null());
        let rev = r.control.contribution.preference().revision;
        r.control.contribution.receive_status(ContributionStatus { free_seat: true, in_chain: true, revision: rev, ..Default::default() });
        r.control.contribution.receive_summary(ContributionSummary { days: 30, balance_mc: 777, ..Default::default() });
        r.control.live.set_connected(true);
        r.control.live.set_chain(Some("c1".into()));
        let s = &c.ask(r#"{"op":"status"}"#).await["status"];
        assert_eq!(s["connected"], true);
        assert_eq!(s["chain_id"], "c1");
        assert_eq!(s["contribution"]["free_seat"], true);
        assert_eq!(s["summary"]["balance_mc"], 777);
        let m = c.ask(r#"{"op":"models"}"#).await;
        assert_eq!(m["models"][0]["id"], "qwen");
        assert_eq!(m["models"][0]["loaded"], true);
        let bad = c.ask("not json").await;
        assert_eq!((bad["ok"].clone(), bad["error"].clone()), (json!(false), json!("bad_request")));
    }

    #[tokio::test]
    async fn follow_sends_on_change() {
        let r = rig("follow");
        let mut c = Client::connect(&r.sock).await;
        c.write(r#"{"op":"follow"}"#).await;
        assert_eq!(c.read().await["status"]["engine"], "ready");
        r.engine.send_replace(EngineState::Busy);
        assert_eq!(c.read().await["status"]["engine"], "working");
        r.control.live.set_leases(2);
        assert_eq!(c.read().await["status"]["active_leases"], 2);
        r.control.live.set_paused(true);
        assert_eq!(c.read().await["status"]["paused"], true);
    }

    #[tokio::test]
    async fn preference_rejects_not_installed() {
        let r = rig("pref");
        let mut c = Client::connect(&r.sock).await;
        let v = c.ask(r#"{"op":"preference","mode":"use","model":"gemma","quant":"Q4_K_M"}"#).await;
        assert_eq!((v["ok"].clone(), v["error"].clone()), (json!(false), json!("not_installed")));
        let v = c.ask(r#"{"op":"preference","mode":"use"}"#).await;
        assert_eq!(v["error"], "bad_request");
        assert_eq!(r.control.contribution.preference().modo, PreferenceMode::Usage, "refusals change nothing");
        // Validation failures leave the next valid change eligible for the cooldown.
        let v = c.ask(r#"{"op":"preference","mode":"free"}"#).await;
        assert_eq!(v["ok"], true);
        assert!(v["revision"].as_u64().unwrap() > 0);
        assert_eq!(r.control.contribution.preference().modo, PreferenceMode::Free);
        let v = c.ask(r#"{"op":"preference","mode":"use","model":"qwen","quant":"Q4_K_M"}"#).await;
        assert_eq!(v["error"], "too_soon");
    }

    #[tokio::test]
    async fn pause_toggles() {
        let r = rig("pause");
        let mut c = Client::connect(&r.sock).await;
        let mut rx = r.control.live.pause_rx();
        assert_eq!(c.ask(r#"{"op":"pause","active":true}"#).await["ok"], true);
        assert!(r.control.live.is_paused());
        rx.changed().await.unwrap();
        assert_eq!(c.ask(r#"{"op":"status"}"#).await["status"]["paused"], true);
        assert_eq!(c.ask(r#"{"op":"pause","active":false}"#).await["ok"], true);
        assert!(!r.control.live.is_paused());
        assert_eq!(c.ask(r#"{"op":"status"}"#).await["status"]["paused"], false);
    }

    #[tokio::test]
    async fn two_clients_at_once() {
        let r = rig("two");
        let mut follower = Client::connect(&r.sock).await;
        follower.write(r#"{"op":"follow"}"#).await;
        follower.read().await;
        // A subscription must leave the listener free to handle other requests.
        let mut other = Client::connect(&r.sock).await;
        assert_eq!(other.ask(r#"{"op":"status"}"#).await["ok"], true);
        assert_eq!(other.ask(r#"{"op":"pause","active":true}"#).await["ok"], true);
        assert_eq!(follower.read().await["status"]["paused"], true);
    }
}
