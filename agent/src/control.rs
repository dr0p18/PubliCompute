//! The local control protocol uses one JSON line per request and reply.
//! A follow request keeps the connection open and emits a snapshot whenever
//! its contents change or as a heartbeat. The default socket is private. A configured socket
//! path permits access by the agent's group. The session and preference
//! store own the states exposed here.

use std::{
    fs,
    io::{self, ErrorKind, Read},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, AtomicU32, Ordering},
        Arc, Mutex, OnceLock,
    },
    time::{Duration, Instant},
};

use anyhow::{Context, Result};
use publicompute_proto::{
    common::ModelRef,
    node::{ContributionStatus, ContributionSummary},
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::watch;

#[cfg(unix)]
use pause_file::PauseFile;

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
    pub network: crate::network::Store,
    connected: AtomicBool,
    leases: AtomicU32,
    chain: Mutex<Option<String>>,
    paused: watch::Sender<bool>,
    #[cfg(unix)]
    pause_file: Option<PauseFile>,
    #[cfg(unix)]
    pause_lock: Arc<tokio::sync::Mutex<()>>,
    #[cfg(all(test, unix))]
    after_rename: Mutex<Option<Box<dyn FnOnce() -> Result<()> + Send>>>,
    tick: watch::Sender<u64>,
    pub changes: crate::model_change::Changes,
}

impl Live {
    #[cfg(test)]
    pub fn new() -> Arc<Self> {
        Arc::new(Self::with_pause(false))
    }

    #[cfg(unix)]
    pub fn open(data_dir: &Path) -> Result<Arc<Self>> {
        let file = PauseFile::open(data_dir)?;
        let mut live = Self::with_pause(file.read()?);
        live.network = crate::network::Store::open(file.directory());
        live.pause_file = Some(file);
        Ok(Arc::new(live))
    }

    #[cfg(not(unix))]
    pub fn open(data_dir: &Path) -> Result<Arc<Self>> {
        let path = data_dir.join("paused.json");
        let paused = read_pause(fs::File::open(&path)).with_context(|| format!("read {}", path.display()))?;
        let mut live = Self::with_pause(paused);
        live.network = crate::network::Store::open(Arc::new(data_dir.to_owned()));
        Ok(Arc::new(live))
    }

    fn with_pause(paused: bool) -> Self {
        Self {
            network: Default::default(),
            connected: AtomicBool::new(false),
            leases: AtomicU32::new(0),
            chain: Mutex::new(None),
            paused: watch::channel(paused).0,
            #[cfg(unix)]
            pause_file: None,
            #[cfg(unix)]
            pause_lock: Default::default(),
            #[cfg(all(test, unix))]
            after_rename: Mutex::new(None),
            tick: watch::channel(0).0,
            changes: Default::default(),
        }
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

    pub fn has_chain(&self) -> bool {
        self.chain.lock().unwrap().is_some()
    }

    pub fn is_paused(&self) -> bool {
        *self.paused.borrow()
    }

    #[cfg(test)]
    pub fn set_paused(&self, v: bool) {
        self.publish_pause(v);
    }

    #[cfg(unix)]
    async fn save_pause(self: &Arc<Self>, v: bool, resume: impl FnOnce() + Send + 'static) -> Value {
        let Ok(lock) = self.pause_lock.clone().try_lock_owned() else {
            return json!({"ok": false, "error": "busy", "text": "a pause state change is in progress"});
        };
        let commands = self.changes.commands.clone().lock_owned().await;
        if v {
            if let Err(err) = self.changes.can_cancel() {
                return json!({"ok": false, "error": "model_change_failed", "text": err.to_string()});
            }
        }
        let live = self.clone();
        // Keep both locks through publication even if the request is cancelled.
        tokio::task::spawn_blocking(move || {
            let sync = if let Some(file) = &live.pause_file {
                if let Err(err) = file.replace(v) {
                    return json!({"ok": false, "error": "internal", "text": format!("{err:#}")});
                }
                let sync = file.sync();
                #[cfg(test)]
                let sync = sync.and_then(|()| {
                    let hook = live.after_rename.lock().unwrap().take();
                    hook.map_or(Ok(()), |hook| hook())
                });
                sync
            } else {
                Ok(())
            };
            // The new value is visible after rename, including when directory sync fails.
            live.publish_pause(v);
            if v {
                live.changes.cancel();
                live.notify();
            } else {
                resume();
            }
            drop(commands);
            drop(lock);
            match sync {
                Ok(()) => json!({"ok": true}),
                Err(err) => {
                    let text = "pause state applied but durability is uncertain";
                    tracing::warn!(error = %format_args!("{err:#}"), paused = v, "{text}");
                    json!({"ok": false, "error": "durability_uncertain", "text": text, "paused": v})
                }
            }
        }).await.unwrap_or_else(|err| json!({"ok": false, "error": "internal", "text": err.to_string()}))
    }

    #[cfg(any(test, unix))]
    fn publish_pause(&self, v: bool) {
        if self.paused.send_replace(v) != v {
            self.notify();
        }
    }

    /// Subscribe to changes the session must publish as availability updates.
    pub fn pause_rx(&self) -> watch::Receiver<bool> {
        self.paused.subscribe()
    }
}

fn read_pause(file: io::Result<fs::File>) -> Result<bool> {
    const MAX_PAUSE_BYTES: u64 = 64;
    let file = match file {
        Ok(file) => file,
        Err(err) if err.kind() == ErrorKind::NotFound => return Ok(false),
        Err(err) => return Err(err.into()),
    };
    let meta = file.metadata()?;
    anyhow::ensure!(meta.is_file() && meta.len() <= MAX_PAUSE_BYTES, "expected a small regular pause file");
    let mut bytes = Vec::new();
    file.take(MAX_PAUSE_BYTES + 1).read_to_end(&mut bytes)?;
    anyhow::ensure!(bytes.len() as u64 <= MAX_PAUSE_BYTES, "pause file is too large");
    Ok(serde_json::from_slice(&bytes)?)
}

#[cfg(unix)]
pub(crate) mod pause_file {
    use super::*;
    use std::{io::Write, os::unix::fs::MetadataExt};

    use rustix::{
        fs::{self as unix_fs, AtFlags, FlockOperation, Mode, OFlags},
        process::geteuid,
    };

    const NAME: &str = "paused.json";

    pub(super) struct PauseFile {
        dir: Arc<fs::File>,
        path: PathBuf,
    }

    impl PauseFile {
        pub(super) fn open(path: &Path) -> Result<Self> {
            let dir = (|| -> Result<fs::File> {
                let path = path.components().collect::<PathBuf>();
                let flags = OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC;
                let dir = fs::File::from(unix_fs::open(path, flags, Mode::empty())?);
                anyhow::ensure!(dir.metadata()?.uid() == geteuid().as_raw(), "data-dir must be owned by the agent user");
                unix_fs::flock(&dir, FlockOperation::NonBlockingLockExclusive).context("lock data-dir")?;
                unix_fs::fchmod(&dir, Mode::from_bits_truncate(0o700))?;
                Ok(dir)
            })().with_context(|| format!("open data-dir {}", path.display()))?;
            Ok(Self { dir: Arc::new(dir), path: path.join(NAME) })
        }

        pub(super) fn directory(&self) -> Arc<fs::File> {
            self.dir.clone()
        }

        pub(super) fn read(&self) -> Result<bool> {
            let flags = OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC;
            let file = unix_fs::openat(&*self.dir, NAME, flags, Mode::empty());
            read_pause(file.map(fs::File::from).map_err(io::Error::from))
                .with_context(|| format!("read {}", self.path.display()))
        }

        pub(super) fn replace(&self, paused: bool) -> Result<()> {
            replace_state_file(&self.dir, NAME, if paused { b"true\n" } else { b"false\n" })
                .with_context(|| format!("save pause state to {}", self.path.display()))
        }

        pub(super) fn sync(&self) -> Result<()> {
            self.dir.sync_all().with_context(|| format!("sync data-dir after replacing {}", self.path.display()))
        }
    }

    pub(crate) fn replace_state_file(dir: &fs::File, name: &str, bytes: &[u8]) -> io::Result<()> {
        let tmp = format!(".{name}-{:016x}.tmp", rand::random::<u64>());
        let mode = Mode::from_bits_truncate(0o600);
        let flags = OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL | OFlags::CLOEXEC;
        let mut output = fs::File::from(unix_fs::openat(dir, &tmp, flags, mode)?);
        let result = (|| -> io::Result<()> {
            unix_fs::fchmod(&output, mode)?;
            output.write_all(bytes)?;
            output.sync_all()?;
            unix_fs::renameat(dir, &tmp, dir, name)?;
            Ok(())
        })();
        if result.is_err() {
            let _ = unix_fs::unlinkat(dir, &tmp, AtFlags::empty());
        }
        result
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
    /// Short English explanation from the coordinator; empty from older coordinators.
    detail: String,
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
    weights_model: Option<String>,
    weights_quant: Option<String>,
    loaded_model: Option<String>,
    loaded_quant: Option<String>,
    download: Option<crate::model_change::DownloadView>,
    model_error: Option<String>,
    active_leases: u32,
    chain_id: Option<String>,
    gpu: Option<GpuView>,
    gpu_lost: bool,
    gpu_lost_reason: Option<&'static str>,
    contribution: Option<ContributionView>,
    summary: Option<SummaryView>,
    network: Option<crate::network::Reading>,
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
    model: Mutex<ModelState>,
    last_change: Mutex<Option<Instant>>,
    store: Option<Arc<crate::models::Store>>,
    stage_only: bool,
    stages: OnceLock<Arc<crate::stages::Stages>>,
}

struct ModelState {
    reference: ModelRef,
    path: Option<PathBuf>,
    error: Option<String>,
    verifying: bool,
    needs_repair: bool,
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
        EngineState::Parked => "parked",
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
        detail: s.detail.clone(),
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
    ) -> Self {
        Self {
            contribution, gpu, live, engine, last_change: Mutex::new(None), store: None,
            stage_only: false, stages: OnceLock::new(),
            model: Mutex::new(ModelState {
                reference: model, path: model_path, error: None, verifying: false, needs_repair: false,
            }),
        }
    }

    pub fn stage_only(mut self, stage_only: bool) -> Self {
        self.stage_only = stage_only;
        self
    }

    pub fn attach_stages(&self, stages: Arc<crate::stages::Stages>) {
        let _ = self.stages.set(stages);
    }

    pub fn with_models(mut self, store: Option<Arc<crate::models::Store>>, error: Option<String>) -> Arc<Self> {
        self.store = store;
        self.model.get_mut().unwrap().error = error;
        Arc::new(self)
    }

    pub fn verifying_weights(&self, verifying: bool) {
        self.model.lock().unwrap().verifying = verifying;
        self.live.notify();
    }

    pub fn model_verified(&self, reference: ModelRef, path: Option<PathBuf>, error: Option<String>, needs_repair: bool) {
        *self.model.lock().unwrap() = ModelState { reference, path, error, verifying: false, needs_repair };
        self.live.notify();
    }

    // The caller holds the command lock so a pause or a new choice cannot race this request.
    pub fn resume_saved_model(&self) {
        if self.live.is_paused() || self.live.changes.view().is_some() {
            return;
        }
        {
            let active = self.model.lock().unwrap();
            if active.verifying || !active.reference.model_id.is_empty() {
                return;
            }
        }
        let preference = self.contribution.preference();
        let Some(model) = preference.model.as_ref()
            .filter(|_| preference.modo == PreferenceMode::Usage)
            .and_then(|m| crate::models::find(&m.id, Some(&m.quant))) else { return };
        let download = self.store.as_ref().context("the model store is unavailable")
            .and_then(|store| self.live.changes.request(&self.live, store, model, None));
        if let Err(err) = download {
            tracing::warn!(model = %model.id, "cannot prepare selected weights: {err:#}");
        }
    }

    fn models(&self) -> Vec<LocalModel> {
        let engine = *self.engine.borrow();
        let active = self.model.lock().unwrap();
        let model = &active.reference;
        let loaded = self.model_loaded(&active, engine, !self.gpu.unavailable());
        let mut models = Vec::new();
        if !model.model_id.is_empty() {
            let bytes = active.path.as_ref().and_then(|p| std::fs::symlink_metadata(p).ok()).filter(|m| m.is_file()).map(|m| m.len()).unwrap_or(0);
            let state = if loaded {
                "ready"
            } else if self.stage_only && active.weights_ready() {
                "weights ready"
            } else {
                active.engine_name(engine)
            };
            models.push(LocalModel { id: model.model_id.clone(), quant: model.quantization.clone(), state, loaded, bytes });
        }
        if self.store.is_some() {
            models.extend(crate::models::MANIFEST.iter()
                .filter(|m| m.id != model.model_id || m.quant != model.quantization)
                .map(|m| LocalModel { id: m.id.clone(), quant: m.quant.clone(), state: "available", loaded: false, bytes: m.bytes }));
        }
        models
    }

    fn model_loaded(&self, model: &ModelState, engine: EngineState, gpu_available: bool) -> bool {
        if !model.verified() || !gpu_available {
            return false;
        }
        (!self.stage_only && matches!(engine, EngineState::Ready | EngineState::Busy | EngineState::Settling))
            || self.stages.get().is_some_and(|stages| stages.model_loaded())
    }

    fn status(&self) -> Status {
        let engine = *self.engine.borrow();
        let model = self.model.lock().unwrap();
        let p = self.contribution.preference();
        let gpu = self.gpu.letture().into_iter().next().map(|l| GpuView {
            name: l.name,
            vram_used_bytes: l.vram_usata,
            vram_total_bytes: l.vram_totale,
            temperature_celsius: l.temperature,
        });
        let gpu_loss = self.gpu.loss();
        let loaded = self.model_loaded(&model, engine, !self.gpu.unavailable());
        let weights_ready = model.weights_ready();
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
            engine: model.engine_name(engine),
            weights_model: weights_ready.then(|| model.reference.model_id.clone()),
            weights_quant: weights_ready.then(|| model.reference.quantization.clone()),
            loaded_model: loaded.then(|| model.reference.model_id.clone()),
            loaded_quant: loaded.then(|| model.reference.quantization.clone()),
            download: self.live.changes.view(),
            model_error: model.error.clone(),
            active_leases: self.live.leases.load(Ordering::Relaxed),
            chain_id: self.live.chain.lock().unwrap().clone(),
            gpu,
            gpu_lost: gpu_loss.is_some(),
            gpu_lost_reason: gpu_loss.map(crate::gpu::ReadFailure::reason),
            contribution: self.contribution.state().as_ref().map(contribution_view),
            summary: self.contribution.summary().as_ref().map(summary_view),
            network: self.live.network.enabled().then(|| self.live.network.reading().unwrap_or_default()),
        }
    }

    async fn set_preference(&self, mode: &str, model: Option<String>, quant: Option<String>, reply: Option<tokio::sync::oneshot::Receiver<()>>) -> Result<Value, Refusal> {
        let bad = |t: &str| Refusal("bad_request", t.to_string());
        let internal = |e: anyhow::Error| Refusal("model_change_failed", format!("{e:#}"));
        let _commands = self.live.changes.commands.lock().await;
        let mut last = self.last_change.lock().unwrap();
        let (mode, selected) = match mode {
            "free" => (PreferenceMode::Free, None),
            "use" => {
                let id = model.ok_or_else(|| bad("mode use needs a model"))?;
                let (active_id, active_quant, needs_repair) = {
                    let active = self.model.lock().unwrap();
                    (active.reference.model_id.clone(), active.reference.quantization.clone(), active.needs_repair)
                };
                if needs_repair || id != active_id || quant.as_ref().is_some_and(|q| *q != active_quant) {
                    let Some((store, model)) = self.store.as_ref().zip(crate::models::find(&id, quant.as_deref())) else {
                        return Err(Refusal("not_installed", format!("model {id} is not installed on this machine")));
                    };
                    if self.live.is_paused() {
                        return Err(Refusal("paused", "resume the agent before changing models".into()));
                    }
                    let (installed, missing) = self.live.changes.request(&self.live, store, model, reply).map_err(internal)?;
                    return Ok(if installed {
                        json!({"ok": true, "state": "switching", "text": format!("switching to {id}")})
                    } else {
                        let gib = crate::models::gib(missing);
                        json!({"ok": true, "state": "downloading", "missing_bytes": missing,
                            "text": format!("downloading {id} ({gib:.1} GiB remaining), see publicompute status")})
                    });
                }
                (PreferenceMode::Usage, Some(ModelSelection { id, quant: active_quant }))
            }
            other => return Err(Refusal("bad_request", format!("unknown mode \"{other}\": use or free"))),
        };
        let pending = self.live.changes.can_cancel().map_err(internal)?;
        // Allow the host to override a pending switch even during the preference cooldown.
        if !pending && last.is_some_and(|t| t.elapsed() < PREFERENCE_GAP) {
            return Err(Refusal("too_soon", "preference changed a moment ago: wait a few seconds".into()));
        }
        let text = selected.as_ref().map_or_else(|| "FARM mode: set".into(), |m| format!("using {}", m.id));
        let p = self.contribution.imposta(mode, selected).map_err(internal)?;
        self.live.changes.cancel();
        *last = Some(Instant::now());
        self.live.notify();
        Ok(json!({"ok": true, "revision": p.revision, "text": text}))
    }

    /// Handle requests that complete with a single reply.
    #[cfg(unix)]
    async fn handle(self: &Arc<Self>, req: Request, reply: Option<tokio::sync::oneshot::Receiver<()>>) -> Value {
        let refused = |r: Refusal| json!({"ok": false, "error": r.0, "text": r.1});
        match req {
            Request::Status => json!({"ok": true, "status": self.status()}),
            Request::Models => json!({"ok": true, "models": self.models()}),
            Request::Pause { active } => {
                let control = self.clone();
                self.live.save_pause(active, move || control.resume_saved_model()).await
            }
            Request::Preference { mode, model, quant } => self.set_preference(&mode, model, quant, reply).await.unwrap_or_else(refused),
            Request::Follow => unreachable!("follow is served by `follow`"),
        }
    }
}

impl ModelState {
    fn verified(&self) -> bool {
        !self.reference.model_id.is_empty() && !self.verifying && !self.needs_repair
    }

    fn weights_ready(&self) -> bool {
        self.path.is_some() && self.verified()
    }

    fn engine_name(&self, engine: EngineState) -> &'static str {
        if self.verifying {
            "verifying weights"
        } else if self.needs_repair {
            "weights error"
        } else {
            engine_name(engine)
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
    use std::{
        io,
        os::unix::{fs::FileTypeExt, net},
    };
    use tokio::{
        io::{AsyncBufRead, AsyncBufReadExt, AsyncRead, AsyncWrite, AsyncWriteExt, BufReader},
        net::UnixListener,
        sync::{OwnedSemaphorePermit, Semaphore},
        time::{self, Instant},
    };
    use tracing::{debug, info};

    const MAX_CONNECTIONS: usize = 32;
    const MAX_FOLLOWERS: usize = 8;
    const MAX_REQUEST_LINE: usize = 64 * 1024;
    const IO_TIMEOUT: Duration = Duration::from_secs(10);
    const FOLLOW_HEARTBEAT: Duration = Duration::from_secs(30);

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
            let connections = Arc::new(Semaphore::new(MAX_CONNECTIONS));
            let followers = Arc::new(Semaphore::new(MAX_FOLLOWERS));
            tokio::spawn(async move {
                loop {
                    match listener.accept().await {
                        Ok((stream, _)) => {
                            let Ok(slot) = connections.clone().try_acquire_owned() else { continue };
                            let control = control.clone();
                            let followers = followers.clone();
                            let (r, w) = stream.into_split();
                            tokio::spawn(client(control, r, w, followers, slot));
                        }
                        Err(e) => debug!("accept on the control socket: {e}"),
                    }
                }
            });
            Ok(())
        }
    }

    async fn send(w: &mut (impl AsyncWrite + Unpin), v: &Value) -> io::Result<()> {
        let mut line = v.to_string();
        line.push('\n');
        time::timeout(IO_TIMEOUT, w.write_all(line.as_bytes())).await?
    }

    async fn read_request(reader: &mut (impl AsyncBufRead + Unpin), line: &mut Vec<u8>) -> io::Result<bool> {
        line.clear();
        loop {
            let bytes = reader.fill_buf().await?;
            if bytes.is_empty() {
                return Ok(false);
            }
            let end = bytes.iter().position(|byte| *byte == b'\n');
            let len = end.unwrap_or(bytes.len());
            if line.len() + len > MAX_REQUEST_LINE {
                return Err(io::ErrorKind::InvalidData.into());
            }
            line.extend_from_slice(&bytes[..len]);
            reader.consume(len + usize::from(end.is_some()));
            if end.is_some() {
                return Ok(true);
            }
        }
    }

    #[cfg(test)]
    #[tokio::test]
    async fn oversized_partial_line_cannot_grow_the_request_buffer() {
        let bytes = vec![b' '; MAX_REQUEST_LINE + 1];
        let mut reader = BufReader::new(bytes.as_slice());
        let mut line = Vec::with_capacity(MAX_REQUEST_LINE);
        let err = read_request(&mut reader, &mut line).await.unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidData);
        assert_eq!(line.capacity(), MAX_REQUEST_LINE);
    }

    #[cfg(test)]
    #[tokio::test]
    async fn blocked_follower_write_expires_and_releases_both_slots() {
        use tokio::io::{duplex, split, AsyncReadExt};

        let dir = super::pause_tests::data_dir("write-backpressure");
        let (control, _engine) = super::tests::test_control(&dir);
        control.live.set_chain(Some("x".repeat(128 * 1024)));
        let connections = Arc::new(Semaphore::new(1));
        let followers = Arc::new(Semaphore::new(1));
        let (server, mut peer) = duplex(4096);
        let (r, w) = split(server);
        let slot = connections.clone().try_acquire_owned().unwrap();
        let task = tokio::spawn(client(control.clone(), r, w, followers.clone(), slot));
        let started = Instant::now();
        peer.write_all(b"{\"op\":\"follow\"}\n").await.unwrap();
        let first = time::timeout(Duration::from_secs(2), peer.read_u8()).await.unwrap().unwrap();
        assert_eq!(connections.available_permits(), 0);
        assert_eq!(followers.available_permits(), 0);

        time::timeout(IO_TIMEOUT + Duration::from_secs(2), task).await.unwrap().unwrap();
        assert!(started.elapsed() >= IO_TIMEOUT);
        let mut partial = vec![first];
        time::timeout(Duration::from_secs(2), peer.read_to_end(&mut partial)).await.unwrap().unwrap();
        assert!(!partial.contains(&b'\n'));
        assert_eq!(connections.available_permits(), 1);
        assert_eq!(followers.available_permits(), 1);

        control.live.set_chain(None);
        let (server, mut peer) = duplex(4096);
        let (r, w) = split(server);
        let slot = connections.clone().try_acquire_owned().unwrap();
        let task = tokio::spawn(client(control, r, w, followers.clone(), slot));
        peer.write_all(b"{\"op\":\"follow\"}\n").await.unwrap();
        let mut line = String::new();
        time::timeout(Duration::from_secs(2), BufReader::new(&mut peer).read_line(&mut line)).await.unwrap().unwrap();
        assert_eq!(serde_json::from_str::<Value>(&line).unwrap()["ok"], true);
        drop(peer);
        time::timeout(Duration::from_secs(2), task).await.unwrap().unwrap();
        fs::remove_dir_all(dir).unwrap();
    }

    async fn client(
        control: Arc<Control>,
        r: impl AsyncRead + Unpin,
        mut w: impl AsyncWrite + Unpin,
        followers: Arc<Semaphore>,
        _slot: OwnedSemaphorePermit,
    ) {
        let mut reader = BufReader::new(r);
        let mut line = Vec::with_capacity(MAX_REQUEST_LINE);
        while let Ok(Ok(true)) = time::timeout(IO_TIMEOUT, read_request(&mut reader, &mut line)).await {
            let (sent, ready) = tokio::sync::oneshot::channel();
            let reply = match serde_json::from_slice::<Request>(&line) {
                Ok(Request::Follow) => {
                    let Ok(_slot) = followers.try_acquire() else {
                        let _ = send(&mut w, &json!({"ok": false, "error": "busy", "text": "too many status followers"})).await;
                        return;
                    };
                    drop(line);
                    // Any input ends a subscription, including bytes already buffered.
                    tokio::select! {
                        biased;
                        _ = reader.fill_buf() => {}
                        _ = follow(&control, &mut w) => {}
                    }
                    return;
                }
                Ok(req) => control.handle(req, Some(ready)).await,
                Err(e) => json!({"ok": false, "error": "bad_request", "text": e.to_string()}),
            };
            let result = send(&mut w, &reply).await;
            let _ = sent.send(());
            if result.is_err() {
                return;
            }
        }
    }

    async fn follow(c: &Control, w: &mut (impl AsyncWrite + Unpin)) {
        let mut tick = c.live.tick.subscribe();
        let mut pref = c.contribution.subscribe();
        let mut engine = c.engine.clone();
        let mut gpu = c.gpu.device_rx();
        let mut poll = tokio::time::interval(FOLLOW_POLL);
        let mut last: Option<Status> = None;
        let mut heartbeat = Instant::now();
        loop {
            let now = c.status();
            if last.as_ref() != Some(&now) || Instant::now() >= heartbeat {
                if send(w, &json!({"ok": true, "status": now})).await.is_err() {
                    return;
                }
                last = Some(now);
                heartbeat = Instant::now() + FOLLOW_HEARTBEAT;
            }
            tokio::select! {
                _ = tick.changed() => {}
                _ = pref.changed() => {}
                _ = engine.changed() => {}
                _ = gpu.changed() => {}
                _ = poll.tick() => {}
                _ = time::sleep_until(heartbeat) => {}
            }
        }
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::contribution::Preference;
    use tokio::{
        io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader},
        net::UnixStream,
        time::{sleep, timeout},
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
        let (control, engine) = test_control(&dir);
        let sock = dir.join("agent.sock");
        test_listener(sock.clone()).serve(control.clone()).unwrap();
        Rig { sock, control, engine }
    }

    pub(super) fn test_control(dir: &Path) -> (Arc<Control>, watch::Sender<EngineState>) {
        let pref = Preference { modo: PreferenceMode::Usage, model: Some(ModelSelection { id: "qwen".into(), quant: "Q4_K_M".into() }), revision: 0 };
        let contribution = Contribution::open(dir, pref).unwrap();
        let (engine_tx, engine_rx) = watch::channel(EngineState::Ready);
        let model = ModelRef { model_id: "qwen".into(), quantization: "Q4_K_M".into(), weights_sha256: vec![] };
        let control = Arc::new(Control::new(contribution, Arc::new(Telemetria::default()), Live::new(), engine_rx, model, Some(dir.join("model.gguf"))));
        (control, engine_tx)
    }

    #[test]
    fn status_uses_the_same_engine_state_during_transitions() {
        use std::{sync::{mpsc, Barrier}, thread};

        let dir = super::pause_tests::data_dir("engine-transition");
        let (control, engine) = test_control(&dir);
        let start = Arc::new(Barrier::new(2));
        let writer = thread::spawn({
            let start = start.clone();
            move || {
                start.wait();
                for _ in 0..10_000 {
                    engine.send_replace(EngineState::Quarantined);
                    engine.send_replace(EngineState::Ready);
                }
            }
        });
        let (done, wait) = mpsc::channel();
        let reader = thread::spawn(move || {
            start.wait();
            for _ in 0..10_000 {
                let status = control.status();
                assert_eq!(status.loaded_model.as_deref(), (status.engine == "ready").then_some("qwen"));
            }
            done.send(()).unwrap();
        });
        wait.recv_timeout(Duration::from_secs(5)).expect("status snapshots did not complete");
        reader.join().unwrap();
        writer.join().unwrap();
        fs::remove_dir_all(dir).unwrap();
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

    async fn expect_closed(mut stream: UnixStream, wait: Duration) {
        match timeout(wait, stream.read_u8()).await.expect("socket still open") {
            Ok(byte) => panic!("unexpected reply byte: {byte}"),
            Err(err) => assert!(matches!(err.kind(), ErrorKind::UnexpectedEof | ErrorKind::ConnectionReset), "{err}"),
        }
    }

    #[tokio::test]
    async fn request_line_limit() {
        let r = rig("line-limit");
        let mut client = Client::connect(&r.sock).await;
        let mut line = br#"{"op":"status"}"#.to_vec();
        line.resize(64 * 1024, b' ');
        line.push(b'\n');
        client.1.write_all(&line).await.unwrap();
        assert_eq!(client.read().await["ok"], true);

        let mut stream = UnixStream::connect(&r.sock).await.unwrap();
        line[64 * 1024] = b' ';
        stream.write_all(&line).await.unwrap();
        expect_closed(stream, Duration::from_secs(2)).await;
        assert_eq!(client.ask(r#"{"op":"status"}"#).await["ok"], true);
    }

    #[tokio::test]
    async fn request_deadline_covers_silence_and_partial_lines() {
        let r = rig("request-deadline");
        let silent = UnixStream::connect(&r.sock).await.unwrap();
        let mut partial = UnixStream::connect(&r.sock).await.unwrap();
        partial.write_all(b"{").await.unwrap();
        let started = tokio::time::Instant::now();
        sleep(Duration::from_secs(6)).await;
        partial.write_all(b" ").await.unwrap();
        timeout(Duration::from_secs(6), async {
            expect_closed(partial, Duration::from_secs(6)).await;
            expect_closed(silent, Duration::from_secs(6)).await;
        }).await.unwrap();
        assert!(started.elapsed() >= Duration::from_secs(9));
    }

    #[tokio::test]
    async fn connection_limit_releases_slots() {
        let r = rig("connection-limit");
        let mut clients = Vec::new();
        for _ in 0..32 {
            let mut client = Client::connect(&r.sock).await;
            assert_eq!(client.ask(r#"{"op":"status"}"#).await["ok"], true);
            clients.push(client);
        }
        expect_closed(UnixStream::connect(&r.sock).await.unwrap(), Duration::from_secs(2)).await;
        drop(clients.pop());
        timeout(Duration::from_secs(2), async {
            loop {
                let mut client = Client::connect(&r.sock).await;
                if client.1.write_all(b"{\"op\":\"status\"}\n").await.is_ok()
                    && matches!(client.0.next_line().await, Ok(Some(_)))
                {
                    break;
                }
                sleep(Duration::from_millis(10)).await;
            }
        }).await.expect("a closed connection still holds its slot");
    }

    #[tokio::test]
    async fn follower_limit_leaves_room_for_status_and_releases_slots() {
        let r = rig("follower-limit");
        let mut followers = Vec::new();
        for _ in 0..8 {
            let mut client = Client::connect(&r.sock).await;
            assert_eq!(client.ask(r#"{"op":"follow"}"#).await["ok"], true);
            followers.push(client);
        }
        let mut extra = Client::connect(&r.sock).await;
        assert_eq!(extra.ask(r#"{"op":"follow"}"#).await["error"], "busy");
        let mut client = Client::connect(&r.sock).await;
        assert_eq!(client.ask(r#"{"op":"status"}"#).await["ok"], true);

        let mut follower = followers.pop().unwrap();
        follower.1.shutdown().await.unwrap();
        assert!(timeout(Duration::from_secs(2), follower.0.next_line()).await.unwrap().unwrap().is_none());
        assert_eq!(client.ask(r#"{"op":"follow"}"#).await["ok"], true);
    }

    #[tokio::test]
    async fn follow_closes_on_first_byte_even_if_buffered() {
        let r = rig("follow-extra-byte");
        let mut pipelined = UnixStream::connect(&r.sock).await.unwrap();
        pipelined.write_all(b"{\"op\":\"follow\"}\nx").await.unwrap();
        expect_closed(pipelined, Duration::from_secs(2)).await;

        let mut follower = Client::connect(&r.sock).await;
        assert_eq!(follower.ask(r#"{"op":"follow"}"#).await["ok"], true);
        follower.1.write_all(b"x").await.unwrap();
        assert!(timeout(Duration::from_secs(2), follower.0.next_line()).await.unwrap().unwrap().is_none());
    }

    #[tokio::test]
    async fn follow_repeats_an_unchanged_snapshot() {
        let r = rig("follow-heartbeat");
        let mut follower = Client::connect(&r.sock).await;
        let first = follower.ask(r#"{"op":"follow"}"#).await;
        let started = tokio::time::Instant::now();
        let line = timeout(Duration::from_secs(35), follower.0.next_line()).await.unwrap().unwrap().unwrap();
        assert_eq!(serde_json::from_str::<Value>(&line).unwrap(), first);
        assert!(started.elapsed() >= Duration::from_secs(29));
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
    async fn pause_and_farm_cancel_a_pending_download() {
        let store = crate::models::tests::fixture_store("control-model-cancel");
        let model = crate::models::find("qwen3.6-35b-a3b", None).unwrap();
        fs::File::create(store.partial(model)).unwrap().set_len(model.bytes - 1).unwrap();
        let dir = super::pause_tests::data_dir("control-model-cancel");
        let (test, _) = test_control(&dir);
        let control = Control::new(test.contribution.clone(), test.gpu.clone(), test.live.clone(),
            test.engine.clone(), test.model.lock().unwrap().reference.clone(), None).with_models(Some(store), None);
        for command in [
            Request::Preference { mode: "free".into(), model: None, quant: None },
            Request::Pause { active: true },
        ] {
            let (sent, ready) = tokio::sync::oneshot::channel();
            let response = control.handle(Request::Preference {
                mode: "use".into(), model: Some("qwen3.6-35b-a3b".into()), quant: None,
            }, Some(ready)).await;
            assert_eq!(response["state"], "downloading");
            assert_eq!(control.live.changes.inventory().unwrap().status(), publicompute_proto::node::ModelStatus::Downloading);
            assert_eq!(control.handle(command, None).await["ok"], true);
            assert!(control.status().download.is_none());
            let _ = sent.send(());
        }
        assert!(control.live.is_paused());
        assert_eq!(control.contribution.preference().modo, PreferenceMode::Free);
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
    async fn busy_pause_requests_leave_room_for_status() {
        let r = rig("pause-busy");
        let _guard = r.control.live.pause_lock.clone().try_lock_owned().unwrap();
        for active in [true, false].into_iter().cycle().take(64) {
            let mut client = Client::connect(&r.sock).await;
            let reply = client.ask(&json!({"op": "pause", "active": active}).to_string()).await;
            assert_eq!(reply["ok"], false);
            assert_eq!(reply["error"], "busy");
        }
        let mut client = Client::connect(&r.sock).await;
        assert_eq!(client.ask(r#"{"op":"status"}"#).await["status"]["paused"], false);
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

#[cfg(all(test, unix))]
mod pause_tests {
    use super::*;
    use crate::contribution::Preference;

    pub(super) fn data_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("publicompute-pause-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[tokio::test]
    async fn pause_and_start_survive_reopening() {
        let dir = data_dir("reopen");
        let live = Live::open(&dir).unwrap();
        assert!(!live.is_paused());
        assert_eq!(live.save_pause(true, || {}).await["ok"], true);
        drop(live);

        let live = Live::open(&dir).unwrap();
        assert!(live.is_paused());
        assert!(*live.pause_rx().borrow());
        assert_eq!(live.save_pause(false, || {}).await["ok"], true);
        drop(live);

        assert!(!Live::open(&dir).unwrap().is_paused());
        fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn persisted_pause_defers_the_saved_download_until_resume() {
        let dir = data_dir("paused-download");
        let store = Arc::new(crate::models::Store::new(&dir));
        store.prepare().unwrap();
        let model = crate::models::find("qwen3.6-35b-a3b", None).unwrap();
        let preference = Preference {
            modo: PreferenceMode::Usage,
            model: Some(ModelSelection { id: model.id.clone(), quant: model.quant.clone() }),
            revision: 0,
        };
        let saved = Contribution::open(&dir, preference).unwrap().preference();
        fs::File::create(store.partial(model)).unwrap().set_len(model.bytes - 1).unwrap();
        let live = Live::open(&dir).unwrap();
        assert_eq!(live.save_pause(true, || {}).await["ok"], true);
        drop(live);

        let live = Live::open(&dir).unwrap();
        let contribution = Contribution::open(&dir, Preference { modo: PreferenceMode::Free, model: None, revision: 0 }).unwrap();
        let (_engine, engine_rx) = watch::channel(EngineState::Parked);
        let control = Control::new(contribution.clone(), Arc::new(Telemetria::default()), live.clone(), engine_rx, ModelRef::default(), None)
            .with_models(Some(store.clone()), None);
        let _worker = store.worker.lock().await;
        {
            let _commands = live.changes.commands.lock().await;
            control.model_verified(ModelRef::default(), None, None, false);
            control.resume_saved_model();
        }
        assert!(control.status().paused);
        assert!(control.status().download.is_none());
        assert_eq!(control.handle(Request::Pause { active: false }, None).await["ok"], true);
        let download = live.changes.inventory().unwrap();
        assert_eq!(download.status(), publicompute_proto::node::ModelStatus::Downloading);
        assert_eq!(download.downloaded_bytes, model.bytes - 1);
        assert_eq!(contribution.preference(), saved);
        assert_eq!(control.handle(Request::Pause { active: true }, None).await["ok"], true);
        fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn failed_pause_write_is_refused_without_publishing() {
        let dir = data_dir("write-error");
        let live = Live::open(&dir).unwrap();
        assert!(!live.is_paused());
        let contribution = Contribution::open(&dir, Preference { modo: PreferenceMode::Free, model: None, revision: 0 }).unwrap();
        let (_engine, engine_rx) = watch::channel(EngineState::Ready);
        let control = Arc::new(Control::new(contribution, Arc::new(Telemetria::default()), live.clone(), engine_rx, ModelRef::default(), None));
        assert_eq!(control.handle(Request::Pause { active: true }, None).await["ok"], true);

        let rx = live.pause_rx();
        let tick = live.tick.subscribe();
        let path = dir.join("paused.json");
        let backup = dir.join("paused.bak");
        fs::rename(&path, &backup).unwrap();
        fs::create_dir(&path).unwrap();
        let reply = control.handle(Request::Pause { active: false }, None).await;
        assert_eq!(reply["ok"], false);
        assert_eq!(reply["error"], "internal");
        assert!(live.is_paused());
        assert!(!rx.has_changed().unwrap());
        assert!(!tick.has_changed().unwrap());
        assert!(!fs::read_dir(&dir).unwrap().any(|entry| entry.unwrap().file_name().to_string_lossy().starts_with(".paused-")));
        fs::remove_dir(&path).unwrap();
        fs::rename(backup, path).unwrap();
        drop(control);
        drop(live);
        assert!(Live::open(&dir).unwrap().is_paused());
        fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn failed_pause_and_farm_writes_preserve_a_pending_download() {
        let dir = data_dir("model-cancel-write-failure");
        let live = Live::open(&dir).unwrap();
        let contribution = Contribution::open(&dir, Preference {
            modo: PreferenceMode::Usage, model: None, revision: 0,
        }).unwrap();
        let (_engine, engine_rx) = watch::channel(EngineState::Ready);
        let store = crate::models::tests::fixture_store("model-cancel-write-failure");
        let model = crate::models::find("qwen3.6-35b-a3b", None).unwrap();
        fs::File::create(store.partial(model)).unwrap().set_len(model.bytes - 1).unwrap();
        let control = Control::new(contribution.clone(), Arc::new(Telemetria::default()), live.clone(), engine_rx, ModelRef::default(), None)
            .with_models(Some(store), None);
        let (sent, ready) = tokio::sync::oneshot::channel();
        let reply = control.handle(Request::Preference { mode: "use".into(), model: Some(model.id.clone()), quant: None }, Some(ready)).await;
        assert_eq!(reply["state"], "downloading");
        let download = live.changes.inventory();

        fs::create_dir(dir.join("paused.json")).unwrap();
        assert_eq!(control.handle(Request::Pause { active: true }, None).await["error"], "internal");
        assert!(!live.is_paused());
        assert_eq!(live.changes.inventory(), download);

        fs::create_dir(dir.join("contributo.json.tmp")).unwrap();
        let farm = Request::Preference { mode: "free".into(), model: None, quant: None };
        assert_eq!(control.handle(farm, None).await["error"], "model_change_failed");
        assert_eq!(contribution.preference().modo, PreferenceMode::Usage);
        assert_eq!(live.changes.inventory(), download);

        fs::remove_dir(dir.join("paused.json")).unwrap();
        assert_eq!(control.handle(Request::Pause { active: true }, None).await["ok"], true);
        assert!(live.changes.inventory().is_none());
        let _ = sent.send(());
    }

    #[tokio::test]
    async fn cancelled_pause_keeps_worker_lock_until_publication() {
        use tokio::{sync::oneshot, time::timeout};

        let dir = data_dir("concurrent");
        let live = Live::open(&dir).unwrap();
        let contribution = Contribution::open(&dir, Preference { modo: PreferenceMode::Free, model: None, revision: 0 }).unwrap();
        let (_engine, engine_rx) = watch::channel(EngineState::Ready);
        let control = Arc::new(Control::new(contribution, Arc::new(Telemetria::default()), live.clone(), engine_rx, ModelRef::default(), None));
        let (renamed, wait) = oneshot::channel();
        let (resume, gate) = std::sync::mpsc::channel();
        *live.after_rename.lock().unwrap() = Some(Box::new(move || {
            renamed.send(()).unwrap();
            gate.recv_timeout(Duration::from_secs(5))?;
            Ok(())
        }));
        let mut rx = live.pause_rx();
        let pause = tokio::spawn({
            let control = control.clone();
            async move { control.handle(Request::Pause { active: true }, None).await }
        });
        timeout(Duration::from_secs(2), wait).await.unwrap().unwrap();
        let reply = timeout(Duration::from_secs(2), control.handle(Request::Pause { active: false }, None)).await.unwrap();
        assert_eq!(reply["error"], "busy");
        assert_eq!(fs::read_to_string(dir.join("paused.json")).unwrap(), "true\n");
        let status = control.handle(Request::Status, None).await;
        assert_eq!(status["status"]["paused"], false);
        assert!(!rx.has_changed().unwrap());

        pause.abort();
        assert!(pause.await.unwrap_err().is_cancelled());
        let reply = timeout(Duration::from_secs(2), control.handle(Request::Pause { active: false }, None)).await.unwrap();
        assert_eq!(reply["error"], "busy");
        resume.send(()).unwrap();
        drop(timeout(Duration::from_secs(2), live.pause_lock.lock()).await.unwrap());
        assert!(rx.has_changed().unwrap());
        assert!(*rx.borrow_and_update());
        assert_eq!(control.handle(Request::Status, None).await["status"]["paused"], true);
        assert_eq!(control.handle(Request::Pause { active: false }, None).await["ok"], true);
        assert!(rx.has_changed().unwrap());
        assert!(!*rx.borrow());
        assert_eq!(fs::read_to_string(dir.join("paused.json")).unwrap(), "false\n");
        fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn pause_state_stays_in_the_open_directory_with_private_permissions() {
        use std::os::unix::fs::{symlink, PermissionsExt};

        let root = data_dir("anchored");
        let dir = root.join("state");
        let moved = root.join("moved");
        let target = root.join("target");
        fs::create_dir(&dir).unwrap();
        fs::write(&target, "keep").unwrap();
        symlink(&target, dir.join("paused.json.tmp")).unwrap();
        let live = Live::open(&dir).unwrap();
        fs::write(dir.join("paused.json"), "false\n").unwrap();
        fs::set_permissions(dir.join("paused.json"), fs::Permissions::from_mode(0o666)).unwrap();
        fs::rename(&dir, &moved).unwrap();
        fs::create_dir(&dir).unwrap();
        assert_eq!(live.save_pause(true, || {}).await["ok"], true);
        let path = moved.join("paused.json");
        assert_eq!(fs::read_to_string(&path).unwrap(), "true\n");
        assert_eq!(fs::metadata(path).unwrap().permissions().mode() & 0o777, 0o600);
        assert!(!dir.join("paused.json").exists());
        assert_eq!(fs::read_to_string(target).unwrap(), "keep");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rejects_symlinks_special_files_and_oversized_pause_files() {
        use std::os::unix::fs::symlink;

        let dir = data_dir("invalid-state");
        let path = dir.join("paused.json");
        let link = dir.join("link");
        symlink(&dir, &link).unwrap();
        assert!(Live::open(&link.join("")).is_err());
        symlink(dir.join("missing"), &path).unwrap();
        assert!(Live::open(&dir).is_err());
        fs::remove_file(&path).unwrap();
        fs::write(&path, format!("true{}", " ".repeat(64))).unwrap();
        assert!(Live::open(&dir).is_err());
        #[cfg(not(target_vendor = "apple"))]
        {
            use rustix::fs::{mkfifoat, Mode, CWD};

            fs::remove_file(&path).unwrap();
            mkfifoat(CWD, &path, Mode::from_bits_truncate(0o600)).unwrap();
            let (done, wait) = std::sync::mpsc::channel();
            let check = std::thread::spawn({
                let dir = dir.clone();
                move || done.send(Live::open(&dir).is_err()).unwrap()
            });
            assert!(wait.recv_timeout(Duration::from_secs(2)).expect("opening a FIFO blocked"));
            check.join().unwrap();
        }
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn data_dir_lock_is_exclusive_until_last_owner_drops() {
        let dir = data_dir("exclusive");
        let live = Live::open(&dir).unwrap();
        let other_owner = live.clone();
        drop(live);
        match Live::open(&dir.join(".")) {
            Err(err) => assert!(format!("{err:#}").contains("lock data-dir")),
            Ok(_) => panic!("data-dir opened while another owner still held it"),
        }
        drop(other_owner);
        assert!(Live::open(&dir).is_ok());
        fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn sync_error_after_rename_reports_the_published_pause_state() {
        let dir = data_dir("sync-error");
        let live = Live::open(&dir).unwrap();
        let contribution = Contribution::open(&dir, Preference { modo: PreferenceMode::Free, model: None, revision: 0 }).unwrap();
        let (_engine, engine_rx) = watch::channel(EngineState::Ready);
        let control = Arc::new(Control::new(contribution, Arc::new(Telemetria::default()), live.clone(), engine_rx, ModelRef::default(), None));
        let mut rx = live.pause_rx();
        for active in [true, false] {
            *live.after_rename.lock().unwrap() = Some(Box::new(|| anyhow::bail!("sync data-dir failed")));
            let reply = control.handle(Request::Pause { active }, None).await;
            assert_eq!(reply["ok"], false);
            assert_eq!(reply["error"], "durability_uncertain");
            assert_eq!(reply["text"], "pause state applied but durability is uncertain");
            assert_eq!(reply["paused"], active);
            assert!(rx.has_changed().unwrap());
            assert_eq!(*rx.borrow_and_update(), active);
            assert_eq!(fs::read_to_string(dir.join("paused.json")).unwrap(), format!("{active}\n"));
        }
        fs::remove_dir_all(dir).unwrap();
    }
}
