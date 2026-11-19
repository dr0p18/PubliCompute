//! The supervisor separates process liveness, HTTP readiness and successful
//! inference. A health response and the expected weights establish readiness.
//! A one-token probe establishes that the engine can actually generate.
//!
//! Startup enters Ready only after the checks pass. Ready accepts a lease and
//! moves to Busy. A clean finish returns to Ready. Abandonment enters Settling,
//! which blocks new leases while the engine finishes the request. The coordinator
//! still sees idle availability until settling fails. A fault or failed settle
//! enters Recovering. A successful probe returns to Ready. A stuck process or an
//! exhausted restart budget enters Quarantined.
//!
//! The gate serializes leases, probes and recovery so a probe cannot compete
//! with active generation.

use std::{
    collections::VecDeque,
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::{Duration, Instant},
};

use anyhow::{anyhow, bail, Context, Result};
use publicompute_proto::node::{EngineHealth, LeaseRejectReason};
use tokio::sync::{watch, OwnedMutexGuard};
use tracing::{error, info, warn};

use crate::gpu::{Device, DeviceState, Telemetria};

use super::{classify, managed, progress::Progress, stats::Stats, Engine, ErrorClass, FaultCause};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineState {
    Starting,
    Ready,
    Busy,
    /// Wait for an abandoned request to clear before accepting another lease.
    Settling,
    /// The stage owns the GPU until its process has been reaped.
    Parked,
    Recovering,
    Quarantined,
}

#[derive(Debug, Clone)]
pub struct SupervisorCfg {
    /// When set, the supervisor owns the engine process.
    pub managed: Option<managed::ManagedCfg>,
    /// Weights path expected in the server's properties response.
    pub expected_model: Option<PathBuf>,
    /// Interval between probes while no lease holds the engine.
    pub probe_interval: Duration,
    /// Time allowed for the engine to clear an abandoned request.
    pub settle: Duration,
    pub max_restarts: u32,
    pub restart_window: Duration,
    /// Delay before retrying recovery. Later attempts double it up to the cap.
    pub backoff_base: Duration,
    pub backoff_max: Duration,
    /// Optional preventive restart interval for an idle managed process.
    pub recycle: Option<Duration>,
    /// Deadline for the first health response from a new child process.
    pub start_timeout: Duration,
    /// Interval for background checks and maintenance.
    pub maintenance_tick: Duration,
}

/// Lease outcomes that determine the next supervisor state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeaseOutcome {
    /// Generation completed or the server rejected the request without an engine fault.
    Clean,
    /// A request failed without a classified fault. Probe before admitting more work.
    RequestFailed,
    /// The caller left while the engine might still be working. Wait for it to become idle.
    Abandoned,
    /// The engine failed and needs recovery.
    Fault(FaultCause),
    /// The agent is shutting down, so skip recovery.
    Shutdown,
}

pub struct Reject {
    pub reason: LeaseRejectReason,
    pub detail: &'static str,
}

pub struct Supervisor {
    pub engine: Arc<Engine>,
    pub stats: Arc<Stats>,
    pub progress: Arc<Progress>,
    cfg: SupervisorCfg,
    state: watch::Sender<EngineState>,
    gpu: Option<Arc<Telemetria>>,
    handled_loss: AtomicU64,
    stopping: AtomicBool,
    gate: Arc<tokio::sync::Mutex<()>>,
    child: tokio::sync::Mutex<Option<managed::Process>>,
    restarts: Mutex<VecDeque<Instant>>,
    process_started: Mutex<Instant>,
    last_probe: Mutex<Instant>,
}

enum Restart {
    /// The child is still alive after forced termination.
    Unkillable(anyhow::Error),
    Failed(anyhow::Error),
}

/// Keep the engine gate through the switch, including shutdown of the stage.
pub struct ParkedEngine {
    sup: Arc<Supervisor>,
    gate: OwnedMutexGuard<()>,
    pub gpu: Option<String>,
}

impl ParkedEngine {
    pub fn can_start_stage(&self) -> bool {
        self.sup.state() == EngineState::Parked && self.sup.can_run()
    }

    pub fn returning(&self) {
        self.sup.set_state(EngineState::Recovering);
    }

    pub fn unpark(self) {
        self.returning();
        tokio::spawn(async move {
            let Self { sup, gate, .. } = self;
            sup.resume_managed().await;
            drop(gate);
        });
    }
}

impl Supervisor {
    #[cfg(test)]
    pub fn new(engine: Arc<Engine>, cfg: SupervisorCfg, stats: Arc<Stats>, progress: Arc<Progress>) -> Arc<Self> {
        Self::with_state(engine, cfg, stats, progress, watch::channel(EngineState::Starting).0, None)
    }

    pub fn with_state(
        engine: Arc<Engine>,
        cfg: SupervisorCfg,
        stats: Arc<Stats>,
        progress: Arc<Progress>,
        state: watch::Sender<EngineState>,
        gpu: Option<Arc<Telemetria>>,
    ) -> Arc<Self> {
        Arc::new(Self {
            engine,
            stats,
            progress,
            cfg,
            state,
            gpu,
            handled_loss: AtomicU64::new(0),
            stopping: AtomicBool::new(false),
            gate: Arc::new(tokio::sync::Mutex::new(())),
            child: tokio::sync::Mutex::new(None),
            restarts: Mutex::new(VecDeque::new()),
            process_started: Mutex::new(Instant::now()),
            last_probe: Mutex::new(Instant::now()),
        })
    }

    pub fn state(&self) -> EngineState {
        *self.state.borrow()
    }

    pub fn subscribe(&self) -> watch::Receiver<EngineState> {
        self.state.subscribe()
    }

    fn set_state(&self, s: EngineState) {
        self.state.send_if_modified(|state| {
            // Preserve the GPU watcher's state if a lease or probe finishes while it stops the child.
            if *state == s || (matches!(s, EngineState::Ready | EngineState::Busy | EngineState::Settling) && !self.can_run()) {
                return false;
            }
            *state = s;
            true
        });
    }

    /// Report stall age only while a lease is active.
    pub fn health(&self) -> EngineHealth {
        let secs = if self.state() == EngineState::Busy { self.progress.since_last().as_secs().min(u64::from(u32::MAX)) as u32 } else { 0 };
        self.stats.health(secs)
    }

    /// Check readiness before the first session. A failed probe starts recovery.
    /// A managed process that cannot start or loads unexpected weights makes
    /// startup fail.
    pub async fn start(self: &Arc<Self>) -> Result<()> {
        let waiting = self.gpu.is_some() && !self.can_run();
        if !waiting {
            tokio::select! {
                biased;
                _ = self.wait_for_gpu_loss() => {}
                result = self.start_engine() => result?,
            }
        }
        if let Some(gpu) = &self.gpu {
            let me = self.clone();
            let device = gpu.device_rx();
            tokio::spawn(async move { me.monitor_gpu(device, waiting).await });
        }
        let me = self.clone();
        tokio::spawn(async move { me.maintenance_loop().await });
        Ok(())
    }

    async fn start_engine(self: &Arc<Self>) -> Result<()> {
        info!(phase = "agent alive", "agent started");
        let gate = self.gate.clone().lock_owned().await;
        if let Some(m) = &self.cfg.managed {
            managed::ensure_port_free(m.port).await?;
            *self.child.lock().await = Some(self.spawn_child(m)?);
            *self.process_started.lock().unwrap() = Instant::now();
            self.wait_health(true).await.context("managed llama-server did not become ready")?;
        }
        match self.bring_up().await {
            Ok(()) => self.set_state(EngineState::Ready),
            Err(BringUp::Fatal(e)) => return Err(e),
            Err(BringUp::Retry(e)) => {
                warn!(phase = "inference healthy", "engine failed its startup check; starting recovery: {e:#}");
                let me = self.clone();
                tokio::spawn(async move {
                    me.recover(FaultCause::ProbeFailed, false).await;
                    drop(gate);
                });
            }
        }
        Ok(())
    }

    pub fn try_park(self: &Arc<Self>) -> Option<ParkedEngine> {
        let cfg = self.cfg.managed.as_ref()?;
        let gate = self.gate.clone().try_lock_owned().ok()?;
        // The GPU monitor can change availability without taking the gate.
        if !self.state.send_if_modified(|state| {
            if *state != EngineState::Ready || !self.can_run() {
                return false;
            }
            *state = EngineState::Parked;
            true
        }) {
            return None;
        }
        let gpu = self.gpu.as_ref().and_then(|gpu| match &gpu.device_rx().borrow().device {
            Device::Ready(uuid) => Some(uuid.clone()),
            _ => None,
        }).or_else(|| cfg.gpu_index.clone());
        Some(ParkedEngine { sup: self.clone(), gate, gpu })
    }

    async fn resume_managed(self: &Arc<Self>) {
        let Some(cfg) = &self.cfg.managed else { return };
        if !self.can_run() {
            // The GPU monitor resumes the engine once the selected device returns.
            return;
        }
        let start = async {
            self.restart_child(cfg).await.map_err(|e| match e {
                Restart::Unkillable(e) | Restart::Failed(e) => e,
            })?;
            self.bring_up().await.map_err(|e| match e {
                BringUp::Fatal(e) | BringUp::Retry(e) => e,
            })?;
            self.check_child_running().await
        };
        let result = tokio::select! {
            biased;
            _ = self.wait_for_gpu_loss() => return,
            result = start => result,
        };
        match result {
            Ok(()) => self.set_state(EngineState::Ready),
            Err(e) => {
                warn!("could not resume llama-server: {e:#}");
                self.recover(FaultCause::Process, false).await;
            }
        }
    }

    fn can_run(&self) -> bool {
        !self.stopping.load(Ordering::Acquire) && self.gpu.as_ref().map_or(true, |gpu| {
            let rx = gpu.device_rx();
            let state = rx.borrow();
            matches!(state.device, Device::Ready(_))
                && state.loss_generation == self.handled_loss.load(Ordering::Acquire)
        })
    }

    fn spawn_child(&self, cfg: &managed::ManagedCfg) -> Result<managed::Process> {
        if self.stopping.load(Ordering::Acquire) {
            bail!("agent is stopping");
        }
        let device = self.gpu.as_ref().map(|gpu| gpu.device_rx());
        let state = device.as_ref().map(watch::Receiver::borrow);
        match state.as_deref() {
            Some(state) if state.loss_generation == self.handled_loss.load(Ordering::Acquire) => {
                match &state.device {
                    Device::Ready(uuid) => cfg.spawn(Some(uuid)),
                    _ => bail!("selected GPU unavailable"),
                }
            }
            Some(_) => bail!("selected GPU unavailable"),
            None => cfg.spawn(None),
        }
    }

    async fn wait_for_gpu_loss(&self) {
        if let Some(gpu) = &self.gpu {
            let generation = self.handled_loss.load(Ordering::Acquire);
            let _ = gpu.device_rx().wait_for(|state| state.loss_generation != generation).await;
        } else {
            std::future::pending::<()>().await;
        }
    }

    async fn monitor_gpu(self: Arc<Self>, mut device: watch::Receiver<DeviceState>, mut waiting: bool) {
        loop {
            let handled = self.handled_loss.load(Ordering::Acquire);
            let state = match device.wait_for(|state| state.loss_generation != handled
                || (waiting && matches!(state.device, Device::Ready(_)))).await {
                Ok(state) => state.clone(),
                Err(_) => return,
            };
            if self.stopping.load(Ordering::Acquire) {
                return;
            }
            // Stop an unavailable GPU's child without waiting for a lease or recovery backoff.
            self.state.send_if_modified(|engine| {
                if matches!(engine, EngineState::Parked | EngineState::Recovering) {
                    return false;
                }
                *engine = EngineState::Recovering;
                true
            });
            if state.loss_generation != handled && self.state() != EngineState::Parked {
                warn!("selected GPU unavailable; stopping llama-server");
                if let Err(e) = self.stop_child().await {
                    error!("cannot stop llama-server after GPU loss: {e:#}");
                }
            }
            if device.wait_for(|state| matches!(state.device, Device::Ready(_))).await.is_err() {
                return;
            }
            let _gate = self.gate.lock().await;
            // Check the driver again because it can disappear during engine startup.
            {
                let state = device.borrow();
                if !matches!(state.device, Device::Ready(_)) {
                    continue;
                }
                self.handled_loss.store(state.loss_generation, Ordering::Release);
            }
            if self.can_run() {
                if waiting {
                    waiting = false;
                    if let Some(cfg) = &self.cfg.managed {
                        if let Err(e) = managed::ensure_port_free(cfg.port).await {
                            self.quarantine(&format!("{e:#}"));
                            continue;
                        }
                    }
                }
                info!("selected GPU available; restarting llama-server");
                self.recover(FaultCause::Process, true).await;
            }
        }
    }

    pub(crate) async fn stop_child(&self) -> Result<()> {
        let mut child = self.child.lock().await;
        if let (Some(cfg), Some(process)) = (&self.cfg.managed, child.as_mut()) {
            managed::stop(process, cfg.stop_grace, cfg.kill_wait, "llama-server").await?;
            *child = None;
        }
        Ok(())
    }

    /// Stop and reap the child during agent shutdown.
    pub async fn shutdown(&self) {
        self.stopping.store(true, Ordering::Release);
        if let Err(e) = self.stop_child().await {
            error!("stop llama-server: {e:#}");
        }
    }

    /// Check whether the engine can accept a lease without reserving it.
    /// Use HOST_BUSY during recovery so the coordinator keeps the advertised
    /// weights in its inventory.
    pub fn check_ready(&self) -> std::result::Result<(), Reject> {
        if !self.can_run() {
            return Err(Reject { reason: LeaseRejectReason::HostBusy, detail: "engine unavailable" });
        }
        match self.state() {
            EngineState::Starting => Err(Reject { reason: LeaseRejectReason::HostBusy, detail: "engine starting" }),
            EngineState::Ready => Ok(()),
            EngineState::Busy => Err(Reject { reason: LeaseRejectReason::HostBusy, detail: "engine busy" }),
            EngineState::Settling => Err(Reject { reason: LeaseRejectReason::HostBusy, detail: "motore in assestamento" }),
            EngineState::Parked => Err(Reject { reason: LeaseRejectReason::ModelNotReady, detail: "engine parked for a stage" }),
            EngineState::Recovering => Err(Reject { reason: LeaseRejectReason::HostBusy, detail: "recovering the engine" }),
            EngineState::Quarantined => Err(Reject { reason: LeaseRejectReason::HostBusy, detail: "engine quarantined" }),
        }
    }

    /// Acquire the gate and reserve the engine until the lease guard is finished.
    pub fn try_begin(self: &Arc<Self>) -> std::result::Result<LeaseGuard, Reject> {
        self.check_ready()?;
        let Ok(guard) = self.gate.clone().try_lock_owned() else {
            return Err(Reject { reason: LeaseRejectReason::HostBusy, detail: "checking the engine" });
        };
        // Recheck availability under the gate because recovery can start while acquiring it.
        self.check_ready()?;
        self.set_state(EngineState::Busy);
        self.progress.begin_work();
        Ok(LeaseGuard { sup: self.clone(), guard: Some(guard) })
    }

    fn finish_with(self: &Arc<Self>, guard: OwnedMutexGuard<()>, outcome: LeaseOutcome) {
        if !self.can_run() || (self.gpu.is_some() && self.state() == EngineState::Recovering) {
            return;
        }
        match outcome {
            LeaseOutcome::Clean | LeaseOutcome::Shutdown => {
                self.set_state(EngineState::Ready);
                drop(guard);
            }
            LeaseOutcome::RequestFailed => {
                // Keep availability stable during the confirmation probe. The gate blocks new leases.
                self.set_state(EngineState::Ready);
                let me = self.clone();
                spawn_bg(async move {
                    me.probe_or_recover().await;
                    drop(guard);
                });
            }
            LeaseOutcome::Abandoned => {
                self.set_state(EngineState::Settling);
                let me = self.clone();
                spawn_bg(async move {
                    let settled = tokio::select! {
                        _ = me.wait_for_gpu_loss() => return,
                        result = me.engine.wait_idle(me.cfg.settle) => result,
                    };
                    match settled {
                        Ok(()) => {
                            info!(phase = "engine ready", "engine idle after request cancellation");
                            me.set_state(EngineState::Ready);
                        }
                        Err(e) => {
                            let cause = match classify(&e) {
                                ErrorClass::Fault(c) => c,
                                _ => FaultCause::ProbeFailed,
                            };
                            warn!("engine did not settle after revocation: {e:#}");
                            me.recover(cause, false).await;
                        }
                    }
                    drop(guard);
                });
            }
            LeaseOutcome::Fault(cause) => {
                // Publish the recovery state before the session releases the failed lease.
                self.set_state(EngineState::Recovering);
                let me = self.clone();
                spawn_bg(async move {
                    me.recover(cause, false).await;
                    drop(guard);
                });
            }
        }
    }

    async fn probe_or_recover(self: &Arc<Self>) {
        let probe = tokio::select! {
            _ = self.wait_for_gpu_loss() => return,
            result = self.engine.probe() => result,
        };
        match probe {
            Ok(d) => {
                self.stats.probe(true);
                *self.last_probe.lock().unwrap() = Instant::now();
                info!(phase = "inference healthy", ms = d.as_millis() as u64, "probe passed");
            }
            Err(e) => {
                self.stats.probe(false);
                warn!(phase = "inference healthy", "probe failed: {e:#}");
                self.recover(FaultCause::ProbeFailed, false).await;
            }
        }
    }

    /// Check the health endpoint and loaded weights before testing generation.
    async fn bring_up(&self) -> std::result::Result<(), BringUp> {
        self.engine.health().await.map_err(BringUp::Retry)?;
        info!(phase = "engine ready", "/health ok");
        match (self.engine.loaded_model_path().await, &self.cfg.expected_model) {
            (Err(e), _) => return Err(BringUp::Retry(e.context("read /props"))),
            (Ok(Some(loaded)), Some(expected)) => {
                let same = std::fs::canonicalize(&loaded).ok().zip(std::fs::canonicalize(expected).ok()).is_some_and(|(a, b)| a == b) || loaded == expected.to_string_lossy();
                if !same {
                    let msg = format!("llama-server loaded {loaded:?} instead of {}", expected.display());
                    if self.cfg.managed.is_some() {
                        // Unexpected weights are fatal when the supervisor chose the file itself.
                        return Err(BringUp::Fatal(anyhow!(msg)));
                    }
                    warn!(phase = "engine ready", "{msg}");
                }
            }
            _ => {}
        }
        info!(phase = "engine ready", "model verified");
        match self.engine.probe().await {
            Ok(d) => {
                self.stats.probe(true);
                *self.last_probe.lock().unwrap() = Instant::now();
                info!(phase = "inference healthy", ms = d.as_millis() as u64, "probe passed");
                Ok(())
            }
            Err(e) => {
                self.stats.probe(false);
                Err(BringUp::Retry(e))
            }
        }
    }

    /// Wait for health after spawning the child. Return an error if it exits before responding.
    async fn wait_health(&self, initial: bool) -> Result<()> {
        let deadline = Instant::now() + self.cfg.start_timeout;
        loop {
            self.check_child_running().await?;
            if self.engine.health().await.is_ok() {
                return Ok(());
            }
            if Instant::now() >= deadline {
                bail!("llama-server did not answer /health after {} s{}", self.cfg.start_timeout.as_secs(), if initial { "" } else { " after restart" });
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
    }

    async fn check_child_running(&self) -> Result<()> {
        let mut child = self.child.lock().await;
        let c = child.as_mut().context("llama-server stopped during startup")?;
        if let Some(st) = c.child.try_wait()? {
            bail!("llama-server exited during startup ({st}); see its log");
        }
        Ok(())
    }

    async fn restart_child(&self, m: &managed::ManagedCfg) -> std::result::Result<(), Restart> {
        let mut slot = self.child.lock().await;
        if let Some(c) = slot.as_mut() {
            // Confirm the entire group has exited before reusing the GPU.
            managed::stop(c, m.stop_grace, m.kill_wait, "llama-server").await.map_err(Restart::Unkillable)?;
        }
        *slot = None;
        managed::ensure_port_free(m.port).await.map_err(Restart::Failed)?;
        let child = self.spawn_child(m).map_err(Restart::Failed)?;
        *slot = Some(child);
        *self.process_started.lock().unwrap() = Instant::now();
        drop(slot);
        self.wait_health(false).await.map_err(Restart::Failed)
    }

    /// Return the tracked child PID for supervisor tests.
    #[cfg(test)]
    pub async fn child_pid_for_test(&self) -> u32 {
        self.child.lock().await.as_ref().unwrap().child.id().unwrap()
    }

    fn restarts_in_window(&self) -> usize {
        let mut r = self.restarts.lock().unwrap();
        while r.front().is_some_and(|t| t.elapsed() > self.cfg.restart_window) {
            r.pop_front();
        }
        r.len()
    }

    fn quarantine(&self, why: &str) {
        error!(phase = "inference healthy", "Engine quarantined: {why}. Restart the agent to retry.");
        self.set_state(EngineState::Quarantined);
    }

    fn backoff(&self, attempt: u32) -> Duration {
        self.cfg.backoff_base.saturating_mul(1u32 << attempt.min(10)).min(self.cfg.backoff_max)
    }

    /// Recover while holding the gate. Managed recovery stops and reaps the child
    /// before spawning and checking its replacement. External recovery keeps
    /// probing with backoff because the process belongs to the operator.
    async fn recover(self: &Arc<Self>, cause: FaultCause, preventive: bool) {
        if !self.can_run() {
            return;
        }
        tokio::select! {
            biased;
            _ = self.wait_for_gpu_loss() => {}
            _ = self.recover_engine(cause, preventive) => {}
        }
    }

    async fn recover_engine(&self, cause: FaultCause, preventive: bool) {
        self.set_state(EngineState::Recovering);
        warn!(phase = "engine ready", cause = cause.as_str(), preventive, "recovering the engine");
        let mut preventive = preventive;
        let mut attempt = 0u32;
        loop {
            if attempt > 0 {
                tokio::time::sleep(self.backoff(attempt - 1)).await;
            }
            if !self.can_run() {
                return;
            }
            if let Some(m) = &self.cfg.managed {
                let n = self.restarts_in_window();
                if !preventive && n as u32 >= self.cfg.max_restarts {
                    self.quarantine(&format!(
                        "{n} restarts in {} min did not recover the engine (initial fault: {})",
                        self.cfg.restart_window.as_secs() / 60,
                        cause.as_str()
                    ));
                    return;
                }
                self.stats.engine_restart();
                if !preventive {
                    self.restarts.lock().unwrap().push_back(Instant::now());
                }
                info!(phase = "engine ready", attempt = attempt + 1, "restarting llama-server");
                match self.restart_child(m).await {
                    Ok(()) => {}
                    Err(Restart::Unkillable(e)) => {
                        self.quarantine(&format!("process still running after SIGKILL: {e:#}"));
                        return;
                    }
                    Err(Restart::Failed(e)) => {
                        warn!(phase = "engine ready", "restart failed: {e:#}");
                        preventive = false;
                        attempt += 1;
                        continue;
                    }
                }
            }
            preventive = false;
            let ready = match self.bring_up().await {
                Ok(()) if self.cfg.managed.is_some() => self.check_child_running().await.map_err(BringUp::Retry),
                result => result,
            };
            match ready {
                Ok(()) => {
                    info!(phase = "inference healthy", "engine recovered");
                    self.set_state(EngineState::Ready);
                    return;
                }
                Err(BringUp::Fatal(e)) => {
                    self.quarantine(&format!("{e:#}"));
                    return;
                }
                Err(BringUp::Retry(e)) => {
                    warn!(phase = "inference healthy", attempt = attempt + 1, "engine health check still failing: {e:#}");
                    attempt += 1;
                }
            }
        }
    }

    /// Maintain an idle engine under the gate. This covers child exit, scheduled
    /// recycling and periodic inference probes.
    async fn maintenance_loop(self: Arc<Self>) {
        let mut tick = tokio::time::interval(self.cfg.maintenance_tick);
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            tick.tick().await;
            if self.state() != EngineState::Ready {
                continue;
            }
            let Ok(guard) = self.gate.clone().try_lock_owned() else { continue };
            if self.state() != EngineState::Ready {
                continue;
            }
            let exited = match self.child.lock().await.as_mut() {
                Some(c) => c.child.try_wait().ok().flatten(),
                None => None,
            };
            if let Some(st) = exited {
                warn!(phase = "engine ready", "managed llama-server exited ({st})");
                self.recover(FaultCause::Process, false).await;
            } else if self.cfg.managed.is_some() && self.cfg.recycle.is_some_and(|r| self.process_started.lock().unwrap().elapsed() >= r) {
                info!("recycling idle llama-server");
                self.recover(FaultCause::Process, true).await;
            } else if self.last_probe.lock().unwrap().elapsed() >= self.cfg.probe_interval {
                self.probe_or_recover().await;
            }
            drop(guard);
        }
    }
}

enum BringUp {
    Fatal(anyhow::Error),
    Retry(anyhow::Error),
}

fn spawn_bg(f: impl std::future::Future<Output = ()> + Send + 'static) {
    match tokio::runtime::Handle::try_current() {
        Ok(h) => {
            h.spawn(f);
        }
        Err(_) => warn!("engine maintenance requires a runtime"),
    }
}

/// Reserve the engine for one lease. Dropping the guard without finishing
/// treats the lease as abandoned.
pub struct LeaseGuard {
    sup: Arc<Supervisor>,
    guard: Option<OwnedMutexGuard<()>>,
}

impl LeaseGuard {
    pub fn finish(mut self, outcome: LeaseOutcome) {
        if let Some(g) = self.guard.take() {
            self.sup.finish_with(g, outcome);
        }
    }
}

impl Drop for LeaseGuard {
    fn drop(&mut self) {
        if let Some(g) = self.guard.take() {
            self.sup.finish_with(g, LeaseOutcome::Abandoned);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;

    use super::*;
    use crate::{
        engine::{llama::LlamaEngine, Limits},
        fake_llama::{self, Fake, FakeState},
    };

    fn limits(probe_ms: u64) -> Limits {
        Limits {
            prefill_stall: Duration::from_millis(1500),
            decode_stall: Duration::from_millis(1500),
            request_max: Duration::from_secs(60),
            call_timeout: Duration::from_secs(5),
            step_timeout: Duration::from_secs(5),
            probe_timeout: Duration::from_millis(probe_ms),
        }
    }

    fn cfg(managed: Option<managed::ManagedCfg>, expected: Option<&str>) -> SupervisorCfg {
        SupervisorCfg {
            managed,
            expected_model: expected.map(PathBuf::from),
            probe_interval: Duration::from_secs(3600),
            settle: Duration::from_millis(600),
            max_restarts: 2,
            restart_window: Duration::from_secs(600),
            backoff_base: Duration::from_millis(50),
            backoff_max: Duration::from_millis(200),
            recycle: None,
            start_timeout: Duration::from_secs(10),
            maintenance_tick: Duration::from_millis(100),
        }
    }

    fn make(url: String, probe_ms: u64, cfg: SupervisorCfg) -> Arc<Supervisor> {
        let progress = Arc::new(Progress::default());
        let engine = Arc::new(Engine::Llama(LlamaEngine::new(url, limits(probe_ms), progress.clone(), true).unwrap()));
        Supervisor::new(engine, cfg, Arc::new(Stats::new(None)), progress)
    }

    async fn ext(probe_ms: u64) -> (Fake, Arc<Supervisor>) {
        let f = fake_llama::spawn(0, FakeState::new("/modelli/finto.gguf")).await.unwrap();
        let sup = make(f.url(), probe_ms, cfg(None, None));
        (f, sup)
    }

    async fn wait_state(sup: &Supervisor, want: EngineState, secs: u64) {
        let t0 = Instant::now();
        while sup.state() != want {
            assert!(t0.elapsed() < Duration::from_secs(secs), "state {:?}, expected {want:?} after {secs} s", sup.state());
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    }

    fn free_port() -> u16 {
        std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port()
    }

    fn fake_bin() -> PathBuf {
        let p = std::env::current_exe().unwrap().parent().unwrap().parent().unwrap().join("fake-llama-server");
        assert!(p.exists(), "missing {}; run cargo build -p publicompute-agent", p.display());
        p
    }

    fn alive(pid: u32) -> bool {
        unsafe { libc::kill(pid as i32, 0) == 0 }
    }

    fn managed_cfg(tag: &str, port: u16, extra: &[&str]) -> managed::ManagedCfg {
        managed::ManagedCfg {
            bin: fake_bin(),
            model_path: PathBuf::from("/modelli/finto.gguf"),
            port,
            gpu_index: None,
            extra_args: extra.iter().map(|s| s.to_string()).collect(),
            log_path: std::env::temp_dir().join(format!("publicompute-fake-{}-{tag}.log", std::process::id())),
            stop_grace: Duration::from_secs(2),
            kill_wait: Duration::from_secs(3),
        }
    }

    async fn child_pid(sup: &Supervisor) -> u32 {
        sup.child.lock().await.as_ref().unwrap().child.id().unwrap()
    }

    #[tokio::test]
    async fn gpu_lost_blocks_restarts() {
        let gpu = Arc::new(Telemetria::default());
        gpu.set_device_for_test(Device::Lost(crate::gpu::ReadFailure::NoDevice));
        let progress = Arc::new(Progress::default());
        let url = "http://127.0.0.1:1".to_string();
        let engine = Arc::new(Engine::Llama(LlamaEngine::new(url, limits(20), progress.clone(), true).unwrap()));
        let cfg = cfg(Some(managed_cfg("gpu-lost", 1, &[])), None);
        let sup = Supervisor::with_state(
            engine, cfg, Arc::new(Stats::new(None)), progress, watch::channel(EngineState::Ready).0, Some(gpu),
        );
        assert!(sup.check_ready().is_err());
        sup.recover(FaultCause::Process, false).await;
        assert_eq!(sup.stats.run().engine_restarts, 0);
        assert!(sup.child.lock().await.is_none());
    }

    #[tokio::test]
    async fn startup_without_gpu_returns_before_engine_is_ready() {
        let gpu = Arc::new(Telemetria::default());
        gpu.set_device_for_test(Device::Checking);
        let progress = Arc::new(Progress::default());
        let engine = Arc::new(Engine::Mock(crate::engine::mock::MockEngine {
            per_token_delay: Duration::ZERO,
            cheat: false,
            hang: Default::default(),
            limits: limits(20),
            progress: progress.clone(),
        }));
        let sup = Supervisor::with_state(
            engine, cfg(None, None), Arc::new(Stats::new(None)), progress,
            watch::channel(EngineState::Starting).0, Some(gpu.clone()),
        );
        sup.start().await.unwrap();
        assert_eq!(sup.state(), EngineState::Starting);
        assert!(sup.check_ready().is_err());
        gpu.set_device_for_test(Device::Ready("GPU-test".into()));
        wait_state(&sup, EngineState::Ready, 1).await;
    }

    /// A health response alone cannot make the engine eligible for leases.
    #[tokio::test]
    async fn healthy_but_unresponsive_engine_is_not_ready() {
        let (f, sup) = ext(400).await;
        f.state.hang_completion.store(true, Ordering::SeqCst);
        sup.start().await.unwrap();
        // The initial probe must pass before startup can advertise readiness.
        assert_eq!(sup.state(), EngineState::Starting);
        assert!(sup.try_begin().is_err());
        tokio::time::sleep(Duration::from_millis(800)).await;
        assert_eq!(sup.state(), EngineState::Recovering, "a successful health check is not enough");
        assert!(sup.stats.run().probes_failed >= 1);
        f.state.hang_completion.store(false, Ordering::SeqCst);
        wait_state(&sup, EngineState::Ready, 10).await;
        assert!(sup.try_begin().is_ok());
    }

    /// A revoked lease holds the engine unavailable until its abandoned request clears.
    #[tokio::test]
    async fn abandoned_request_blocks_offers_until_engine_is_idle() {
        let (f, sup) = ext(400).await;
        sup.start().await.unwrap();
        assert_eq!(sup.state(), EngineState::Ready);
        let g = sup.try_begin().ok().unwrap();
        f.state.busy.store(true, Ordering::SeqCst);
        g.finish(LeaseOutcome::Abandoned);
        assert_eq!(sup.state(), EngineState::Settling, "an abandoned lease is not an engine fault");
        assert!(sup.try_begin().is_err());
        tokio::time::sleep(Duration::from_millis(300)).await;
        assert!(sup.try_begin().is_err(), "still busy");
        f.state.busy.store(false, Ordering::SeqCst);
        wait_state(&sup, EngineState::Ready, 5).await;
        assert_eq!(sup.stats.run().engine_restarts, 0);
    }

    /// A cancelled lease that settles successfully returns to Ready without recovery
    /// or an extra probe.
    #[tokio::test]
    async fn cancellation_transitions_through_settling() {
        let (f, sup) = ext(400).await;
        sup.start().await.unwrap();
        let g = sup.try_begin().ok().unwrap();
        f.state.busy.store(true, Ordering::SeqCst);
        g.finish(LeaseOutcome::Abandoned);
        let t0 = Instant::now();
        while t0.elapsed() < Duration::from_millis(250) {
            assert_eq!(sup.state(), EngineState::Settling, "settling should not trigger recovery");
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        f.state.busy.store(false, Ordering::SeqCst);
        let t0 = Instant::now();
        while sup.state() != EngineState::Ready {
            assert_eq!(sup.state(), EngineState::Settling);
            assert!(t0.elapsed() < Duration::from_secs(5), "engine did not become ready again");
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert_eq!(sup.stats.run().engine_restarts, 0);
        assert_eq!(sup.stats.run().probes_failed, 0);
    }

    /// A settle deadline failure triggers recovery with backoff.
    #[tokio::test]
    async fn engine_that_never_frees_goes_to_recovery() {
        let (f, sup) = ext(300).await;
        sup.start().await.unwrap();
        let g = sup.try_begin().ok().unwrap();
        f.state.busy.store(true, Ordering::SeqCst);
        g.finish(LeaseOutcome::Abandoned);
        tokio::time::sleep(Duration::from_millis(2200)).await;
        assert_eq!(sup.state(), EngineState::Recovering);
        assert!(sup.stats.run().probes_failed >= 1, "recovery should run a probe after settling");
        f.state.busy.store(false, Ordering::SeqCst);
        wait_state(&sup, EngineState::Ready, 10).await;
    }

    /// An unfinished guard follows the abandoned-lease path when dropped.
    #[tokio::test]
    async fn dropped_guard_is_treated_as_abandoned() {
        let (f, sup) = ext(400).await;
        sup.start().await.unwrap();
        let g = sup.try_begin().ok().unwrap();
        f.state.busy.store(true, Ordering::SeqCst);
        drop(g);
        assert_eq!(sup.state(), EngineState::Settling);
        f.state.busy.store(false, Ordering::SeqCst);
        wait_state(&sup, EngineState::Ready, 5).await;
    }

    /// A periodic probe must leave an active lease uninterrupted.
    #[tokio::test]
    async fn periodic_probe_runs_when_idle_and_never_during_a_lease() {
        let f = fake_llama::spawn(0, FakeState::new("/m.gguf")).await.unwrap();
        let mut c = cfg(None, None);
        c.probe_interval = Duration::from_millis(200);
        let sup = make(f.url(), 1000, c);
        sup.start().await.unwrap();
        tokio::time::sleep(Duration::from_millis(900)).await;
        assert!(sup.stats.run().probes_ok >= 3, "probes_ok = {}", sup.stats.run().probes_ok);
        let g = sup.try_begin().ok().unwrap();
        let during = f.state.completions.load(Ordering::SeqCst);
        tokio::time::sleep(Duration::from_millis(800)).await;
        assert_eq!(f.state.completions.load(Ordering::SeqCst), during, "do not probe during a lease");
        g.finish(LeaseOutcome::Clean);
        tokio::time::sleep(Duration::from_millis(600)).await;
        assert!(f.state.completions.load(Ordering::SeqCst) > during);
    }

    /// A replacement process becomes ready only after its weights and probe are checked.
    #[tokio::test]
    async fn managed_recovers_after_restart() {
        let port = free_port();
        let flag = std::env::temp_dir().join(format!("publicompute-hang-first-{}", std::process::id()));
        let _ = std::fs::remove_file(&flag);
        let m = managed_cfg("recover", port, &["--fake-hang-first-run", flag.to_str().unwrap()]);
        let sup = make(format!("http://127.0.0.1:{port}"), 500, cfg(Some(m), Some("/modelli/finto.gguf")));
        sup.start().await.unwrap();
        // The first child serves health but hangs on completion, so startup stays pending.
        assert_eq!(sup.state(), EngineState::Starting);
        let old = child_pid(&sup).await;
        assert!(sup.try_begin().is_err());
        wait_state(&sup, EngineState::Ready, 20).await;
        let new = child_pid(&sup).await;
        assert_ne!(old, new);
        assert!(!alive(old), "the replaced process should have exited");
        assert!(alive(new));
        assert_eq!(sup.stats.run().engine_restarts, 1);
        assert!(sup.stats.run().probes_ok >= 1);
        assert_eq!(sup.engine.loaded_model_path().await.unwrap().as_deref(), Some("/modelli/finto.gguf"));
        sup.shutdown().await;
        assert!(!alive(new));
        let _ = std::fs::remove_file(&flag);
    }

    /// Exhausting the managed restart budget quarantines the engine.
    #[tokio::test]
    async fn managed_goes_to_quarantine_after_k_restarts() {
        let port = free_port();
        let m = managed_cfg("quarantine", port, &["--fake-always-hang"]);
        let sup = make(format!("http://127.0.0.1:{port}"), 300, cfg(Some(m), None));
        sup.start().await.unwrap();
        wait_state(&sup, EngineState::Quarantined, 30).await;
        assert_eq!(sup.stats.run().engine_restarts, 2, "max_restarts = 2");
        assert!(sup.try_begin().is_err());
        let pid = child_pid(&sup).await;
        assert!(alive(pid), "recovery must stop retrying");
        // Quarantine prevents further automatic attempts.
        tokio::time::sleep(Duration::from_millis(600)).await;
        assert_eq!(sup.stats.run().engine_restarts, 2);
        assert_eq!(sup.state(), EngineState::Quarantined);
        sup.shutdown().await;
        assert!(!alive(pid));
    }

    #[tokio::test]
    async fn managed_refuses_to_start_on_busy_port() {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = l.local_addr().unwrap().port();
        let sup = make(format!("http://127.0.0.1:{port}"), 300, cfg(Some(managed_cfg("busy", port, &[])), None));
        let err = sup.start().await.unwrap_err();
        assert!(format!("{err:#}").contains("already in use"), "{err:#}");
    }

    #[tokio::test]
    async fn managed_wrong_model_is_fatal_but_external_only_warns() {
        let port = free_port();
        let sup = make(format!("http://127.0.0.1:{port}"), 300, cfg(Some(managed_cfg("model", port, &[])), Some("/other/model.gguf")));
        let err = sup.start().await.unwrap_err();
        assert!(format!("{err:#}").contains("instead of"), "{err:#}");
        sup.shutdown().await;

        let (_f, sup) = {
            let f = fake_llama::spawn(0, FakeState::new("/modelli/finto.gguf")).await.unwrap();
            let sup = make(f.url(), 300, cfg(None, Some("/other/model.gguf")));
            (f, sup)
        };
        sup.start().await.unwrap();
        assert_eq!(sup.state(), EngineState::Ready);
    }

    /// Preventive recycling waits for the active lease to finish.
    #[tokio::test]
    async fn recycle_waits_for_the_lease_to_end() {
        let port = free_port();
        let mut c = cfg(Some(managed_cfg("recycle", port, &[])), None);
        c.recycle = Some(Duration::from_millis(800));
        let sup = make(format!("http://127.0.0.1:{port}"), 1000, c);
        sup.start().await.unwrap();
        assert_eq!(sup.state(), EngineState::Ready);
        let old = child_pid(&sup).await;
        let g = sup.try_begin().ok().unwrap();
        tokio::time::sleep(Duration::from_millis(1800)).await;
        assert_eq!(sup.stats.run().engine_restarts, 0, "do not restart during a lease");
        assert_eq!(child_pid(&sup).await, old);
        g.finish(LeaseOutcome::Clean);
        let t0 = Instant::now();
        while sup.stats.run().engine_restarts == 0 {
            assert!(t0.elapsed() < Duration::from_secs(10), "preventively restarting the engine failed");
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        wait_state(&sup, EngineState::Ready, 10).await;
        assert!(!alive(old));
        sup.shutdown().await;
    }

    /// An external engine with a failed lease stays in recovery until it passes a probe.
    #[tokio::test]
    async fn fault_outcome_triggers_recovery_and_returns_to_ready() {
        let (f, sup) = ext(400).await;
        sup.start().await.unwrap();
        let g = sup.try_begin().ok().unwrap();
        f.state.hang_completion.store(true, Ordering::SeqCst);
        g.finish(LeaseOutcome::Fault(FaultCause::Watchdog));
        assert_eq!(sup.state(), EngineState::Recovering);
        tokio::time::sleep(Duration::from_millis(700)).await;
        assert_eq!(sup.state(), EngineState::Recovering);
        f.state.hang_completion.store(false, Ordering::SeqCst);
        wait_state(&sup, EngineState::Ready, 10).await;
        assert_eq!(sup.stats.run().engine_restarts, 0, "external mode must not restart the engine");
    }
}
