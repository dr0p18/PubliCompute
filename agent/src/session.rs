//! The control stream carries offers, lease updates and heartbeats. Each accepted
//! lease gets a separate work stream. Send the acceptance and an immediate
//! heartbeat on the control stream. Wait for the lease renewal before opening
//! the work stream, since the renewal confirms that the coordinator has recorded
//! the acceptance.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use anyhow::{bail, Context, Result};
use prost_types::Timestamp;
use publicompute_proto::{
    common::ModelRef,
    node::{
        coordinator_message, lease_update, node_message, stage_status::State as StageState, work_offer, work_stream_coordinator_message, work_stream_node_message, request_result,
        AgentFeature, Availability, AvailabilityUpdate, Completion, ContributionStatus, ContributionSummary, CoordinatorMessage,
        GoodbyeReason, Heartbeat, Hello, InstalledModel, LeaseAccept, Leaving, ModelDirective, ModelDirectiveStatus, RejectReason,
        LeaseComplete, LeaseProgress, LeaseReject, LeaseRejectReason, LeaseRelease, LeaseReleaseReason, LeaseRevokeReason, LogicCommitment,
        MeasurementPong, ModelInventory, ModelStatus, NodeMessage, PeerProbeReady, PositionLogprobs, RequestFailure, RequestFailureCode, RequestResult,
        StageStatus, Telemetry, Timing, TokenDelta, TokenLogprob, Usage, VerifyResult, VerifyTask, WorkOffer, WorkStreamCoordinatorMessage,
        WorkStreamNodeMessage, WorkStreamOpen,
    },
    PROTOCOL_VERSION,
};
use tokio::sync::{mpsc, oneshot, watch};
use tracing::{debug, info, warn};

use crate::{
    contribution::Contribution,
    economics::{self, HostCosts},
    engine::{
        classify,
        output::{wait_for_saturation, OutputStalled},
        progress::{Phase, Progress},
        stages::delta_channel,
        supervisor::{EngineState, LeaseGuard, LeaseOutcome, Supervisor},
        Engine, ErrorClass,
    },
    framed::{self, FrameReader},
    gpu::Telemetria,
    logic,
    network,
    transport,
};

pub struct NodeConfig {
    pub model: ModelRef,
    pub costs: Option<HostCosts>,
    pub hardware: publicompute_proto::node::HardwareProfile,
    /// Engine health and execution reservation.
    pub sup: Arc<Supervisor>,
    /// Stage process manager, when this agent runs a stage.
    pub stages: Option<Arc<crate::stages::Stages>>,
    /// The second engine on a node that can switch from managed llama-server.
    pub stage_engine: Option<Arc<Engine>>,
    /// Region hint sent in the handshake. An empty value leaves it unset.
    pub region: String,
    /// Measured NAT classification sent in the handshake.
    pub nat_class: String,
    /// Public bridge endpoint ID. Without it, chain links use direct TCP.
    pub iroh_node_id: String,
    /// Local chat capacity for an HTTP engine. A stage chain provides its own capacity.
    pub slots: usize,
    /// Stored contribution preference and the coordinator's latest status and summary.
    pub contribution: Arc<Contribution>,
    /// Latest sampled GPU telemetry, retained across reconnects.
    pub gpu: Arc<Telemetria>,
    /// Weights path used to report actual bytes. Without a path, the reported size is zero.
    pub model_path: Option<std::path::PathBuf>,
    /// First shutdown signal requests an acknowledged departure. A closed sender
    /// leaves the request pending. A second signal uses the immediate shutdown path.
    pub output: watch::Receiver<bool>,
    /// Session-owned state exposed by the local control socket.
    pub live: Arc<crate::control::Live>,
    pub model_store: Option<Arc<crate::models::Store>>,
}

impl NodeConfig {
    pub async fn commit_model(&self, change: crate::model_change::Change) -> bool {
        let store = self.model_store.as_ref().expect("switch needs a model store");
        let committed = self.live.changes.commit(change, store.clone(), &self.contribution).await;
        self.live.notify();
        committed
    }

    pub fn gpu_profile_missing(&self) -> bool {
        self.hardware.gpus.is_empty() && self.gpu.monitored()
    }
}

/// Session state retained across reconnects.
#[derive(Default)]
pub struct Shared {
    /// Measured token throughput for estimating the cost of later jobs.
    pub tokens_per_sec: Mutex<Option<f64>>,
    /// Highest directive epoch observed. Require a strictly increasing epoch.
    pub epoca_direttiva: std::sync::atomic::AtomicU64,
}

/// Grace period requested from the coordinator and limit on waiting for departure.
pub const GRAZIA_USCITA_S: u32 = 300;
pub const ATTESA_USCITA: Duration = Duration::from_secs(GRAZIA_USCITA_S as u64 + 5);
const CLOSE_IO_TIMEOUT: Duration = Duration::from_secs(2);

/// Format the user-facing shutdown message. Non-head stages do not track active
/// answers directly, so they show a generic chain drain message instead of a count.
fn departure_notice(role: Option<crate::stages::Role>, has_chain: bool, active_answers: usize) -> String {
    match role {
        Some(crate::stages::Role::Middle | crate::stages::Role::Tail) => {
            "waiting for the chain to finish its answers, Ctrl+C again to quit now".into()
        }
        None if has_chain && active_answers == 0 => {
            "waiting for the chain to finish its answers, Ctrl+C again to quit now".into()
        }
        _ if active_answers == 1 => {
            "waiting for 1 answer to finish, Ctrl+C again to quit now".into()
        }
        _ if active_answers > 1 => {
            format!("waiting for {active_answers} answers to finish, Ctrl+C again to quit now")
        }
        _ => "shutting down, Ctrl+C again to quit now".into(),
    }
}

/// Return the file size in bytes, or zero when it cannot be measured.
fn dimensione_modello(path: &Option<std::path::PathBuf>) -> u64 {
    path.as_ref().and_then(|p| std::fs::metadata(p).ok()).map(|m| m.len()).unwrap_or(0)
}

/// Build the weights inventory with the measured file size and slot count.
fn inventory(cfg: &NodeConfig, engine: &Engine) -> ModelInventory {
    let mut models = Vec::new();
    if !cfg.model.model_id.is_empty() {
        models.push(InstalledModel {
            model: Some(cfg.model.clone()),
            status: ModelStatus::Ready as i32,
            loaded: !matches!(engine, Engine::Stages(_)) && single_loaded(cfg),
            size_bytes: dimensione_modello(&cfg.model_path),
            slots: capacita(cfg, engine) as u32,
            downloaded_bytes: 0,
        });
    }
    if let Some(transfer) = cfg.live.changes.inventory() {
        models.push(transfer);
    }
    ModelInventory { models }
}

fn single_loaded(cfg: &NodeConfig) -> bool {
    !cfg.gpu.unavailable() && matches!(cfg.sup.state(), EngineState::Ready | EngineState::Busy | EngineState::Settling)
}

fn features(cfg: &NodeConfig) -> Vec<i32> {
    let mut features = vec![AgentFeature::Contribution as i32];
    if cfg.stages.is_some() {
        features.push(AgentFeature::Stages as i32);
        // The stage binary shipped with the agent applies the assigned engine profile.
        features.push(AgentFeature::StageEngineProfile as i32);
    }
    if cfg.stage_engine.is_some() {
        features.push(AgentFeature::EngineSwitch as i32);
    }
    if cfg.live.network.enabled() {
        features.push(AgentFeature::NetworkMeasurement as i32);
    }
    if cfg.stages.as_ref().and_then(|s| s.bridge()).is_some() {
        features.push(AgentFeature::PeerEcho as i32);
    }
    features
}

#[derive(Debug)]
pub enum SessionEnd {
    Shutdown,
    SwitchModel(crate::model_change::Change),
    Reconnect,
    /// The coordinator rejects the identity, so the agent must enroll again.
    Unauthenticated,
    Banned,
    /// Reconnecting cannot change the installed release or the wire protocol.
    IncompatibleRelease(IncompatibleRelease),
    Lost(anyhow::Error),
}

#[derive(Default)]
struct LeaseCounters {
    requests: u32,
    tokens: u64,
    /// Once LeaseComplete is sent, the coordinator can reuse the capacity while
    /// the work task finishes closing its stream.
    completing: bool,
    /// Confirm engine health after the lease if a request failure could not be classified.
    unclassified_failure: bool,
}

struct ActiveLease {
    id: String,
    stage: bool,
    progress: Arc<Mutex<LeaseCounters>>,
    /// Concurrent chats have independent progress. A lease without its own
    /// counter uses the engine's shared counter.
    own: Option<Arc<Progress>>,
    confirm: Option<oneshot::Sender<()>>,
    task: tokio::task::JoinHandle<()>,
    /// Reserving the engine transfers it to the lease guard. Dropping the guard
    /// treats the lease as abandoned.
    guard: Option<LeaseGuard>,
}

/// Track work streams and the engine reservation separately. Concurrent chats
/// share one reservation from the first acceptance until the last completion.
/// An exclusive lease keeps its own guard.
#[derive(Default)]
struct Leases {
    map: HashMap<String, ActiveLease>,
    comune: Option<LeaseGuard>,
    /// Run a probe after the final concurrent lease if any lease had an unclassified request error.
    sospetto: bool,
}

impl Leases {
    fn attive(&self) -> usize {
        self.map.values().filter(|l| !l.progress.lock().unwrap().completing).count()
    }

    fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Finish the shared reservation with the given outcome, if one exists.
    fn chiudi_comune(&mut self, outcome: LeaseOutcome) {
        if let Some(g) = self.comune.take() {
            g.finish(outcome);
        }
    }
}

/// Read chat capacity from the chain assignment or the configured HTTP slot count.
fn capacita(cfg: &NodeConfig, engine: &Engine) -> usize {
    match (engine, &cfg.stages) {
        (Engine::Stages(_), Some(s)) => s.slots(),
        _ => cfg.slots.clamp(1, 16),
    }
}

/// Decide whether leases share the engine guard and report independent progress.
fn condivide(cfg: &NodeConfig, engine: &Engine) -> bool {
    matches!(engine, Engine::Stages(_)) || capacita(cfg, engine) > 1
}

struct LeaseDone {
    id: String,
    chat_id: Option<String>,
    result: Result<(u64, Duration)>,
}

pub async fn run(conn: quinn::Connection, cfg: &NodeConfig, engine: Arc<Engine>, shared: Arc<Shared>, mut shutdown: watch::Receiver<bool>) -> SessionEnd {
    match run_inner(&conn, cfg, engine, shared, &mut shutdown).await {
        Ok(end) => end,
        Err(e) => match e.downcast::<IncompatibleRelease>() {
            Ok(rejection) => {
                conn.close(0u32.into(), b"");
                SessionEnd::IncompatibleRelease(rejection)
            }
            Err(e) => match conn.close_reason().as_ref().and_then(transport::app_close_code) {
                Some(transport::CLOSE_UNAUTHENTICATED) => SessionEnd::Unauthenticated,
                Some(transport::CLOSE_BANNED) => SessionEnd::Banned,
                Some(transport::CLOSE_REPLACED) => {
                    SessionEnd::Lost(anyhow::anyhow!("session replaced; another agent may be using this identity"))
                }
                _ => SessionEnd::Lost(e),
            },
        },
    }
}

async fn run_inner(
    conn: &quinn::Connection,
    cfg: &NodeConfig,
    engine: Arc<Engine>,
    shared: Arc<Shared>,
    shutdown: &mut watch::Receiver<bool>,
) -> Result<SessionEnd> {
    let (send, recv) = conn.open_bi().await?;

    // A single writer assigns sequence numbers to match the order on the wire.
    let (out_tx, mut out_rx) = mpsc::channel::<node_message::Body>(64);
    let (probe_tx, mut probe_rx) = mpsc::channel::<node_message::Body>(2);
    let writer = tokio::spawn(async move {
        let mut send = send;
        let mut seq = 0u64;
        loop {
            let body = tokio::select! {
                biased;
                body = out_rx.recv() => match body {
                    Some(body) => body,
                    None => break,
                },
                Some(body) = probe_rx.recv() => body,
            };
            seq += 1;
            if framed::write(&mut send, &NodeMessage { seq, body: Some(body) }).await.is_err() {
                break;
            }
        }
        if send.finish().is_ok() {
            let _ = send.stopped().await;
        }
    });
    let out = Out(out_tx);

    let (in_tx, mut in_rx) = mpsc::channel::<CoordinatorMessage>(64);
    tokio::spawn(async move {
        let mut r = FrameReader::new(recv);
        while let Ok(Some(m)) = r.read::<CoordinatorMessage>().await {
            if in_tx.send(m).await.is_err() {
                break;
            }
        }
    });

    let inv_iniziale = inventory(cfg, &engine);
    let mut gpu_rx = cfg.gpu.device_rx();
    let mut loss_generation = gpu_rx.borrow_and_update().loss_generation;
    // A reconnect can follow a release while the single engine is still settling.
    let mut waiting_for_single = cfg.stage_engine.is_some();
    let mut last_avail = if cfg.gpu_profile_missing() {
        Availability::PausedByHost
    } else {
        session_availability(cfg, cfg.sup.state(), waiting_for_single)
    };
    out.send(node_message::Body::Hello(Hello {
        protocol_version: Some(PROTOCOL_VERSION),
        agent_version: env!("CARGO_PKG_VERSION").to_string(),
        hardware: Some(cfg.hardware.clone()),
        models: Some(inv_iniziale.clone()),
        availability: last_avail as i32,
        region: cfg.region.clone(),
        nat_class: cfg.nat_class.clone(),
        iroh_node_id: cfg.iroh_node_id.clone(),
        contribution: Some(cfg.contribution.to_proto(&modelli_di(&inv_iniziale))),
        features: features(cfg),
    }))
    .await?;

    let welcome = loop {
        tokio::select! {
            biased;
            message = in_rx.recv() => break message.and_then(|m| m.body),
            change = cfg.live.changes.wait_switch() => {
                if cfg.live.changes.claim(&change).await {
                    drop(out);
                    let _ = tokio::time::timeout(CLOSE_IO_TIMEOUT, writer).await;
                    conn.close(0u32.into(), b"");
                    return Ok(SessionEnd::SwitchModel(change));
                }
            }
        }
    };
    let welcome = match welcome {
        Some(coordinator_message::Body::Welcome(w)) => w,
        Some(coordinator_message::Body::Error(e)) => {
            if let Some(rejection) = IncompatibleRelease::from_protocol(&e) {
                return Err(rejection.into());
            }
            bail!("coordinator rejected the session: {} ({:?})", e.message, e.code());
        }
        _ => bail!("expected Welcome"),
    };
    let hb_every = welcome
        .heartbeat_interval
        .and_then(|d| Duration::try_from(d).ok())
        .filter(|d| !d.is_zero())
        .unwrap_or(Duration::from_secs(10));
    if welcome.min_agent_version != env!("CARGO_PKG_VERSION") {
        return Err(IncompatibleRelease(format!("the coordinator requires agent version {}", welcome.min_agent_version)).into());
    }
    if welcome.protocol_version != Some(PROTOCOL_VERSION) {
        return Err(IncompatibleRelease("the coordinator uses an unsupported protocol version".into()).into());
    }
    info!(session = %welcome.session_id, tier = ?welcome.tier(), "sessione aperta");
    cfg.live.set_connected(true);
    let local_time = network::unix_seconds() as i64;
    let clock_offset = crate::stages::clock_offset_from_welcome(welcome.server_time.as_ref(), local_time);
    let bridge = welcome.peer_probes.then(|| cfg.stages.as_ref().and_then(|s| s.bridge())).flatten();
    if let Some(s) = &cfg.stages {
        // Use the handshake's signing key to verify capabilities for stage assignments.
        if let Err(e) = s.set_capability_key(&welcome.capability_pub) {
            warn!("could not save the capability signing key: {e:#}");
        }
        // Correct for clock skew when stages check capability expiry.
        let scarto = clock_offset;
        s.set_clock_offset(scarto);
        info!(clock_offset_s = scarto, "clock offset relative to the coordinator");
        if scarto.abs() > 60 {
            warn!(clock_offset_s = scarto, "clock differs from the coordinator by more than 60 s; synchronize it");
        }
        if scarto.abs() > 240 {
            warn!(clock_offset_s = scarto, "clock offset exceeds 4 min; stage TLS may fail because certificates are not yet valid");
        }
    }

    let mut heartbeat = tokio::time::interval(hb_every);
    heartbeat.tick().await; // Consume the immediate timer tick so the next tick follows the heartbeat interval.
    let (done_tx, mut done_rx) = mpsc::channel::<LeaseDone>(4);
    let mut lease = Leases::default();
    let mut draining = false;
    let mut state_rx = cfg.sup.subscribe();
    state_rx.mark_changed();
    let mut pause_rx = cfg.live.pause_rx();
    pause_rx.mark_changed();
    let mut model_rx = cfg.live.changes.subscribe();
    // Subscribe to preference changes only when the peer supports updates.
    let mut pref_rx = cfg.contribution.subscribe();
    pref_rx.mark_unchanged();
    let mut ultimo_inventario = inv_iniziale;
    let mut uscita_rx = cfg.output.clone();
    if *uscita_rx.borrow() {
        uscita_rx.mark_changed();
    }
    let mut uscita_chiusa = false;
    let mut output: Option<Instant> = None;
    let mut conto = tokio::time::interval(Duration::from_secs(10));
    conto.tick().await;
    let mut ultimo_stato: Option<ContributionStatus> = None;
    let mut ultimo_riepilogo: Option<ContributionSummary> = None;
    // Subscribe to stage state changes when this agent owns a stage process.
    let (stage_tx, mut stage_rx) = mpsc::channel::<StageStatus>(32);
    if let Some(s) = &cfg.stages {
        s.connect_peer(stage_tx);
    }
    let mut echo = tokio::task::JoinSet::new();
    let mut bandwidth = tokio::task::JoinSet::new();
    let mut rtt_tick = tokio::time::interval(Duration::from_millis(500));
    rtt_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let mut bandwidth_tick = tokio::time::interval(Duration::from_secs(30));
    bandwidth_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut rtt = [0; network::RTT_SAMPLES];
    let mut rtt_count = 0;
    cfg.live.network.begin_session(network::unix_seconds());
    let mut model_departure = None;
    let mut leave_ok = false;
    let mut pending_stage_status = None;
    // Race the full control loop against the switch deadline. A full output
    // queue can suspend any branch, including heartbeats and inventory updates.
    let run_control = async {
        loop {
            let change = cfg.live.changes.switching();
            let chain_active = cfg.stages.as_ref().is_some_and(|s| s.has_chain());
            let idle = lease.is_empty() && !chain_active;
            if let Some(change) = change {
                if model_departure.is_none() && output.is_none() {
                    let deadline = change.deadline().unwrap();
                    let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
                    let grazia_s = remaining.as_secs().min(u64::from(GRAZIA_USCITA_S)) as u32;
                    out.send(node_message::Body::Leaving(Leaving { grazia_s })).await?;
                    model_departure = Some(deadline);
                }
                if output.is_none() && idle && leave_ok && cfg.live.changes.claim(&change).await {
                    break Ok(SessionEnd::SwitchModel(change));
                }
            } else if let Some(deadline) = model_departure {
                // Leaving is irrevocable on this connection. Reconnect if the
                // operator cancels or replaces the request during the drain.
                if output.is_none() && (leave_ok && idle || deadline <= tokio::time::Instant::now()) {
                    break Ok(SessionEnd::Reconnect);
                }
            }
            let generation = gpu_rx.borrow_and_update().loss_generation;
            if let Some(s) = &cfg.stages {
                if generation != loss_generation {
                    s.release_current("selected GPU unavailable");
                } else if cfg.live.is_paused() {
                    s.release_current("agent paused by host");
                }
            }
            let released = gpu_loss_leases(generation, &mut loss_generation, &mut lease, &cfg.live);
            let inv = inventory(cfg, &engine);
            if inv != ultimo_inventario {
                out.send(node_message::Body::ModelInventory(inv.clone())).await?;
                ultimo_inventario = inv;
            }
            if cfg.gpu_profile_missing() {
                if !cfg.gpu.unavailable() && (cfg.model.model_id.is_empty() || cfg.sup.state() == EngineState::Ready) {
                    // Reconnect to refresh the coordinator inventory because the hardware profile is sent in Hello.
                    break Ok(SessionEnd::Reconnect);
                }
            } else {
                // Let a terminal StageStatus reach the scheduler before the
                // availability change can dissolve the chain on its own.
                let state = match (cfg.stages.as_ref().and_then(|s| s.switch_state()), cfg.sup.state()) {
                    (Some(EngineState::Parked), _) => EngineState::Parked,
                    (Some(EngineState::Recovering), EngineState::Ready) => EngineState::Recovering,
                    (_, state) => state,
                };
                let availability = session_availability(cfg, state, waiting_for_single);
                send_availability(&out, availability, &mut last_avail).await?;
                if availability == Availability::Idle {
                    waiting_for_single = false;
                }
            }
            send_releases(&out, released, LeaseReleaseReason::EngineFailure).await?;
            if draining && lease.is_empty() {
                break Ok(SessionEnd::Shutdown);
            }
            let scadenza_uscita = output
                .map(|t| tokio::time::Instant::from_std(t + ATTESA_USCITA))
                .unwrap_or_else(|| tokio::time::Instant::now() + Duration::from_secs(3600));
            let idle = last_avail == Availability::Idle && lease.is_empty()
                && !cfg.live.has_chain() && !draining && output.is_none();
            if !idle {
                bandwidth.abort_all();
            }
            tokio::select! {
                m = in_rx.recv() => {
                    let Some(m) = m else { break Err(anyhow::anyhow!("control stream closed")) };
                    if matches!(m.body.as_ref(), Some(coordinator_message::Body::WorkOffer(_) | coordinator_message::Body::StageAssign(_))) {
                        bandwidth.abort_all();
                    }
                    match m.body {
                        Some(coordinator_message::Body::WorkOffer(offer)) => {
                            let engine = if cfg.stages.as_ref().and_then(|s| s.switch_state()) == Some(EngineState::Parked) {
                                cfg.stage_engine.as_ref().unwrap_or(&engine)
                            } else {
                                &engine
                            };
                            handle_offer(conn, cfg, engine, &shared, &out, &done_tx, &mut lease, draining || output.is_some() || model_departure.is_some() || waiting_for_single, offer).await?;
                            cfg.live.set_leases(lease.attive() as u32);
                        }
                        Some(coordinator_message::Body::LeaseUpdate(u)) => {
                            let Some(l) = lease.map.get_mut(&u.lease_id) else { continue };
                            match u.status {
                                Some(lease_update::Status::RenewedUntil(_)) => {
                                    if let Some(c) = l.confirm.take() {
                                        let _ = c.send(());
                                    }
                                }
                                Some(lease_update::Status::Revoked(r)) => {
                                    warn!(lease = %l.id, reason = r, "lease revoked");
                                    on_revoked(cfg, &mut lease, &u.lease_id, r);
                                }
                                None => {}
                            }
                        }
                        Some(coordinator_message::Body::StageAssign(a)) => match &cfg.stages {
                            Some(s) if cfg.stage_engine.is_some() => {
                                let allow_new = !(cfg.live.is_paused() || cfg.gpu.unavailable() || draining || output.is_some()
                                    || model_departure.is_some() || cfg.live.changes.switching().is_some());
                                match s.queue_switch(a, &cfg.sup, allow_new) {
                                    Ok(()) => cfg.live.set_chain(s.serving_chain()),
                                    Err(status) => out.send(node_message::Body::StageStatus(status)).await?,
                                }
                            }
                            Some(_) if cfg.live.is_paused() || cfg.gpu.unavailable() || cfg.gpu_profile_missing() => {
                                out.send(node_message::Body::StageStatus(StageStatus {
                                    chain_id: a.chain_id,
                                    assign_epoch: a.assign_epoch,
                                    state: StageState::Failed as i32,
                                    error: if cfg.live.is_paused() { "agent paused by host" } else { "selected GPU unavailable" }.into(),
                                    ..Default::default()
                                })).await?;
                            }
                            Some(s) if (model_departure.is_some() || cfg.live.changes.switching().is_some()) && !s.matches_active(&a) => {
                                out.send(node_message::Body::StageStatus(StageStatus {
                                    chain_id: a.chain_id,
                                    assign_epoch: a.assign_epoch,
                                    state: StageState::Stopped as i32,
                                    error: "agent is changing models".into(),
                                    ..Default::default()
                                })).await?;
                            }
                            Some(s) => s.queue_assignment(a),
                            None => {
                                let error = if cfg.model.model_id.is_empty() {
                                    "no verified model weights"
                                } else {
                                    "start the agent with --stage-bin to run stages"
                                };
                                warn!(chain = %a.chain_id, "{error}");
                                out.send(node_message::Body::StageStatus(StageStatus {
                                    chain_id: a.chain_id,
                                    assign_epoch: a.assign_epoch,
                                    state: StageState::Failed as i32,
                                    error: error.into(),
                                    ..Default::default()
                                }))
                                .await?;
                            }
                        },
                        Some(coordinator_message::Body::StageRelease(r)) => {
                            if let Some(s) = &cfg.stages {
                                let closing = s.queue_release(r);
                                if cfg.stage_engine.is_some() && (closing || s.switch_state().is_none()) {
                                    // Even a refused assignment needs a fresh IDLE after release.
                                    waiting_for_single = true;
                                    send_availability(&out, availability_now(EngineState::Recovering, &cfg.live, &cfg.gpu), &mut last_avail).await?;
                                }
                                if closing {
                                    cfg.live.set_chain(None);
                                    send_releases(&out, abort_stage_leases(&mut lease), LeaseReleaseReason::Unspecified).await?;
                                    cfg.live.set_leases(lease.attive() as u32);
                                }
                            }
                        }
                        Some(coordinator_message::Body::MeasurementPing(p)) => {
                            let _ = probe_tx.try_send(node_message::Body::MeasurementPong(MeasurementPong { probe_id: p.probe_id, nonce: p.nonce }));
                        }
                        Some(coordinator_message::Body::PeerProbe(p)) => {
                            if let (Some(bridge), Some(request)) = (&bridge, crate::peer_echo::Request::from_proto(&p, clock_offset)) {
                                if p.receive {
                                    if let Ok(slot) = probe_tx.try_reserve() {
                                        if bridge.echo.prepare(request) {
                                            slot.send(node_message::Body::PeerProbeReady(PeerProbeReady { nonce: p.nonce }));
                                        }
                                    }
                                } else if echo.is_empty() {
                                    let bridge = bridge.clone();
                                    echo.spawn(async move { bridge.probe(request).await });
                                }
                            }
                        }
                        Some(coordinator_message::Body::Drain(d)) => {
                            info!(reason = %d.reason, "coordinator requested shutdown");
                            draining = true;
                        }
                        Some(coordinator_message::Body::Error(e)) => warn!(code = ?e.code(), msg = %e.message, seq = e.in_reply_to_seq, "coordinator error"),
                        Some(coordinator_message::Body::ContributionStatus(st)) => {
                            let new = cfg.contribution.receive_status(st.clone());
                            if new {
                                cfg.live.notify();
                            }
                            if new && ultimo_stato.as_ref() != Some(&st) {
                                info!(
                                    modo = ?st.effective_mode(), model = %st.serving.as_ref().map(|m| m.model_id.as_str()).unwrap_or(""),
                                    in_catena = st.in_chain, posto_gratuito = st.free_seat, reason = ?st.reason(), revision = st.revision,
                                    "contribution status"
                                );
                                ultimo_stato = Some(st);
                            }
                        }
                        Some(coordinator_message::Body::ContributionSummary(r)) => {
                            if ultimo_riepilogo.as_ref() != Some(&r) {
                                info!(
                                    giorni = r.days, chat_gratuite = r.free_chats, valore_gratuito_mc = r.free_value_mc, guadagnati_mc = r.earned_mc,
                                    saldo_mc = r.balance_mc, "contribution summary"
                                );
                                ultimo_riepilogo = Some(r.clone());
                            }
                            cfg.contribution.receive_summary(r);
                            cfg.live.notify();
                        }
                        Some(coordinator_message::Body::ModelDirective(d)) => {
                            let st = rispondi_direttiva(cfg, &shared, d);
                            out.send(node_message::Body::ModelDirectiveStatus(st)).await?;
                        }
                        Some(coordinator_message::Body::LeaveOk(l)) => {
                            info!(interrupted_leases = l.lease_interrotte, "coordinator confirmed departure");
                            leave_ok = true;
                            if output.is_some() {
                                break Ok(SessionEnd::Shutdown);
                            }
                        }
                        Some(other) => debug!(?other, "unhandled message"),
                        None => {}
                    }
                }
                _ = heartbeat.tick() => {
                    out.send(node_message::Body::Heartbeat(heartbeat_msg(&lease, cfg, &shared))).await?;
                }
                _ = rtt_tick.tick(), if cfg.live.network.enabled() && rtt_count < rtt.len() => {
                    rtt[rtt_count] = network::micros(conn.rtt());
                    rtt_count += 1;
                    if rtt_count == rtt.len() {
                        let (median, _) = network::median_and_p95(&mut rtt);
                        cfg.live.network.record_rtt(median, network::unix_seconds());
                        cfg.live.notify();
                    }
                }
                _ = bandwidth_tick.tick(), if cfg.live.network.enabled() => {
                    if idle && bandwidth.is_empty() && cfg.live.network.due(network::unix_seconds())
                        && !cfg.stages.as_ref().is_some_and(|s| s.network_busy())
                    {
                        bandwidth.spawn(network::measure());
                    }
                }
                Some(result) = bandwidth.join_next() => {
                    if let Ok(result) = result {
                        cfg.live.network.record_bandwidth(result, network::unix_seconds());
                        cfg.live.notify();
                    }
                }
                Some(result) = echo.join_next() => {
                    if let Ok(result) = result {
                        let _ = probe_tx.try_send(node_message::Body::PeerProbeResult(result));
                    }
                }
                _ = pref_rx.changed() => {
                    let p = cfg.contribution.to_proto(&modelli_di(&ultimo_inventario));
                    info!(modo = ?p.mode(), revision = p.revision, "contribution preference changed");
                    out.send(node_message::Body::Contribution(p)).await?;
                }
                r = uscita_rx.changed(), if !uscita_chiusa && output.is_none() => {
                    match r {
                        Err(_) => uscita_chiusa = true,
                        Ok(()) if !*uscita_rx.borrow() => {}
                        Ok(()) => {
                            let role = cfg.stages.as_ref().and_then(|s| s.active_role());
                            let has_chain = cfg.stages.as_ref().is_some_and(|s| s.has_chain());
                            let notice = departure_notice(role, has_chain, lease.attive());
                            warn!(max_seconds = GRAZIA_USCITA_S, "{notice}");
                            output = Some(Instant::now());
                            conto.reset();
                            if leave_ok {
                                break Ok(SessionEnd::Shutdown);
                            }
                            if model_departure.is_none() {
                                out.send(node_message::Body::Leaving(Leaving { grazia_s: GRAZIA_USCITA_S })).await?;
                            }
                        }
                    }
                }
                _ = conto.tick(), if output.is_some() => {
                    let trascorsi = output.map(|t| t.elapsed()).unwrap_or_default();
                    let rimasti = ATTESA_USCITA.saturating_sub(trascorsi).as_secs().min(u64::from(GRAZIA_USCITA_S));
                    let role = cfg.stages.as_ref().and_then(|s| s.active_role());
                    let has_chain = cfg.stages.as_ref().is_some_and(|s| s.has_chain());
                    let notice = departure_notice(role, has_chain, lease.attive());
                    info!(seconds = rimasti, "{notice} ({rimasti} s left)");
                }
                _ = tokio::time::sleep_until(scadenza_uscita), if output.is_some() => {
                    warn!("departure acknowledgment timed out; exiting");
                    break Ok(SessionEnd::Shutdown);
                }
                _ = state_rx.changed() => {}
                _ = pause_rx.changed() => {}
                _ = model_rx.changed() => {}
                _ = async {
                    if let Some(deadline) = model_departure {
                        tokio::time::sleep_until(deadline).await;
                    } else {
                        std::future::pending::<()>().await;
                    }
                }, if cfg.live.changes.switching().is_none() && output.is_none() => {}
                _ = gpu_rx.changed() => {}
                Some(st) = stage_rx.recv(), if cfg.stages.is_some() => {
                    let terminal = matches!(st.state(), StageState::Stopped | StageState::Failed);
                    let current = cfg.stage_engine.is_none()
                        || cfg.stages.as_ref().is_some_and(|s| s.stage_status_current(&st));
                    if current {
                        cfg.live.set_chain((!terminal).then(|| st.chain_id.clone()));
                        cfg.live.notify();
                    }
                    pending_stage_status = Some(st.clone());
                    out.send(node_message::Body::StageStatus(st)).await?;
                    pending_stage_status = None;
                    if current && terminal {
                        if cfg.stage_engine.is_some() && !waiting_for_single {
                            waiting_for_single = true;
                            send_availability(&out, availability_now(EngineState::Recovering, &cfg.live, &cfg.gpu), &mut last_avail).await?;
                        }
                        send_releases(&out, abort_stage_leases(&mut lease), LeaseReleaseReason::Unspecified).await?;
                        cfg.live.set_leases(lease.attive() as u32);
                    }
                }
                Some(done) = done_rx.recv() => {
                    on_lease_done(cfg, &shared, &out, &mut last_avail, &mut lease, done, waiting_for_single).await?;
                    cfg.live.set_leases(lease.attive() as u32);
                }
                _ = shutdown.changed() => {
                    break Ok(SessionEnd::Shutdown);
                }
            }
        }
    };
    let result = {
        tokio::pin!(run_control);
        loop {
            tokio::select! {
                biased;
                change = cfg.live.changes.wait_deadline() => {
                    if cfg.live.changes.claim(&change).await {
                        warn!("model switch deadline reached; releasing active work");
                        break Ok(SessionEnd::SwitchModel(change));
                    }
                }
                result = &mut run_control => break result,
            }
        }
    };
    // Stop network work on every exit path, including the model switch deadline.
    drop(bandwidth);
    drop(echo);
    drop(probe_tx);
    if let Some(bridge) = bridge {
        bridge.echo.clear();
    }
    let graceful = result.is_ok();
    let released = if graceful {
        abort_leases(&mut lease, LeaseOutcome::Shutdown)
    } else {
        // Dropping the guard on disconnect marks work as abandoned.
        for l in lease.map.values() {
            l.task.abort();
        }
        Vec::new()
    };
    drop(lease);
    cfg.live.set_leases(0);
    cfg.live.set_chain(None);

    let confirm_departure = async {
        if !matches!(result, Ok(SessionEnd::SwitchModel(_))) || leave_ok {
            return CLOSE_IO_TIMEOUT;
        }
        let start = Instant::now();
        let ack = async {
            if model_departure.is_none() && output.is_none() {
                out.send(node_message::Body::Leaving(Leaving { grazia_s: 0 })).await?;
            }
            while let Some(message) = in_rx.recv().await {
                if matches!(message.body, Some(coordinator_message::Body::LeaveOk(_))) {
                    break;
                }
            }
            Ok::<_, anyhow::Error>(())
        };
        let _ = tokio::time::timeout(CLOSE_IO_TIMEOUT, ack).await;
        CLOSE_IO_TIMEOUT.saturating_sub(start.elapsed())
    };
    // Collect the terminal reports before Goodbye, which ends the coordinator's
    // control reader. Drain the queue while stopping so the worker can report.
    let stop_stages = async {
        let mut updates: Vec<_> = pending_stage_status.into_iter().collect();
        if let Some(s) = &cfg.stages {
            let stop = s.stop_all();
            tokio::pin!(stop);
            loop {
                tokio::select! {
                    _ = &mut stop => break,
                    Some(st) = stage_rx.recv() => updates.push(st),
                }
            }
            while let Ok(st) = stage_rx.try_recv() {
                updates.push(st);
            }
        }
        updates
    };
    let (close_budget, stage_updates) = tokio::join!(confirm_departure, stop_stages);
    cfg.live.set_connected(false);
    let close_stream = async {
        send_releases(&out, released, LeaseReleaseReason::Shutdown).await?;
        for st in stage_updates {
            out.send(node_message::Body::StageStatus(st)).await?;
        }
        if graceful && !cfg.stages.as_ref().is_some_and(|s| s.has_chain()) {
            out.send(goodbye(GoodbyeReason::HostStopped)).await?;
        }
        drop(out);
        // Wait for the transport to acknowledge the queued frames before closing.
        writer.await?;
        Ok::<_, anyhow::Error>(())
    };
    if !matches!(tokio::time::timeout(close_budget, close_stream).await, Ok(Ok(()))) {
        debug!("control stream closed before departure messages were flushed");
    }
    conn.close(0u32.into(), b"");
    result
}

struct Out(mpsc::Sender<node_message::Body>);

impl Out {
    async fn send(&self, b: node_message::Body) -> Result<()> {
        self.0.send(b).await.context("writer closed")
    }
}

fn goodbye(r: GoodbyeReason) -> node_message::Body {
    node_message::Body::Goodbye(publicompute_proto::node::Goodbye { reason: r as i32 })
}

fn gpu_loss_leases(generation: u64, seen: &mut u64, lease: &mut Leases, live: &crate::control::Live) -> Vec<String> {
    if generation == *seen {
        return Vec::new();
    }
    *seen = generation;
    let released = abort_leases(lease, LeaseOutcome::Shutdown);
    live.set_leases(0);
    released
}

fn abort_leases(lease: &mut Leases, outcome: LeaseOutcome) -> Vec<String> {
    let mut active = std::mem::take(&mut lease.map);
    for l in active.values() {
        l.task.abort();
    }
    for l in active.values_mut() {
        if let Some(g) = l.guard.take() {
            g.finish(outcome);
        }
    }
    lease.chiudi_comune(outcome);
    active.into_keys().collect()
}

fn abort_stage_leases(lease: &mut Leases) -> Vec<String> {
    let mut released = Vec::new();
    lease.map.retain(|id, active| {
        if active.stage {
            active.task.abort();
            released.push(id.clone());
        }
        !active.stage
    });
    released
}

async fn send_releases(out: &Out, released: Vec<String>, reason: LeaseReleaseReason) -> Result<()> {
    for lease_id in released {
        out.send(node_message::Body::LeaseRelease(LeaseRelease { lease_id, reason: reason as i32 })).await?;
    }
    Ok(())
}

/// Provide the advertised weights fingerprint when encoding a contribution preference.
fn modelli_di(inv: &ModelInventory) -> Vec<ModelRef> {
    inv.models.iter().filter(|m| m.status() == ModelStatus::Ready).filter_map(|m| m.model.clone()).collect()
}

/// A directive can confirm the weights the engine is already serving.
/// Reject an unsupported selection or a non-increasing epoch.
pub fn rispondi_direttiva(cfg: &NodeConfig, shared: &Shared, d: ModelDirective) -> ModelDirectiveStatus {
    use publicompute_proto::node::{model_directive_status::State, ModelRole};
    use std::sync::atomic::Ordering;
    let base = ModelDirectiveStatus { epoch: d.epoch, model: d.model.clone(), ..Default::default() };
    if d.epoch <= shared.epoca_direttiva.load(Ordering::SeqCst) {
        info!(epoch = d.epoch, "rejected model directive with a stale epoch");
        return ModelDirectiveStatus { state: State::Rejected as i32, reject: RejectReason::StaleEpoch as i32, ..base };
    }
    shared.epoca_direttiva.fetch_max(d.epoch, Ordering::SeqCst);
    let stesso = d.model.as_ref().is_some_and(|m| {
        m.model_id == cfg.model.model_id
            && m.quantization == cfg.model.quantization
            && (m.weights_sha256.is_empty() || m.weights_sha256 == cfg.model.weights_sha256)
    });
    let ruolo_singolo = matches!(d.role(), ModelRole::Unspecified | ModelRole::Single);
    if stesso && ruolo_singolo {
        let reject = if cfg.sup.state() == EngineState::Parked {
            Some(RejectReason::InChain)
        } else if !single_loaded(cfg) || matches!(&*cfg.sup.engine, Engine::Stages(_)) {
            Some(RejectReason::Busy)
        } else {
            None
        };
        if let Some(reject) = reject {
            return ModelDirectiveStatus { state: State::Rejected as i32, reject: reject as i32, ..base };
        }
        info!(epoch = d.epoch, reason = %d.reason, "requested model is already loaded");
        ModelDirectiveStatus { state: State::Serving as i32, ..base }
    } else {
        info!(epoch = d.epoch, reason = %d.reason, "model directive requires an engine switch");
        ModelDirectiveStatus {
            state: State::Rejected as i32,
            reject: RejectReason::Unsupported as i32,
            error: "this agent cannot switch models while running".into(),
            ..base
        }
    }
}

/// Reconnecting cannot change the release or wire protocol in use.
#[derive(Debug)]
pub struct IncompatibleRelease(String);

impl IncompatibleRelease {
    pub fn from_protocol(error: &publicompute_proto::common::Error) -> Option<Self> {
        use publicompute_proto::common::ErrorCode;
        matches!(error.code(), ErrorCode::AgentTooOld | ErrorCode::UnsupportedProtocol)
            .then(|| Self(error.message.clone()))
    }
}

impl std::fmt::Display for IncompatibleRelease {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for IncompatibleRelease {}

/// Report busy engines as IDLE because the coordinator already tracks lease
/// occupancy. Availability reflects the engine's ability to serve.
fn availability_of(s: EngineState) -> Availability {
    match s {
        // Settling delays new work without making a temporary cancellation look like a fault.
        EngineState::Ready | EngineState::Busy | EngineState::Settling | EngineState::Parked => Availability::Idle,
        EngineState::Starting | EngineState::Recovering => Availability::EngineRecovering,
        EngineState::Quarantined => Availability::EngineQuarantined,
    }
}

/// A local pause overrides engine availability so the scheduler stops offering work.
fn availability_now(state: EngineState, live: &crate::control::Live, gpu: &Telemetria) -> Availability {
    if live.is_paused() || gpu.unavailable() {
        Availability::PausedByHost
    } else {
        availability_of(state)
    }
}

fn session_availability(cfg: &NodeConfig, state: EngineState, waiting_for_single: bool) -> Availability {
    match availability_now(state, &cfg.live, &cfg.gpu) {
        Availability::Idle if waiting_for_single && state != EngineState::Ready => Availability::EngineRecovering,
        availability => availability,
    }
}

#[cfg(test)]
async fn sync_availability(out: &Out, sup: &Supervisor, live: &crate::control::Live, gpu: &Telemetria, last: &mut Availability) -> Result<()> {
    send_availability(out, availability_now(sup.state(), live, gpu), last).await
}

async fn send_availability(out: &Out, a: Availability, last: &mut Availability) -> Result<()> {
    if a != *last {
        info!(availability = ?a, "availability changed");
        *last = a;
        out.send(node_message::Body::Availability(AvailabilityUpdate { availability: a as i32 })).await?;
    }
    Ok(())
}

/// Send actual engine progress with each heartbeat. The heartbeat itself
/// does not advance the sequence counter.
fn heartbeat_msg(lease: &Leases, cfg: &NodeConfig, shared: &Shared) -> Heartbeat {
    // Exclude completed leases from renewal even if their work streams are still closing.
    let mut ids: Vec<&String> = lease.map.keys().collect();
    ids.sort();
    let leases = ids
        .into_iter()
        .filter_map(|id| {
            let l = &lease.map[id];
            let p = l.progress.lock().unwrap();
            (!p.completing).then(|| lease_progress(&l.id, &p, l.own.as_deref().unwrap_or(&cfg.sup.progress).snapshot()))
        })
        .collect();
    let tps = shared.tokens_per_sec.lock().unwrap().unwrap_or(0.0) as f32;
    Heartbeat {
        node_time: Some(Timestamp::from(std::time::SystemTime::now())),
        telemetry: Some(Telemetry { gpus: cfg.gpu.gpus(), recent_tokens_per_sec: tps, engine: Some(cfg.sup.health()) }),
        leases,
        network: cfg.live.network.report(),
    }
}

fn lease_progress(id: &str, p: &LeaseCounters, snap: crate::engine::progress::Snapshot) -> LeaseProgress {
    LeaseProgress {
        lease_id: id.to_string(),
        requests_completed: p.requests,
        tokens_generated: p.tokens,
        phase: snap.phase.to_proto() as i32,
        progress_seq: snap.seq,
        prompt_tokens_processed: snap.prompt_tokens_processed,
    }
}

/// Recover the engine on a no-progress revocation. Other revocations can
/// leave work running, so let the abandoned request settle before accepting
/// another lease.
fn on_revoked(cfg: &NodeConfig, leases: &mut Leases, id: &str, reason: i32) {
    let Some(mut l) = leases.map.remove(id) else { return };
    // Aborting a stage task drops the cancellation sender. The head receives
    // a cancel for that request and can release its slot independently.
    l.task.abort();
    if l.stage {
        return;
    }
    let outcome = if reason == LeaseRevokeReason::NoProgress as i32 {
        LeaseOutcome::Fault(crate::engine::FaultCause::NoProgress)
    } else {
        if reason == LeaseRevokeReason::Cancelled as i32
            || reason == LeaseRevokeReason::Reassigned as i32
            || reason == LeaseRevokeReason::TrustDowngraded as i32
        {
            cfg.sup.stats.request_cancelled();
        }
        LeaseOutcome::Abandoned
    };
    if let Some(g) = l.guard.take() {
        g.finish(outcome);
    } else if leases.comune.is_some() && (leases.is_empty() || matches!(outcome, LeaseOutcome::Fault(_))) {
        // Release a shared reservation on the last lease or immediately after an engine fault.
        leases.chiudi_comune(outcome);
    }
}

/// On an engine fault publish recovering availability before releasing the
/// lease. The coordinator must see that the engine is unavailable before
/// it can schedule replacement work.
async fn on_lease_done(
    cfg: &NodeConfig,
    shared: &Shared,
    out: &Out,
    last_avail: &mut Availability,
    lease: &mut Leases,
    done: LeaseDone,
    waiting_for_single: bool,
) -> Result<()> {
    // A queued result can arrive after cancellation removed its lease.
    let Some(mut own) = lease.map.remove(&done.id) else { return Ok(()) };
    let suspicious = own.progress.lock().unwrap().unclassified_failure;
    let mut guard = own.guard.take();
    // Keep the shared guard until the final lease finishes, unless an engine fault requires immediate recovery.
    let comune = !own.stage && guard.is_none() && lease.comune.is_some();
    if comune {
        lease.sospetto |= suspicious;
        if lease.is_empty() || done.result.as_ref().is_err_and(|error| matches!(classify(error), ErrorClass::Fault(_))) {
            guard = lease.comune.take();
        }
    }
    let suspicious = suspicious || (comune && lease.sospetto);
    match done.result {
        Ok((tokens, gen_time)) => {
            if let Some(g) = guard {
                g.finish(if suspicious { LeaseOutcome::RequestFailed } else { LeaseOutcome::Clean });
            }
            if tokens > 0 && !gen_time.is_zero() {
                let tps = tokens as f64 / gen_time.as_secs_f64();
                *shared.tokens_per_sec.lock().unwrap() = Some(tps);
                info!(lease = %done.id, tokens, tokens_per_sec = format!("{tps:.1}"), "lease completata");
            }
        }
        Err(e) => {
            let class = classify(&e);
            if let Some(chat_id) = &done.chat_id {
                let code = match class {
                    ErrorClass::Fault(cause) => cause.as_str(),
                    ErrorClass::Request(code) => code.as_str_name(),
                    _ if e.is::<OutputStalled>() => "output_stalled",
                    _ => "work_stream_error",
                };
                info!(chat = %chat_id, lease = %done.id, code, "releasing failed chat lease");
            } else {
                warn!(lease = %done.id, err = format!("{e:#}"), "releasing failed lease");
            }
            let outcome = match class {
                ErrorClass::Fault(c) => LeaseOutcome::Fault(c),
                ErrorClass::Request(_) => LeaseOutcome::Clean,
                ErrorClass::Other => LeaseOutcome::Abandoned,
            };
            if let Some(g) = guard {
                g.finish(outcome);
            }
            if !own.stage {
                let availability = session_availability(cfg, cfg.sup.state(), waiting_for_single);
                send_availability(out, availability, last_avail).await?;
            }
            let reason = if e.is::<OutputStalled>() || e.is::<framed::ChatStreamError>() {
                LeaseReleaseReason::Unspecified
            } else {
                LeaseReleaseReason::EngineFailure
            };
            out.send(node_message::Body::LeaseRelease(LeaseRelease { lease_id: done.id, reason: reason as i32 })).await?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn handle_offer(
    conn: &quinn::Connection,
    cfg: &NodeConfig,
    engine: &Arc<Engine>,
    shared: &Arc<Shared>,
    out: &Out,
    done_tx: &mpsc::Sender<LeaseDone>,
    lease: &mut Leases,
    draining: bool,
    offer: WorkOffer,
) -> Result<()> {
    let lease_id = offer.lease.as_ref().map(|l| l.lease_id.clone()).unwrap_or_default();
    let reject = |reason: LeaseRejectReason, detail: &str| {
        info!(lease = %lease_id, ?reason, detail, "lease offer rejected");
        node_message::Body::LeaseReject(LeaseReject { lease_id: lease_id.clone(), reason: reason as i32, detail: detail.to_string() })
    };
    if cfg.live.changes.switching().is_some() {
        return out.send(reject(LeaseRejectReason::HostBusy, "agent is changing models")).await;
    }
    if cfg.live.is_paused() || cfg.gpu.unavailable() || cfg.gpu_profile_missing() {
        return out.send(reject(LeaseRejectReason::HostBusy, "host unavailable")).await;
    }
    let stage = matches!(&**engine, Engine::Stages(_));
    let capacity = capacita(cfg, engine);
    let attive = lease.attive();
    if attive >= capacity || draining {
        return out.send(reject(LeaseRejectReason::HostBusy, "lease already active")).await;
    }
    // A completed stream can remain open after its engine reservation becomes unnecessary.
    if attive == 0 {
        for l in lease.map.values_mut() {
            if let Some(g) = l.guard.take() {
                g.finish(LeaseOutcome::Clean);
            }
        }
        lease.chiudi_comune(LeaseOutcome::Clean);
    }
    // An active group of chats already owns the guard. Only a lease that
    // needs a new reservation checks the supervisor's idle availability.
    let condiviso = attive > 0 && lease.comune.is_some();
    if !stage && !condiviso {
        if let Err(r) = cfg.sup.check_ready() {
            // A parked or recovering engine still has the GGUF on disk.
            let reason = match r.reason {
                LeaseRejectReason::ModelNotReady => LeaseRejectReason::HostBusy,
                reason => reason,
            };
            return out.send(reject(reason, r.detail)).await;
        }
    }
    if let Engine::Stages(e) = &**engine {
        if !e.pronto() {
            return out.send(reject(LeaseRejectReason::HostBusy, "chain not ready; no head in service")).await;
        }
    }
    if offer.model.as_ref() != Some(&cfg.model) {
        return out.send(reject(LeaseRejectReason::ModelNotReady, "requested model does not match the loaded model")).await;
    }
    let logic = offer.logic.unwrap_or_default();
    if !logic::is_supported(logic.format_version) {
        return out.send(reject(LeaseRejectReason::HostBusy, "unsupported LOGIC format")).await;
    }
    match (&offer.task, &**engine) {
        (Some(work_offer::Task::Chat(_)), _) => {}
        (Some(work_offer::Task::Batch(_) | work_offer::Task::Verify(_)), Engine::Stages(_)) => {
            return out.send(reject(LeaseRejectReason::HostBusy, "a stage node can only serve chat for its own chain")).await;
        }
        (Some(work_offer::Task::Batch(_) | work_offer::Task::Verify(_)), _) => {}
        _ => return out.send(reject(LeaseRejectReason::HostBusy, "unsupported task")).await,
    }
    let tps = *shared.tokens_per_sec.lock().unwrap();
    if !economics::is_profitable(offer.payout.as_ref(), cfg.costs.as_ref(), tps) {
        return out.send(reject(LeaseRejectReason::Unprofitable, "payment does not cover electricity")).await;
    }
    // Validate the offer before reserving the engine so earlier exits need no guard cleanup.
    let guard = if stage || condiviso {
        None
    } else {
        match cfg.sup.try_begin() {
            Ok(g) => Some(g),
            Err(r) => return out.send(reject(LeaseRejectReason::HostBusy, r.detail)).await,
        }
    };

    out.send(node_message::Body::LeaseAccept(LeaseAccept { lease_id: lease_id.clone() })).await?;
    let progress = Arc::new(Mutex::new(LeaseCounters::default()));
    // Give concurrent leases separate counters so a healthy chat cannot mask a stalled one.
    let own = (condivide(cfg, engine) && matches!(offer.task, Some(work_offer::Task::Chat(_)))).then(|| Arc::new(Progress::default()));
    let (confirm_tx, confirm_rx) = oneshot::channel();
    // Send a heartbeat immediately so the renewal confirms the acceptance before work begins.
    out.send(node_message::Body::Heartbeat(Heartbeat {
        node_time: Some(Timestamp::from(std::time::SystemTime::now())),
        telemetry: None,
        network: None,
        leases: vec![lease_progress(&lease_id, &LeaseCounters::default(), own.as_deref().unwrap_or(&cfg.sup.progress).snapshot())],
    }))
    .await?;

    let (conn2, engine2, sup2, progress2, done2, id2, own2) =
        (conn.clone(), engine.clone(), cfg.sup.clone(), progress.clone(), done_tx.clone(), lease_id.clone(), own.clone());
    let chat_id = match &offer.task {
        Some(work_offer::Task::Chat(chat)) => Some(chat.chat_id.clone()),
        _ => None,
    };
    let task = tokio::spawn(async move {
        let result = run_lease(conn2, &id2, offer, engine2, sup2, confirm_rx, progress2, own2).await;
        let _ = done2.send(LeaseDone { id: id2, chat_id, result }).await;
    });
    let (guard_propria, comune) = if condivide(cfg, engine) { (None, guard) } else { (guard, None) };
    if comune.is_some() {
        lease.comune = comune;
        lease.sospetto = false;
    }
    lease.map.insert(lease_id.clone(), ActiveLease { id: lease_id, stage, progress, own, confirm: Some(confirm_tx), task, guard: guard_propria });
    Ok(())
}

async fn run_lease(
    conn: quinn::Connection,
    lease_id: &str,
    offer: WorkOffer,
    engine: Arc<Engine>,
    sup: Arc<Supervisor>,
    confirmed: oneshot::Receiver<()>,
    progress: Arc<Mutex<LeaseCounters>>,
    own: Option<Arc<Progress>>,
) -> Result<(u64, Duration)> {
    let logic_params = offer.logic.unwrap_or_default();
    tokio::time::timeout(Duration::from_secs(30), confirmed)
        .await
        .context("lease confirmation timed out")?
        .context("lease cancelled before confirmation")?;

    let (mut send, recv) = conn.open_bi().await?;
    let mut r = FrameReader::new(recv);
    let open = WorkStreamNodeMessage { body: Some(work_stream_node_message::Body::Open(WorkStreamOpen { lease_id: lease_id.to_string() })) };
    let task = match offer.task {
        Some(work_offer::Task::Chat(chat)) => {
            let mut send = framed::ChatWriter::new(send);
            send.write(&open).await?;
            let result = run_chat(&mut send, &mut r, &engine, &sup, &progress, own.as_ref(), lease_id, chat, &logic_params).await?;
            progress.lock().unwrap().completing = true;
            send.write(&lease_complete(lease_id)).await?;
            send.finish()?;
            let _ = tokio::time::timeout(Duration::from_secs(5), r.read::<WorkStreamCoordinatorMessage>()).await;
            return Ok(result);
        }
        task => task,
    };
    framed::write(&mut send, &open).await?;
    let batch = match task {
        Some(work_offer::Task::Batch(b)) => b,
        Some(work_offer::Task::Verify(v)) => {
            run_verify(&mut send, &mut r, &engine, &sup, v).await?;
            progress.lock().unwrap().completing = true;
            sup.progress.set_phase(Phase::Finishing);
            finish_lease(&mut send, &mut r, lease_id).await?;
            return Ok((0, Duration::ZERO));
        }
        _ => bail!("unsupported task"),
    };

    let (mut total_tokens, mut total_gen) = (0u64, Duration::ZERO);
    for req in &batch.requests {
        sup.progress.set_phase(Phase::BetweenRequests);
        let t0 = Instant::now();
        let outcome = match engine.generate(req, logic_params.top_k).await {
            Ok(g) => Ok(g),
            Err(e) => match classify(&e) {
                ErrorClass::Fault(c) => {
                    sup.stats.request_failed(c.as_str());
                    return Err(e.context(format!("request {}", req.request_id)));
                }
                // A request rejection leaves the engine eligible for later work.
                ErrorClass::Request(code) => Err((code, format!("{e:#}"))),
                ErrorClass::Other => {
                    progress.lock().unwrap().unclassified_failure = true;
                    Err((RequestFailureCode::EngineError, format!("{e:#}")))
                }
            },
        };
        sup.progress.set_phase(Phase::Finishing);
        let (result, n, gen_time) = match outcome {
            Ok(g) => {
                let result = completion_result(&req.request_id, g, &logic_params)?;
                sup.stats.request_completed(t0.elapsed());
                result
            }
            Err((code, detail)) => {
                warn!(request = %req.request_id, ?code, %detail, "request failed");
                sup.stats.request_failed(&format!("{code:?}"));
                let result = RequestResult {
                    request_id: req.request_id.clone(),
                    outcome: Some(request_result::Outcome::Failure(RequestFailure { code: code as i32, detail })),
                };
                (result, 0, Duration::ZERO)
            }
        };
        framed::write(&mut send, &WorkStreamNodeMessage { body: Some(work_stream_node_message::Body::RequestResult(result)) }).await?;
        loop {
            let m: WorkStreamCoordinatorMessage = r.read().await?.context("coordinator closed the work stream")?;
            match m.body {
                Some(work_stream_coordinator_message::Body::ResultAck(a)) => {
                    if !a.accepted {
                        warn!(request = %a.request_id, reason = %a.reason, "result rejected");
                    }
                    break;
                }
                Some(work_stream_coordinator_message::Body::CancelRequest(_)) | None => {}
            }
        }
        // Count delivery of a completed request as useful progress.
        sup.progress.tick();
        total_tokens += u64::from(n);
        total_gen += gen_time;
        let mut p = progress.lock().unwrap();
        p.requests += 1;
        p.tokens += u64::from(n);
    }
    progress.lock().unwrap().completing = true;
    sup.progress.set_phase(Phase::Finishing);
    finish_lease(&mut send, &mut r, lease_id).await?;
    Ok((total_tokens, total_gen))
}

/// Build a completed result with its token commitment, output IDs and generation time.
fn completion_result(request_id: &str, g: crate::engine::Generation, logic_params: &publicompute_proto::node::LogicParams) -> Result<(RequestResult, u32, Duration)> {
    let n = g.token_ids.len() as u32;
    let commitment = logic::encode(logic_params.format_version, logic_params.top_k, &g.probs, &g.token_ids)?;
    let gen_time = g.generation;
    let result = RequestResult {
        request_id: request_id.to_string(),
        outcome: Some(request_result::Outcome::Completion(Completion {
            text: g.text,
            token_ids: g.token_ids,
            finish_reason: g.finish_reason as i32,
            usage: Some(Usage { prompt_tokens: g.prompt_tokens, completion_tokens: n, cached_prompt_tokens: g.cached_prompt_tokens }),
            slot_report: g.slot_report,
            logic: Some(LogicCommitment {
                format_version: logic_params.format_version,
                data: commitment,
                verification_incomplete: g.verification_incomplete,
            }),
            prompt_tokens_sha256: g.prompt_tokens_sha256.to_vec(),
            timing: Some(Timing { prefill: prost_types::Duration::try_from(g.prefill).ok(), generation: prost_types::Duration::try_from(g.generation).ok() }),
        })),
    };
    Ok((result, n, gen_time))
}

/// Run one streamed chat per lease. Forward text deltas as they arrive,
/// propagate cancellation from the coordinator and send the final request result.
#[allow(clippy::too_many_arguments)]
async fn run_chat(
    send: &mut framed::ChatWriter,
    r: &mut FrameReader,
    engine: &Arc<Engine>,
    sup: &Supervisor,
    progress: &Arc<Mutex<LeaseCounters>>,
    own: Option<&Arc<Progress>>,
    lease_id: &str,
    chat: publicompute_proto::node::ChatTask,
    logic_params: &publicompute_proto::node::LogicParams,
) -> Result<(u64, Duration)> {
    let req = chat.request.context("chat is missing its request")?;
    let t0 = Instant::now();
    let (dtx, mut drx, mut saturated) = delta_channel();
    let (cancel_tx, cancel_rx) = oneshot::channel::<()>();
    let mut cancel_tx = Some(cancel_tx);
    let lp: Arc<Progress> = own.cloned().unwrap_or_else(|| sup.progress.clone());
    let (e2, lp2) = (engine.clone(), lp.clone());
    let request_id = req.request_id.clone();
    let top_k = logic_params.top_k;
    let chat_id = chat.chat_id;
    let trace_id = chat_id.clone();
    let gen = tokio::spawn(async move {
        let context = crate::engine::ChatContext { id: &trace_id, owner: &chat.affinity_owner, member_seat: chat.member_seat, member_key: &chat.member_key };
        e2.chat_lease(&req, context, top_k, &lp2, Some(dtx), Some(cancel_rx)).await
    });
    let (n, gen_time) = {
        let write_reply = async {
            let mut first = true;
            while let Some(queued) = drx.recv().await {
                let delta = queued.item;
                if let Some(slot) = delta.slot_taken {
                    send.write(&WorkStreamNodeMessage { body: Some(work_stream_node_message::Body::SlotTaken(slot)) }).await?;
                    continue;
                }
                send.write(&WorkStreamNodeMessage {
                    body: Some(work_stream_node_message::Body::TokenDelta(TokenDelta {
                        request_id: request_id.clone(), index: delta.index, text: delta.text, token_ids: delta.token_ids,
                    })),
                }).await?;
                if first {
                    first = false;
                    info!(chat = %chat_id, lease = lease_id, request = %request_id, elapsed_ms = t0.elapsed().as_millis(), "first delta forwarded");
                }
            }
            if !matches!(&**engine, Engine::Stages(_)) {
                sup.progress.set_phase(Phase::Finishing);
            }
            lp.set_phase(Phase::Finishing);
            let generation = gen.await.context("chat task")?;
            let (result, count, elapsed) = chat_result(generation, &request_id, logic_params, sup, progress, t0)?;
            let code = match &result.outcome {
                Some(request_result::Outcome::Failure(failure)) => failure.code().as_str_name(),
                _ => "ok",
            };
            send.write(&WorkStreamNodeMessage { body: Some(work_stream_node_message::Body::RequestResult(result)) }).await?;
            info!(chat = %chat_id, lease = lease_id, request = %request_id, code, tokens = count, elapsed_ms = t0.elapsed().as_millis(), "chat result forwarded");
            Ok::<_, anyhow::Error>((count, elapsed))
        };
        let read_commands = async {
            loop {
                let message = read_chat_command(r).await?;
                if let Some(work_stream_coordinator_message::Body::CancelRequest(cancel)) = message.body {
                    if cancel.request_id == request_id {
                        info!(chat = %chat_id, lease = lease_id, request = %cancel.request_id, "coordinator cancelled the chat");
                        if let Some(sender) = cancel_tx.take() {
                            let _ = sender.send(());
                        }
                    }
                }
            }
        };
        tokio::select! {
            biased;
            _ = wait_for_saturation(&mut saturated) => Err(OutputStalled.into()),
            result = write_reply => result,
            result = read_commands => result,
        }
    }?;
    loop {
        let m = read_chat_command(r).await?;
        if let Some(work_stream_coordinator_message::Body::ResultAck(a)) = m.body {
            if !a.accepted {
                info!(chat = %chat_id, lease = lease_id, request = %a.request_id, "chat result rejected");
            }
            break;
        }
    }
    if !matches!(&**engine, Engine::Stages(_)) {
        sup.progress.tick();
    }
    if !Arc::ptr_eq(&lp, &sup.progress) {
        lp.tick();
    }
    {
        let mut p = progress.lock().unwrap();
        p.requests += 1;
        p.tokens += u64::from(n);
    }
    Ok((u64::from(n), gen_time))
}

async fn read_chat_command(reader: &mut FrameReader) -> Result<WorkStreamCoordinatorMessage> {
    reader.read().await.and_then(|message| message.context("coordinator closed the work stream")).context(framed::ChatStreamError)
}

fn chat_result(
    generation: Result<crate::engine::Generation>,
    request_id: &str,
    logic_params: &publicompute_proto::node::LogicParams,
    sup: &Supervisor,
    progress: &Mutex<LeaseCounters>,
    started: Instant,
) -> Result<(RequestResult, u32, Duration)> {
    Ok(match generation {
        Ok(g) => {
            sup.stats.request_completed(started.elapsed());
            completion_result(request_id, g, logic_params)?
        }
        Err(e) if e.is::<OutputStalled>() => return Err(e),
        Err(e) => match classify(&e) {
            ErrorClass::Fault(c) => {
                sup.stats.request_failed(c.as_str());
                return Err(e.context(format!("request {request_id}")));
            }
            ErrorClass::Request(code) => {
                sup.stats.request_failed(&format!("{code:?}"));
                (RequestResult { request_id: request_id.to_string(), outcome: Some(request_result::Outcome::Failure(RequestFailure { code: code as i32, detail: String::new() })) }, 0, Duration::ZERO)
            }
            ErrorClass::Other => {
                progress.lock().unwrap().unclassified_failure = true;
                (
                    RequestResult {
                        request_id: request_id.to_string(),
                        outcome: Some(request_result::Outcome::Failure(RequestFailure { code: RequestFailureCode::EngineError as i32, detail: String::new() })),
                    },
                    0,
                    Duration::ZERO,
                )
            }
        },
    })
}

fn lease_complete(lease_id: &str) -> WorkStreamNodeMessage {
    WorkStreamNodeMessage { body: Some(work_stream_node_message::Body::LeaseComplete(LeaseComplete { lease_id: lease_id.to_string() })) }
}

async fn finish_lease(send: &mut quinn::SendStream, r: &mut FrameReader, lease_id: &str) -> Result<()> {
    framed::write(send, &lease_complete(lease_id)).await?;
    send.finish()?;
    // Keep the work stream until the coordinator closes it after recording completion.
    let _ = tokio::time::timeout(Duration::from_secs(5), r.read::<WorkStreamCoordinatorMessage>()).await;
    Ok(())
}

/// Include the verifier's prompt token count so the coordinator can check the claimed usage.
fn verify_result(verify_id: &str, res: crate::engine::Verification) -> VerifyResult {
    VerifyResult {
        verify_id: verify_id.to_string(),
        positions: res
            .positions
            .into_iter()
            .map(|p| PositionLogprobs {
                position: p.position,
                top: p.top.into_iter().map(|(token_id, logprob)| TokenLogprob { token_id, logprob }).collect(),
                chosen_logprob: p.chosen_logprob.unwrap_or(f32::NEG_INFINITY),
            })
            .collect(),
        prompt_tokens_sha256: res.prompt_tokens_sha256.to_vec(),
        prompt_tokens: res.prompt_tokens,
    }
}

/// Recompute the requested positions from the supplied output without
/// identifying the node that produced it.
async fn run_verify(send: &mut quinn::SendStream, r: &mut FrameReader, engine: &Engine, sup: &Supervisor, v: VerifyTask) -> Result<()> {
    let req = v.request.clone().context("verification is missing its request")?;
    let res = engine.verify(&req, &v.output_token_ids, &v.positions, v.top_k).await.context("recomputing for verification")?;
    let msg = verify_result(&v.verify_id, res);
    sup.progress.tick(); // A completed verification step contributes to reported progress.
    sup.progress.set_phase(Phase::Finishing);
    framed::write(send, &WorkStreamNodeMessage { body: Some(work_stream_node_message::Body::VerifyResult(msg)) }).await?;
    let ack: WorkStreamCoordinatorMessage = r.read().await?.context("stream closed before the result")?;
    if let Some(work_stream_coordinator_message::Body::ResultAck(a)) = ack.body {
        if !a.accepted {
            warn!(reason = %a.reason, "coordinator rejected verification");
        }
    }
    info!(positions = v.positions.len(), "verification sent");
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;

    use super::*;
    use crate::{
        engine::{
            fault, llama::LlamaEngine, progress::Progress, stats::Stats, supervisor::SupervisorCfg, ErrorClass, FaultCause, Limits,
        },
        fake_llama::{self, Fake, FakeState},
    };

    /// Verification reports prompt usage measured independently of the submitted result.
    #[test]
    fn verify_result_carries_prompt_token_count() {
        let res = crate::engine::Verification { prompt_tokens_sha256: [7; 32], prompt_tokens: 42, positions: vec![] };
        let m = verify_result("r0", res);
        assert_eq!(m.prompt_tokens, 42);
        assert_eq!(m.prompt_tokens_sha256, vec![7u8; 32]);
        assert_eq!(m.verify_id, "r0");
    }

    fn contributo_prova(name: &str) -> Arc<Contribution> {
        use crate::contribution::{PreferenceMode, ModelSelection, Preference};
        let d = std::env::temp_dir().join(format!("publicompute-sess-{}-{name}-{}", std::process::id(), rand::random::<u32>()));
        std::fs::create_dir_all(&d).unwrap();
        Contribution::open(&d, Preference { modo: PreferenceMode::Usage, model: Some(ModelSelection { id: "".into(), quant: "".into(), weights_sha256: None }), revision: 0 }).unwrap()
    }

    async fn node(probe_ms: u64) -> (Fake, NodeConfig) {
        let f = fake_llama::spawn(0, FakeState::new("/m.gguf")).await.unwrap();
        let progress = Arc::new(Progress::default());
        let limits = Limits {
            prefill_stall: Duration::from_millis(800),
            decode_stall: Duration::from_millis(600),
            request_max: Duration::from_secs(60),
            call_timeout: Duration::from_secs(5),
            step_timeout: Duration::from_secs(5),
            probe_timeout: Duration::from_millis(probe_ms),
        };
        let engine = Arc::new(Engine::Llama(LlamaEngine::new(f.url(), limits, progress.clone(), true).unwrap()));
        let sup = Supervisor::new(
            engine,
            SupervisorCfg {
                managed: None,
                expected_model: None,
                probe_interval: Duration::from_secs(3600),
                settle: Duration::from_millis(500),
                max_restarts: 3,
                restart_window: Duration::from_secs(600),
                backoff_base: Duration::from_millis(50),
                backoff_max: Duration::from_millis(200),
                recycle: None,
                start_timeout: Duration::from_secs(5),
                maintenance_tick: Duration::from_millis(100),
            },
            Arc::new(Stats::new(None)),
            progress,
        );
        sup.start().await.unwrap();
        let cfg = NodeConfig { model: ModelRef::default(), costs: None, hardware: Default::default(), sup, stages: None, stage_engine: None, region: String::new(), nat_class: String::new(), iroh_node_id: String::new(), slots: 1, contribution: contributo_prova("nodo"), gpu: Default::default(), model_path: None, output: watch::channel(false).1, live: crate::control::Live::new(), model_store: None };
        (f, cfg)
    }

    fn active(cfg: &NodeConfig, id: &str) -> ActiveLease {
        ActiveLease {
            id: id.into(),
            stage: false,
            progress: Arc::new(Mutex::new(LeaseCounters::default())),
            own: None,
            confirm: None,
            task: tokio::spawn(std::future::pending::<()>()),
            guard: Some(cfg.sup.try_begin().ok().unwrap()),
        }
    }

    fn single_lease(l: ActiveLease) -> Leases {
        let mut v = Leases::default();
        v.map.insert(l.id.clone(), l);
        v
    }

    fn drain(rx: &mut mpsc::Receiver<node_message::Body>) -> Vec<node_message::Body> {
        let mut v = vec![];
        while let Ok(m) = rx.try_recv() {
            v.push(m);
        }
        v
    }

    /// The handshake sends a configured region and leaves the field empty otherwise.
    #[tokio::test]
    async fn hello_carries_configured_region() {
        for regione in ["EU-IT", ""] {
            let (_f, mut cfg) = node(400).await;
            cfg.region = regione.to_string();
            let (client, server) = loopback().await;
            let engine = cfg.sup.engine.clone();
            let (stop_tx, stop_rx) = watch::channel(false);
            let task = tokio::spawn(async move { run(client, &cfg, engine, Arc::new(Shared::default()), stop_rx).await });
            let (_send, recv) = server.accept_bi().await.unwrap();
            let mut r = FrameReader::new(recv);
            let primo: NodeMessage = r.read().await.unwrap().expect("Hello");
            match primo.body {
                Some(node_message::Body::Hello(h)) => assert_eq!(h.region, regione),
                altro => panic!("expected Hello, got {altro:?}"),
            }
            let _ = stop_tx.send(true);
            server.close(0u32.into(), b"fine");
            let _ = tokio::time::timeout(Duration::from_secs(5), task).await;
        }
    }

    async fn wait_ready(cfg: &NodeConfig) {
        let t0 = Instant::now();
        while cfg.sup.state() != EngineState::Ready {
            assert!(t0.elapsed() < Duration::from_secs(10), "state {:?}", cfg.sup.state());
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    }

    /// A stalled decode must advertise recovery before releasing the lease as failed.
    #[tokio::test]
    async fn engine_fault_sends_recovering_before_release() {
        let (f, cfg) = node(400).await;
        f.state.decode_stall_after.store(2, Ordering::SeqCst);
        let mut lease = single_lease(active(&cfg, "L1"));
        let req = publicompute_proto::node::InferenceRequest {
            request_id: "r".into(),
            messages: vec![publicompute_proto::node::ChatMessage { role: publicompute_proto::node::ChatRole::User as i32, content: "x".into() }],
            ..Default::default()
        };
        let t0 = Instant::now();
        let err = cfg.sup.engine.generate(&req, 3).await.unwrap_err();
        assert!(t0.elapsed() < Duration::from_secs(4));
        assert!(matches!(classify(&err), ErrorClass::Fault(FaultCause::Watchdog)), "{err:#}");

        let (tx, mut rx) = mpsc::channel(8);
        let out = Out(tx);
        let mut last = Availability::Idle;
        let shared = Shared::default();
        on_lease_done(&cfg, &shared, &out, &mut last, &mut lease, LeaseDone { id: "L1".into(), chat_id: None, result: Err(err) }, false).await.unwrap();
        let msgs = drain(&mut rx);
        match msgs.as_slice() {
            [node_message::Body::Availability(a), node_message::Body::LeaseRelease(r)] => {
                assert_eq!(a.availability, Availability::EngineRecovering as i32);
                assert_eq!(r.reason, LeaseReleaseReason::EngineFailure as i32);
                assert_eq!(r.lease_id, "L1");
            }
            other => panic!("sequenza inattesa: {other:?}"),
        }
        assert!(lease.is_empty());
        // Check that scheduling resumes only after recovery.
        assert!(cfg.sup.check_ready().is_err());
        f.state.decode_stall_after.store(-1, Ordering::SeqCst);
        wait_ready(&cfg).await;
    }

    /// A transport failure can abandon an active request. Give the engine time
    /// to clear it without restarting.
    #[tokio::test]
    async fn non_engine_error_releases_and_only_settles() {
        let (_f, cfg) = node(400).await;
        let mut lease = single_lease(active(&cfg, "L1"));
        let (tx, mut rx) = mpsc::channel(8);
        let mut last = Availability::Idle;
        on_lease_done(
            &cfg,
            &Shared::default(),
            &Out(tx),
            &mut last,
            &mut lease,
            LeaseDone { id: "L1".into(), chat_id: None, result: Err(anyhow::anyhow!("work stream closed")) },
            false,
        )
        .await
        .unwrap();
        let msgs = drain(&mut rx);
        assert!(
            matches!(msgs.as_slice(), [node_message::Body::Availability(_), node_message::Body::LeaseRelease(_)] | [node_message::Body::LeaseRelease(_)]),
            "{msgs:?}"
        );
        wait_ready(&cfg).await;
        assert_eq!(cfg.sup.stats.run().engine_restarts, 0);
    }

    /// A cancellation revocation must not be classified as an engine failure.
    #[tokio::test]
    async fn cancelled_revoke_is_not_an_engine_failure() {
        let (_f, cfg) = node(400).await;
        let mut l = single_lease(active(&cfg, "L1"));
        on_revoked(&cfg, &mut l, "L1", LeaseRevokeReason::Cancelled as i32);
        wait_ready(&cfg).await;
        let c = cfg.sup.stats.run();
        assert_eq!((c.requests_cancelled, c.engine_restarts, c.requests_failed, c.probes_failed), (1, 0, 0, 0));
        assert_eq!(c.probes_ok, 1, "expected just the startup probe");
    }

    /// A revoked lease can leave an engine slot occupied.
    #[tokio::test]
    async fn revoke_while_engine_busy_blocks_offers_until_free() {
        let (f, cfg) = node(400).await;
        let mut l = single_lease(active(&cfg, "L1"));
        f.state.busy.store(true, Ordering::SeqCst);
        on_revoked(&cfg, &mut l, "L1", LeaseRevokeReason::Reassigned as i32);
        tokio::time::sleep(Duration::from_millis(250)).await;
        assert!(cfg.sup.try_begin().is_err());
        f.state.busy.store(false, Ordering::SeqCst);
        wait_ready(&cfg).await;
        assert!(cfg.sup.try_begin().is_ok());
    }

    /// A no-progress revocation starts engine recovery.
    #[tokio::test]
    async fn no_progress_revoke_is_an_engine_fault() {
        let (f, cfg) = node(400).await;
        let mut l = single_lease(active(&cfg, "L1"));
        f.state.hang_completion.store(true, Ordering::SeqCst);
        on_revoked(&cfg, &mut l, "L1", LeaseRevokeReason::NoProgress as i32);
        assert_eq!(cfg.sup.state(), EngineState::Recovering);
        tokio::time::sleep(Duration::from_millis(600)).await;
        assert_eq!(cfg.sup.state(), EngineState::Recovering);
        f.state.hang_completion.store(false, Ordering::SeqCst);
        wait_ready(&cfg).await;
        assert_eq!(cfg.sup.stats.run().requests_cancelled, 0);
    }



    use crate::{
        engine::stages::StadiEngine,
        stages::{
            tests::{assignment, cfg_test, prossimo, tempdir},
            Stages, StagesConfig,
        },
    };
    use publicompute_proto::node::{
        stage_status::State, work_stream_coordinator_message as wc, work_stream_node_message as wn, CancelRequest, ChatMessage, ChatTask, FinishReason, InferenceRequest, Lease,
        LogicParams, ResultAck, SamplingParams, WorkStreamCoordinatorMessage as WcMsg, WorkStreamNodeMessage as WnMsg,
    };

    /// Open a local QUIC connection pair using fixture certificates.
    async fn loopback() -> (quinn::Connection, quinn::Connection) {
        loopback_with_window(None).await
    }

    async fn loopback_with_window(window: Option<u32>) -> (quinn::Connection, quinn::Connection) {
        let ck = rcgen::generate_simple_self_signed(vec!["localhost".to_string()]).unwrap();
        let cert = ck.cert.der().clone();
        let key = rustls::pki_types::PrivatePkcs8KeyDer::from(ck.signing_key.serialize_der());
        let mut tls = rustls::ServerConfig::builder_with_provider(Arc::new(rustls::crypto::ring::default_provider()))
            .with_protocol_versions(&[&rustls::version::TLS13])
            .unwrap()
            .with_no_client_auth()
            .with_single_cert(vec![cert.clone()], key.into())
            .unwrap();
        tls.alpn_protocols = vec![transport::ALPN.to_vec()];
        let mut sc = quinn::ServerConfig::with_crypto(Arc::new(quinn::crypto::rustls::QuicServerConfig::try_from(tls).unwrap()));
        if let Some(window) = window {
            let mut transport = quinn::TransportConfig::default();
            transport.stream_receive_window(window.into());
            sc.transport_config(Arc::new(transport));
        }
        let ep = quinn::Endpoint::server(sc, "127.0.0.1:0".parse().unwrap()).unwrap();
        let addr = ep.local_addr().unwrap();
        let server = tokio::spawn(async move { ep.accept().await.unwrap().await.unwrap() });
        let t = transport::Transport::new(addr, "localhost".into(), cert.to_vec()).unwrap();
        let client = t.connect_anonymous().await.unwrap();
        (client, server.await.unwrap())
    }

    /// Start a fixture head with a service connection ready for streamed work.
    async fn nodo_testa(name: &str, env: Vec<(String, String)>) -> (NodeConfig, Arc<Engine>, Arc<Stages>) {
        let d = tempdir(name);
        let stages = Stages::new(StagesConfig { env, ..cfg_test(&d) });
        let (tx, mut rx) = mpsc::channel(16);
        stages.connect_peer(tx);
        stages.queue_assignment(assignment("catena-x", 1, 1, 2, 0));
        let loading = prossimo(&mut rx).await;
        assert_eq!(loading.state(), State::Loading, "{loading:?}");
        assert_eq!(prossimo(&mut rx).await.state(), State::Ready);
        let progress = Arc::new(Progress::default());
        let limits = Limits {
            prefill_stall: Duration::from_secs(10),
            decode_stall: Duration::from_secs(10),
            request_max: Duration::from_secs(60),
            call_timeout: Duration::from_secs(5),
            step_timeout: Duration::from_secs(5),
            probe_timeout: Duration::from_secs(5),
        };
        let engine = Arc::new(Engine::Stages(StadiEngine { stages: stages.clone(), attive: Default::default(), limits, progress: progress.clone() }));
        let sup = Supervisor::new(
            engine.clone(),
            SupervisorCfg {
                managed: None,
                expected_model: None,
                probe_interval: Duration::from_secs(3600),
                settle: Duration::from_millis(500),
                max_restarts: 3,
                restart_window: Duration::from_secs(600),
                backoff_base: Duration::from_millis(50),
                backoff_max: Duration::from_millis(200),
                recycle: None,
                start_timeout: Duration::from_secs(5),
                maintenance_tick: Duration::from_millis(100),
            },
            Arc::new(Stats::new(None)),
            progress,
        );
        sup.start().await.unwrap();
        let model = ModelRef { model_id: "m".into(), quantization: "q".into(), weights_sha256: vec![7; 32] };
        (NodeConfig { model, costs: None, hardware: Default::default(), sup, stages: Some(stages.clone()), stage_engine: None, region: String::new(), nat_class: String::new(), iroh_node_id: String::new(), slots: 1, contribution: contributo_prova(name), gpu: Default::default(), model_path: None, output: watch::channel(false).1, live: crate::control::Live::new(), model_store: None }, engine, stages)
    }

    fn offerta_chat(cfg: &NodeConfig, request_id: &str, text: &str, max_tokens: u32) -> WorkOffer {
        WorkOffer {
            lease: Some(Lease { lease_id: "L-chat".into(), ..Default::default() }),
            model: Some(cfg.model.clone()),
            logic: Some(LogicParams { format_version: logic::LOGIC_FORMAT, top_k: 5 }),
            payout: None,
            task: Some(work_offer::Task::Chat(ChatTask {
                chat_id: "test-chat".into(),
                affinity_owner: Vec::new(),
                member_seat: false,
                member_key: Vec::new(),
                request: Some(InferenceRequest {
                    request_id: request_id.into(),
                    messages: vec![ChatMessage { role: publicompute_proto::node::ChatRole::User as i32, content: text.into() }],
                    sampling: Some(SamplingParams { max_tokens, ..Default::default() }),
                }),
            })),
        }
    }

    /// Act as the work-stream peer. Collect deltas and the result, acknowledge
    /// it and wait for completion. Optionally cancel after a chosen delta.
    async fn coordinatore(server: quinn::Connection, request_id: String, annulla_dopo: Option<usize>) -> (Vec<TokenDelta>, RequestResult) {
        let (mut send, recv) = server.accept_bi().await.unwrap();
        let mut r = FrameReader::new(recv);
        let open: WnMsg = r.read().await.unwrap().unwrap();
        assert!(matches!(open.body, Some(wn::Body::Open(ref o)) if o.lease_id == "L-chat"), "{open:?}");
        let mut deltas = Vec::new();
        let mut taken = None;
        let risultato = loop {
            let m: WnMsg = r.read().await.unwrap().expect("stream closed");
            match m.body {
                Some(wn::Body::TokenDelta(d)) => {
                    deltas.push(d);
                    if annulla_dopo == Some(deltas.len()) {
                        framed::write(&mut send, &WcMsg { body: Some(wc::Body::CancelRequest(CancelRequest { request_id: request_id.clone() })) }).await.unwrap();
                    }
                }
                Some(wn::Body::SlotTaken(slot)) => {
                    assert!(deltas.is_empty(), "admission must precede tokens");
                    taken = Some(slot);
                }
                Some(wn::Body::RequestResult(res)) => break res,
                altro => panic!("unexpected message: {altro:?}"),
            }
        };
        if let Some(request_result::Outcome::Completion(c)) = &risultato.outcome {
            assert_eq!(taken, c.slot_report.as_ref().map(|r| publicompute_proto::common::SlotTaken {
                slot: r.slot, generation: r.generation,
            }));
        }
        framed::write(&mut send, &WcMsg { body: Some(wc::Body::ResultAck(ResultAck { request_id: request_id.clone(), accepted: true, reason: String::new() })) }).await.unwrap();
        let done: WnMsg = r.read().await.unwrap().expect("LeaseComplete");
        assert!(matches!(done.body, Some(wn::Body::LeaseComplete(_))), "{done:?}");
        (deltas, risultato)
    }

    async fn lancia_lease(client: quinn::Connection, cfg: &NodeConfig, engine: Arc<Engine>, offerta: WorkOffer) -> Result<(u64, Duration)> {
        let (ctx, crx) = oneshot::channel();
        ctx.send(()).unwrap();
        run_lease(client, "L-chat", offerta, engine, cfg.sup.clone(), crx, Arc::new(Mutex::new(LeaseCounters::default())), None).await
    }

    /// Exercise assignment, readiness, offer acceptance and a streamed result
    /// over a local QUIC session.
    #[tokio::test]
    async fn chain_head_chat_streams_deltas_and_result() {
        let (cfg, engine, stages) = nodo_testa("chat-lease", vec![]).await;
        assert_eq!(features(&cfg), vec![AgentFeature::Contribution as i32, AgentFeature::Stages as i32, AgentFeature::StageEngineProfile as i32]);
        let (client, server) = loopback().await;
        let coord = tokio::spawn(coordinatore(server, "r1".into(), None));
        let mut offerta = offerta_chat(&cfg, "r1", "cached-ciao", 50);
        let Some(work_offer::Task::Chat(chat)) = &mut offerta.task else { panic!("expected chat offer") };
        chat.affinity_owner = vec![0x1f; 16];
        chat.member_seat = true;
        chat.member_key = vec![0x3f; 16];
        let (tokens, _) = lancia_lease(client, &cfg, engine, offerta).await.unwrap();
        assert_eq!(tokens, 5);
        let (deltas, res) = coord.await.unwrap();
        // A token without display text belongs to the next fragment. The end token can have an empty fragment.
        let visto: Vec<(u32, &str, Vec<u32>)> = deltas.iter().map(|d| (d.index, d.text.as_str(), d.token_ids.clone())).collect();
        assert_eq!(visto, vec![(0, "Ciao ", vec![101]), (1, "\u{e8}", vec![102, 103]), (3, " un test", vec![104]), (4, "", vec![2])]);
        assert!(deltas.iter().all(|d| d.request_id == "r1"));
        assert_eq!(res.request_id, "r1");
        let Some(request_result::Outcome::Completion(c)) = res.outcome else { panic!("expected Completion, got {res:?}") };
        assert_eq!(c.text, "Ciao \u{e8} un test");
        assert_eq!(c.token_ids, vec![101, 102, 103, 104, 2]);
        assert_eq!(c.finish_reason, FinishReason::Stop as i32);
        let usage = c.usage.unwrap();
        assert_eq!(usage.completion_tokens, 5);
        assert_eq!(usage.cached_prompt_tokens, 4);
        assert_eq!(c.slot_report, Some(publicompute_proto::common::SlotReport { slot: 0, generation: 1, parked: true }));
        assert_eq!(c.prompt_tokens_sha256.len(), 32);
        assert_eq!(c.logic.unwrap().format_version, logic::LOGIC_FORMAT);
        assert!(c.timing.unwrap().prefill.is_some());
        stages.stop_all().await;
    }

    #[tokio::test]
    async fn coordinator_cancel_ends_chat_as_cancelled() {
        // Slow the fixture so the first delta leaves time for the cancellation to take effect.
        let (cfg, engine, stages) = nodo_testa("chat-annulla", vec![("FAKE_STAGE_TOKEN_DELAY_MS".into(), "150".into())]).await;
        let (client, server) = loopback().await;
        let coord = tokio::spawn(coordinatore(server, "r2".into(), Some(1)));
        let (tokens, _) = lancia_lease(client, &cfg, engine, offerta_chat(&cfg, "r2", "ciao", 50)).await.unwrap();
        let (deltas, res) = coord.await.unwrap();
        let Some(request_result::Outcome::Completion(c)) = res.outcome else { panic!("{res:?}") };
        assert_eq!(c.finish_reason, FinishReason::Cancelled as i32);
        assert!(tokens < 5 && !deltas.is_empty(), "token {tokens}, frammenti {}", deltas.len());
        stages.stop_all().await;
    }

    #[tokio::test]
    async fn head_rejection_is_a_request_failure() {
        let (cfg, engine, stages) = nodo_testa("chat-errore", vec![]).await;
        let (client, server) = loopback().await;
        let coord = tokio::spawn(coordinatore(server, "r3".into(), None));
        lancia_lease(client, &cfg, engine, offerta_chat(&cfg, "r3", "rifiuta", 50)).await.unwrap();
        let (deltas, res) = coord.await.unwrap();
        assert!(deltas.is_empty());
        let Some(request_result::Outcome::Failure(f)) = res.outcome else { panic!("{res:?}") };
        assert_eq!(f.code, RequestFailureCode::ContextTooLong as i32);
        stages.stop_all().await;
    }

    #[tokio::test]
    async fn head_capacity_rejection_survives_the_work_stream() {
        let (cfg, engine, stages) = nodo_testa("chat-protected-slots", vec![]).await;
        let (client, server) = loopback().await;
        let coord = tokio::spawn(coordinatore(server, "r4".into(), None));
        lancia_lease(client, &cfg, engine, offerta_chat(&cfg, "r4", "capacity-unavailable", 50)).await.unwrap();
        let (deltas, result) = coord.await.unwrap();
        assert!(deltas.is_empty());
        let Some(request_result::Outcome::Failure(failure)) = result.outcome else { panic!("{result:?}") };
        assert_eq!(failure.code, RequestFailureCode::CapacityUnavailable as i32);
        stages.stop_all().await;
    }

    /// Drive one lease through stream open, deltas, result acknowledgment and
    /// completion. Optionally cancel after the requested number of deltas.
    async fn stream(mut send: quinn::SendStream, recv: quinn::RecvStream, annulla_dopo: Option<usize>) -> (String, Vec<TokenDelta>, RequestResult) {
        let mut r = FrameReader::new(recv);
        let open: WnMsg = r.read().await.unwrap().unwrap();
        let Some(wn::Body::Open(o)) = open.body else { panic!("expected Open, got {open:?}") };
        let (mut deltas, mut rid) = (Vec::new(), String::new());
        let risultato = loop {
            let m: WnMsg = r.read().await.unwrap().expect("stream closed");
            match m.body {
                Some(wn::Body::TokenDelta(d)) => {
                    rid = d.request_id.clone();
                    deltas.push(d);
                    if annulla_dopo == Some(deltas.len()) {
                        framed::write(&mut send, &WcMsg { body: Some(wc::Body::CancelRequest(CancelRequest { request_id: rid.clone() })) }).await.unwrap();
                    }
                }
                Some(wn::Body::RequestResult(res)) => break res,
                altro => panic!("unexpected message: {altro:?}"),
            }
        };
        framed::write(&mut send, &WcMsg { body: Some(wc::Body::ResultAck(ResultAck { request_id: risultato.request_id.clone(), accepted: true, reason: String::new() })) }).await.unwrap();
        let done: WnMsg = r.read().await.unwrap().expect("LeaseComplete");
        assert!(matches!(done.body, Some(wn::Body::LeaseComplete(_))), "{done:?}");
        let _ = rid;
        (o.lease_id, deltas, risultato)
    }

    /// Four concurrent leases share one head. Cancelling one preserves the
    /// others' output. The head reports RUNNING when work starts and READY
    /// when the last request ends.
    #[tokio::test]
    async fn four_concurrent_leases_and_independent_cancellation() {
        let (cfg, engine, stages) = nodo_testa("chat-quattro", vec![("FAKE_STAGE_TOKEN_DELAY_MS".into(), "80".into())]).await;
        let (tx, mut stati) = mpsc::channel(16);
        stages.connect_peer(tx);
        let (client, server) = loopback().await;
        let coord = tokio::spawn(async move {
            let mut flussi = Vec::new();
            for _ in 0..4 {
                let (send, recv) = server.accept_bi().await.unwrap();
                // Associate the cancellation by stream order because the request name
                // is unavailable before the first delta.
                let cancel = (flussi.len() == 2).then_some(1);
                flussi.push(tokio::spawn(stream(send, recv, cancel)));
            }
            let mut v = Vec::new();
            for f in flussi {
                v.push(f.await.unwrap());
            }
            v
        });
        let mut task = Vec::new();
        for i in 0..4 {
            let (c, cf, e) = (client.clone(), cfg.sup.clone(), engine.clone());
            let offerta = offerta_chat(&cfg, &format!("r{i}"), "ciao", 50);
            task.push(tokio::spawn(async move {
                let (ctx, crx) = oneshot::channel();
                ctx.send(()).unwrap();
                run_lease(c, &format!("L{i}"), offerta, e, cf, crx, Arc::new(Mutex::new(LeaseCounters::default())), None).await
            }));
            // Open streams sequentially so their order matches the accepted leases.
            tokio::time::sleep(Duration::from_millis(40)).await;
        }
        for t in task {
            t.await.unwrap().unwrap();
        }
        let flussi = coord.await.unwrap();
        assert_eq!(flussi.len(), 4);
        let (mut completi, mut annullati) = (0, 0);
        for (_lease, deltas, res) in &flussi {
            let Some(request_result::Outcome::Completion(c)) = &res.outcome else { panic!("{res:?}") };
            if c.finish_reason == FinishReason::Cancelled as i32 {
                annullati += 1;
                assert!(c.token_ids.len() < 5);
            } else {
                completi += 1;
                assert_eq!(c.text, "Ciao \u{e8} un test");
                assert_eq!(c.token_ids, vec![101, 102, 103, 104, 2]);
                assert_eq!(deltas.iter().map(|d| d.text.as_str()).collect::<String>(), "Ciao \u{e8} un test");
            }
        }
        assert_eq!((completi, annullati), (3, 1));
        assert_eq!(prossimo(&mut stati).await.state(), State::Running);
        assert_eq!(prossimo(&mut stati).await.state(), State::Ready);
        stages.stop_all().await;
    }

    /// Reject an offer beyond stage capacity and hold the shared reservation
    /// until the last lease ends.
    #[tokio::test]
    async fn chain_head_accepts_offers_up_to_slot_count() {
        let (cfg, engine, stages) = nodo_testa("chat-capacita", vec![]).await;
        let (client, _server) = loopback().await;
        let (tx, mut rx) = mpsc::channel(64);
        let (done_tx, _done_rx) = mpsc::channel(8);
        let mut lease = Leases::default();
        let shared = Arc::new(Shared::default());
        for i in 0..5 {
            let mut o = offerta_chat(&cfg, &format!("r{i}"), "ciao", 5);
            o.lease.as_mut().unwrap().lease_id = format!("L{i}");
            handle_offer(&client, &cfg, &engine, &shared, &Out(tx.clone()), &done_tx, &mut lease, false, o).await.unwrap();
        }
        let msgs = drain(&mut rx);
        let accettate = msgs.iter().filter(|m| matches!(m, node_message::Body::LeaseAccept(_))).count();
        let rifiutate: Vec<_> = msgs.iter().filter_map(|m| if let node_message::Body::LeaseReject(r) = m { Some((r.lease_id.clone(), r.reason())) } else { None }).collect();
        assert_eq!(accettate, 4);
        assert_eq!(rifiutate, vec![("L4".to_string(), LeaseRejectReason::HostBusy)]);
        assert_eq!(cfg.sup.state(), EngineState::Ready);
        assert!(lease.comune.is_none());
        assert_eq!(heartbeat_msg(&lease, &cfg, &Shared::default()).leases.len(), 4);
        for i in 0..3 {
            on_revoked(&cfg, &mut lease, &format!("L{i}"), LeaseRevokeReason::Cancelled as i32);
            assert_eq!(cfg.sup.state(), EngineState::Ready, "stage leases do not reserve the supervisor");
        }
        on_revoked(&cfg, &mut lease, "L3", LeaseRevokeReason::Cancelled as i32);
        wait_ready(&cfg).await;
        assert!(lease.is_empty());
        stages.stop_all().await;
    }

    /// Read received cancellations from the fixture stage log.
    fn annulli_ricevuti(name: &str) -> usize {
        let log = std::env::temp_dir().join(format!("publicompute-stadi-{name}-{}", std::process::id())).join("log/stage-catena-x-1.log");
        std::fs::read_to_string(log).unwrap_or_default().lines().filter(|r| r.contains("Cancel received for request")).count()
    }

    async fn accept_partial_chat(server: &quinn::Connection) -> (quinn::SendStream, quinn::RecvStream) {
        let (send, mut recv) = server.accept_bi().await.unwrap();
        let mut byte = [0];
        recv.read_exact(&mut byte).await.unwrap();
        assert!((1..128).contains(&byte[0]));
        let mut payload = vec![0; byte[0] as usize];
        recv.read_exact(&mut payload).await.unwrap();
        let open = <WnMsg as prost::Message>::decode(payload.as_slice()).unwrap();
        assert!(matches!(open.body, Some(wn::Body::Open(_))));
        recv.read_exact(&mut byte).await.unwrap();
        assert_ne!(byte[0] & 0x80, 0, "expected the start of a large delta");
        (send, recv)
    }

    fn start_chat_task(client: quinn::Connection, cfg: &NodeConfig, engine: Arc<Engine>, request_id: &str, text: &str) -> tokio::task::JoinHandle<Result<(u64, Duration)>> {
        let sup = cfg.sup.clone();
        let offer = offerta_chat(cfg, request_id, text, 50);
        tokio::spawn(async move {
            let (tx, rx) = oneshot::channel();
            tx.send(()).unwrap();
            run_lease(client, "L-chat", offer, engine, sup, rx, Arc::new(Mutex::new(LeaseCounters::default())), None).await
        })
    }

    async fn wait_for_head_cancel(name: &str) {
        tokio::time::timeout(Duration::from_secs(5), async {
            while annulli_ricevuti(name) == 0 {
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        }).await.expect("the head did not receive Cancel");
    }

    async fn assert_chat_reset(mut recv: quinn::RecvStream) {
        tokio::time::timeout(Duration::from_secs(5), async {
            let mut buf = [0; 4096];
            loop {
                match recv.read(&mut buf).await {
                    Ok(Some(_)) => {}
                    Err(quinn::ReadError::Reset(code)) => {
                        assert_eq!(code, 1u32.into());
                        break;
                    }
                    other => panic!("expected a reset, got {other:?}"),
                }
            }
        }).await.unwrap();
    }

    #[tokio::test]
    async fn stalled_chat_reads_cancel_while_another_chat_completes() {
        let name = "chat-stream-stalled";
        let (cfg, engine, stages) = nodo_testa(name, vec![]).await;
        let (client, server) = loopback_with_window(Some(1024)).await;
        let stalled = start_chat_task(client.clone(), &cfg, engine.clone(), "stalled", "stall-chat-output");
        let (mut commands, blocked) = tokio::time::timeout(Duration::from_secs(5), accept_partial_chat(&server)).await.unwrap();
        framed::write(&mut commands, &WcMsg { body: Some(wc::Body::CancelRequest(CancelRequest { request_id: "stalled".into() })) }).await.unwrap();
        wait_for_head_cancel(name).await;

        let peer = tokio::spawn(coordinatore(server.clone(), "healthy".into(), None));
        let healthy = start_chat_task(client.clone(), &cfg, engine.clone(), "healthy", "ciao");
        assert_eq!(tokio::time::timeout(Duration::from_secs(5), healthy).await.unwrap().unwrap().unwrap().0, 5);
        assert_eq!(peer.await.unwrap().0.len(), 4);
        assert!(!stalled.is_finished());
        stalled.abort();
        assert!(stalled.await.unwrap_err().is_cancelled());
        assert_chat_reset(blocked).await;
        assert_healthy_chat(&client, &server, &cfg, engine).await;
        stages.stop_all().await;
    }

    #[tokio::test]
    async fn saturated_chat_cancels_generation_and_resets_the_stream() {
        let name = "chat-stream-saturated";
        let (cfg, engine, stages) = nodo_testa(name, vec![]).await;
        let (client, server) = loopback_with_window(Some(1024)).await;
        let stalled = start_chat_task(client.clone(), &cfg, engine.clone(), "stalled", "flood-chat-output");
        let (_commands, blocked) = tokio::time::timeout(Duration::from_secs(5), accept_partial_chat(&server)).await.unwrap();
        let error = tokio::time::timeout(Duration::from_secs(5), stalled).await.unwrap().unwrap().unwrap_err();
        assert!(error.is::<OutputStalled>(), "{error:#}");
        wait_for_head_cancel(name).await;
        assert_chat_reset(blocked).await;
        assert_healthy_chat(&client, &server, &cfg, engine).await;
        stages.stop_all().await;
    }

    async fn assert_healthy_chat(client: &quinn::Connection, server: &quinn::Connection, cfg: &NodeConfig, engine: Arc<Engine>) {
        let peer = tokio::spawn(coordinatore(server.clone(), "after-reset".into(), None));
        let chat = start_chat_task(client.clone(), cfg, engine, "after-reset", "ciao");
        assert_eq!(tokio::time::timeout(Duration::from_secs(5), chat).await.unwrap().unwrap().unwrap().0, 5);
        assert_eq!(peer.await.unwrap().0.len(), 4);
    }

    #[tokio::test]
    async fn saturated_deltas_cancel_only_their_head_request() {
        let name = "head-deltas-saturated";
        let (cfg, engine, stages) = nodo_testa(name, vec![]).await;
        let Engine::Stages(engine) = &*engine else { unreachable!() };
        let request = |id, prompt| {
            let Some(work_offer::Task::Chat(chat)) = offerta_chat(&cfg, id, prompt, 50).task else { unreachable!() };
            chat.request.unwrap()
        };
        let stalled = request("stalled", "stall-chat-output");
        let healthy = request("healthy", "ciao");
        let (deltas, _unread, full) = crate::engine::output::channel(2, 256 << 10);
        let (first, second) = tokio::join!(engine.chat(&stalled, Some(deltas), None), engine.chat(&healthy, None, None));
        assert!(first.unwrap_err().is::<OutputStalled>());
        assert!(*full.borrow());
        assert_eq!(second.unwrap().token_ids.len(), 5);
        wait_for_head_cancel(name).await;
        stages.stop_all().await;
    }

    #[tokio::test]
    async fn partial_chat_frame_timeout_resets_the_stream() {
        let (client, server) = loopback_with_window(Some(1024)).await;
        let (send, _recv) = client.open_bi().await.unwrap();
        let task = tokio::spawn(async move {
            let mut writer = framed::ChatWriter::new(send);
            writer.write_with_timeout(&WnMsg { body: Some(wn::Body::TokenDelta(TokenDelta {
                text: "x".repeat(512 << 10), ..Default::default()
            })) }, Duration::from_secs(1)).await.unwrap_err()
        });
        let (_commands, mut blocked) = tokio::time::timeout(Duration::from_secs(5), server.accept_bi()).await.unwrap().unwrap();
        blocked.read_exact(&mut [0; 1]).await.unwrap();
        let error = task.await.unwrap();
        assert!(error.is::<framed::ChatStreamError>());
        assert!(format!("{error:#}").contains("chat frame write timed out"), "{error:#}");
        assert_chat_reset(blocked).await;
    }

    /// A single lease revocation reaches the corresponding head request and
    /// leaves another active chat running.
    #[tokio::test]
    async fn revoking_lease_cancels_request_on_head() {
        let name = "chat-revoca";
        // Delay tokens so revocation arrives while the requests are active.
        let (cfg, engine, stages) = nodo_testa(name, vec![("FAKE_STAGE_TOKEN_DELAY_MS".into(), "400".into())]).await;
        let (client, _server) = loopback().await;
        let (tx, _rx) = mpsc::channel(64);
        let (done_tx, _done_rx) = mpsc::channel(8);
        let mut lease = Leases::default();
        let shared = Arc::new(Shared::default());
        for i in 0..2 {
            let mut o = offerta_chat(&cfg, &format!("r{i}"), "ciao", 50);
            o.lease.as_mut().unwrap().lease_id = format!("L{i}");
            handle_offer(&client, &cfg, &engine, &shared, &Out(tx.clone()), &done_tx, &mut lease, false, o).await.unwrap();
        }
        assert_eq!(lease.map.len(), 2);
        for l in lease.map.values_mut() {
            let _ = l.confirm.take().unwrap().send(());
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
        assert_eq!(annulli_ricevuti(name), 0);
        on_revoked(&cfg, &mut lease, "L0", LeaseRevokeReason::Reassigned as i32);
        let t0 = Instant::now();
        while annulli_ricevuti(name) < 1 {
            assert!(t0.elapsed() < Duration::from_secs(3), "head did not receive Cancel after revocation");
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
        assert_eq!(annulli_ricevuti(name), 1, "expected one Cancel for one revocation");
        assert_eq!(lease.map.len(), 1);
        on_revoked(&cfg, &mut lease, "L1", LeaseRevokeReason::Cancelled as i32);
        let t0 = Instant::now();
        while annulli_ricevuti(name) < 2 {
            assert!(t0.elapsed() < Duration::from_secs(3), "missing the second Cancel");
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        stages.stop_all().await;
    }

    async fn esito_offerta(cfg: &NodeConfig, engine: &Arc<Engine>, client: &quinn::Connection, offerta: WorkOffer) -> Option<LeaseRejectReason> {
        let (tx, mut rx) = mpsc::channel(8);
        let (done_tx, _done_rx) = mpsc::channel(1);
        let mut lease = Leases::default();
        handle_offer(client, cfg, engine, &Arc::new(Shared::default()), &Out(tx), &done_tx, &mut lease, false, offerta).await.unwrap();
        match drain(&mut rx).into_iter().next() {
            Some(node_message::Body::LeaseReject(r)) => Some(r.reason()),
            _ => None,
        }
    }

    #[tokio::test]
    async fn wrong_node_rejects_offers() {
        let (cfg, engine, stages) = nodo_testa("offerte", vec![]).await;
        let (client, _server) = loopback().await;
        let mut batch = offerta_chat(&cfg, "r", "x", 5);
        batch.task = Some(work_offer::Task::Batch(publicompute_proto::node::BatchTask { requests: vec![] }));
        assert_eq!(esito_offerta(&cfg, &engine, &client, batch).await, Some(LeaseRejectReason::HostBusy));
        stages.stop_all().await;
        assert_eq!(esito_offerta(&cfg, &engine, &client, offerta_chat(&cfg, "r", "x", 5)).await, Some(LeaseRejectReason::HostBusy));
        let (_f, single) = node(400).await;
        let engine1 = single.sup.engine.clone();
        let mut single = single;
        single.model = cfg.model.clone();
        assert_eq!(esito_offerta(&single, &engine1, &client, offerta_chat(&single, "r", "x", 5)).await, None);
    }

    /// Enforce the configured HTTP slot limit. Keep the shared reservation
    /// until the final revocation and report progress separately for each lease.
    #[tokio::test]
    async fn single_gpu_accepts_chats_up_to_slot_count() {
        let (f, mut cfg) = node(400).await;
        f.state.token_delay_ms.store(300, Ordering::SeqCst);
        cfg.slots = 3;
        let engine = cfg.sup.engine.clone();
        let (client, _server) = loopback().await;
        let (tx, mut rx) = mpsc::channel(64);
        let (done_tx, _done_rx) = mpsc::channel(8);
        let mut lease = Leases::default();
        let shared = Arc::new(Shared::default());
        for i in 0..4 {
            let mut o = offerta_chat(&cfg, &format!("r{i}"), "ciao", 5);
            o.lease.as_mut().unwrap().lease_id = format!("L{i}");
            handle_offer(&client, &cfg, &engine, &shared, &Out(tx.clone()), &done_tx, &mut lease, false, o).await.unwrap();
        }
        let msgs = drain(&mut rx);
        let accettate = msgs.iter().filter(|m| matches!(m, node_message::Body::LeaseAccept(_))).count();
        let rifiutate: Vec<_> = msgs.iter().filter_map(|m| if let node_message::Body::LeaseReject(r) = m { Some((r.lease_id.clone(), r.reason())) } else { None }).collect();
        assert_eq!(accettate, 3);
        assert_eq!(rifiutate, vec![("L3".to_string(), LeaseRejectReason::HostBusy)]);
        assert_eq!(cfg.sup.state(), EngineState::Busy);
        assert_eq!(heartbeat_msg(&lease, &cfg, &Shared::default()).leases.len(), 3);
        assert!(lease.map.values().all(|l| l.own.is_some()));
        for i in 0..2 {
            on_revoked(&cfg, &mut lease, &format!("L{i}"), LeaseRevokeReason::Cancelled as i32);
            assert_eq!(cfg.sup.state(), EngineState::Busy, "one lease is still active");
        }
        on_revoked(&cfg, &mut lease, "L2", LeaseRevokeReason::Cancelled as i32);
        f.state.busy.store(false, Ordering::SeqCst);
        wait_ready(&cfg).await;
        assert!(lease.is_empty());
        // The default one-slot engine rejects a second concurrent chat.
        cfg.slots = 1;
        for i in 0..2 {
            let mut o = offerta_chat(&cfg, &format!("s{i}"), "ciao", 5);
            o.lease.as_mut().unwrap().lease_id = format!("S{i}");
            handle_offer(&client, &cfg, &engine, &shared, &Out(tx.clone()), &done_tx, &mut lease, false, o).await.unwrap();
        }
        let msgs = drain(&mut rx);
        assert_eq!(msgs.iter().filter(|m| matches!(m, node_message::Body::LeaseAccept(_))).count(), 1);
        assert_eq!(msgs.iter().filter(|m| matches!(m, node_message::Body::LeaseReject(_))).count(), 1);
    }

    /// The HTTP backend must deliver deltas and a final result over the same
    /// work protocol as the stage backend.
    #[tokio::test]
    async fn single_gpu_chat_streams_deltas_and_result() {
        let (_f, cfg) = node(400).await;
        let engine = cfg.sup.engine.clone();
        let (client, server) = loopback().await;
        let coord = tokio::spawn(coordinatore(server, "r1".into(), None));
        let (tokens, _) = lancia_lease(client, &cfg, engine, offerta_chat(&cfg, "r1", "ciao", 0)).await.unwrap();
        assert_eq!(tokens, 12);
        let (deltas, res) = coord.await.unwrap();
        assert!(!deltas.is_empty());
        let Some(request_result::Outcome::Completion(c)) = res.outcome else { panic!("expected Completion, got {res:?}") };
        assert_eq!(deltas.iter().map(|d| d.text.as_str()).collect::<String>(), c.text);
        assert_eq!(c.token_ids.len(), 12);
        assert_eq!(c.finish_reason, FinishReason::Stop as i32);
        assert_eq!(c.logic.unwrap().format_version, logic::LOGIC_FORMAT);
    }

    /// Preserve conversation history and role strings when forwarding to the head.
    #[test]
    fn head_receives_full_conversation() {
        use crate::engine::stages::messaggi_testa;
        use publicompute_proto::node::ChatRole::*;
        let m = |r, c: &str| ChatMessage { role: r as i32, content: c.into() };
        let req = InferenceRequest { messages: vec![m(System, "sii breve"), m(User, "prima"), m(Assistant, "ok"), m(User, "ultima")], ..Default::default() };
        let v: Vec<(String, String)> = messaggi_testa(&req).unwrap().into_iter().map(|m| (m.role, m.text)).collect();
        assert_eq!(
            v,
            [("system", "sii breve"), ("user", "prima"), ("assistant", "ok"), ("user", "ultima")].map(|(a, b)| (a.to_string(), b.to_string()))
        );
        assert!(messaggi_testa(&InferenceRequest { messages: vec![m(System, "x")], ..Default::default() }).is_none());
    }

    /// A heartbeat carries the engine phase, progress sequence and telemetry.
    #[tokio::test]
    async fn heartbeat_carries_phase_and_progress() {
        let (_f, cfg) = node(400).await;
        let l = single_lease(active(&cfg, "L1"));
        cfg.sup.progress.set_phase(Phase::Decode);
        cfg.sup.progress.tick();
        let hb = heartbeat_msg(&l, &cfg, &Shared::default());
        let p = &hb.leases[0];
        assert_eq!(p.phase, publicompute_proto::node::LeasePhase::Decode as i32);
        assert_eq!(p.progress_seq, cfg.sup.progress.snapshot().seq);
        assert!(hb.telemetry.unwrap().engine.is_some());
        let _ = fault(FaultCause::Timeout, "x"); // Exercise fault construction with the same type used by engine backends.
    }

    /// Independent counters let the coordinator detect a stalled prefill even
    /// while another chat on the same head continues decoding.
    #[tokio::test]
    async fn stage_node_sends_per_lease_heartbeat() {
        let (cfg, engine, stages) = nodo_testa("hb-per-lease", vec![("FAKE_STAGE_TOKEN_DELAY_MS".into(), "100".into())]).await;
        let Engine::Stages(e) = &*engine else { panic!("stage engine") };
        let req = |id: &str, text: &str| match offerta_chat(&cfg, id, text, 50).task {
            Some(work_offer::Task::Chat(c)) => c.request.unwrap(),
            _ => unreachable!(),
        };
        let (pa, pb) = (Arc::new(Progress::default()), Arc::new(Progress::default()));
        let mk = |id: &str, own: &Arc<Progress>| ActiveLease {
            id: id.into(),
            stage: false,
            progress: Arc::new(Mutex::new(LeaseCounters::default())),
            own: Some(own.clone()),
            confirm: None,
            task: tokio::spawn(std::future::pending::<()>()),
            guard: None,
        };
        let mut l = Leases::default();
        l.map.insert("LA".into(), mk("LA", &pa));
        l.map.insert("LB".into(), mk("LB", &pb));
        let ra = req("ra", "ciao");
        let rb = req("rb", "blocca questa");
        let (e1, e2) = (engine.clone(), engine.clone());
        let (pa2, pb2) = (pa.clone(), pb.clone());
        let a = tokio::spawn(async move { let Engine::Stages(e) = &*e1 else { unreachable!() }; e.chat_lease(&ra, Default::default(), &pa2, None, None).await });
        let (ctx, crx) = oneshot::channel();
        let b = tokio::spawn(async move { let Engine::Stages(e) = &*e2 else { unreachable!() }; e.chat_lease(&rb, Default::default(), &pb2, None, Some(crx)).await });
        let _ = e;
        let hb = |l: &Leases| {
            let h = heartbeat_msg(l, &cfg, &Shared::default());
            let get = |id: &str| h.leases.iter().find(|x| x.lease_id == id).cloned().unwrap();
            (get("LA"), get("LB"))
        };
        tokio::time::sleep(Duration::from_millis(250)).await;
        let (a1, b1) = hb(&l);
        assert_eq!(a1.phase, publicompute_proto::node::LeasePhase::Decode as i32);
        assert_eq!(b1.phase, publicompute_proto::node::LeasePhase::Prefill as i32);
        assert_eq!(b1.progress_seq, 1, "progress sequence starts at 1");
        a.await.unwrap().unwrap();
        let (a2, b2) = hb(&l);
        assert!(a2.progress_seq > a1.progress_seq, "first lease progresses: {} -> {}", a1.progress_seq, a2.progress_seq);
        assert_eq!(b2.progress_seq, b1.progress_seq, "second lease is idle");
        assert_eq!(b2.phase, publicompute_proto::node::LeasePhase::Prefill as i32);
        let _ = ctx.send(());
        let _ = b.await;
        stages.stop_all().await;
    }


    use publicompute_proto::{
        common::ProtocolVersion,
        node::{
            model_directive_status::State as StatoDirettiva, ContributionMode, ContributionPreference, LeaveOk, ModelRole,
            Welcome,
        },
    };

    fn modello_prova() -> ModelRef {
        ModelRef { model_id: "qwen-prova".into(), quantization: "Q4_K_M".into(), weights_sha256: vec![5; 32] }
    }

    /// Create a fixture with a persisted preference for its loaded weights.
    async fn nodo_con_modello(name: &str) -> (Fake, NodeConfig) {
        use crate::contribution::{PreferenceMode, ModelSelection, Preference};
        let (f, mut cfg) = node(400).await;
        cfg.model = modello_prova();
        let d = std::env::temp_dir().join(format!("publicompute-sess-{}-{name}-{}", std::process::id(), rand::random::<u32>()));
        std::fs::create_dir_all(&d).unwrap();
        let pref = Preference { modo: PreferenceMode::Usage, model: Some(ModelSelection { id: "qwen-prova".into(), quant: "Q4_K_M".into(), weights_sha256: None }), revision: 0 };
        cfg.contribution = Contribution::open(&d, pref).unwrap();
        (f, cfg)
    }

    fn welcome(minor: Option<u32>, min_agent: &str, heartbeat_ms: u32) -> Welcome {
        Welcome {
            session_id: "s_prova".into(),
            heartbeat_interval: prost_types::Duration::try_from(Duration::from_millis(u64::from(heartbeat_ms))).ok(),
            min_agent_version: if min_agent.is_empty() { env!("CARGO_PKG_VERSION") } else { min_agent }.into(),
            protocol_version: minor.map(|m| ProtocolVersion { major: 0, minor: m }),
            ..Default::default()
        }
    }

    /// Drive a session against a scripted coordinator.
    struct Finta {
        send: quinn::SendStream,
        r: FrameReader,
        hello: Hello,
        stop_tx: watch::Sender<bool>,
        task: tokio::task::JoinHandle<SessionEnd>,
        _server: quinn::Connection,
        seq: u64,
    }

    impl Finta {
        async fn start(cfg: NodeConfig, welcome: Welcome) -> (Finta, Arc<Shared>) {
            Self::start_with_window(cfg, welcome, None).await
        }

        async fn start_with_window(cfg: NodeConfig, welcome: Welcome, window: Option<u32>) -> (Finta, Arc<Shared>) {
            Self::start_reply(cfg, coordinator_message::Body::Welcome(welcome), window).await
        }

        async fn start_reply(cfg: NodeConfig, reply: coordinator_message::Body, window: Option<u32>) -> (Finta, Arc<Shared>) {
            let (client, server) = loopback_with_window(window).await;
            let engine = cfg.sup.engine.clone();
            let (stop_tx, stop_rx) = watch::channel(false);
            let shared = Arc::new(Shared::default());
            let shared2 = shared.clone();
            let task = tokio::spawn(async move { run(client, &cfg, engine, shared2, stop_rx).await });
            let (send, recv) = server.accept_bi().await.unwrap();
            let mut r = FrameReader::new(recv);
            let primo: NodeMessage = r.read().await.unwrap().expect("Hello");
            let Some(node_message::Body::Hello(hello)) = primo.body else { panic!("expected Hello, got {primo:?}") };
            let mut f = Finta { send, r, hello, stop_tx, task, _server: server, seq: 0 };
            f.manda(reply).await;
            (f, shared)
        }

        async fn manda(&mut self, b: coordinator_message::Body) {
            self.seq += 1;
            framed::write(&mut self.send, &CoordinatorMessage { seq: self.seq, body: Some(b) }).await.unwrap();
        }

        /// Read the next node message. Return None when the deadline expires.
        async fn read(&mut self, ms: u64) -> Option<node_message::Body> {
            match tokio::time::timeout(Duration::from_millis(ms), self.r.read::<NodeMessage>()).await {
                Ok(Ok(Some(m))) => m.body,
                _ => None,
            }
        }

        /// Skip unrelated messages until the selector accepts one.
        async fn attendi<T>(&mut self, ms: u64, mut f: impl FnMut(&node_message::Body) -> Option<T>) -> Option<T> {
            let done = Instant::now() + Duration::from_millis(ms);
            while Instant::now() < done {
                let resto = done.saturating_duration_since(Instant::now()).as_millis() as u64;
                let b = self.read(resto.max(1)).await?;
                if let Some(t) = f(&b) {
                    return Some(t);
                }
            }
            None
        }

        async fn done(self) -> SessionEnd {
            tokio::time::timeout(Duration::from_secs(5), self.task).await.expect("the session should end").unwrap()
        }

        async fn acknowledge_leaving(&mut self, remaining: u32) {
            let grace = self.attendi(3000, |msg| match msg {
                node_message::Body::Leaving(l) => Some(l.grazia_s),
                _ => None,
            }).await.expect("model switch must announce departure");
            assert_eq!(grace, GRAZIA_USCITA_S);
            self.manda(coordinator_message::Body::LeaveOk(LeaveOk { lease_interrotte: remaining })).await;
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn saved_bandwidth_stays_out_of_hello_on_legacy_reconnect() {
        let dir = tempdir("legacy-network");
        let live = crate::control::Live::open(&dir).unwrap();
        live.network.record_bandwidth(network::Bandwidth::Measured {
            down: 120_000_000, up: 40_000_000, complete: true,
        }, network::unix_seconds());
        for _ in 0..2 {
            let (_fake, mut cfg) = node(1000).await;
            cfg.live = live.clone();
            let (mut peer, _) = Finta::start(cfg, welcome(Some(PROTOCOL_VERSION.minor), "", 25)).await;
            assert!(peer.hello.hardware.as_ref().unwrap().network.is_none());
            let up = peer.attendi(2000, |body| match body {
                node_message::Body::Heartbeat(h) => h.network.as_ref().map(|n| n.upload_bits_per_sec),
                _ => None,
            }).await;
            assert_eq!(up, Some(40_000_000));
            peer.stop_tx.send(true).unwrap();
            peer.done().await;
        }
        drop(live);
        std::fs::remove_dir_all(dir).unwrap();
    }

    async fn dual_node(name: &str, stage_env: &[(&str, &str)], llama_args: &[&str]) -> (NodeConfig, std::path::PathBuf, u16) {
        let dir = tempdir(name);
        let config = cfg_test(&dir);
        let model_path = config.model_path.clone();
        let stages = Stages::new(StagesConfig {
            env: stage_env.iter().map(|(key, value)| (key.to_string(), value.to_string())).collect(),
            ..config
        });
        let progress = Arc::new(Progress::default());
        let limits = Limits {
            prefill_stall: Duration::from_secs(4),
            decode_stall: Duration::from_secs(4),
            request_max: Duration::from_secs(10),
            call_timeout: Duration::from_secs(3),
            step_timeout: Duration::from_secs(3),
            probe_timeout: Duration::from_secs(2),
        };
        let stage_engine = Arc::new(Engine::Stages(StadiEngine {
            stages: stages.clone(), attive: Default::default(), limits: limits.clone(), progress: Default::default(),
        }));
        let port = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
        let bin = std::env::current_exe().unwrap().parent().unwrap().parent().unwrap().join("fake-llama-server");
        let llama = Arc::new(Engine::Llama(LlamaEngine::new(format!("http://127.0.0.1:{port}"), limits, progress.clone(), true).unwrap()));
        let gpu = Arc::new(Telemetria::default());
        gpu.set_device_for_test(crate::gpu::Device::Ready("GPU-test".into()));
        let sup = Supervisor::with_state(
            llama,
            SupervisorCfg {
                managed: Some(crate::engine::managed::ManagedCfg {
                    bin, model_path: model_path.clone(), port, gpu_index: None,
                    extra_args: llama_args.iter().map(|arg| arg.to_string()).collect(),
                    log_path: dir.join("llama.log"), stop_grace: Duration::from_millis(600), kill_wait: Duration::from_secs(2),
                }),
                expected_model: Some(model_path.clone()),
                probe_interval: Duration::from_secs(60),
                settle: Duration::from_secs(2),
                max_restarts: 3,
                restart_window: Duration::from_secs(60),
                backoff_base: Duration::from_millis(50),
                backoff_max: Duration::from_millis(100),
                recycle: Some(Duration::from_secs(2)),
                start_timeout: Duration::from_secs(5),
                maintenance_tick: Duration::from_millis(30),
            },
            Arc::new(Stats::new(None)), progress, watch::channel(EngineState::Starting).0, Some(gpu.clone()),
        );
        sup.start().await.unwrap();
        let model = assignment("ad", 1, 1, 2, 0).model.unwrap();
        let mut hardware = publicompute_proto::node::HardwareProfile::default();
        hardware.gpus.push(Default::default());
        (NodeConfig {
            model, hardware, costs: None, sup, stages: Some(stages), stage_engine: Some(stage_engine), slots: 2,
            region: "EU".into(), nat_class: String::new(), iroh_node_id: String::new(),
            contribution: contributo_prova(name), gpu, model_path: Some(model_path), output: watch::channel(false).1,
            live: crate::control::Live::new(),
            model_store: None,
        }, dir, port)
    }

    impl Finta {
        async fn stage(&mut self, want: State) -> StageStatus {
            self.attendi(7000, |msg| match msg {
                node_message::Body::StageStatus(status) if status.state() == want => Some(status.clone()),
                _ => None,
            }).await.unwrap_or_else(|| panic!("missing stage status {want:?}"))
        }

        async fn accept_chat(&mut self, offer: WorkOffer) {
            let lease_id = offer.lease.as_ref().unwrap().lease_id.clone();
            self.manda(coordinator_message::Body::WorkOffer(offer)).await;
            self.attendi(3000, |msg| match msg {
                node_message::Body::LeaseAccept(accept) if accept.lease_id == lease_id => Some(()),
                node_message::Body::LeaseReject(reject) => panic!("chat rejected: {reject:?}"),
                _ => None,
            }).await.expect("chat offer accepted");
        }

        async fn update_lease(&mut self, status: lease_update::Status) {
            self.manda(coordinator_message::Body::LeaseUpdate(publicompute_proto::node::LeaseUpdate {
                lease_id: "L-chat".into(), status: Some(status),
            })).await;
        }

        async fn availability(&mut self, want: Availability) {
            self.attendi(5000, |msg| match msg {
                node_message::Body::Availability(update) if update.availability() == want => Some(()),
                _ => None,
            }).await.unwrap_or_else(|| panic!("missing availability {want:?}"));
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn dual_mode_switches_and_renews_an_active_chat() {
        let (cfg, dir, _) = dual_node("dual-chat", &[("FAKE_STAGE_TOKEN_DELAY_MS", "350")], &["--fake-start-delay-ms", "250"]).await;
        let initial = inventory(&cfg, &cfg.sup.engine).models.remove(0);
        let offer = offerta_chat(&cfg, "chat", "ciao", 8);
        let blocked = offerta_chat(&cfg, "blocked", "blocca", 8);
        let sup = cfg.sup.clone();
        let pid = sup.child_pid_for_test().await;
        let stages = cfg.stages.clone().unwrap();
        let (mut peer, _) = Finta::start(cfg, welcome(Some(PROTOCOL_VERSION.minor), "", 100)).await;
        assert_eq!(peer.hello.features, vec![AgentFeature::Contribution as i32, AgentFeature::Stages as i32, AgentFeature::StageEngineProfile as i32, AgentFeature::EngineSwitch as i32]);

        let mut a = assignment("ad", 1, 1, 2, 0);
        peer.accept_chat(offer.clone()).await;
        peer.manda(coordinator_message::Body::StageAssign(a.clone())).await;
        assert_eq!(peer.stage(State::Stopped).await.error, "busy");
        assert_eq!(pid, sup.child_pid_for_test().await);
        peer.manda(coordinator_message::Body::StageRelease(publicompute_proto::node::StageRelease {
            chain_id: a.chain_id.clone(), assign_epoch: a.assign_epoch, reason: "busy".into(),
        })).await;
        peer.availability(Availability::EngineRecovering).await;
        assert_eq!(sup.state(), EngineState::Busy);
        peer.update_lease(lease_update::Status::Revoked(LeaseRevokeReason::Cancelled as i32)).await;
        peer.availability(Availability::Idle).await;
        assert_eq!(sup.state(), EngineState::Ready);

        a.assign_epoch += 1;
        peer.manda(coordinator_message::Body::StageAssign(a.clone())).await;
        let installed = peer.attendi(5000, |msg| match msg {
            node_message::Body::ModelInventory(inv) if !inv.models[0].loaded => Some(inv.models[0].clone()),
            _ => None,
        }).await.expect("unloaded model in inventory");
        assert_eq!(installed, InstalledModel { loaded: false, ..initial.clone() });
        peer.stage(State::Ready).await;
        assert_eq!(sup.state(), EngineState::Parked);
        assert_eq!(unsafe { libc::kill(pid as i32, 0) }, -1, "llama-server still present");

        peer.accept_chat(offer.clone()).await;
        peer.update_lease(lease_update::Status::RenewedUntil(Timestamp::from(std::time::SystemTime::now()))).await;
        let chat = tokio::spawn(coordinatore(peer._server.clone(), "chat".into(), None));
        peer.stage(State::Running).await;
        a.capability = vec![7, 8, 9];
        peer.manda(coordinator_message::Body::StageAssign(a.clone())).await;
        peer.stage(State::Running).await;
        assert_eq!(std::fs::read(dir.join(format!("tls/capability-ad-{}.bin", a.assign_epoch))).unwrap(), a.capability);
        let (_, result) = tokio::time::timeout(Duration::from_secs(7), chat).await.unwrap().unwrap();
        let Some(request_result::Outcome::Completion(c)) = result.outcome else { panic!("{result:?}") };
        assert_eq!(c.text, "Ciao è un test");
        assert_eq!(sup.state(), EngineState::Parked);

        peer.accept_chat(blocked).await;
        peer.update_lease(lease_update::Status::RenewedUntil(Timestamp::from(std::time::SystemTime::now()))).await;
        peer.stage(State::Running).await;
        let (_commands, recv) = peer._server.accept_bi().await.unwrap();
        let mut blocked_stream = FrameReader::new(recv);
        assert!(matches!(blocked_stream.read::<WnMsg>().await.unwrap().unwrap().body, Some(wn::Body::Open(_))));
        tokio::time::sleep(Duration::from_millis(850)).await;
        assert_eq!(sup.stats.run().engine_restarts, 0, "parked engine must not recycle");
        assert_eq!(sup.stats.run().probes_ok, 1, "parked engine must not probe");
        peer.manda(coordinator_message::Body::StageRelease(publicompute_proto::node::StageRelease {
            chain_id: a.chain_id.clone(), assign_epoch: a.assign_epoch, reason: "test release".into(),
        })).await;
        let mut in_transit = offer.clone();
        in_transit.lease.as_mut().unwrap().lease_id = "in-transit".into();
        peer.manda(coordinator_message::Body::WorkOffer(in_transit)).await;
        peer.availability(Availability::EngineRecovering).await;
        let released = peer.attendi(3000, |msg| match msg {
            node_message::Body::LeaseRelease(release) => Some(release.clone()),
            _ => None,
        }).await.expect("chain lease released");
        assert_eq!(released.lease_id, "L-chat");
        assert!(blocked_stream.read::<WnMsg>().await.is_err(), "released chat stream should reset");
        let rejected = peer.attendi(3000, |msg| match msg {
            node_message::Body::LeaseReject(reject) => Some(reject.clone()),
            _ => None,
        }).await.expect("in-flight offer rejected");
        assert_eq!((rejected.lease_id.as_str(), rejected.reason()), ("in-transit", LeaseRejectReason::HostBusy));
        peer.stage(State::Stopped).await;
        assert!(stages.head().is_none());
        let installed = peer.attendi(5000, |msg| match msg {
            node_message::Body::ModelInventory(inv) if inv.models[0].loaded => Some(inv.models[0].clone()),
            _ => None,
        }).await.expect("loaded model in inventory");
        assert_eq!(installed, initial);
        peer.availability(Availability::Idle).await;
        assert_ne!(pid, sup.child_pid_for_test().await);
        assert_eq!(sup.stats.run().engine_restarts, 0);

        // An in-flight renewal of the closed epoch must leave Llama available.
        peer.manda(coordinator_message::Body::StageAssign(a)).await;
        peer.accept_chat(offer).await;
        peer.update_lease(lease_update::Status::RenewedUntil(Timestamp::from(std::time::SystemTime::now()))).await;
        let (_, result) = tokio::time::timeout(Duration::from_secs(7), coordinatore(peer._server.clone(), "chat".into(), None)).await.unwrap();
        let Some(request_result::Outcome::Completion(c)) = result.outcome else { panic!("{result:?}") };
        assert_ne!(c.text, "Ciao è un test");
        peer.stop_tx.send(true).unwrap();
        assert!(matches!(peer.done().await, SessionEnd::Shutdown));
        sup.shutdown().await;
    }

    #[tokio::test]
    async fn dual_mode_resumes_after_stage_failure() {
        let (cfg, _, _) = dual_node("dual-stage-failure", &[("FAKE_STAGE_EXIT_AFTER_MS", "500")], &[]).await;
        let sup = cfg.sup.clone();
        let pid = sup.child_pid_for_test().await;
        let (mut peer, _) = Finta::start(cfg, welcome(Some(PROTOCOL_VERSION.minor), "", 100)).await;
        peer.manda(coordinator_message::Body::StageAssign(assignment("ad", 1, 1, 2, 0))).await;
        peer.stage(State::Ready).await;
        peer.stage(State::Failed).await;
        peer.availability(Availability::EngineRecovering).await;
        peer.availability(Availability::Idle).await;
        assert_ne!(pid, sup.child_pid_for_test().await);
        peer.stop_tx.send(true).unwrap();
        assert!(matches!(peer.done().await, SessionEnd::Shutdown));
        sup.shutdown().await;
    }

    #[tokio::test]
    async fn dual_mode_stops_stage_on_gpu_loss_during_loading() {
        use crate::gpu::{Device, ReadFailure};
        let (cfg, _, port) = dual_node("dual-gpu-loss", &[("FAKE_STAGE_START_DELAY_MS", "2000")], &[]).await;
        let (gpu, sup) = (cfg.gpu.clone(), cfg.sup.clone());
        let (mut peer, _) = Finta::start(cfg, welcome(Some(PROTOCOL_VERSION.minor), "", 100)).await;
        peer.manda(coordinator_message::Body::StageAssign(assignment("ad", 1, 1, 2, 0))).await;
        peer.stage(State::Loading).await;
        gpu.set_device_for_test(Device::Lost(ReadFailure::NoDevice));
        peer.availability(Availability::PausedByHost).await;
        peer.stage(State::Stopped).await;
        assert_eq!(sup.state(), EngineState::Recovering);
        assert!(tokio::net::TcpStream::connect(("127.0.0.1", port)).await.is_err());
        assert_eq!(sup.stats.run().engine_restarts, 0);
        gpu.set_device_for_test(Device::Ready("GPU-test".into()));
        peer.availability(Availability::Idle).await;
        peer.stop_tx.send(true).unwrap();
        assert!(matches!(peer.done().await, SessionEnd::Shutdown));
        sup.shutdown().await;
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn dual_mode_control_eof_during_park_cancels_stage_start() {
        let (cfg, dir, _) = dual_node("dual-eof", &[], &["--fake-ignore-term"]).await;
        let sup = cfg.sup.clone();
        let pid = sup.child_pid_for_test().await;
        let (mut peer, _) = Finta::start(cfg, welcome(Some(PROTOCOL_VERSION.minor), "", 100)).await;
        peer.manda(coordinator_message::Body::StageAssign(assignment("ad", 1, 1, 2, 0))).await;
        let mut state = sup.subscribe();
        tokio::time::timeout(Duration::from_secs(5), state.wait_for(|s| *s == EngineState::Parked)).await.unwrap().unwrap();
        peer.send.finish().unwrap();
        assert!(matches!(peer.done().await, SessionEnd::Lost(_)));
        tokio::time::timeout(Duration::from_secs(7), state.wait_for(|s| *s == EngineState::Ready)).await.unwrap().unwrap();
        assert!(!dir.join("log/stage-ad-1.log").exists());
        assert_ne!(pid, sup.child_pid_for_test().await);
        sup.shutdown().await;
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn dual_mode_refuses_stage_if_single_exit_cannot_be_confirmed() {
        let (cfg, dir, _) = dual_node("dual-stop-error", &[], &[]).await;
        let stages = cfg.stages.as_ref().unwrap();
        let (tx, mut rx) = mpsc::channel(8);
        stages.connect_peer(tx);
        let pid = cfg.sup.child_pid_for_test().await as libc::pid_t;
        // Consume the exit before the stage worker calls wait on the child.
        assert_eq!(unsafe { libc::kill(pid, libc::SIGKILL) }, 0);
        assert_eq!(unsafe { libc::waitpid(pid, std::ptr::null_mut(), 0) }, pid);
        stages.queue_switch(assignment("ad", 1, 1, 2, 0), &cfg.sup, true).unwrap();
        let status = prossimo(&mut rx).await;
        assert_eq!(status.state(), State::Failed);
        assert!(status.error.contains("could not stop llama-server"));
        assert!(!dir.join("log/stage-ad-1.log").exists());
        assert_ne!(cfg.sup.state(), EngineState::Ready);
        cfg.sup.shutdown().await;
    }

    #[tokio::test]
    async fn dual_mode_does_not_use_another_server_on_the_llama_port() {
        let (cfg, _, port) = dual_node("dual-busy-port", &[], &[]).await;
        let sup = cfg.sup.clone();
        let same_model = cfg.model_path.as_ref().unwrap().to_str().unwrap().to_owned();
        let (mut peer, _) = Finta::start(cfg, welcome(Some(PROTOCOL_VERSION.minor), "", 100)).await;
        peer.manda(coordinator_message::Body::StageAssign(assignment("ad", 1, 1, 2, 0))).await;
        peer.stage(State::Ready).await;
        let foreign = fake_llama::spawn(port, FakeState::new(&same_model)).await.unwrap();
        peer.manda(coordinator_message::Body::StageRelease(publicompute_proto::node::StageRelease {
            chain_id: "ad".into(), assign_epoch: 1, reason: "test release".into(),
        })).await;
        peer.availability(Availability::EngineRecovering).await;
        peer.availability(Availability::EngineQuarantined).await;
        assert_eq!(foreign.state.completions.load(Ordering::SeqCst), 0);
        peer.stop_tx.send(true).unwrap();
        assert!(matches!(peer.done().await, SessionEnd::Shutdown));
        sup.shutdown().await;
    }

    #[tokio::test]
    async fn hello_carries_preference() {
        let (_f, cfg) = nodo_con_modello("hello").await;
        let rev = cfg.contribution.preference().revision;
        let (fin, _) = Finta::start(cfg, welcome(Some(PROTOCOL_VERSION.minor), "", 60_000)).await;
        let p: &ContributionPreference = fin.hello.contribution.as_ref().expect("preferenza nell'Hello");
        assert_eq!(p.mode(), ContributionMode::Use);
        assert_eq!(p.model.as_ref().unwrap(), &modello_prova(), "model with the inventory fingerprint");
        assert_eq!(p.revision, rev);
        assert_eq!(fin.hello.features, vec![AgentFeature::Contribution as i32]);
        assert_eq!(fin.hello.protocol_version.as_ref().unwrap().minor, PROTOCOL_VERSION.minor);
        let _ = fin.stop_tx.send(true);
    }

    #[tokio::test]
    async fn preference_change_is_sent() {
        let (_f, cfg) = nodo_con_modello("cambio").await;
        let contribution = cfg.contribution.clone();
        let (mut fin, _) = Finta::start(cfg, welcome(Some(PROTOCOL_VERSION.minor), "", 60_000)).await;
        let rev0 = fin.hello.contribution.as_ref().unwrap().revision;
        tokio::time::sleep(Duration::from_millis(50)).await;
        contribution.imposta(crate::contribution::PreferenceMode::Free, None).unwrap();
        let p = fin
            .attendi(2000, |b| match b {
                node_message::Body::Contribution(p) => Some(p.clone()),
                _ => None,
            })
            .await
            .expect("expected the preference update after the change");
        assert_eq!(p.mode(), ContributionMode::Free);
        assert!(p.model.is_none());
        assert!(p.revision > rev0);
        let _ = fin.stop_tx.send(true);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn hello_has_no_ready_weights_during_the_first_download() {
        use crate::contribution::PreferenceMode;
        use crate::models::{find, tests::fixture_store};
        let (_f, mut cfg) = nodo_con_modello("hello-without-weights").await;
        cfg.model = ModelRef::default();
        cfg.model_path = None;
        cfg.contribution.imposta(PreferenceMode::Free, None).unwrap();
        let (peer, _) = Finta::start(cfg, welcome(Some(PROTOCOL_VERSION.minor), "", 60_000)).await;
        assert!(peer.hello.models.as_ref().unwrap().models.is_empty());
        assert_eq!(peer.hello.contribution.as_ref().unwrap().mode(), ContributionMode::Free);
        peer.stop_tx.send(true).unwrap();
        assert!(matches!(peer.done().await, SessionEnd::Shutdown));

        let (_f, mut cfg) = nodo_con_modello("hello-first-download").await;
        cfg.model = ModelRef::default();
        cfg.model_path = None;
        let store = fixture_store("hello-first-download");
        let download = find("qwen3.6-35b-a3b", None).unwrap();
        std::fs::File::create(store.partial(download)).unwrap().set_len(download.bytes - 1).unwrap();
        let (sent, ready) = oneshot::channel();
        cfg.live.changes.request(&cfg.live, &store, download, Some(ready)).unwrap();
        let live = cfg.live.clone();
        let (peer, _) = Finta::start(cfg, welcome(Some(PROTOCOL_VERSION.minor), "", 60_000)).await;
        let models = &peer.hello.models.as_ref().unwrap().models;
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].model.as_ref(), Some(&download.reference()));
        assert_eq!(models[0].status(), ModelStatus::Downloading);
        assert!(!models[0].loaded);
        live.changes.cancel();
        let _ = sent.send(());
        peer.stop_tx.send(true).unwrap();
        assert!(matches!(peer.done().await, SessionEnd::Shutdown));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn hello_only_advertises_the_active_model_as_ready() {
        use crate::models::{find, tests::fixture_store};
        let (_f, cfg) = nodo_con_modello("hello-download").await;
        let store = fixture_store("hello-download");
        let download = find("qwen3.6-35b-a3b", None).unwrap();
        std::fs::File::create(store.partial(download)).unwrap().set_len(download.bytes - 1).unwrap();
        let (sent, ready) = oneshot::channel();
        cfg.live.changes.request(&cfg.live, &store, download, Some(ready)).unwrap();
        let active = cfg.model.clone();
        let live = cfg.live.clone();
        let (peer, _) = Finta::start(cfg, welcome(Some(PROTOCOL_VERSION.minor), "", 60_000)).await;
        let models = &peer.hello.models.as_ref().unwrap().models;
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].model.as_ref(), Some(&active));
        assert_eq!(models[0].status(), ModelStatus::Ready);
        assert_eq!(models[1].model.as_ref(), Some(&download.reference()));
        assert_eq!(models[1].status(), ModelStatus::Downloading);
        assert!(!models[1].loaded);
        live.changes.cancel();
        let _ = sent.send(());
        peer.stop_tx.send(true).unwrap();
        assert!(matches!(peer.done().await, SessionEnd::Shutdown));
    }

    #[cfg(unix)]
    async fn drain_for_model_switch(force: bool) {
        use crate::models::{find, tests::{cached_weights, fixture_store}};
        let name = if force { "switch-deadline" } else { "switch-drain" };
        let (_f, mut cfg) = nodo_con_modello(name).await;
        let store = fixture_store(name);
        let model = find("qwen3.6-35b-a3b", None).unwrap();
        cached_weights(&store, model);
        let mut new = offerta_chat(&cfg, "r2", "second chat", 4);
        new.lease.as_mut().unwrap().lease_id = "L-next".into();
        let first = offerta_chat(&cfg, "r1", "first chat", 4);
        let live = cfg.live.clone();
        let pref = cfg.contribution.clone();
        cfg.model_store = Some(store.clone());
        let (mut peer, _) = Finta::start(cfg, welcome(Some(PROTOCOL_VERSION.minor), "", 60_000)).await;
        peer.accept_chat(first).await;
        live.changes.request(&live, &store, model, None).unwrap();
        tokio::time::timeout(Duration::from_secs(3), live.changes.wait_switch()).await.unwrap();
        peer.acknowledge_leaving(1).await;
        peer.manda(coordinator_message::Body::WorkOffer(new)).await;
        let rejected = peer.attendi(3000, |msg| match msg {
            node_message::Body::LeaseReject(r) if r.lease_id == "L-next" => Some(r.clone()),
            _ => None,
        }).await.unwrap();
        assert_eq!(rejected.detail, "agent is changing models");
        assert!(store.active(|_, _| true).unwrap().is_none());
        assert!(!peer.task.is_finished());
        if force {
            live.changes.expire_switch();
            let released = peer.attendi(3000, |msg| match msg {
                node_message::Body::LeaseRelease(r) if r.lease_id == "L-chat" => Some(r.clone()),
                _ => None,
            }).await.unwrap();
            assert_eq!(released.reason(), LeaseReleaseReason::Shutdown);
        } else {
            peer.update_lease(lease_update::Status::Revoked(LeaseRevokeReason::Cancelled as i32)).await;
        }
        let SessionEnd::SwitchModel(change) = peer.done().await else { panic!("expected to leave for the model switch") };
        assert!(live.changes.commit(change, store.clone(), &pref).await);
        assert_eq!(store.active(|_, _| true).unwrap().unwrap().0.id, model.id);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn switch_waits_for_the_existing_lease_and_refuses_new_offers() {
        drain_for_model_switch(false).await;
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn switch_waits_for_a_chain_even_without_a_local_chat() {
        use crate::models::{find, tests::{cached_weights, fixture_store}};
        let (cfg, _, _) = nodo_testa("chain-model-switch", vec![]).await;
        let store = fixture_store("chain-model-switch-weights");
        let model = find("qwen3.6-35b-a3b", None).unwrap();
        cached_weights(&store, model);
        let live = cfg.live.clone();
        let (mut peer, _) = Finta::start(cfg, welcome(Some(PROTOCOL_VERSION.minor), "", 60_000)).await;
        peer.manda(coordinator_message::Body::MeasurementPing(Default::default())).await;
        peer.attendi(3000, |msg| matches!(msg, node_message::Body::MeasurementPong(_)).then_some(())).await.unwrap();
        live.changes.request(&live, &store, model, None).unwrap();
        tokio::time::timeout(Duration::from_secs(3), live.changes.wait_switch()).await.unwrap();
        peer.acknowledge_leaving(0).await;

        peer.manda(coordinator_message::Body::StageAssign(assignment("new", 1, 1, 2, 0))).await;
        assert_eq!(peer.stage(State::Stopped).await.error, "agent is changing models");
        assert!(!peer.task.is_finished());
        peer.manda(coordinator_message::Body::StageRelease(publicompute_proto::node::StageRelease {
            chain_id: "catena-x".into(), assign_epoch: 1, reason: "chain finished".into(),
        })).await;
        assert!(matches!(peer.done().await, SessionEnd::SwitchModel(_)));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn switch_deadline_reports_terminal_stage_before_goodbye() {
        use crate::models::{find, tests::{cached_weights, fixture_store}};
        let (cfg, _, _) = dual_node("chain-switch-terminal", &[], &[]).await;
        let supervisor = cfg.sup.clone();
        let store = fixture_store("chain-switch-terminal-weights");
        let model = find("qwen3.6-35b-a3b", None).unwrap();
        cached_weights(&store, model);
        let live = cfg.live.clone();
        let (mut peer, _) = Finta::start(cfg, welcome(Some(PROTOCOL_VERSION.minor), "", 60_000)).await;
        peer.manda(coordinator_message::Body::StageAssign(assignment("ad", 1, 1, 2, 0))).await;
        peer.stage(State::Ready).await;
        live.changes.request(&live, &store, model, None).unwrap();
        peer.acknowledge_leaving(0).await;
        assert!(peer.read(100).await.is_none_or(|msg| !matches!(msg, node_message::Body::Goodbye(_))));
        assert!(!peer.task.is_finished());
        live.changes.expire_switch();
        let terminal_first = peer.attendi(3000, |msg| match msg {
            node_message::Body::StageStatus(s) if s.chain_id == "ad" && s.state() == State::Stopped => Some(true),
            node_message::Body::Goodbye(_) => Some(false),
            _ => None,
        }).await;
        assert_eq!(terminal_first, Some(true));
        peer.attendi(3000, |msg| matches!(msg, node_message::Body::Goodbye(_)).then_some(())).await.expect("Goodbye");
        assert!(matches!(peer.done().await, SessionEnd::SwitchModel(_)));
        supervisor.shutdown().await;
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn switch_deadline_interrupts_control_send_when_peer_does_not_read() {
        use crate::models::{find, tests::{cached_weights, fixture_store}};
        let (_fake, cfg) = nodo_con_modello("switch-backpressure").await;
        let first = offerta_chat(&cfg, "r1", "first chat", 4);
        let store = fixture_store("switch-backpressure");
        let model = find("qwen3.6-35b-a3b", None).unwrap();
        cached_weights(&store, model);
        let live = cfg.live.clone();
        let (mut peer, _) = Finta::start_with_window(cfg, welcome(Some(PROTOCOL_VERSION.minor), "", 60_000), Some(1024)).await;
        peer.accept_chat(first).await;
        live.changes.request(&live, &store, model, None).unwrap();
        peer.acknowledge_leaving(1).await;
        // Pongs overflow the receive window and the bounded writer queue.
        for _ in 0..160 {
            peer.manda(coordinator_message::Body::MeasurementPing(publicompute_proto::node::MeasurementPing {
                nonce: vec![1; 256], ..Default::default()
            })).await;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert!(!peer.task.is_finished());
        live.changes.expire_switch();
        let end = tokio::time::timeout(CLOSE_IO_TIMEOUT + Duration::from_secs(1), &mut peer.task).await.unwrap().unwrap();
        assert!(matches!(end, SessionEnd::SwitchModel(_)));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn switch_releases_existing_work_when_the_deadline_expires() {
        drain_for_model_switch(true).await;
    }

    #[tokio::test]
    async fn rejects_welcome_without_current_protocol() {
        let (_f, cfg) = nodo_con_modello("protocol").await;
        let (fin, _) = Finta::start(cfg, welcome(Some(2), "", 60_000)).await;
        assert!(matches!(fin.done().await, SessionEnd::IncompatibleRelease(error) if error.0.contains("unsupported protocol version")));
    }

    #[tokio::test]
    async fn shows_update_message_from_coordinator() {
        use publicompute_proto::common::{Error, ErrorCode};

        let (_f, cfg) = nodo_con_modello("reject-release").await;
        let message = "the coordinator requires agent version 0.1.7";
        let error = Error { code: ErrorCode::AgentTooOld as i32, message: message.into(), ..Default::default() };
        let (fin, _) = Finta::start_reply(cfg, coordinator_message::Body::Error(error), None).await;
        assert!(matches!(fin.done().await, SessionEnd::IncompatibleRelease(error) if error.0 == message));
    }

    #[tokio::test]
    async fn session_keeps_received_status() {
        let (_f, cfg) = nodo_con_modello("stato").await;
        let contribution = cfg.contribution.clone();
        let rev = contribution.preference().revision;
        let (mut fin, _) = Finta::start(cfg, welcome(Some(PROTOCOL_VERSION.minor), "", 60_000)).await;
        fin.manda(coordinator_message::Body::ContributionStatus(ContributionStatus { free_seat: true, in_chain: true, revision: rev, ..Default::default() })).await;
        let t0 = Instant::now();
        while contribution.state().is_none() {
            assert!(t0.elapsed() < Duration::from_secs(3), "contribution status was not saved");
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert!(contribution.state().unwrap().free_seat);
        let _ = fin.stop_tx.send(true);
    }

    #[tokio::test]
    async fn session_keeps_received_summary() {
        let (_f, cfg) = nodo_con_modello("riepilogo").await;
        let contribution = cfg.contribution.clone();
        let (mut fin, _) = Finta::start(cfg, welcome(Some(PROTOCOL_VERSION.minor), "", 60_000)).await;
        fin.manda(coordinator_message::Body::ContributionSummary(ContributionSummary { days: 30, balance_mc: 777, ..Default::default() })).await;
        let t0 = Instant::now();
        while contribution.summary().is_none() {
            assert!(t0.elapsed() < Duration::from_secs(3), "contribution summary was not saved");
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert_eq!(contribution.summary().unwrap().balance_mc, 777);
        let _ = fin.stop_tx.send(true);
    }

    #[tokio::test]
    async fn announced_shutdown_sends_leaving_and_waits_for_ack() {
        let (_f, mut cfg) = nodo_con_modello("uscita").await;
        let (uscita_tx, uscita_rx) = watch::channel(false);
        cfg.output = uscita_rx;
        let (mut fin, _) = Finta::start(cfg, welcome(Some(PROTOCOL_VERSION.minor), "", 60_000)).await;
        uscita_tx.send(true).unwrap();
        let l = fin
            .attendi(2000, |b| match b {
                node_message::Body::Leaving(l) => Some(l.clone()),
                _ => None,
            })
            .await
            .expect("expected Leaving after the first signal");
        // A graceful exit leaves 300 s for the answers in flight to finish.
        assert_eq!(l.grazia_s, GRAZIA_USCITA_S);
        // Wait for acknowledgment before sending the final goodbye.
        let saluto = fin.attendi(400, |b| matches!(b, node_message::Body::Goodbye(_)).then_some(())).await;
        assert!(saluto.is_none(), "Goodbye arrived before LeaveOk");
        fin.manda(coordinator_message::Body::LeaveOk(LeaveOk { lease_interrotte: 0 })).await;
        let saluto = fin.attendi(2000, |b| matches!(b, node_message::Body::Goodbye(_)).then_some(())).await;
        assert!(saluto.is_some(), "expected Goodbye after LeaveOk");
        assert!(matches!(fin.done().await, SessionEnd::Shutdown));
    }

    #[tokio::test]
    async fn second_signal_exits_immediately() {
        let (_f, mut cfg) = nodo_con_modello("secondo").await;
        let (uscita_tx, uscita_rx) = watch::channel(false);
        cfg.output = uscita_rx;
        let (mut fin, _) = Finta::start(cfg, welcome(Some(PROTOCOL_VERSION.minor), "", 60_000)).await;
        uscita_tx.send(true).unwrap();
        fin.attendi(2000, |b| matches!(b, node_message::Body::Leaving(_)).then_some(())).await.expect("Leaving");
        // A second shutdown signal ends the wait for acknowledgment.
        let t0 = Instant::now();
        fin.stop_tx.send(true).unwrap();
        let saluto = fin.attendi(1500, |b| matches!(b, node_message::Body::Goodbye(_)).then_some(())).await;
        assert!(saluto.is_some(), "expected Goodbye right after the second signal");
        assert!(t0.elapsed() < Duration::from_secs(1));
        assert!(matches!(fin.done().await, SessionEnd::Shutdown));
    }

    #[tokio::test]
    async fn minimum_version_stops_agent() {
        let (_f, cfg) = nodo_con_modello("minima").await;
        let (fin, _) = Finta::start(cfg, welcome(Some(PROTOCOL_VERSION.minor), "99.0.0", 60_000)).await;
        match fin.done().await {
            SessionEnd::IncompatibleRelease(error) => assert_eq!(error.0, "the coordinator requires agent version 99.0.0"),
            altro => panic!("expected an agent version error, got {altro:?}"),
        }
    }

    #[test]
    fn departure_notice_uses_generic_text_for_non_head_stages() {
        assert_eq!(GRAZIA_USCITA_S, 300);
        assert_eq!(ATTESA_USCITA, Duration::from_secs(305));
        assert_eq!(
            departure_notice(Some(crate::stages::Role::Head), true, 3),
            "waiting for 3 answers to finish, Ctrl+C again to quit now"
        );
        assert_eq!(
            departure_notice(Some(crate::stages::Role::Head), true, 1),
            "waiting for 1 answer to finish, Ctrl+C again to quit now"
        );
        assert_eq!(
            departure_notice(Some(crate::stages::Role::Middle), true, 0),
            "waiting for the chain to finish its answers, Ctrl+C again to quit now"
        );
        assert_eq!(
            departure_notice(Some(crate::stages::Role::Tail), true, 0),
            "waiting for the chain to finish its answers, Ctrl+C again to quit now"
        );
    }

    #[tokio::test]
    async fn directive_for_loaded_model_reports_serving() {
        let (_f, cfg) = nodo_con_modello("dir1").await;
        let shared = Shared::default();
        let d = ModelDirective { model: Some(modello_prova()), epoch: 1, reason: "test".into(), ..Default::default() };
        let st = rispondi_direttiva(&cfg, &shared, d);
        assert_eq!(st.state(), StatoDirettiva::Serving);
        assert_eq!(st.epoch, 1);
        let (mut fin, _) = Finta::start(cfg, welcome(Some(PROTOCOL_VERSION.minor), "", 60_000)).await;
        fin.manda(coordinator_message::Body::ModelDirective(ModelDirective { model: Some(modello_prova()), epoch: 7, ..Default::default() })).await;
        let st = fin
            .attendi(2000, |b| match b {
                node_message::Body::ModelDirectiveStatus(s) => Some(s.clone()),
                _ => None,
            })
            .await
            .expect("ModelDirectiveStatus");
        assert_eq!((st.state(), st.epoch), (StatoDirettiva::Serving, 7));
        let _ = fin.stop_tx.send(true);
    }

    #[tokio::test]
    async fn directive_for_other_model_reports_unsupported() {
        let (_f, cfg) = nodo_con_modello("dir2").await;
        let shared = Shared::default();
        let altro = ModelRef { model_id: "gemma-26b".into(), quantization: "IQ4_XS".into(), weights_sha256: vec![9; 32] };
        let st = rispondi_direttiva(&cfg, &shared, ModelDirective { model: Some(altro), epoch: 3, ..Default::default() });
        assert_eq!((st.state(), st.reject()), (StatoDirettiva::Rejected, RejectReason::Unsupported));
        // Matching weights alone cannot satisfy a request to become a standby stage.
        let st = rispondi_direttiva(
            &cfg,
            &shared,
            ModelDirective { model: Some(modello_prova()), epoch: 4, role: ModelRole::StageStandby as i32, ..Default::default() },
        );
        assert_eq!((st.state(), st.reject()), (StatoDirettiva::Rejected, RejectReason::Unsupported));
    }

    #[tokio::test]
    async fn ignores_directive_from_old_epoch() {
        let (_f, cfg) = nodo_con_modello("dir3").await;
        let shared = Shared::default();
        let d = |epoch| ModelDirective { model: Some(modello_prova()), epoch, ..Default::default() };
        assert_eq!(rispondi_direttiva(&cfg, &shared, d(5)).state(), StatoDirettiva::Serving);
        for vecchia in [5, 4, 1] {
            let st = rispondi_direttiva(&cfg, &shared, d(vecchia));
            assert_eq!((st.state(), st.reject()), (StatoDirettiva::Rejected, RejectReason::StaleEpoch), "epoca {vecchia}");
        }
        assert_eq!(rispondi_direttiva(&cfg, &shared, d(6)).state(), StatoDirettiva::Serving);
    }

    #[tokio::test]
    async fn inventory_has_actual_sizes() {
        let (_f, mut cfg) = nodo_con_modello("inv").await;
        let file = std::env::temp_dir().join(format!("publicompute-inv-{}-{}.gguf", std::process::id(), rand::random::<u32>()));
        std::fs::write(&file, vec![0u8; 12_345]).unwrap();
        cfg.model_path = Some(file.clone());
        assert_eq!(inventory(&cfg, &cfg.sup.engine).models[0].size_bytes, 12_345);
        let (mut fin, _) = Finta::start(cfg, welcome(Some(PROTOCOL_VERSION.minor), "", 100)).await;
        let m = &fin.hello.models.as_ref().unwrap().models[0];
        assert_eq!((m.size_bytes, m.loaded), (12_345, true), "dimensione vera nell'Hello");
        // Polling resends the inventory only when its file size changes.
        std::fs::write(&file, vec![0u8; 20_000]).unwrap();
        let inv = fin
            .attendi(3000, |b| match b {
                node_message::Body::ModelInventory(i) => Some(i.clone()),
                _ => None,
            })
            .await
            .expect("inventario rimandato");
        assert_eq!(inv.models[0].size_bytes, 20_000);
        let altro = fin.attendi(500, |b| matches!(b, node_message::Body::ModelInventory(_)).then_some(())).await;
        assert!(altro.is_none(), "unchanged inventory should not be sent again");
        let _ = fin.stop_tx.send(true);
        let _ = std::fs::remove_file(&file);
    }

    #[test]
    fn gpu_back_keeps_manual_pause() {
        let live = crate::control::Live::new();
        let gpu = Telemetria::default();
        gpu.set_device_for_test(crate::gpu::Device::Lost(crate::gpu::ReadFailure::Timeout));
        assert_eq!(availability_now(EngineState::Ready, &live, &gpu), Availability::PausedByHost);
        live.set_paused(true);
        gpu.set_device_for_test(crate::gpu::Device::Ready("GPU-test".into()));
        assert!(live.is_paused());
        assert_eq!(availability_now(EngineState::Ready, &live, &gpu), Availability::PausedByHost);
        live.set_paused(false);
        assert_eq!(availability_now(EngineState::Ready, &live, &gpu), Availability::Idle);
    }

    #[tokio::test]
    async fn gpu_loss_releases_lease_even_when_device_is_already_back() {
        use crate::gpu::{Device, ReadFailure};

        let gpu = Telemetria::default();
        let live = crate::control::Live::new();
        live.set_leases(1);
        gpu.set_device_for_test(Device::Ready("GPU-test".into()));
        let mut rx = gpu.device_rx();
        let mut seen = rx.borrow_and_update().loss_generation;
        let task = tokio::spawn(std::future::pending());
        let abort = task.abort_handle();
        let mut leases = single_lease(ActiveLease {
            id: "L1".into(),
            stage: false,
            progress: Default::default(),
            own: None,
            confirm: None,
            task,
            guard: None,
        });
        gpu.set_device_for_test(Device::Lost(ReadFailure::NoDevice));
        gpu.set_device_for_test(Device::Ready("GPU-test".into()));

        rx.changed().await.unwrap();
        let generation = rx.borrow_and_update().loss_generation;
        assert!(gpu.loss().is_none());
        let released = gpu_loss_leases(generation, &mut seen, &mut leases, &live);
        assert!(leases.is_empty());
        tokio::time::timeout(Duration::from_secs(1), async {
            while !abort.is_finished() {
                tokio::task::yield_now().await;
            }
        }).await.unwrap();

        let (tx, mut messages) = mpsc::channel(1);
        send_releases(&Out(tx), released, LeaseReleaseReason::EngineFailure).await.unwrap();
        assert!(matches!(messages.try_recv().unwrap(), node_message::Body::LeaseRelease(release) if
            release.lease_id == "L1" && release.reason() == LeaseReleaseReason::EngineFailure));
        assert!(gpu_loss_leases(generation, &mut seen, &mut leases, &live).is_empty());
    }

    #[tokio::test]
    async fn heartbeat_carries_gpu_telemetry() {
        let (_f, cfg) = nodo_con_modello("tele").await;
        assert!(heartbeat_msg(&Leases::default(), &cfg, &Shared::default()).telemetry.unwrap().gpus.is_empty(), "CPU-only node should report no GPU");
        cfg.gpu.imposta(&crate::gpu::interpreta("RTX Test, 8192, 2048, 55, 63, 120, 555.42"));
        let g = heartbeat_msg(&Leases::default(), &cfg, &Shared::default()).telemetry.unwrap().gpus;
        assert_eq!(g.len(), 1);
        assert_eq!((g[0].utilization_percent, g[0].temperature_celsius, g[0].vram_used_bytes), (55, 63, 2048 << 20));
    }

    /// Settling after cancellation keeps the coordinator-facing availability at
    /// IDLE because the engine is clearing an abandoned request.
    #[tokio::test]
    async fn cancellation_preserves_availability() {
        let (f, cfg) = node(400).await;
        let mut l = single_lease(active(&cfg, "L1"));
        f.state.busy.store(true, Ordering::SeqCst);
        let (tx, mut rx) = mpsc::channel(8);
        let out = Out(tx);
        let mut last = availability_now(cfg.sup.state(), &cfg.live, &cfg.gpu);
        assert_eq!(last, Availability::Idle);
        on_revoked(&cfg, &mut l, "L1", LeaseRevokeReason::Cancelled as i32);
        assert_eq!(cfg.sup.state(), EngineState::Settling);
        sync_availability(&out, &cfg.sup, &cfg.live, &cfg.gpu, &mut last).await.unwrap();
        assert!(drain(&mut rx).is_empty(), "no AvailabilityUpdate while settling");
        f.state.busy.store(false, Ordering::SeqCst);
        wait_ready(&cfg).await;
        sync_availability(&out, &cfg.sup, &cfg.live, &cfg.gpu, &mut last).await.unwrap();
        assert!(drain(&mut rx).is_empty(), "no AvailabilityUpdate even after the engine is ready");
        assert_eq!(cfg.sup.stats.run().engine_restarts, 0);
    }

    /// A local pause must be advertised to the coordinator and cleared on resume.
    #[tokio::test]
    async fn pause_announces_paused_by_host() {
        let (_f, cfg) = node(400).await;
        let (tx, mut rx) = mpsc::channel(8);
        let out = Out(tx);
        let mut last = availability_now(cfg.sup.state(), &cfg.live, &cfg.gpu);
        cfg.live.set_paused(true);
        sync_availability(&out, &cfg.sup, &cfg.live, &cfg.gpu, &mut last).await.unwrap();
        assert!(matches!(drain(&mut rx).as_slice(), [node_message::Body::Availability(a)] if a.availability == Availability::PausedByHost as i32));
        cfg.live.set_paused(false);
        sync_availability(&out, &cfg.sup, &cfg.live, &cfg.gpu, &mut last).await.unwrap();
        assert!(matches!(drain(&mut rx).as_slice(), [node_message::Body::Availability(a)] if a.availability == Availability::Idle as i32));
    }

    #[tokio::test]
    async fn paused_session_rejects_a_stage_assignment_in_transit() {
        let (cfg, _, stages) = nodo_testa("paused-stage-assign", vec![]).await;
        stages.stop_all().await;
        cfg.live.set_paused(true);
        let (mut coordinator, _) = Finta::start(cfg, welcome(Some(PROTOCOL_VERSION.minor), "", 60_000)).await;
        coordinator.manda(coordinator_message::Body::StageAssign(assignment("paused-chain", 2, 1, 2, 0))).await;
        let status = coordinator.attendi(2000, |body| match body {
            node_message::Body::StageStatus(status) => Some(status.clone()),
            _ => None,
        }).await.unwrap();
        assert_eq!((status.chain_id.as_str(), status.assign_epoch, status.state()), ("paused-chain", 2, State::Failed));
        assert!(status.error.contains("paused"));
        assert!(stages.head().is_none());
    }
}
