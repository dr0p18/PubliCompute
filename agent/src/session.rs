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
        coordinator_message, lease_update, node_message, work_offer, work_stream_coordinator_message, work_stream_node_message, request_result,
        AgentFeature, Availability, AvailabilityUpdate, Completion, ContributionStatus, ContributionSummary, CoordinatorMessage,
        GoodbyeReason, Heartbeat, Hello, InstalledModel, LeaseAccept, Leaving, ModelDirective, ModelDirectiveStatus, RejectReason,
        LeaseComplete, LeaseProgress, LeaseReject, LeaseRejectReason, LeaseRelease, LeaseReleaseReason, LeaseRevokeReason, LogicCommitment,
        MeasurementPong, ModelInventory, ModelStatus, NodeMessage, PositionLogprobs, RequestFailure, RequestFailureCode, RequestResult,
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
        progress::{Phase, Progress},
        supervisor::{EngineState, LeaseGuard, LeaseOutcome, Supervisor},
        Engine, ErrorClass,
    },
    framed::{self, FrameReader},
    gpu::Telemetria,
    logic,
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
}

impl NodeConfig {
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
pub const GRAZIA_USCITA_S: u32 = 60;
pub const ATTESA_USCITA: Duration = Duration::from_secs(65);

/// Return the file size in bytes, or zero when it cannot be measured.
fn dimensione_modello(path: &Option<std::path::PathBuf>) -> u64 {
    path.as_ref().and_then(|p| std::fs::metadata(p).ok()).map(|m| m.len()).unwrap_or(0)
}

/// Build the weights inventory with the measured file size and slot count.
fn inventory(cfg: &NodeConfig, engine: &Engine) -> ModelInventory {
    ModelInventory {
        models: vec![InstalledModel {
            model: Some(cfg.model.clone()),
            status: ModelStatus::Ready as i32,
            loaded: true,
            size_bytes: dimensione_modello(&cfg.model_path),
            slots: capacita(cfg, engine) as u32,
            downloaded_bytes: 0,
        }],
    }
}

#[derive(Debug)]
pub enum SessionEnd {
    Shutdown,
    GpuReady,
    /// The coordinator rejects the identity, so the agent must enroll again.
    Unauthenticated,
    Banned,
    /// The coordinator requires a higher agent release. Reconnecting cannot satisfy the requirement.
    TroppoVecchio(String),
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
    result: Result<(u64, Duration)>,
}

pub async fn run(conn: quinn::Connection, cfg: &NodeConfig, engine: Arc<Engine>, shared: Arc<Shared>, mut shutdown: watch::Receiver<bool>) -> SessionEnd {
    match run_inner(&conn, cfg, engine, shared, &mut shutdown).await {
        Ok(end) => end,
        Err(e) if e.downcast_ref::<VersioneTroppoVecchia>().is_some() => {
            SessionEnd::TroppoVecchio(e.downcast_ref::<VersioneTroppoVecchia>().map(|v| v.to_string()).unwrap_or_default())
        }
        Err(e) => match conn.close_reason().as_ref().and_then(transport::app_close_code) {
            Some(transport::CLOSE_UNAUTHENTICATED) => SessionEnd::Unauthenticated,
            Some(transport::CLOSE_BANNED) => SessionEnd::Banned,
            Some(transport::CLOSE_REPLACED) => {
                SessionEnd::Lost(anyhow::anyhow!("session replaced; another agent may be using this identity"))
            }
            _ => SessionEnd::Lost(e),
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
    let writer = tokio::spawn(async move {
        let mut send = send;
        let mut seq = 0u64;
        while let Some(body) = out_rx.recv().await {
            seq += 1;
            if framed::write(&mut send, &NodeMessage { seq, body: Some(body) }).await.is_err() {
                break;
            }
        }
        let _ = send.finish();
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
    let mut last_avail = if cfg.gpu_profile_missing() {
        Availability::PausedByHost
    } else {
        availability_now(cfg.sup.state(), &cfg.live, &cfg.gpu)
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
        features: vec![AgentFeature::Contribution as i32],
    }))
    .await?;

    let welcome = match in_rx.recv().await.and_then(|m| m.body) {
        Some(coordinator_message::Body::Welcome(w)) => w,
        Some(coordinator_message::Body::Error(e)) => bail!("coordinator rejected the session: {} ({:?})", e.message, e.code()),
        _ => bail!("expected Welcome"),
    };
    let hb_every = welcome
        .heartbeat_interval
        .and_then(|d| Duration::try_from(d).ok())
        .filter(|d| !d.is_zero())
        .unwrap_or(Duration::from_secs(10));
    // Exit on an unsupported minimum release because reconnecting would repeat the rejection.
    if !versione_sufficiente(env!("CARGO_PKG_VERSION"), &welcome.min_agent_version) {
        return Err(anyhow::Error::new(VersioneTroppoVecchia { minima: welcome.min_agent_version.clone() }));
    }
    // Gate control message bodies on the negotiated protocol minor. A peer
    // that does not recognize these bodies can close the connection.
    let minor2 = welcome.protocol_version.as_ref().is_some_and(|v| v.major == PROTOCOL_VERSION.major && v.minor >= 2);
    info!(session = %welcome.session_id, tier = ?welcome.tier(), protocollo_minor2 = minor2, "sessione aperta");
    cfg.live.set_connected(true);
    if !minor2 {
        info!("coordinator accepts preferences only in Hello; free placement is unavailable");
    }
    if let Some(s) = &cfg.stages {
        // Use the handshake's signing key to verify capabilities for stage assignments.
        if let Err(e) = s.set_capability_key(&welcome.capability_pub) {
            warn!("could not save the capability signing key: {e:#}");
        }
        // Correct for clock skew when stages check capability expiry.
        let ora_locale = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0);
        let scarto = crate::stages::clock_offset_from_welcome(welcome.server_time.as_ref(), ora_locale);
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

    let result = loop {
        let generation = gpu_rx.borrow_and_update().loss_generation;
        let released = gpu_loss_leases(generation, &mut loss_generation, &mut lease, &cfg.live);
        if cfg.gpu_profile_missing() {
            if cfg.sup.state() == EngineState::Ready {
                // Reconnect to refresh the coordinator inventory because the hardware profile is sent in Hello.
                break Ok(SessionEnd::GpuReady);
            }
        } else {
            sync_availability(&out, &cfg.sup, &cfg.live, &cfg.gpu, &mut last_avail).await?;
        }
        send_releases(&out, released, LeaseReleaseReason::EngineFailure).await?;
        let scadenza_uscita = output
            .map(|t| tokio::time::Instant::from_std(t + ATTESA_USCITA))
            .unwrap_or_else(|| tokio::time::Instant::now() + Duration::from_secs(3600));
        tokio::select! {
            m = in_rx.recv() => {
                let Some(m) = m else { break Err(anyhow::anyhow!("control stream closed")) };
                match m.body {
                    Some(coordinator_message::Body::WorkOffer(offer)) => {
                        handle_offer(conn, cfg, &engine, &shared, &out, &done_tx, &mut lease, draining || output.is_some(), offer).await?;
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
                        Some(s) => s.queue_assignment(a),
                        None => {
                            warn!(chain = %a.chain_id, "StageAssign requires --engine stages");
                            out.send(node_message::Body::StageStatus(StageStatus {
                                chain_id: a.chain_id,
                                assign_epoch: a.assign_epoch,
                                state: publicompute_proto::node::stage_status::State::Failed as i32,
                                error: "start the agent with --engine stages to run stages".into(),
                                ..Default::default()
                            }))
                            .await?;
                        }
                    },
                    Some(coordinator_message::Body::StageRelease(r)) => {
                        if let Some(s) = &cfg.stages {
                            s.queue_release(r);
                        }
                    }
                    Some(coordinator_message::Body::MeasurementPing(p)) => {
                        out.send(node_message::Body::MeasurementPong(MeasurementPong { probe_id: p.probe_id, nonce: p.nonce })).await?;
                    }
                    Some(coordinator_message::Body::Drain(d)) => {
                        info!(reason = %d.reason, "coordinator requested shutdown");
                        draining = true;
                        if lease.is_empty() {
                            out.send(goodbye(GoodbyeReason::HostStopped)).await?;
                            break Ok(SessionEnd::Shutdown);
                        }
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
                        ferma_subito(&out, &mut lease).await?;
                        break Ok(SessionEnd::Shutdown);
                    }
                    Some(other) => debug!(?other, "unhandled message"),
                    None => {}
                }
            }
            _ = heartbeat.tick() => {
                out.send(node_message::Body::Heartbeat(heartbeat_msg(&lease, cfg, &shared))).await?;
                // Publish a new inventory when measured size or capacity changes.
                let inv = inventory(cfg, &engine);
                if inv != ultimo_inventario {
                    out.send(node_message::Body::ModelInventory(inv.clone())).await?;
                    ultimo_inventario = inv;
                }
            }
            _ = pref_rx.changed() => {
                if minor2 {
                    let p = cfg.contribution.to_proto(&modelli_di(&ultimo_inventario));
                    info!(modo = ?p.mode(), revision = p.revision, "contribution preference changed");
                    out.send(node_message::Body::Contribution(p)).await?;
                } else {
                    debug!("coordinator accepts preferences only in Hello; skipping live update");
                }
            }
            r = uscita_rx.changed(), if !uscita_chiusa && output.is_none() => {
                match r {
                    Err(_) => uscita_chiusa = true,
                    Ok(()) if !*uscita_rx.borrow() => {}
                    Ok(()) if minor2 => {
                        warn!("shutting down within {GRAZIA_USCITA_S} s; press Ctrl-C again to exit now");
                        output = Some(Instant::now());
                        conto.reset();
                        out.send(node_message::Body::Leaving(Leaving { grazia_s: GRAZIA_USCITA_S })).await?;
                    }
                    Ok(()) => {
                        // If the peer cannot acknowledge departure, take the immediate shutdown path.
                        ferma_subito(&out, &mut lease).await?;
                        break Ok(SessionEnd::Shutdown);
                    }
                }
            }
            _ = conto.tick(), if output.is_some() => {
                let trascorsi = output.map(|t| t.elapsed()).unwrap_or_default();
                let rimasti = ATTESA_USCITA.saturating_sub(trascorsi).as_secs().min(u64::from(GRAZIA_USCITA_S));
                info!(seconds = rimasti, lease = lease.map.len(), "shutting down in {rimasti} s");
            }
            _ = tokio::time::sleep_until(scadenza_uscita), if output.is_some() => {
                warn!("departure acknowledgment timed out; exiting");
                ferma_subito(&out, &mut lease).await?;
                break Ok(SessionEnd::Shutdown);
            }
            _ = state_rx.changed() => {}
            _ = pause_rx.changed() => {}
            _ = gpu_rx.changed() => {}
            Some(st) = stage_rx.recv(), if cfg.stages.is_some() => {
                let attivo = !matches!(st.state(), publicompute_proto::node::stage_status::State::Stopped | publicompute_proto::node::stage_status::State::Failed);
                cfg.live.set_chain(attivo.then(|| st.chain_id.clone()));
                out.send(node_message::Body::StageStatus(st)).await?;
            }
            Some(done) = done_rx.recv() => {
                on_lease_done(cfg, &shared, &out, &mut last_avail, &mut lease, done).await?;
                cfg.live.set_leases(lease.attive() as u32);
                if draining {
                    out.send(goodbye(GoodbyeReason::HostStopped)).await?;
                    break Ok(SessionEnd::Shutdown);
                }
            }
            _ = shutdown.changed() => {
                ferma_subito(&out, &mut lease).await?;
                break Ok(SessionEnd::Shutdown);
            }
        }
    };
    cfg.live.set_connected(false);
    cfg.live.set_leases(0);
    cfg.live.set_chain(None);
    // Dropping the guard on disconnect marks in-flight work as abandoned so the
    // engine can settle before accepting another lease.
    for l in lease.map.values() {
        l.task.abort();
    }
    // Stop the stage when its assignment's session ends. Report STOPPED if
    // the stream still works and let the next assignment determine what starts.
    if let Some(s) = &cfg.stages {
        s.stop_all().await;
        while let Ok(st) = stage_rx.try_recv() {
            let _ = out.send(node_message::Body::StageStatus(st)).await;
        }
    }
    drop(out);
    // Give queued shutdown messages time to reach the peer.
    let _ = tokio::time::timeout(Duration::from_secs(2), writer).await;
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

/// Release in-flight leases and send a goodbye without waiting for departure acknowledgment.
async fn ferma_subito(out: &Out, lease: &mut Leases) -> Result<()> {
    let released = abort_leases(lease, LeaseOutcome::Shutdown);
    send_releases(out, released, LeaseReleaseReason::Shutdown).await?;
    out.send(goodbye(GoodbyeReason::HostStopped)).await
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

async fn send_releases(out: &Out, released: Vec<String>, reason: LeaseReleaseReason) -> Result<()> {
    for lease_id in released {
        out.send(node_message::Body::LeaseRelease(LeaseRelease { lease_id, reason: reason as i32 })).await?;
    }
    Ok(())
}

/// Provide the advertised weights fingerprint when encoding a contribution preference.
fn modelli_di(inv: &ModelInventory) -> Vec<ModelRef> {
    inv.models.iter().filter_map(|m| m.model.clone()).collect()
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

/// Compare release components numerically. An absent or malformed minimum does not block startup.
pub fn versione_sufficiente(attuale: &str, minima: &str) -> bool {
    fn parse(v: &str) -> Option<[u64; 3]> {
        let core = v.split('-').next()?;
        let mut it = core.split('.').map(|p| p.parse::<u64>().ok());
        Some([it.next()??, it.next()??, it.next()??])
    }
    match (parse(attuale), parse(minima)) {
        (Some(a), Some(m)) => a >= m,
        _ => true,
    }
}

/// Treat the minimum release refusal as fatal so the agent does not reconnect indefinitely.
#[derive(Debug)]
pub struct VersioneTroppoVecchia {
    pub minima: String,
}

impl std::fmt::Display for VersioneTroppoVecchia {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "coordinator requires agent version {} or later (running {})", self.minima, env!("CARGO_PKG_VERSION"))
    }
}

impl std::error::Error for VersioneTroppoVecchia {}

/// Report busy engines as IDLE because the coordinator already tracks lease
/// occupancy. Availability reflects the engine's ability to serve.
fn availability_of(s: EngineState) -> Availability {
    match s {
        // Settling delays new work without making a temporary cancellation look like a fault.
        EngineState::Ready | EngineState::Busy | EngineState::Settling => Availability::Idle,
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

async fn sync_availability(out: &Out, sup: &Supervisor, live: &crate::control::Live, gpu: &Telemetria, last: &mut Availability) -> Result<()> {
    let a = availability_now(sup.state(), live, gpu);
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
) -> Result<()> {
    // A queued result can arrive after cancellation removed its lease.
    let Some(mut own) = lease.map.remove(&done.id) else { return Ok(()) };
    let suspicious = own.progress.lock().unwrap().unclassified_failure;
    let mut guard = own.guard.take();
    // Keep the shared guard until the final lease finishes, unless an engine fault requires immediate recovery.
    let comune = guard.is_none() && lease.comune.is_some();
    if comune {
        lease.sospetto |= suspicious;
        if done.result.is_ok() {
            if lease.is_empty() {
                guard = lease.comune.take();
            }
        } else {
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
            warn!(lease = %done.id, err = format!("{e:#}"), "releasing failed lease");
            let outcome = match classify(&e) {
                ErrorClass::Fault(c) => LeaseOutcome::Fault(c),
                ErrorClass::Request(_) => LeaseOutcome::Clean,
                ErrorClass::Other => LeaseOutcome::Abandoned,
            };
            if let Some(g) = guard {
                g.finish(outcome);
            }
            sync_availability(out, &cfg.sup, &cfg.live, &cfg.gpu, last_avail).await?;
            out.send(node_message::Body::LeaseRelease(LeaseRelease { lease_id: done.id, reason: LeaseReleaseReason::EngineFailure as i32 })).await?;
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
    if cfg.live.is_paused() || cfg.gpu.unavailable() || cfg.gpu_profile_missing() {
        return out.send(reject(LeaseRejectReason::HostBusy, "host unavailable")).await;
    }
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
    if !condiviso {
        if let Err(r) = cfg.sup.check_ready() {
            return out.send(reject(r.reason, r.detail)).await;
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
        // Stage chats require a connected head.
        (Some(work_offer::Task::Chat(_)), Engine::Stages(e)) => {
            if !e.pronto() {
                return out.send(reject(LeaseRejectReason::ModelNotReady, "chain not ready; no head in service")).await;
            }
        }
        (Some(work_offer::Task::Chat(_)), Engine::Llama(_) | Engine::Mock(_)) => {}
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
    let guard = if condiviso {
        None
    } else {
        match cfg.sup.try_begin() {
            Ok(g) => Some(g),
            Err(r) => return out.send(reject(r.reason, r.detail)).await,
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
        leases: vec![lease_progress(&lease_id, &LeaseCounters::default(), own.as_deref().unwrap_or(&cfg.sup.progress).snapshot())],
    }))
    .await?;

    let (conn2, engine2, sup2, progress2, done2, id2, own2) =
        (conn.clone(), engine.clone(), cfg.sup.clone(), progress.clone(), done_tx.clone(), lease_id.clone(), own.clone());
    let task = tokio::spawn(async move {
        let result = run_lease(conn2, &id2, offer, engine2, sup2, confirm_rx, progress2, own2).await;
        let _ = done2.send(LeaseDone { id: id2, result }).await;
    });
    let (guard_propria, comune) = if condivide(cfg, engine) { (None, guard) } else { (guard, None) };
    if comune.is_some() {
        lease.comune = comune;
        lease.sospetto = false;
    }
    lease.map.insert(lease_id.clone(), ActiveLease { id: lease_id, progress, own, confirm: Some(confirm_tx), task, guard: guard_propria });
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
    framed::write(
        &mut send,
        &WorkStreamNodeMessage { body: Some(work_stream_node_message::Body::Open(WorkStreamOpen { lease_id: lease_id.to_string() })) },
    )
    .await?;
    let batch = match offer.task {
        Some(work_offer::Task::Batch(b)) => b,
        Some(work_offer::Task::Verify(v)) => {
            run_verify(&mut send, &mut r, &engine, &sup, v).await?;
            progress.lock().unwrap().completing = true;
            sup.progress.set_phase(Phase::Finishing);
            finish_lease(&mut send, &mut r, lease_id).await?;
            return Ok((0, Duration::ZERO));
        }
        Some(work_offer::Task::Chat(c)) => {
            let esito = run_chat(&mut send, &mut r, &engine, &sup, &progress, own.as_ref(), lease_id, c, &logic_params).await?;
            progress.lock().unwrap().completing = true;
            sup.progress.set_phase(Phase::Finishing);
            if let Some(o) = &own {
                o.set_phase(Phase::Finishing);
            }
            finish_lease(&mut send, &mut r, lease_id).await?;
            return Ok(esito);
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
            usage: Some(Usage { prompt_tokens: g.prompt_tokens, completion_tokens: n }),
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
    send: &mut quinn::SendStream,
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
    let (dtx, mut drx) = mpsc::unbounded_channel();
    let (cancel_tx, cancel_rx) = oneshot::channel::<()>();
    let mut cancel_tx = Some(cancel_tx);
    let lp: Arc<Progress> = own.cloned().unwrap_or_else(|| sup.progress.clone());
    let (e2, req2, lp2) = (engine.clone(), req.clone(), lp.clone());
    let top_k = logic_params.top_k;
    let gen = tokio::spawn(async move { e2.chat_lease(&req2, top_k, &lp2, Some(dtx), Some(cancel_rx)).await });
    loop {
        tokio::select! {
            d = drx.recv() => match d {
                Some(d) => {
                    framed::write(send, &WorkStreamNodeMessage {
                        body: Some(work_stream_node_message::Body::TokenDelta(TokenDelta {
                            request_id: req.request_id.clone(), index: d.index, text: d.text, token_ids: d.token_ids,
                        })),
                    }).await?;
                }
                // The closed delta channel ends streaming. The task result supplies the final outcome.
                None => break,
            },
            m = r.read::<WorkStreamCoordinatorMessage>() => match m? {
                Some(m) => {
                    if let Some(work_stream_coordinator_message::Body::CancelRequest(c)) = m.body {
                        if c.request_id == req.request_id {
                            info!(lease = lease_id, request = %c.request_id, "coordinator cancelled the chat");
                            if let Some(t) = cancel_tx.take() {
                                let _ = t.send(());
                            }
                        }
                    }
                }
                None => {
                    // Dropping the stage request cancels work after the coordinator closes its stream.
                    cancel_tx.take();
                    bail!("coordinator closed the work stream");
                }
            },
        }
    }
    sup.progress.set_phase(Phase::Finishing);
    lp.set_phase(Phase::Finishing);
    let esito = gen.await.context("chat task")?;
    let (result, n, gen_time) = match esito {
        Ok(g) => {
            sup.stats.request_completed(t0.elapsed());
            completion_result(&req.request_id, g, logic_params)?
        }
        Err(e) => match classify(&e) {
            ErrorClass::Fault(c) => {
                sup.stats.request_failed(c.as_str());
                return Err(e.context(format!("request {}", req.request_id)));
            }
            ErrorClass::Request(code) => {
                warn!(request = %req.request_id, ?code, "chat failed: {e:#}");
                sup.stats.request_failed(&format!("{code:?}"));
                (RequestResult { request_id: req.request_id.clone(), outcome: Some(request_result::Outcome::Failure(RequestFailure { code: code as i32, detail: format!("{e:#}") })) }, 0, Duration::ZERO)
            }
            ErrorClass::Other => {
                progress.lock().unwrap().unclassified_failure = true;
                (
                    RequestResult {
                        request_id: req.request_id.clone(),
                        outcome: Some(request_result::Outcome::Failure(RequestFailure { code: RequestFailureCode::EngineError as i32, detail: format!("{e:#}") })),
                    },
                    0,
                    Duration::ZERO,
                )
            }
        },
    };
    framed::write(send, &WorkStreamNodeMessage { body: Some(work_stream_node_message::Body::RequestResult(result)) }).await?;
    loop {
        let m: WorkStreamCoordinatorMessage = r.read().await?.context("coordinator closed the work stream")?;
        if let Some(work_stream_coordinator_message::Body::ResultAck(a)) = m.body {
            if !a.accepted {
                warn!(request = %a.request_id, reason = %a.reason, "result rejected");
            }
            break;
        }
    }
    sup.progress.tick();
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

async fn finish_lease(send: &mut quinn::SendStream, r: &mut FrameReader, lease_id: &str) -> Result<()> {
    framed::write(
        send,
        &WorkStreamNodeMessage {
            body: Some(work_stream_node_message::Body::LeaseComplete(LeaseComplete { lease_id: lease_id.to_string() })),
        },
    )
    .await?;
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
        Contribution::open(&d, Preference { modo: PreferenceMode::Usage, model: Some(ModelSelection { id: "".into(), quant: "".into() }), revision: 0 }).unwrap()
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
        let cfg = NodeConfig { model: ModelRef::default(), costs: None, hardware: Default::default(), sup, stages: None, region: String::new(), nat_class: String::new(), iroh_node_id: String::new(), slots: 1, contribution: contributo_prova("nodo"), gpu: Default::default(), model_path: None, output: watch::channel(false).1, live: crate::control::Live::new() };
        (f, cfg)
    }

    fn active(cfg: &NodeConfig, id: &str) -> ActiveLease {
        ActiveLease {
            id: id.into(),
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
        on_lease_done(&cfg, &shared, &out, &mut last, &mut lease, LeaseDone { id: "L1".into(), result: Err(err) }).await.unwrap();
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
            LeaseDone { id: "L1".into(), result: Err(anyhow::anyhow!("work stream closed")) },
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
        let sc = quinn::ServerConfig::with_crypto(Arc::new(quinn::crypto::rustls::QuicServerConfig::try_from(tls).unwrap()));
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
        (NodeConfig { model, costs: None, hardware: Default::default(), sup, stages: Some(stages.clone()), region: String::new(), nat_class: String::new(), iroh_node_id: String::new(), slots: 1, contribution: contributo_prova(name), gpu: Default::default(), model_path: None, output: watch::channel(false).1, live: crate::control::Live::new() }, engine, stages)
    }

    fn offerta_chat(cfg: &NodeConfig, request_id: &str, text: &str, max_tokens: u32) -> WorkOffer {
        WorkOffer {
            lease: Some(Lease { lease_id: "L-chat".into(), ..Default::default() }),
            model: Some(cfg.model.clone()),
            logic: Some(LogicParams { format_version: logic::LOGIC_FORMAT, top_k: 5 }),
            payout: None,
            task: Some(work_offer::Task::Chat(ChatTask {
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
        let risultato = loop {
            let m: WnMsg = r.read().await.unwrap().expect("stream closed");
            match m.body {
                Some(wn::Body::TokenDelta(d)) => {
                    deltas.push(d);
                    if annulla_dopo == Some(deltas.len()) {
                        framed::write(&mut send, &WcMsg { body: Some(wc::Body::CancelRequest(CancelRequest { request_id: request_id.clone() })) }).await.unwrap();
                    }
                }
                Some(wn::Body::RequestResult(res)) => break res,
                altro => panic!("unexpected message: {altro:?}"),
            }
        };
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
        let (client, server) = loopback().await;
        let coord = tokio::spawn(coordinatore(server, "r1".into(), None));
        let offerta = offerta_chat(&cfg, "r1", "ciao", 50);
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
        assert_eq!(c.usage.unwrap().completion_tokens, 5);
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
        assert_eq!(cfg.sup.state(), EngineState::Busy);
        assert_eq!(heartbeat_msg(&lease, &cfg, &Shared::default()).leases.len(), 4);
        for i in 0..3 {
            on_revoked(&cfg, &mut lease, &format!("L{i}"), LeaseRevokeReason::Cancelled as i32);
            assert_eq!(cfg.sup.state(), EngineState::Busy, "one lease is still active");
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
        assert_eq!(esito_offerta(&cfg, &engine, &client, offerta_chat(&cfg, "r", "x", 5)).await, Some(LeaseRejectReason::ModelNotReady));
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
        let a = tokio::spawn(async move { let Engine::Stages(e) = &*e1 else { unreachable!() }; e.chat_lease(&ra, &pa2, None, None).await });
        let (ctx, crx) = oneshot::channel();
        let b = tokio::spawn(async move { let Engine::Stages(e) = &*e2 else { unreachable!() }; e.chat_lease(&rb, &pb2, None, Some(crx)).await });
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
        let pref = Preference { modo: PreferenceMode::Usage, model: Some(ModelSelection { id: "qwen-prova".into(), quant: "Q4_K_M".into() }), revision: 0 };
        cfg.contribution = Contribution::open(&d, pref).unwrap();
        (f, cfg)
    }

    fn welcome(minor: Option<u32>, min_agent: &str, heartbeat_ms: u32) -> Welcome {
        Welcome {
            session_id: "s_prova".into(),
            heartbeat_interval: prost_types::Duration::try_from(Duration::from_millis(u64::from(heartbeat_ms))).ok(),
            min_agent_version: min_agent.into(),
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
            let (client, server) = loopback().await;
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
            f.manda(coordinator_message::Body::Welcome(welcome)).await;
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
    }

    #[tokio::test]
    async fn hello_carries_preference() {
        let (_f, cfg) = nodo_con_modello("hello").await;
        let rev = cfg.contribution.preference().revision;
        let (fin, _) = Finta::start(cfg, welcome(Some(2), "", 60_000)).await;
        let p: &ContributionPreference = fin.hello.contribution.as_ref().expect("preferenza nell'Hello");
        assert_eq!(p.mode(), ContributionMode::Use);
        assert_eq!(p.model.as_ref().unwrap(), &modello_prova(), "model with the inventory fingerprint");
        assert_eq!(p.revision, rev);
        assert!(fin.hello.features.contains(&(AgentFeature::Contribution as i32)));
        assert_eq!(fin.hello.protocol_version.as_ref().unwrap().minor, 2);
        let _ = fin.stop_tx.send(true);
    }

    #[tokio::test]
    async fn preference_change_is_sent() {
        let (_f, cfg) = nodo_con_modello("cambio").await;
        let contribution = cfg.contribution.clone();
        let (mut fin, _) = Finta::start(cfg, welcome(Some(2), "", 60_000)).await;
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

    #[tokio::test]
    async fn skips_preference_update_with_old_coordinator() {
        let (_f, cfg) = nodo_con_modello("vecchio").await;
        let contribution = cfg.contribution.clone();
        // Without negotiated support for preference updates, send the preference only in the handshake.
        let (mut fin, _) = Finta::start(cfg, welcome(None, "", 60_000)).await;
        assert!(fin.hello.contribution.is_some());
        contribution.imposta(crate::contribution::PreferenceMode::Free, None).unwrap();
        let visto = fin
            .attendi(700, |b| match b {
                node_message::Body::Contribution(_) | node_message::Body::ModelDirectiveStatus(_) | node_message::Body::Leaving(_) => Some(()),
                _ => None,
            })
            .await;
        assert!(visto.is_none(), "no preference update for a coordinator without support");
        let _ = fin.stop_tx.send(true);
    }

    #[tokio::test]
    async fn session_keeps_received_status() {
        let (_f, cfg) = nodo_con_modello("stato").await;
        let contribution = cfg.contribution.clone();
        let rev = contribution.preference().revision;
        let (mut fin, _) = Finta::start(cfg, welcome(Some(2), "", 60_000)).await;
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
        let (mut fin, _) = Finta::start(cfg, welcome(Some(2), "", 60_000)).await;
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
        let (mut fin, _) = Finta::start(cfg, welcome(Some(2), "", 60_000)).await;
        uscita_tx.send(true).unwrap();
        let l = fin
            .attendi(2000, |b| match b {
                node_message::Body::Leaving(l) => Some(l.clone()),
                _ => None,
            })
            .await
            .expect("expected Leaving after the first signal");
        assert_eq!(l.grazia_s, 60);
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
        let (mut fin, _) = Finta::start(cfg, welcome(Some(2), "", 60_000)).await;
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
    async fn shutdown_with_old_coordinator() {
        let (_f, mut cfg) = nodo_con_modello("uscita-vecchio").await;
        let (uscita_tx, uscita_rx) = watch::channel(false);
        cfg.output = uscita_rx;
        let (mut fin, _) = Finta::start(cfg, welcome(None, "", 60_000)).await;
        uscita_tx.send(true).unwrap();
        let mut visto_leaving = false;
        let saluto = fin
            .attendi(2000, |b| {
                visto_leaving |= matches!(b, node_message::Body::Leaving(_));
                matches!(b, node_message::Body::Goodbye(_)).then_some(())
            })
            .await;
        assert!(saluto.is_some() && !visto_leaving, "without departure support, send Goodbye on the first signal");
        assert!(matches!(fin.done().await, SessionEnd::Shutdown));
    }

    #[tokio::test]
    async fn minimum_version_stops_agent() {
        let (_f, cfg) = nodo_con_modello("minima").await;
        let (fin, _) = Finta::start(cfg, welcome(Some(2), "99.0.0", 60_000)).await;
        match fin.done().await {
            SessionEnd::TroppoVecchio(msg) => assert!(msg.contains("99.0.0") && msg.contains("requires agent version"), "{msg}"),
            altro => panic!("expected an agent version error, got {altro:?}"),
        }
        assert!(versione_sufficiente("0.1.0", ""));
        assert!(versione_sufficiente("0.1.0", "0.1.0"));
        assert!(versione_sufficiente("0.2.0", "0.1.9"));
        assert!(!versione_sufficiente("0.1.0", "0.1.1"));
        assert!(!versione_sufficiente("0.9.9", "1.0.0"));
    }

    #[tokio::test]
    async fn directive_for_loaded_model_reports_serving() {
        let (_f, cfg) = nodo_con_modello("dir1").await;
        let shared = Shared::default();
        let d = ModelDirective { model: Some(modello_prova()), epoch: 1, reason: "test".into(), ..Default::default() };
        let st = rispondi_direttiva(&cfg, &shared, d);
        assert_eq!(st.state(), StatoDirettiva::Serving);
        assert_eq!(st.epoch, 1);
        let (mut fin, _) = Finta::start(cfg, welcome(Some(2), "", 60_000)).await;
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
        let (mut fin, _) = Finta::start(cfg, welcome(Some(2), "", 100)).await;
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
}
