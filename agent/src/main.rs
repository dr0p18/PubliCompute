//! The agent enrolls with the coordinator, advertises local capacity and
//! runs leases through a selected engine. Managed HTTP mode owns the child
//! process. External mode observes a server started by the operator. Stage
//! mode starts a process for a coordinator assignment and routes chats to the
//! head's service socket when this node owns the head. These modes share
//! the same local control interface.

mod contribution;
mod control;
mod economics;
mod engine;
mod enroll;
mod nat;
mod network;
mod peer_echo;
mod bridge;
#[cfg(test)]
mod fake_llama;
mod framed;
mod gpu;
mod hardware;
mod identity;
mod logic;
mod model_change;
mod models;
mod session;
mod stages;
mod transport;

use std::{io::Read, path::PathBuf, sync::Arc, time::Duration};

use anyhow::{bail, Context, Result};
use clap::{ArgAction, Parser, Subcommand, ValueEnum};
use publicompute_proto::common::{EngineKind, ModelRef, Money};
use sha2::{Digest, Sha256};
use tracing::{error, info, warn};

const MODEL_SWITCH_EXIT_CODE: i32 = 75;

#[derive(Clone, Copy, ValueEnum)]
enum EngineArg {
    /// Deterministic engine for development and tests.
    Mock,
    /// Local HTTP inference server.
    Llama,
    /// Launch a stage process when the coordinator assigns this node a range.
    Stages,
}

#[derive(Clone, Copy, ValueEnum)]
enum LlamaMode {
    /// Use an existing server and leave its process lifecycle to the operator.
    External,
    /// Own the server process and restart it when supervision requires recovery.
    Managed,
}

#[derive(Parser)]
#[command(version, about = "PubliCompute node agent")]
struct Args {
    /// UDP endpoint of the coordinator.
    #[arg(long, global = true, default_value = "127.0.0.1:7443")]
    coordinator: String,
    /// Server name to verify in the coordinator certificate.
    #[arg(long, global = true, default_value = "localhost")]
    server_name: String,
    /// PEM trust anchor for the coordinator connection.
    #[arg(long, global = true)]
    ca: Option<PathBuf>,
    /// Directory for the enrollment identity and local state.
    /// It must be a real local directory and stay at the same path while the agent runs.
    /// Custom service units specify its real path.
    #[arg(long, default_value = "data/agent")]
    data_dir: PathBuf,
    /// Local control socket path. A configured path grants access to the agent group.
    #[arg(long)]
    control_socket: Option<PathBuf>,
    /// Read the link code from this file in preference to the environment variable.
    #[arg(long)]
    owner_token_file: Option<PathBuf>,
    /// Secret link code used only at enrollment to associate the node with an owner.
    /// Omitting it enrolls anonymously. Prefer a file or environment variable to a
    /// command-line value, which can be visible to other local users.
    #[arg(long, env = "PUBLICOMPUTE_OWNER_TOKEN", hide_env_values = true)]
    owner_token: Option<String>,

    #[arg(long, value_enum, default_value = "mock")]
    engine: EngineArg,
    #[arg(long, default_value = "http://127.0.0.1:8080")]
    llama_url: String,

    /// GGUF weights file. Supply it to derive the ID and hash when omitted.
    #[arg(long)]
    model_path: Option<PathBuf>,
    #[arg(long)]
    model_id: Option<String>,
    #[arg(long, default_value = "")]
    quantization: String,
    /// Hexadecimal SHA-256 of the weights when their file is unavailable.
    #[arg(long)]
    weights_sha256: Option<String>,

    /// Electricity tariff in euros per kilowatt-hour for the offer cost check.
    #[arg(long)]
    electricity_eur_kwh: Option<f64>,
    /// Power draw under load, in watts.
    #[arg(long)]
    power_watts: Option<f64>,
    /// Declared power limit as a percentage of rated power.
    #[arg(long, default_value_t = 70, value_parser = clap::value_parser!(u32).range(30..=100))]
    power_limit_percent: u32,
    /// Select a driver-reported device by index or override the reported name and memory.
    #[arg(long)]
    gpu: Option<String>,
    /// Driver version to report with a manual GPU override. The driver query supplies the default.
    #[arg(long)]
    gpu_driver: Option<String>,
    /// NVIDIA GPU UUID to use and monitor in managed mode.
    #[arg(long, env = "PUBLICOMPUTE_GPU_UUID", conflicts_with = "gpu")]
    gpu_uuid: Option<String>,
    /// Process ownership mode for the HTTP engine.
    #[arg(long, value_enum, default_value = "external")]
    llama_mode: LlamaMode,
    /// Server executable required in managed mode.
    #[arg(long)]
    llama_bin: Option<PathBuf>,
    /// Additional server argument in managed mode. Repeat to pass multiple arguments.
    #[arg(long = "llama-arg", allow_hyphen_values = true)]
    llama_args: Vec<String>,
    /// Number of concurrent chats advertised to the coordinator. The managed server
    /// gets this many parallel slots and a context sized to the slot count times the
    /// per-slot context. Configure an external server to match. For a stage chain,
    /// the assignment supplies its slot count.
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u32).range(1..=16))]
    slots: u32,
    /// Stream completion events to measure progress. Disabling streaming selects the polling path.
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    llama_stream: bool,
    /// Deadline for the one-token probe. This leaves time for a slow token while
    /// bounding the wait for a stalled engine.
    #[arg(long, default_value_t = 30)]
    probe_timeout_secs: u64,
    /// Probe the idle engine at this interval to detect stalls between leases.
    #[arg(long, default_value_t = 300)]
    probe_interval_secs: u64,
    /// Maximum prefill silence. The default lets the agent detect a stall before
    /// the coordinator revokes the lease.
    #[arg(long, default_value_t = 150)]
    prefill_stall_secs: u64,
    /// Maximum decode silence. The default lets the agent release a stalled lease
    /// before the coordinator's progress deadline.
    #[arg(long, default_value_t = 30)]
    decode_stall_secs: u64,
    /// Absolute request deadline. The progress watchdog handles ordinary stalls.
    #[arg(long, default_value_t = 1800)]
    request_max_secs: u64,
    /// Restart budget within the quarantine window, for managed mode.
    #[arg(long, default_value_t = 3)]
    max_restarts: u32,
    /// Window over which restart attempts count toward the budget, in minutes.
    #[arg(long, default_value_t = 60)]
    restart_window_mins: u64,
    /// Give an abandoned request this long to finish before treating the engine
    /// as unhealthy.
    #[arg(long, default_value_t = 60)]
    settle_secs: u64,
    /// Recycle the idle managed process after this many hours. Zero disables recycling.
    #[arg(long, default_value_t = 0.0)]
    llama_recycle_hours: f64,
    /// Delay before the second recovery attempt. Later attempts double it up to two minutes.
    #[arg(long, default_value_t = 5, hide = true)]
    restart_backoff_secs: u64,
    /// Maximum startup wait for the managed server's health endpoint.
    #[arg(long, default_value_t = 300, hide = true)]
    engine_start_timeout_secs: u64,
    /// Grace period between sending SIGTERM and SIGKILL.
    #[arg(long, default_value_t = 10, hide = true)]
    stop_grace_secs: u64,
    /// Wait for exit after forced termination before marking the engine quarantined.
    #[arg(long, default_value_t = 15, hide = true)]
    kill_wait_secs: u64,
    /// Stall the first N fixture generations partway through a response.
    #[arg(long, default_value_t = 0, hide = true)]
    mock_hang: u32,
    /// Fixture delay per token, in milliseconds.
    #[arg(long, default_value_t = 0)]
    mock_delay_ms: u64,
    /// Make the fixture generate fabricated responses to exercise verification.
    #[arg(long, hide = true)]
    mock_cheat: bool,

    /// Initial contribution mode. Usage reserves the card for the selected weights.
    /// Free mode lets the scheduler use it where needed. A stored preference takes priority.
    #[arg(long = "mode", default_value = "usage")]
    modo: String,
    /// USE selection as an ID with optional quantization. Defaults to the installed weights.
    #[arg(long = "selected-model")]
    modello_scelto: Option<String>,

    /// Region advertised in the handshake. An empty value leaves it unset.
    /// An administrative assignment takes precedence over this hint.
    #[arg(long = "region", env = "PUBLICOMPUTE_REGION", default_value = "")]
    region: String,

    /// Skip the NAT probe and advertise an unknown classification.
    #[arg(long = "skip-nat-probe")]
    senza_prova_nat: bool,
    /// STUN host used to classify the node's UDP mapping.
    #[arg(long, global = true, env = "PUBLICOMPUTE_STUN_HOST", default_value = "relay.publicompute.com")]
    stun_host: String,
    /// First UDP port of the STUN target.
    #[arg(long, global = true, env = "PUBLICOMPUTE_STUN_PORT_A", default_value_t = 3478)]
    stun_port_a: u16,
    /// Second UDP port of the STUN target. Compare the two mapped ports to detect a symmetric mapping.
    #[arg(long, global = true, env = "PUBLICOMPUTE_STUN_PORT_B", default_value_t = 3479)]
    stun_port_b: u16,

    /// Relay URL for stage links. The bridge uses this relay with discovery disabled.
    /// Without a URL, stage links use direct TCP.
    #[arg(long, env = "PUBLICOMPUTE_RELAY")]
    relay: Option<String>,

    /// Stage executable for stage mode or switching a managed llama-server to a stage.
    #[arg(long)]
    stage_bin: Option<PathBuf>,
    /// Stage log directory. Defaults to the stage subdirectory of the data directory.
    #[arg(long)]
    stage_log_dir: Option<PathBuf>,
    /// CPU thread count for each stage process.
    #[arg(long, default_value_t = 6)]
    stage_threads: u32,
    /// Number of stage layers to offload to the GPU. Use 999 to offload all layers.
    #[arg(long, default_value_t = 999)]
    stage_ngl: i32,
    /// Additional stage executable argument. Repeat to pass multiple arguments.
    #[arg(long = "stage-arg", allow_hyphen_values = true)]
    stage_args: Vec<String>,
    /// Startup deadline covering weight loading and chain connection setup.
    #[arg(long, default_value_t = 900)]
    stage_load_timeout_secs: u64,
    /// Permit unsigned assignments to start over plaintext. Use only in a test environment.
    #[arg(long = "allow-plaintext-stages", default_value_t = false)]
    stadi_in_chiaro: bool,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Print the NAT class and exit.
    NatCheck,
    /// Verify the coordinator's QUIC endpoint without enrolling.
    CoordinatorCheck,
    /// Replace the installed package directory in one filesystem operation.
    #[cfg(target_os = "linux")]
    #[command(hide = true)]
    SwapInstall { staging: PathBuf, installed: PathBuf },
}

fn main() {
    let args = Args::parse();
    // Clear the link code before runtime threads or children can inherit the environment.
    std::env::remove_var("PUBLICOMPUTE_OWNER_TOKEN");
    std::env::remove_var("CREDENTIALS_DIRECTORY");
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();
    match run_command(args) {
        Ok(code) => std::process::exit(code),
        Err(e) => {
            error!("{e:#}");
            std::process::exit(1);
        }
    }
}

fn run_command(args: Args) -> Result<i32> {
    let local = if args.command.is_none() {
        if args.ca.is_none() {
            bail!("--ca is required for the coordinator QUIC certificate");
        }
        std::fs::create_dir_all(&args.data_dir).with_context(|| format!("create {}", args.data_dir.display()))?;
        let live = control::Live::open(&args.data_dir)?;
        let path = args.control_socket.clone().unwrap_or_else(|| args.data_dir.join("agent.sock"));
        Some((control::Listener::bind(path, args.control_socket.is_some())?, live))
    } else {
        None
    };
    let data_dir_lock = local.as_ref().map(|(_, live)| live.clone());
    let runtime = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
    let result = runtime.block_on(async {
        match &args.command {
            Some(Command::NatCheck) => {
                println!("{}", nat::classifica(&stun_config(&args)).await);
                Ok(0)
            }
            Some(Command::CoordinatorCheck) => {
                let t = coordinator_transport(&args).await?;
                let conn = tokio::time::timeout(Duration::from_secs(10), t.connect_anonymous())
                    .await.context("coordinator QUIC check timed out")??;
                conn.close(0u32.into(), b"");
                println!("Coordinator reachable over QUIC");
                Ok(0)
            }
            #[cfg(target_os = "linux")]
            Some(Command::SwapInstall { staging, installed }) => swap_install(staging, installed).map(|()| 0),
            None => {
                let (listener, live) = local.expect("agent control state");
                run(args, listener, live).await
            }
        }
    });
    runtime.shutdown_timeout(Duration::from_secs(1));
    drop(data_dir_lock);
    result
}

#[cfg(target_os = "linux")]
fn swap_install(staging: &std::path::Path, installed: &std::path::Path) -> Result<()> {
    use std::{ffi::CString, os::unix::ffi::OsStrExt};
    match std::fs::symlink_metadata(installed) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(std::fs::rename(staging, installed)?),
        Err(e) => return Err(e.into()),
        Ok(m) if !m.is_dir() => bail!("{} is not an installation directory", installed.display()),
        Ok(_) => {}
    }
    let staging = CString::new(staging.as_os_str().as_bytes())?;
    let installed = CString::new(installed.as_os_str().as_bytes())?;
    // Exchanging two nonempty directories requires the platform-specific rename operation.
    let rc = unsafe {
        libc::renameat2(libc::AT_FDCWD, staging.as_ptr(), libc::AT_FDCWD, installed.as_ptr(), libc::RENAME_EXCHANGE)
    };
    if rc != 0 {
        return Err(std::io::Error::last_os_error()).context("exchange installation directories");
    }
    Ok(())
}

fn stun_config(args: &Args) -> nat::StunCfg {
    nat::StunCfg::new(args.stun_host.clone(), args.stun_port_a, args.stun_port_b)
}

async fn coordinator_transport(args: &Args) -> Result<transport::Transport> {
    let ca = read_pem_cert(args.ca.as_ref().context("--ca is required for the coordinator QUIC certificate")?)?;
    let addr = tokio::net::lookup_host(&args.coordinator).await?
        .next()
        .context("could not resolve the coordinator address")?;
    transport::Transport::new(addr, args.server_name.clone(), ca)
}

fn read_owner_token(file: Option<&std::path::Path>, fallback: Option<String>) -> Result<Option<String>> {
    let Some(file) = file else { return Ok(fallback) };
    let token = std::fs::read_to_string(file)
        .with_context(|| format!("read owner token file {}", file.display()))?;
    let token = token.trim();
    if token.is_empty() {
        bail!("owner token file is empty");
    }
    Ok(Some(token.to_string()))
}

async fn run(mut args: Args, listener: control::Listener, live: Arc<control::Live>) -> Result<i32> {
    // An externally managed engine can keep serving its old weights after this agent restarts.
    let can_switch = cfg!(unix) && !matches!((args.engine, args.llama_mode), (EngineArg::Llama, LlamaMode::External));
    let mut model_error = None;
    let weights = Arc::new(models::Store::new(&args.data_dir));
    let store = if can_switch {
        match weights.prepare() {
            Ok(()) => Some(weights.clone()),
            Err(err) => {
                let text = format!("cannot open the model directory: {err:#}");
                warn!("{text}");
                model_error = Some(text);
                None
            }
        }
    } else {
        None
    };
    let configured = model_metadata(&args);
    let modo = contribution::PreferenceMode::parse(&args.modo)?;
    let selected = |model: &ModelRef, override_model: Option<&str>| match override_model {
        Some(m) => match m.split_once(':') {
            Some((id, quant)) => contribution::ModelSelection { id: id.into(), quant: quant.into() },
            None => contribution::ModelSelection { id: m.into(), quant: model.quantization.clone() },
        },
        None => contribution::ModelSelection { id: model.model_id.clone(), quant: model.quantization.clone() },
    };
    let initial = configured.as_ref().cloned().unwrap_or_default();
    let contribution = contribution::Contribution::open(&args.data_dir, contribution::Preference {
        modo,
        model: (modo == contribution::PreferenceMode::Usage).then(|| selected(&initial, args.modello_scelto.as_deref())),
        revision: 0,
    })?;
    let telemetria_gpu = Arc::new(gpu::Telemetria::default());
    let (engine_state, engine_rx) = tokio::sync::watch::channel(engine::supervisor::EngineState::Starting);
    let control = control::Control::new(
        contribution.clone(),
        telemetria_gpu.clone(),
        live.clone(),
        engine_rx,
        initial,
        args.model_path.clone(),
    ).with_models(store.clone(), model_error.clone());
    control.verifying_weights(store.is_some() || args.model_path.is_some());
    listener.serve(control.clone())?;

    let (stop_tx, stop_rx) = tokio::sync::watch::channel(false);
    let (uscita_tx, mut uscita_rx) = tokio::sync::watch::channel(false);
    // The first signal gives the session time to announce departure. A second skips the wait.
    let (segnali_tx, segnali_rx) = tokio::sync::mpsc::channel::<()>(4);
    tokio::spawn(inoltra_segnali(segnali_tx));
    tokio::spawn(arresto(segnali_rx, uscita_tx, stop_tx, session::ATTESA_USCITA + Duration::from_secs(5)));

    let configured_path = args.model_path.clone();
    let restore = store.is_some();
    let verify = async {
        tokio::select! {
            verified = tokio::task::spawn_blocking(move || verify_startup_weights(&weights, restore, configured, configured_path)) => {
                verified?.map(Some)
            }
            _ = uscita_rx.wait_for(|stop| *stop) => Ok(None),
        }
    };
    let StartupWeights { model, path, restored, error, needs_repair } =
        match while_offline(&live, store.as_ref(), &contribution, verify).await {
            None => return Ok(MODEL_SWITCH_EXIT_CODE),
            Some(Ok(None)) => return Ok(0),
            Some(Ok(Some(verified))) => verified,
            Some(Err(err)) => return Err(err),
        };
    args.model_path = path;
    if let Some(error) = error {
        model_error = Some(model_error.map_or_else(|| error.clone(), |prior| format!("{prior}; {error}")));
    }
    {
        let _commands = live.changes.commands.lock().await;
        let modello_scelto = selected(&model, args.modello_scelto.as_deref().filter(|_| restored.is_none()));
        if let Some(active) = restored {
            let pref = contribution.preference();
            if pref.modo == contribution::PreferenceMode::Free && active.use_revision == Some(pref.revision) {
                contribution.imposta(contribution::PreferenceMode::Usage, Some(modello_scelto.clone()))?;
            }
        }
        contribution.align_model(modello_scelto)?;
        control.model_verified(model.clone(), args.model_path.clone(), model_error, needs_repair);
    }
    if needs_repair {
        // Keep the socket available for a repair or another model choice.
        let stop = uscita_rx.wait_for(|stop| *stop);
        return Ok(if while_offline(&live, store.as_ref(), &contribution, stop).await.is_none() {
            MODEL_SWITCH_EXIT_CODE
        } else {
            0
        });
    }

    args.owner_token = read_owner_token(args.owner_token_file.as_deref(), args.owner_token.take())?;
    let t = tokio::select! {
        t = coordinator_transport(&args) => t?,
        _ = uscita_rx.wait_for(|stop| *stop) => return Ok(0),
    };
    let cuda_devices = std::env::var_os("CUDA_VISIBLE_DEVICES");
    let gpu_index = args.gpu.as_deref().filter(|gpu| gpu.parse::<u32>().is_ok());
    let selector = args.gpu_uuid.as_deref().or(args.gpu.as_deref()).or_else(|| {
        cuda_devices.as_deref().map(|v| v.to_str().unwrap_or(""))
    });
    let managed_gpu = matches!((args.engine, args.llama_mode), (EngineArg::Llama, LlamaMode::Managed));
    let gpu_uuid = match selector {
        Some(value) if value.len() == 40 && value.starts_with("GPU-") => Some(value.to_owned()),
        Some(value) if value.parse::<u32>().is_ok() => gpu::resolve_index(value).await,
        _ => None,
    };
    let track_gpu = managed_gpu && (selector.is_none() || gpu_uuid.is_some());
    let unresolved_index = gpu_index.is_some() && gpu_uuid.is_none();
    if managed_gpu && !track_gpu {
        warn!("cannot resolve a single NVIDIA GPU UUID; GPU loss monitoring is disabled");
    }
    let first = gpu::read(gpu_uuid.as_deref()).await;
    let detected = first.as_deref().unwrap_or_default().to_vec();
    gpu::avvia_campionatore(telemetria_gpu.clone(), first, track_gpu, gpu_uuid).await;
    if track_gpu && identity::load(&args.data_dir)?.is_none() {
        // Initial enrollment requires an inventory. An enrolled node can reconnect while paused.
        let mut device = telemetria_gpu.device_rx();
        let wait = async {
            tokio::select! {
                ready = device.wait_for(|state| matches!(state.device, gpu::Device::Ready(_))) => { ready?; }
                _ = uscita_rx.wait_for(|stop| *stop) => {}
            }
            Ok::<(), anyhow::Error>(())
        };
        match while_offline(&live, store.as_ref(), &contribution, wait).await {
            None => return Ok(MODEL_SWITCH_EXIT_CODE),
            Some(result) => result?,
        }
        if *uscita_rx.borrow() {
            return Ok(0);
        }
    }
    let selected = if track_gpu { telemetria_gpu.letture() } else { detected };
    let electricity = args.electricity_eur_kwh.map(|e| Money { micros: (e * 1e6).round() as i64, currency: "EUR".into() });
    // An absent driver inventory still permits the agent to start.
    let gpus: Vec<hardware::ManualGpu> = match args.gpu.as_deref().filter(|_| gpu_index.is_none()) {
        Some(forzata) => {
            let mut g = parse_gpu(forzata)?;
            g.driver_version = args.gpu_driver.clone().or_else(|| selected.first().map(|card| card.driver.clone())).unwrap_or_default();
            vec![g]
        }
        None => selected.into_iter().filter(|_| !unresolved_index)
            .map(|l| hardware::ManualGpu { name: l.name, vram_bytes: l.vram_totale, driver_version: l.driver })
            .collect(),
    };
    if gpus.is_empty() {
        info!("no NVIDIA GPU detected; reporting a CPU-only node");
    }
    let kind = match args.engine {
        EngineArg::Mock => EngineKind::Unspecified,
        EngineArg::Llama | EngineArg::Stages => EngineKind::LlamaCpp,
    };
    // Use the filesystem holding the weights to measure free capacity when their path is known.
    let cartella_modelli = args
        .model_path
        .as_ref()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        .filter(|d| !d.as_os_str().is_empty())
        .unwrap_or_else(|| args.data_dir.clone());
    let mut hw = hardware::profile(&gpus, kind, electricity.as_ref(), args.power_limit_percent, &cartella_modelli);
    let costs = match (args.power_watts, electricity.clone()) {
        (Some(w), Some(e)) => Some(economics::HostCosts { power_watts: w, electricity: e }),
        _ => None,
    };

    let secs = Duration::from_secs;
    let limits = engine::Limits {
        prefill_stall: secs(args.prefill_stall_secs),
        decode_stall: secs(args.decode_stall_secs),
        request_max: secs(args.request_max_secs),
        // Short auxiliary calls have their own budget so they cannot consume the completion deadline.
        call_timeout: secs(30),
        // A verification step performs a prefill and one decode, so use the prefill budget.
        step_timeout: secs(args.prefill_stall_secs),
        probe_timeout: secs(args.probe_timeout_secs),
    };
    let progress = Arc::new(engine::progress::Progress::default());
    let stats = Arc::new(engine::stats::Stats::new(Some(&args.data_dir)));
    let stage_only = matches!(args.engine, EngineArg::Stages);
    if stage_only && matches!(args.llama_mode, LlamaMode::Managed) {
        bail!("--llama-mode managed requires a llama engine");
    }
    let stages = match args.stage_bin.as_ref() {
        Some(bin) if stage_only || managed_gpu => {
            Some(stages::Stages::new(stages::StagesConfig {
                stage_bin: bin.clone(),
                model_path: args.model_path.clone().context("--stage-bin requires --model-path pointing to the full GGUF")?,
                weights_sha256: model.weights_sha256.clone(),
                log_dir: args.stage_log_dir.clone().unwrap_or_else(|| args.data_dir.join("stadi")),
                threads: args.stage_threads,
                ngl: args.stage_ngl,
                extra_args: args.stage_args.clone(),
                env: vec![],
                load_timeout: secs(args.stage_load_timeout_secs),
                stop_grace: secs(args.stop_grace_secs),
                kill_wait: secs(args.kill_wait_secs),
                ca_path: args.ca.clone(),
                tls_dir: args.data_dir.join("stadi").join("tls"),
                in_chiaro: args.stadi_in_chiaro,
            }))
        }
        Some(_) => bail!("--stage-bin requires --engine stages or a managed llama engine"),
        None if stage_only => bail!("--engine stages requires --stage-bin"),
        None => None,
    };
    let stage_engine = stages.as_ref().map(|s| Arc::new(engine::Engine::Stages(engine::stages::StadiEngine {
        stages: s.clone(),
        attive: Default::default(),
        limits: limits.clone(),
        // Stage chats track their own progress while llama-server is parked.
        progress: if stage_only { progress.clone() } else { Default::default() },
    })));
    let engine = match args.engine {
        EngineArg::Mock => Arc::new(engine::Engine::Mock(engine::mock::MockEngine {
            per_token_delay: Duration::from_millis(args.mock_delay_ms),
            cheat: args.mock_cheat,
            hang: std::sync::atomic::AtomicU32::new(args.mock_hang),
            limits: limits.clone(),
            progress: progress.clone(),
        })),
        EngineArg::Llama => Arc::new(engine::Engine::Llama(engine::llama::LlamaEngine::new(
            args.llama_url.clone(),
            limits.clone(),
            progress.clone(),
            args.llama_stream,
        )?)),
        EngineArg::Stages => stage_engine.as_ref().expect("stage config is required").clone(),
    };
    let managed = match (args.engine, args.llama_mode) {
        (EngineArg::Llama, LlamaMode::Managed) => Some(engine::managed::ManagedCfg {
            bin: args.llama_bin.clone().context("--llama-mode managed requires --llama-bin")?,
            model_path: args.model_path.clone().context("--llama-mode managed requires --model-path")?,
            port: engine::managed::port_from_url(&args.llama_url)?,
            gpu_index: gpu_index.map(str::to_owned),
            extra_args: engine::managed::args_con_posti(&args.llama_args, args.slots)?,
            log_path: args.data_dir.join("llama-server.log"),
            stop_grace: secs(args.stop_grace_secs),
            kill_wait: secs(args.kill_wait_secs),
        }),
        (EngineArg::Mock, LlamaMode::Managed) => bail!("--llama-mode managed requires --engine llama"),
        _ => None,
    };
    let sup = engine::supervisor::Supervisor::with_state(
        engine.clone(),
        engine::supervisor::SupervisorCfg {
            managed,
            expected_model: args.model_path.clone(),
            probe_interval: secs(args.probe_interval_secs),
            settle: secs(args.settle_secs),
            max_restarts: args.max_restarts,
            restart_window: Duration::from_secs(args.restart_window_mins * 60),
            backoff_base: secs(args.restart_backoff_secs),
            backoff_max: secs(120),
            recycle: (args.llama_recycle_hours > 0.0).then(|| Duration::from_secs_f64(args.llama_recycle_hours * 3600.0)),
            start_timeout: secs(args.engine_start_timeout_secs),
            maintenance_tick: secs(1),
        },
        stats,
        progress,
        engine_state,
        track_gpu.then(|| telemetria_gpu.clone()),
    );
    // Check the engine before connecting so the first advertised state reflects its readiness.
    // Recovery continues in the background if the initial probe does not pass.
    let start_engine = async {
        tokio::select! {
            result = sup.start() => result,
            _ = uscita_rx.wait_for(|stop| *stop) => Ok(()),
        }
    };
    match while_offline(&live, store.as_ref(), &contribution, start_engine).await {
        Some(Ok(())) if !*uscita_rx.borrow() => {}
        outcome => {
            sup.shutdown().await;
            return match outcome {
                Some(result) => result.map(|()| 0),
                None => Ok(MODEL_SWITCH_EXIT_CODE),
            };
        }
    }
    // Report runtime and launch settings for observational verification. Launch settings
    // are available only when the agent owns the engine process.
    let settings = if matches!((args.engine, args.llama_mode), (EngineArg::Llama, LlamaMode::Managed)) {
        engine::managed::settings_from_args(&args.llama_args)
    } else {
        Default::default()
    };
    let build = if track_gpu && sup.state() != engine::supervisor::EngineState::Ready {
        None
    } else {
        engine.build_info().await
    };
    hardware::describe_engine(&mut hw, build, settings);

    // Probe once at startup and share only the NAT classification.
    let nat_class = if args.senza_prova_nat {
        nat::NatClass::Sconosciuto
    } else {
        nat::classifica(&stun_config(&args)).await
    };
    info!("NAT type: {nat_class}");
    // Share one bridge endpoint across the node's assignments. Its key stays in
    // the data directory and only the public endpoint ID is advertised.
    let bridge = match (&args.relay, &stages) {
        (Some(relay), Some(s)) => {
            let chiave = bridge::load_key(&args.data_dir.join("iroh.key"))?;
            let p = bridge::Bridge::new(relay, chiave).await?;
            if !p.in_linea(Duration::from_secs(30)).await {
                warn!("relay {relay} unreachable after 30 s; the bridge will retry");
            }
            info!(id = %p.id(), relay = %relay, "iroh chain bridge ready");
            s.set_bridge(p.clone());
            Some(p)
        }
        (Some(_), None) => {
            // Shared configuration may set a relay even on nodes that never start a stage.
            warn!("--relay requires --stage-bin; ignoring it");
            None
        }
        _ => None,
    };
    let cfg = session::NodeConfig {
        model,
        costs,
        hardware: hw,
        sup: sup.clone(),
        stages,
        stage_engine: stage_engine.filter(|_| !stage_only),
        region: args.region.clone(),
        nat_class: nat_class.as_str().to_string(),
        iroh_node_id: bridge.as_ref().map(|p| p.id()).unwrap_or_default(),
        slots: args.slots as usize,
        contribution,
        gpu: telemetria_gpu,
        model_path: args.model_path.clone(),
        output: uscita_rx,
        live,
        model_store: store,
    };
    let shared = Arc::new(session::Shared::default());

    let live = cfg.live.clone();
    let result = connect_loop(&args, &t, cfg, &engine, &shared, stop_rx).await;
    if let Some(p) = &bridge {
        p.close().await;
    }
    sup.shutdown().await;
    result?;
    Ok(if live.changes.committed() { MODEL_SWITCH_EXIT_CODE } else { 0 })
}

async fn connect_loop(
    args: &Args,
    t: &transport::Transport,
    mut cfg: session::NodeConfig,
    engine: &Arc<engine::Engine>,
    shared: &Arc<session::Shared>,
    stop_rx: tokio::sync::watch::Receiver<bool>,
) -> Result<()> {
    let mut backoff = Duration::from_secs(1);
    let live = cfg.live.clone();
    loop {
        let store = cfg.model_store.clone();
        let contribution = cfg.contribution.clone();
        let attempt = connect(args, t, &mut cfg, engine, &stop_rx, &mut backoff);
        let Some(result) = while_offline(&live, store.as_ref(), &contribution, attempt).await else {
            return Ok(());
        };
        let Some((conn, id)) = result? else { return Ok(()) };
        info!(node = %id.node_id, "connected to the coordinator");
        if let Some(s) = &cfg.stages {
            // Share the enrolled identity with stage processes for mutual TLS.
            if let Err(e) = s.set_identity(&id) {
                warn!("could not write stage TLS files: {e:#}");
            }
        }
        backoff = Duration::from_secs(1);
        match session::run(conn, &cfg, engine.clone(), shared.clone(), stop_rx.clone()).await {
            session::SessionEnd::Shutdown => return Ok(()),
            session::SessionEnd::Reconnect => continue,
            session::SessionEnd::SwitchModel(change) => {
                if cfg.commit_model(change).await {
                    return Ok(());
                }
            }
            session::SessionEnd::Banned => bail!("coordinator banned this node"),
            session::SessionEnd::TroppoVecchio(msg) => bail!("{msg}"),
            session::SessionEnd::Unauthenticated => {
                warn!("coordinator did not recognize the node; enrolling again");
                identity::forget(&args.data_dir)?;
            }
            session::SessionEnd::Lost(e) => {
                warn!("session lost: {e:#}");
                let sleep = sleep_or_stop(&stop_rx, &cfg.output, &mut backoff);
                if while_offline(&live, cfg.model_store.as_ref(), &cfg.contribution, sleep).await.is_none() {
                    return Ok(());
                }
            }
        }
    }
}

// The caller owns the connection and engine lifecycle. Between sessions there is no work to drain.
async fn while_offline<T>(
    live: &control::Live,
    store: Option<&Arc<models::Store>>,
    contribution: &contribution::Contribution,
    work: impl std::future::Future<Output = T>,
) -> Option<T> {
    let Some(store) = store else { return Some(work.await) };
    tokio::pin!(work);
    loop {
        tokio::select! {
            biased;
            change = live.changes.wait_switch() => {
                if live.changes.claim(&change).await {
                    let committed = live.changes.commit(change, store.clone(), contribution).await;
                    live.notify();
                    if committed {
                        return None;
                    }
                }
            }
            result = &mut work => return Some(result),
        }
    }
}

async fn connect(
    args: &Args,
    t: &transport::Transport,
    cfg: &mut session::NodeConfig,
    engine: &Arc<engine::Engine>,
    stop_rx: &tokio::sync::watch::Receiver<bool>,
    backoff: &mut Duration,
) -> Result<Option<(quinn::Connection, identity::Identity)>> {
    let mut leaving = cfg.output.clone();
    loop {
        // Between sessions there is no active connection to notify of shutdown.
        if *stop_rx.borrow() || *cfg.output.borrow() {
            return Ok(None);
        }
        if cfg.gpu_profile_missing() && cfg.sup.state() == engine::supervisor::EngineState::Ready {
            let cards = cfg.gpu.letture();
            if let Some(card) = cards.first() {
                hardware::set_detected_gpu(&mut cfg.hardware, card, engine.build_info().await);
            }
        }
        let id = match identity::load(&args.data_dir)? {
            Some(id) => id,
            None => {
                if cfg.gpu_profile_missing() {
                    let mut state = cfg.sup.subscribe();
                    tokio::select! {
                        ready = state.wait_for(|state| *state == engine::supervisor::EngineState::Ready) => { ready?; }
                        _ = leaving.wait_for(|stop| *stop) => return Ok(None),
                    }
                    continue;
                }
                let result = tokio::select! {
                    id = enroll::enroll(t, &args.data_dir, cfg.hardware.clone(), args.owner_token.as_deref()) => id,
                    _ = leaving.wait_for(|stop| *stop) => return Ok(None),
                };
                match result {
                    Ok(id) => id,
                    Err(e) => {
                        warn!("enrollment failed: {e:#}");
                        sleep_or_stop(stop_rx, &cfg.output, backoff).await;
                        continue;
                    }
                }
            }
        };
        let conn = tokio::select! {
            conn = t.connect(&id) => conn,
            _ = leaving.wait_for(|stop| *stop) => return Ok(None),
        };
        let conn = match conn {
            Ok(c) => c,
            Err(e) => {
                warn!("connection failed: {e:#}");
                sleep_or_stop(stop_rx, &cfg.output, backoff).await;
                continue;
            }
        };
        return Ok(Some((conn, id)));
    }
}

/// Route Ctrl-C and, on Unix, SIGTERM into the same shutdown sequence.
async fn inoltra_segnali(tx: tokio::sync::mpsc::Sender<()>) {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        let mut term = signal(SignalKind::terminate()).ok();
        loop {
            let visto = match term.as_mut() {
                Some(t) => tokio::select! {
                    r = tokio::signal::ctrl_c() => r.is_ok(),
                    r = t.recv() => r.is_some(),
                },
                None => tokio::signal::ctrl_c().await.is_ok(),
            };
            if !visto || tx.send(()).await.is_err() {
                return;
            }
        }
    }
    #[cfg(not(unix))]
    {
        while tokio::signal::ctrl_c().await.is_ok() {
            if tx.send(()).await.is_err() {
                return;
            }
        }
    }
}

/// Request an acknowledged departure on the first signal. A second signal or
/// an expired grace period cancels the shutdown wait.
async fn arresto(
    mut segnali: tokio::sync::mpsc::Receiver<()>,
    uscita_tx: tokio::sync::watch::Sender<bool>,
    stop_tx: tokio::sync::watch::Sender<bool>,
    tetto: Duration,
) {
    if segnali.recv().await.is_none() {
        return;
    }
    info!("shutdown requested; notifying the coordinator");
    let _ = uscita_tx.send(true);
    tokio::select! {
        _ = segnali.recv() => info!("second signal received, exiting immediately"),
        _ = tokio::time::sleep(tetto) => warn!("shutdown grace period of {} s expired", tetto.as_secs()),
    }
    let _ = stop_tx.send(true);
}

async fn sleep_or_stop(stop: &tokio::sync::watch::Receiver<bool>, output: &tokio::sync::watch::Receiver<bool>, backoff: &mut Duration) {
    let mut stop = stop.clone();
    let mut output = output.clone();
    let jitter = Duration::from_millis(rand::random::<u64>() % 500);
    tokio::select! {
        _ = tokio::time::sleep(*backoff + jitter) => {}
        _ = stop.changed() => {}
        _ = output.changed() => {}
    }
    *backoff = (*backoff * 2).min(Duration::from_secs(60));
}

struct StartupWeights {
    model: ModelRef,
    path: Option<PathBuf>,
    restored: Option<models::ActiveModel>,
    error: Option<String>,
    needs_repair: bool,
}

fn verify_startup_weights(
    store: &models::Store,
    restore: bool,
    configured: Result<ModelRef>,
    path: Option<PathBuf>,
) -> Result<StartupWeights> {
    let mut error = None;
    if restore {
        match store.active() {
            Ok(Some((model, active))) => return Ok(StartupWeights {
                model: model.reference(), path: Some(store.path(model)), restored: Some(active), error: None, needs_repair: false,
            }),
            Ok(None) => {}
            Err(err) => {
                let text = format!("cannot restore the active model: {err:#}");
                warn!("{text}; using the configured model");
                error = Some(text);
            }
        }
    }
    let mut model = configured?;
    let mut needs_repair = false;
    if let Some(path) = &path {
        if let Some(known) = store.model_at(path) {
            model = known.reference();
            if let Err(err) = store.verify(known, || true) {
                let text = format!("cannot verify configured model weights: {err:#}");
                warn!("{text}");
                error = Some(error.map_or_else(|| text.clone(), |prior| format!("{prior}; {text}")));
                needs_repair = true;
            }
        } else {
            info!(path = %path.display(), "hashing model weights");
            model.weights_sha256 = sha256_file(path)?;
        }
    }
    Ok(StartupWeights { model, path, restored: None, error, needs_repair })
}

fn model_metadata(args: &Args) -> Result<ModelRef> {
    let (id, hash) = match (&args.model_path, &args.weights_sha256) {
        (Some(p), _) => {
            let id = args.model_id.clone().unwrap_or_else(|| p.file_stem().unwrap_or_default().to_string_lossy().into_owned());
            (id, vec![])
        }
        (None, Some(h)) => {
            let hash = hex::decode(h).context("--weights-sha256 must be hexadecimal")?;
            if hash.len() != 32 {
                bail!("--weights-sha256 must be 32 bytes");
            }
            (args.model_id.clone().context("--weights-sha256 requires --model-id")?, hash)
        }
        (None, None) => bail!("indicare --model-path oppure --model-id e --weights-sha256"),
    };
    Ok(ModelRef { model_id: id, quantization: args.quantization.clone(), weights_sha256: hash })
}

fn sha256_file(p: &PathBuf) -> Result<Vec<u8>> {
    let mut f = std::fs::File::open(p).with_context(|| format!("open {}", p.display()))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(h.finalize().to_vec())
}

fn parse_gpu(s: &str) -> Result<hardware::ManualGpu> {
    let (name, gb) = s.rsplit_once(':').context("expected --gpu in \"name:GB\" format")?;
    let gb: f64 = gb.parse().context("invalid VRAM size in GB")?;
    Ok(hardware::ManualGpu { name: name.to_string(), vram_bytes: (gb * (1u64 << 30) as f64) as u64, driver_version: String::new() })
}

fn read_pem_cert(p: &PathBuf) -> Result<Vec<u8>> {
    let text = std::fs::read_to_string(p).with_context(|| format!("read {}", p.display()))?;
    let b64: String = text
        .lines()
        .skip_while(|l| !l.starts_with("-----BEGIN CERTIFICATE-----"))
        .skip(1)
        .take_while(|l| !l.starts_with("-----END CERTIFICATE-----"))
        .collect();
    if b64.is_empty() {
        bail!("{}: no PEM certificate", p.display());
    }
    base64_decode(&b64)
}

/// Decode the standard Base64 alphabet used by PEM files.
fn base64_decode(s: &str) -> Result<Vec<u8>> {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = Vec::with_capacity(s.len() * 3 / 4);
    let (mut acc, mut bits) = (0u32, 0u32);
    for c in s.bytes().filter(|c| !c.is_ascii_whitespace() && *c != b'=') {
        let v = T.iter().position(|x| *x == c).context("invalid base64 character")? as u32;
        acc = (acc << 6) | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::{mpsc, watch};

    #[test]
    fn owner_token_file_wins() {
        let path = std::env::temp_dir().join(format!("publicompute-owner-token-{}", std::process::id()));
        std::fs::write(&path, "from-file\n").unwrap();
        let token = read_owner_token(Some(&path), Some("from-env".into())).unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!(token.as_deref(), Some("from-file"));
    }

    /// The first signal requests departure. The second ends the grace period.
    #[tokio::test]
    async fn second_signal_exits_immediately() {
        let (tx, rx) = mpsc::channel(4);
        let (uscita_tx, uscita_rx) = watch::channel(false);
        let (stop_tx, stop_rx) = watch::channel(false);
        let task = tokio::spawn(arresto(rx, uscita_tx, stop_tx, Duration::from_secs(60)));
        tx.send(()).await.unwrap();
        let mut u = uscita_rx.clone();
        tokio::time::timeout(Duration::from_secs(2), u.wait_for(|v| *v)).await.expect("expected the departure signal").unwrap();
        tokio::time::sleep(Duration::from_millis(150)).await;
        assert!(!*stop_rx.borrow(), "first signal should allow graceful shutdown");
        let t0 = std::time::Instant::now();
        tx.send(()).await.unwrap();
        let mut s = stop_rx.clone();
        tokio::time::timeout(Duration::from_secs(2), s.wait_for(|v| *v)).await.expect("arresto").unwrap();
        assert!(t0.elapsed() < Duration::from_millis(500));
        task.await.unwrap();
    }

    /// The grace deadline ends shutdown even when no second signal arrives.
    #[tokio::test]
    async fn shutdown_deadline_forces_exit() {
        let (tx, rx) = mpsc::channel(4);
        let (uscita_tx, _uscita_rx) = watch::channel(false);
        let (stop_tx, stop_rx) = watch::channel(false);
        tokio::spawn(arresto(rx, uscita_tx, stop_tx, Duration::from_millis(200)));
        tx.send(()).await.unwrap();
        let mut s = stop_rx.clone();
        tokio::time::timeout(Duration::from_secs(2), s.wait_for(|v| *v)).await.expect("forced shutdown at the deadline").unwrap();
    }
}
