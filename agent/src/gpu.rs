//! The agent queries the driver utility in a child process for inventory and
//! telemetry. This keeps driver library loading outside the agent and lets
//! startup continue with an empty inventory when the utility or card is absent.
//! The coordinator receives only card names and numeric samples.

use std::{
    process::Stdio,
    sync::{Arc, Mutex},
    time::Duration,
};

use publicompute_proto::node::GpuTelemetry;
use tokio::{process::Command, sync::watch};
use tracing::debug;

#[derive(Debug, Clone, PartialEq)]
pub struct Lettura {
    pub name: String,
    pub vram_totale: u64,
    pub vram_usata: u64,
    pub uso_percento: u32,
    pub temperature: u32,
    pub potenza_w: u32,
    pub driver: String,
    pub uuid: String,
}

const CAMPI: &str = "name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw,driver_version,uuid";

/// Interval between telemetry samples, in seconds.
pub const PERIODO: Duration = Duration::from_secs(10);

/// Parse one CSV record per card from the driver utility. Skip malformed
/// records and use zero for readings the driver marks unavailable.
pub fn interpreta(output: &str) -> Vec<Lettura> {
    const MIB: u64 = 1 << 20;
    output
        .lines()
        .filter_map(|row| {
            let c: Vec<&str> = row.split(',').map(str::trim).collect();
            if !(7..=8).contains(&c.len()) || c[0].is_empty() {
                return None;
            }
            let num = |s: &str| s.parse::<f64>().ok().filter(|v| v.is_finite() && *v >= 0.0);
            Some(Lettura {
                name: c[0].to_string(),
                vram_totale: num(c[1])? as u64 * MIB,
                vram_usata: num(c[2]).unwrap_or(0.0) as u64 * MIB,
                uso_percento: num(c[3]).unwrap_or(0.0) as u32,
                temperature: num(c[4]).unwrap_or(0.0) as u32,
                potenza_w: num(c[5]).unwrap_or(0.0).round() as u32,
                driver: c[6].to_string(),
                uuid: c.get(7).copied().unwrap_or_default().to_string(),
            })
        })
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReadFailure {
    NoDevice,
    Error,
    Timeout,
}

impl ReadFailure {
    pub fn reason(self) -> &'static str {
        match self {
            Self::NoDevice => "selected NVIDIA GPU not found",
            Self::Error => "nvidia-smi failed",
            Self::Timeout => "nvidia-smi timed out",
        }
    }
}

async fn query(fields: &str, selector: Option<&str>, limit: Duration) -> Result<String, ReadFailure> {
    let mut command = Command::new("nvidia-smi");
    command
        .args([&format!("--query-gpu={fields}"), "--format=csv,noheader,nounits"])
        .env("LC_ALL", "C")
        .stdin(Stdio::null())
        .kill_on_drop(true);
    if let Some(selector) = selector {
        command.args(["-i", selector]);
    }
    let output = match tokio::time::timeout(limit, command.output()).await {
        Ok(Ok(output)) => output,
        Ok(Err(e)) => {
            debug!("nvidia-smi: {e}");
            return Err(ReadFailure::Error);
        }
        Err(_) => return Err(ReadFailure::Timeout),
    };
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    if [&*stdout, &*stderr].iter().any(|text| {
        text.contains("No devices were found") || text.contains("No devices found")
    }) {
        return Err(ReadFailure::NoDevice);
    }
    if !output.status.success() {
        debug!(status = %output.status, "nvidia-smi failed");
        return Err(ReadFailure::Error);
    }
    if stdout.trim().is_empty() {
        return Err(ReadFailure::NoDevice);
    }
    Ok(stdout.into_owned())
}

pub async fn resolve_index(index: &str) -> Option<String> {
    let uuid = query("uuid", Some(index), LIMITE).await.ok()?;
    let uuid = uuid.trim();
    (uuid.starts_with("GPU-") && uuid.len() == 40).then(|| uuid.to_owned())
}

pub async fn read(uuid: Option<&str>) -> Result<Vec<Lettura>, ReadFailure> {
    let mut readings = interpreta(&query(CAMPI, uuid, LIMITE).await?);
    readings.retain(|card| card.uuid.starts_with("GPU-"));
    if readings.is_empty() {
        return Err(ReadFailure::Error);
    }
    Ok(readings)
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum Device {
    #[default]
    Disabled,
    Checking,
    Ready(String),
    Lost(ReadFailure),
}

#[derive(Clone, Debug, Default)]
pub struct DeviceState {
    pub device: Device,
    pub loss_generation: u64,
}

#[derive(Default)]
struct Monitor {
    uuid: Option<String>,
    missing: u8,
    telemetry_failures: u8,
    state: DeviceState,
}

impl Monitor {
    async fn sample<F: std::future::Future<Output = bool>>(
        &mut self,
        reading: Result<&[Lettura], ReadFailure>,
        present: impl FnOnce(Option<String>) -> F,
    ) -> Option<Lettura> {
        let selected = reading.and_then(|cards| {
            match &self.uuid {
                Some(uuid) => cards.iter().find(|card| &card.uuid == uuid),
                None => cards.first(),
            }
            .ok_or(ReadFailure::NoDevice)
        });
        match selected {
            Ok(card) => {
                self.uuid = Some(card.uuid.clone());
                self.state.device = Device::Ready(card.uuid.clone());
                self.missing = 0;
                self.telemetry_failures = 0;
                Some(card.clone())
            }
            Err(reason) => {
                let lost = if reason == ReadFailure::NoDevice {
                    self.telemetry_failures = 0;
                    self.missing = self.missing.saturating_add(1);
                    self.missing >= 3
                } else {
                    self.missing = 0;
                    self.telemetry_failures = self.telemetry_failures.saturating_add(1);
                    self.telemetry_failures >= 3 && !present(self.uuid.clone()).await
                };
                if lost && !matches!(self.state.device, Device::Lost(_)) {
                    self.state.loss_generation += 1;
                    self.state.device = Device::Lost(reason);
                }
                None
            }
        }
    }
}

/// Bound each driver query so a stalled utility cannot hold up sampling indefinitely.
const LIMITE: Duration = Duration::from_secs(5);

impl Lettura {
    pub fn a_telemetria(&self, index: u32) -> GpuTelemetry {
        GpuTelemetry {
            index: index,
            utilization_percent: self.uso_percento.min(100),
            temperature_celsius: self.temperature,
            power_watts: self.potenza_w,
            vram_used_bytes: self.vram_usata,
        }
    }
}

/// Share the latest readings across coordinator reconnects.
#[derive(Default)]
pub struct Telemetria {
    ultima: Mutex<Vec<GpuTelemetry>>,
    /// Keep inventory fields with the readings for the local control socket.
    letture: Mutex<Vec<Lettura>>,
    device: watch::Sender<DeviceState>,
}

impl Telemetria {
    pub fn device_rx(&self) -> watch::Receiver<DeviceState> {
        self.device.subscribe()
    }

    pub fn loss(&self) -> Option<ReadFailure> {
        match self.device.borrow().device {
            Device::Lost(reason) => Some(reason),
            _ => None,
        }
    }

    pub fn unavailable(&self) -> bool {
        matches!(self.device.borrow().device, Device::Checking | Device::Lost(_))
    }

    pub fn monitored(&self) -> bool {
        self.device.borrow().device != Device::Disabled
    }

    #[cfg(test)]
    pub fn set_device_for_test(&self, device: Device) {
        self.device.send_modify(|state| {
            if matches!(device, Device::Lost(_)) && !matches!(state.device, Device::Lost(_)) {
                state.loss_generation += 1;
            }
            state.device = device;
        });
    }

    pub fn imposta(&self, letture: &[Lettura]) {
        *self.letture.lock().unwrap() = letture.to_vec();
        *self.ultima.lock().unwrap() = letture.iter().enumerate().map(|(i, l)| l.a_telemetria(i as u32)).collect();
    }

    pub fn letture(&self) -> Vec<Lettura> {
        self.letture.lock().unwrap().clone()
    }

    /// Convert the current sample into heartbeat telemetry. An empty inventory yields no entries.
    pub fn gpus(&self) -> Vec<GpuTelemetry> {
        self.ultima.lock().unwrap().clone()
    }
}

async fn publish(t: &Telemetria, monitor: &mut Monitor, managed: bool, reading: Result<Vec<Lettura>, ReadFailure>) {
    if !managed {
        t.imposta(reading.as_deref().unwrap_or_default());
        return;
    }
    let selected = monitor.sample(reading.as_deref().map_err(|reason| *reason), |uuid| async move {
        query("name", uuid.as_deref(), Duration::from_secs(2)).await.is_ok()
    }).await;
    if selected.is_some() || matches!(monitor.state.device, Device::Lost(_)) {
        t.imposta(selected.as_slice());
    }
    t.device.send_if_modified(|state| {
        if state.device == monitor.state.device && state.loss_generation == monitor.state.loss_generation {
            return false;
        }
        *state = monitor.state.clone();
        true
    });
}

pub async fn avvia_campionatore(t: Arc<Telemetria>, first: Result<Vec<Lettura>, ReadFailure>, managed: bool, uuid: Option<String>) {
    let mut monitor = Monitor {
        uuid,
        state: DeviceState { device: Device::Checking, loss_generation: 0 },
        ..Default::default()
    };
    publish(&t, &mut monitor, managed, first).await;
    tokio::spawn(async move {
        let start = tokio::time::Instant::now() + PERIODO;
        let mut tick = tokio::time::interval_at(start, PERIODO);
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            tick.tick().await;
            let reading = read(monitor.uuid.as_deref()).await;
            publish(&t, &mut monitor, managed, reading).await;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn gpu_lost_after_three_failures() {
        let mut monitor = Monitor::default();
        for reason in [ReadFailure::Error, ReadFailure::Timeout] {
            monitor.sample(Err(reason), |_| async { panic!("presence checked too early") }).await;
        }
        assert_eq!(monitor.state.loss_generation, 0);
        monitor.sample(Err(ReadFailure::Timeout), |_| async { false }).await;
        assert_eq!(monitor.state.device, Device::Lost(ReadFailure::Timeout));
        assert_eq!(monitor.state.loss_generation, 1);
    }

    #[tokio::test]
    async fn gpu_back_resumes() {
        let mut monitor = Monitor::default();
        let cards = interpreta("A, 8192, 0, 0, 40, 20, 555, GPU-a");
        monitor.sample(Ok(&cards), |_| async { false }).await;
        for _ in 0..3 {
            monitor.sample(Ok(&interpreta("B, 8192, 0, 0, 40, 20, 555, GPU-b")), |_| async {
                panic!("a different GPU does not count as presence")
            }).await;
        }
        assert_eq!(monitor.state.device, Device::Lost(ReadFailure::NoDevice));
        monitor.sample(Ok(&cards), |_| async { false }).await;
        assert_eq!(monitor.state.device, Device::Ready("GPU-a".into()));
        assert_eq!(monitor.state.loss_generation, 1);
    }

    #[tokio::test]
    async fn blocked_telemetry_during_inference_does_not_lose_gpu() {
        use crate::engine::{mock::MockEngine, progress::Phase, Limits};

        let mut monitor = Monitor::default();
        let cards = interpreta("A, 8192, 0, 0, 40, 20, 555, GPU-a");
        monitor.sample(Ok(&cards), |_| async { false }).await;
        let second = Duration::from_secs(1);
        let engine = MockEngine {
            per_token_delay: Duration::from_millis(10),
            cheat: false,
            hang: Default::default(),
            limits: Limits {
                prefill_stall: second, decode_stall: second, request_max: second,
                call_timeout: second, step_timeout: second, probe_timeout: second,
            },
            progress: Default::default(),
        };
        let request = Default::default();
        let inference = engine.generate(&request, 1);
        tokio::pin!(inference);
        tokio::select! {
            biased;
            _ = &mut inference => panic!("expected inference in progress"),
            _ = async {
                assert_eq!(engine.progress.snapshot().phase, Phase::Decode);
                for _ in 0..6 {
                    monitor.sample(Err(ReadFailure::Timeout), |uuid| async move {
                        uuid.as_deref() == Some("GPU-a")
                    }).await;
                }
            } => {}
        }
        assert_eq!(monitor.state.device, Device::Ready("GPU-a".into()));
        assert_eq!(monitor.state.loss_generation, 0);
        assert!(inference.await.is_ok());
    }

    #[test]
    fn parses_one_gpu() {
        let l = interpreta("NVIDIA GeForce RTX 2080 SUPER, 8192, 1024, 37, 61, 85.43, 555.42\n");
        assert_eq!(l.len(), 1);
        assert_eq!(l[0].name, "NVIDIA GeForce RTX 2080 SUPER");
        assert_eq!(l[0].vram_totale, 8192 << 20);
        assert_eq!(l[0].vram_usata, 1024 << 20);
        assert_eq!((l[0].uso_percento, l[0].temperature, l[0].potenza_w), (37, 61, 85));
        assert_eq!(l[0].driver, "555.42");
        let t = l[0].a_telemetria(0);
        assert_eq!((t.utilization_percent, t.temperature_celsius, t.power_watts, t.vram_used_bytes), (37, 61, 85, 1024 << 20));
    }

    #[test]
    fn unavailable_values_and_malformed_rows() {
        let l = interpreta("RTX Mobile, 6144, 0, 0, 40, [N/A], 551.1\nspazzatura\n\nRTX B, [N/A], 0, 0, 0, 0, 1\n");
        // Unavailable power is a valid reading. A missing capacity makes the record unusable.
        assert_eq!(l.len(), 1);
        assert_eq!(l[0].potenza_w, 0);
    }

    #[test]
    fn telemetry_is_empty_without_gpu() {
        assert!(interpreta("").is_empty());
        let t = Telemetria::default();
        assert!(t.gpus().is_empty());
        t.imposta(&interpreta("A, 1000, 10, 5, 50, 20, 1\nB, 2000, 20, 6, 51, 21, 1"));
        let g = t.gpus();
        assert_eq!(g.len(), 2);
        assert_eq!(g[1].index, 1);
    }
}
