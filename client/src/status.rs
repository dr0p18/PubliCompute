//! Decode the subset of the agent's status that the client displays.
//! Unknown fields can be ignored because the summary only uses local fields.

use serde::Deserialize;

#[derive(Deserialize, Clone, Debug, Default, PartialEq)]
pub struct Preference {
    #[serde(default)]
    pub mode: String,
    pub model: Option<String>,
    pub quant: Option<String>,
}

#[derive(Deserialize, Clone, Debug, PartialEq)]
pub struct Gpu {
    pub name: String,
    pub vram_used_bytes: u64,
    pub vram_total_bytes: u64,
    pub temperature_celsius: u32,
}

#[derive(Deserialize, Clone, Debug, PartialEq)]
pub struct Contribution {
    #[serde(default)]
    pub serving: Option<String>,
    #[serde(default)]
    pub in_chain: bool,
    #[serde(default)]
    pub free_seat: bool,
    #[serde(default)]
    pub reason: String,
    /// Coordinator explanation, already in English; absent from older agents.
    #[serde(default)]
    pub detail: String,
    /// Earliest time a preference change can take effect, in Unix seconds.
    pub stable_until: Option<i64>,
}

#[derive(Deserialize, Clone, Debug, PartialEq)]
pub struct Summary {
    pub days: u32,
    pub free_chats: u32,
    pub service_seconds: u64,
    pub work_seconds: u64,
}

#[derive(Deserialize, Clone, Debug, PartialEq)]
pub struct Network {
    pub download_bits_per_sec: u64,
    pub upload_bits_per_sec: u64,
    pub bandwidth_measured_at: Option<u64>,
    pub coordinator_rtt_micros: Option<u32>,
}

#[derive(Deserialize, Clone, Debug, PartialEq)]
pub struct Download {
    id: String,
    state: String,
    bytes: u64,
    total_bytes: u64,
    error: Option<String>,
}

#[derive(Deserialize, Clone, Debug, PartialEq)]
pub struct Status {
    pub connected: bool,
    pub update_required: Option<String>,
    pub paused: bool,
    #[serde(default)]
    pub preference: Preference,
    #[serde(default)]
    pub engine: String,
    pub loaded_model: Option<String>,
    pub loaded_quant: Option<String>,
    // Verified local weights; in chain mode loaded_model waits for a stage.
    #[serde(default)]
    pub weights_model: Option<String>,
    #[serde(default)]
    pub weights_quant: Option<String>,
    pub weights_sha256: Option<String>,
    pub download: Option<Download>,
    pub model_error: Option<String>,
    #[serde(default)]
    pub active_leases: u32,
    pub chain_id: Option<String>,
    pub gpu: Option<Gpu>,
    #[serde(default)]
    pub gpu_lost: bool,
    pub contribution: Option<Contribution>,
    pub summary: Option<Summary>,
    pub network: Option<Network>,
}

impl Status {
    pub fn update_message(&self) -> Option<String> {
        self.update_required.as_ref().map(|reason| format!("Update required: {reason} — run the installer again"))
    }

    pub fn download_line(&self) -> Option<String> {
        let d = self.download.as_ref()?;
        let gib = |bytes| bytes as f64 / (1u64 << 30) as f64;
        Some(match d.state.as_str() {
            "failed" => format!("! download: {} failed: {}", d.id, d.error.as_deref().unwrap_or("unknown error")),
            "switching" => format!("switching to {} after current work finishes", d.id),
            _ => {
                let verification = if d.state == "verifying" { " (verifying)" } else { "" };
                format!("download: {} {:.1}/{:.1} GiB{verification}", d.id, gib(d.bytes), gib(d.total_bytes))
            }
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Tone {
    Good,
    Plain,
    Warn,
}

/// The agent reports busy without a cause, so the summary uses temperature to identify heat.
const HOT_CELSIUS: u32 = 85;

/// Evaluate the status conditions in priority order and display the first match.
pub const GPU_LOST: &str = "GPU not available: check the driver, then sudo systemctl restart publicompute-agent";

pub fn status_text(status: &Status, now: u64) -> String {
    let (line, tone) = status_line(status);
    let mut text = if tone == Tone::Warn { format!("! {line}") } else { line };
    if let Some(detail) = status.download_line() {
        text.push('\n');
        text.push_str(&detail);
    }
    if let Some(error) = &status.model_error {
        text.push_str(&format!("\n! {error}"));
    }
    let Some(network) = &status.network else { return text };
    let rate = |bits: u64| {
        if bits == 0 { "?".into() } else { format!("{:.1}", bits as f64 / 1e6) }
    };
    let rtt = network.coordinator_rtt_micros.map_or_else(|| "?".into(), |micros| format!("{:.1}", f64::from(micros) / 1000.0));
    text.push_str(&format!(
        "\nnetwork: {}/{} Mbit/s, {rtt} ms",
        rate(network.download_bits_per_sec), rate(network.upload_bits_per_sec)
    ));
    let fresh = network.bandwidth_measured_at
        .and_then(|at| now.checked_sub(at))
        .is_some_and(|age| age <= 8 * 86400);
    if !fresh || network.download_bits_per_sec.min(network.upload_bits_per_sec) < 100_000_000 {
        text.push_str("\nchains: below the network standard (100 Mbit/s up and down)");
    }
    text
}

pub fn status_line(s: &Status) -> (String, Tone) {
    use Tone::*;
    if let Some(message) = s.update_message() {
        return (message, Warn);
    }
    let loaded = s.loaded_model.clone().unwrap_or_else(|| "the model".into());
    let chosen = s.preference.model.clone().unwrap_or_else(|| loaded.clone());
    if s.gpu_lost {
        return (GPU_LOST.into(), Warn);
    }
    if s.engine == "verifying weights" {
        return ("Verifying weights".into(), Plain);
    }
    if s.engine == "weights error" {
        return ("Model weights unavailable".into(), Warn);
    }
    if s.engine == "starting" {
        return (format!("Starting {chosen}"), Plain);
    }
    if !s.connected {
        return ("Not connected to the coordinator — retrying".into(), Warn);
    }
    if s.engine == "quarantined" {
        return ("Engine stopped: the agent needs a restart".into(), Warn);
    }
    if s.paused {
        return ("Paused — resume with [P]".into(), Plain);
    }
    if s.engine == "recovering" {
        return ("Recovering the engine".into(), Warn);
    }
    let Some(c) = &s.contribution else {
        if s.preference.mode == "free" {
            return ("Free card — waiting for work".into(), Plain);
        }
        return (format!("Serving {loaded}"), Plain);
    };
    let serving = c.serving.clone().unwrap_or_else(|| loaded.clone());
    let reason = c.reason.strip_prefix("FREE_SEAT_REASON_").unwrap_or(&c.reason);
    if reason == "BUSY" {
        return match s.gpu.as_ref().map(|g| g.temperature_celsius) {
            Some(t) if t >= HOT_CELSIUS => (format!("Card hot ({t} °C) — it resumes by itself"), Warn),
            _ => ("Card busy with another program — the free seat is suspended".into(), Warn),
        };
    }
    if c.free_seat {
        let chain = if c.in_chain { " (in chain)" } else { "" };
        return (format!("Free seat active on {serving}{chain}"), Good);
    }
    match reason {
        "WARMING_UP" => (format!("Starting {serving} — free seat shortly"), Plain),
        "MODE_FREE" if s.preference.mode == "free" => match c.serving.as_deref() {
            Some(model) => (format!("Free card — serving {model}"), Good),
            None => ("Free card — waiting for work".into(), Plain),
        },
        "MODE_FREE" => (format!("Free card (automatic) — back to use at the next chat on {chosen}"), Plain),
        "NOT_PLACED" => (
            detail_or(&c.detail, format!("Serving {serving}. The card was not placed in a chain of {chosen}")),
            Plain,
        ),
        "UNVERIFIED" => ("Serving, checks in progress — free seat after the first checks".into(), Plain),
        "BELOW_QUALITY" => (format!("Serving below the speed threshold — no free seat on {chosen}"), Warn),
        "NO_ACCOUNT" => ("Card without a linked user — link your user on the site".into(), Warn),
        "MODEL_MISSING" => (format!("{chosen} is not installed on this PC"), Warn),
        "SUSPENDED" => ("Free seat suspended for 24 h after abrupt exits".into(), Warn),
        // The coordinator explains these with live numbers, so its text wins over ours.
        "WAITING_PEERS" => (detail_or(&c.detail, format!("Waiting for other cards to form a chain of {chosen}")), Plain),
        "CHAIN_FULL" => (detail_or(&c.detail, format!("The chains of {chosen} are full — waiting for a seat")), Plain),
        "NETWORK_BELOW_STANDARD" => (
            detail_or(&c.detail, "Network below the chain standard (100 Mbit/s each way, 25 ms)".into()),
            Warn,
        ),
        "VRAM_TOO_SMALL" => (detail_or(&c.detail, format!("This card cannot hold a stage of {chosen}")), Warn),
        "NAT_BLOCKED" => (
            detail_or(&c.detail, "The router blocks incoming connections — the card cannot join a chain".into()),
            Warn,
        ),
        // A reason newer than this client: the detail is the only explanation we have.
        _ => (detail_or(&c.detail, format!("Serving {serving}")), Plain),
    }
}

/// The coordinator's detail with the first letter capitalized, or the fallback when it is empty.
fn detail_or(detail: &str, fallback: String) -> String {
    let mut chars = detail.trim().chars();
    match chars.next() {
        Some(first) => first.to_uppercase().chain(chars).collect(),
        None => fallback,
    }
}
