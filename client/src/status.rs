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
    /// Earliest time a preference change can take effect, in Unix seconds.
    pub stable_until: Option<i64>,
}

#[derive(Deserialize, Clone, Debug, PartialEq)]
pub struct Summary {
    pub days: u32,
    pub free_value_mc: i64,
    pub free_chats: u32,
    pub earned_mc: i64,
    pub service_seconds: u64,
    pub work_seconds: u64,
}

#[derive(Deserialize, Clone, Debug, PartialEq)]
pub struct Status {
    #[serde(default)]
    pub connected: bool,
    #[serde(default)]
    pub paused: bool,
    #[serde(default)]
    pub preference: Preference,
    #[serde(default)]
    pub engine: String,
    pub loaded_model: Option<String>,
    #[serde(default)]
    pub active_leases: u32,
    pub chain_id: Option<String>,
    pub gpu: Option<Gpu>,
    #[serde(default)]
    pub gpu_lost: bool,
    pub contribution: Option<Contribution>,
    pub summary: Option<Summary>,
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

pub fn status_line(s: &Status) -> (String, Tone) {
    use Tone::*;
    let loaded = s.loaded_model.clone().unwrap_or_else(|| "the model".into());
    let chosen = s.preference.model.clone().unwrap_or_else(|| loaded.clone());
    if s.gpu_lost {
        return (GPU_LOST.into(), Warn);
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
        return ("Paused — resume with [P]".into(), Warn);
    }
    if s.engine == "recovering" {
        return ("Recovering the engine".into(), Warn);
    }
    let Some(c) = &s.contribution else {
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
        "MODE_FREE" if s.preference.mode == "free" => (format!("FARM mode — serving {serving}"), Good),
        "MODE_FREE" => (format!("FARM mode (automatic) — back to use at the next chat on {chosen}"), Plain),
        "NOT_PLACED" => (
            format!("Serving {serving}. {chosen} is paid in credits: the card was not placed in one of its chains"),
            Plain,
        ),
        "UNVERIFIED" => ("Serving, checks in progress — free seat after the first checks".into(), Plain),
        "BELOW_QUALITY" => (format!("Serving below the speed threshold — no free seat on {chosen}"), Warn),
        "NO_ACCOUNT" => ("Card without a linked user — link your user on the site".into(), Warn),
        "MODEL_MISSING" => (format!("{chosen} is not installed on this PC"), Warn),
        "SUSPENDED" => ("Free seat suspended for 24 h after abrupt exits".into(), Warn),
        _ => (format!("Serving {serving}"), Plain),
    }
}
