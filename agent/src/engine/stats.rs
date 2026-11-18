//! Engine counters track both process-lifetime and cumulative totals.
//! Write the cumulative snapshot by rename so readers only see complete JSON.

use std::{
    collections::BTreeMap,
    path::PathBuf,
    sync::Mutex,
    time::Duration,
};

use publicompute_proto::node::EngineHealth;
use serde::{Deserialize, Serialize};
use tracing::warn;

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Counters {
    pub requests_completed: u64,
    pub requests_failed: u64,
    /// Failures grouped by request error code or engine fault.
    pub failed_by_cause: BTreeMap<String, u64>,
    pub requests_cancelled: u64,
    pub engine_restarts: u64,
    pub probes_ok: u64,
    pub probes_failed: u64,
    /// Duration sum and count for calculating mean generation time.
    pub inference_ms_total: u64,
    pub inference_count: u64,
}

impl Counters {
    #[allow(dead_code)] // Used by callers that open the process-lifetime snapshot directly.
    pub fn mean_inference_ms(&self) -> Option<f64> {
        (self.inference_count > 0).then(|| self.inference_ms_total as f64 / self.inference_count as f64)
    }
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct FileFormat {
    cumulative: Counters,
}

pub struct Stats {
    path: Option<PathBuf>,
    inner: Mutex<(Counters, Counters)>, // The tuple holds process-lifetime counters first and persistent totals second.
}

impl Stats {
    /// Without a directory, keep both sets of counters in memory.
    pub fn new(dir: Option<&std::path::Path>) -> Self {
        let path = dir.map(|d| d.join("engine-stats.json"));
        let cumulative = path
            .as_ref()
            .and_then(|p| std::fs::read(p).ok())
            .and_then(|b| serde_json::from_slice::<FileFormat>(&b).ok())
            .map(|f| f.cumulative)
            .unwrap_or_default();
        Self { path, inner: Mutex::new((Counters::default(), cumulative)) }
    }

    fn update(&self, f: impl Fn(&mut Counters)) {
        let mut g = self.inner.lock().unwrap();
        f(&mut g.0);
        f(&mut g.1);
        if let Some(p) = &self.path {
            if let Err(e) = write_atomic(p, &FileFormat { cumulative: g.1.clone() }) {
                warn!("could not save counters: {e:#}");
            }
        }
    }

    pub fn request_completed(&self, inference: Duration) {
        self.update(|c| {
            c.requests_completed += 1;
            c.inference_count += 1;
            c.inference_ms_total += inference.as_millis() as u64;
        });
    }

    pub fn request_failed(&self, cause: &str) {
        self.update(|c| {
            c.requests_failed += 1;
            *c.failed_by_cause.entry(cause.to_string()).or_default() += 1;
        });
    }

    pub fn request_cancelled(&self) {
        self.update(|c| c.requests_cancelled += 1);
    }

    pub fn engine_restart(&self) {
        self.update(|c| c.engine_restarts += 1);
    }

    pub fn probe(&self, ok: bool) {
        self.update(|c| if ok { c.probes_ok += 1 } else { c.probes_failed += 1 });
    }

    pub fn run(&self) -> Counters {
        self.inner.lock().unwrap().0.clone()
    }

    #[allow(dead_code)] // Operators can read cumulative totals from the on-disk snapshot.
    pub fn cumulative(&self) -> Counters {
        self.inner.lock().unwrap().1.clone()
    }

    pub fn health(&self, seconds_since_progress: u32) -> EngineHealth {
        let c = self.run();
        EngineHealth {
            requests_completed: c.requests_completed,
            requests_failed: c.requests_failed,
            requests_cancelled: c.requests_cancelled,
            engine_restarts: c.engine_restarts,
            probe_failures: c.probes_failed,
            seconds_since_progress,
        }
    }
}

fn write_atomic(path: &std::path::Path, data: &FileFormat) -> anyhow::Result<()> {
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_vec_pretty(data)?)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cumulative_survives_restart_and_run_does_not() {
        let dir = std::env::temp_dir().join(format!("publicompute-stats-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let _ = std::fs::remove_file(dir.join("engine-stats.json"));
        {
            let s = Stats::new(Some(&dir));
            s.request_completed(Duration::from_millis(100));
            s.request_completed(Duration::from_millis(300));
            s.request_failed("watchdog");
            s.probe(false);
            assert_eq!(s.run().mean_inference_ms(), Some(200.0));
        }
        let s = Stats::new(Some(&dir));
        assert_eq!(s.run(), Counters::default());
        assert_eq!(s.cumulative().requests_completed, 2);
        assert_eq!(s.cumulative().failed_by_cause["watchdog"], 1);
        assert!(!dir.join("engine-stats.json.tmp").exists());
        std::fs::remove_dir_all(&dir).ok();
    }
}
