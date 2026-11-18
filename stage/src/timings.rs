//! Record process-wide time by phase in nanoseconds. Decode covers the
//! engine call. Copy covers batch setup and state reads. Logit access may
//! synchronize with the accelerator, so measure it separately from
//! sampling and from idle network waits.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::{Duration, Instant};

#[derive(Clone, Copy)]
pub enum Fase {
    Decode,
    Campiona,
    Sincro,
    Attesa,
}

static NS: [AtomicU64; 5] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
static BYTE_STATO_INVIATI: AtomicU64 = AtomicU64::new(0);
static RIGHE_STATO_INVIATE: AtomicU64 = AtomicU64::new(0);
static BYTE_STATO_RICEVUTI: AtomicU64 = AtomicU64::new(0);

/// Count state bytes forwarded (only `state`, excluding headers) and the rows (tokens) they contain.
pub fn conta_stato_inviato(byte: usize, rows: usize) {
    BYTE_STATO_INVIATI.fetch_add(byte as u64, Relaxed);
    RIGHE_STATO_INVIATE.fetch_add(rows as u64, Relaxed);
}

pub fn conta_stato_ricevuto(byte: usize) {
    BYTE_STATO_RICEVUTI.fetch_add(byte as u64, Relaxed);
}

static N_DECODE: AtomicU64 = AtomicU64::new(0);
// Draft batches at the tail have their own decode time and count.
static NS_MTP: AtomicU64 = AtomicU64::new(0);
static N_MTP: AtomicU64 = AtomicU64::new(0);

/// One MTP batch of the tail (`Motore::mtp_esegui`).
pub fn aggiungi_mtp(d: Duration) {
    NS_MTP.fetch_add(d.as_nanos() as u64, Relaxed);
    N_MTP.fetch_add(1, Relaxed);
}

pub fn aggiungi(f: Fase, d: Duration) {
    NS[f as usize].fetch_add(d.as_nanos() as u64, Relaxed);
    if matches!(f, Fase::Decode) {
        N_DECODE.fetch_add(1, Relaxed);
    }
}

/// Run `f` and add its duration to the phase.
pub fn misura<T>(f: Fase, g: impl FnOnce() -> T) -> T {
    let t = Instant::now();
    let r = g();
    aggiungi(f, t.elapsed());
    r
}

/// Add elapsed time on scope exit.
pub struct Cronometro(Fase, Instant);

impl Cronometro {
    pub fn new(f: Fase) -> Self {
        Self(f, Instant::now())
    }
}

impl Drop for Cronometro {
    fn drop(&mut self) {
        aggiungi(self.0, self.1.elapsed());
    }
}

pub fn istantanea() -> serde_json::Value {
    let s = |f: Fase| NS[f as usize].load(Relaxed) as f64 / 1e9;
    let mut j = serde_json::json!({
        "decode_s": s(Fase::Decode),
        "n_decode": N_DECODE.load(Relaxed),
        "logits_sync_s": s(Fase::Sincro),
        "sampling_s": s(Fase::Campiona),
        "network_wait_s": s(Fase::Attesa),
        "state_bytes_sent": BYTE_STATO_INVIATI.load(Relaxed),
        "state_rows_sent": RIGHE_STATO_INVIATE.load(Relaxed),
        "state_bytes_received": BYTE_STATO_RICEVUTI.load(Relaxed),
    });
    // Include draft timing only when the tail has executed a draft batch.
    let n_mtp = N_MTP.load(Relaxed);
    if n_mtp > 0 {
        j["mtp_s"] = serde_json::json!(NS_MTP.load(Relaxed) as f64 / 1e9);
        j["n_mtp"] = serde_json::json!(n_mtp);
    }
    j
}

// Per-stage occupancy and step timings.

/// Occupancy divides the sum of generation work intervals by the span from
/// the first step's start to the last step's end. Compute time includes
/// forwarding or sampling. Wait time is time the stage spent waiting for
/// the event before a step. The stage sends cumulative snapshots, so the
/// head needs only the latest one from each stage.
#[derive(Default)]
pub struct StageTimings {
    inizio: Option<Instant>,
    occupato: Duration,
    occupato_passi: Duration,
    primo_passo: Option<Instant>,
    ultimo_passo_fine: Option<Instant>,
    n_passi: u64,
    /// Batch rows -> (step count, total compute time, total wait time)
    per_righe: std::collections::BTreeMap<u32, (u64, Duration, Duration)>,
}

impl StageTimings {
    /// Work message other than `Step` (`Open`, `PrefillBlock`, `Close`). `t0` is when processing began.
    pub fn lavoro(&mut self, t0: Instant) {
        self.inizio.get_or_insert(t0);
        self.occupato += t0.elapsed();
    }

    /// A `Step` with `rows` rows. `timeout` = wait for its event, `t0` = start of processing.
    pub fn step(&mut self, rows: u32, timeout: Duration, t0: Instant) {
        let calcolo = t0.elapsed();
        let done = Instant::now();
        self.inizio.get_or_insert(t0);
        self.primo_passo.get_or_insert(t0);
        self.ultimo_passo_fine = Some(done);
        self.occupato += calcolo;
        self.occupato_passi += calcolo;
        self.n_passi += 1;
        let e = self.per_righe.entry(rows).or_default();
        e.0 += 1;
        e.1 += calcolo;
        e.2 += timeout;
    }

    pub fn json(&self, index: u32, role: &str) -> serde_json::Value {
        let s = |d: Duration| d.as_secs_f64();
        let finestra = match (self.primo_passo, self.ultimo_passo_fine) {
            (Some(a), Some(b)) => b.saturating_duration_since(a).as_secs_f64(),
            _ => 0.0,
        };
        let occ = if finestra > 0.0 { (s(self.occupato_passi) / finestra).min(1.0) } else { 0.0 };
        let per_righe: Vec<_> = self
            .per_righe
            .iter()
            .map(|(&r, &(n, c, a))| {
                serde_json::json!({
                    "rows": r, "n": n, "compute_s": s(c), "wait_s": s(a),
                    "mean_compute_ms": s(c) * 1e3 / n.max(1) as f64, "mean_wait_ms": s(a) * 1e3 / n.max(1) as f64,
                })
            })
            .collect();
        serde_json::json!({
            "index": index,
            "role": role,
            "run_s": self.inizio.map_or(0.0, |t| t.elapsed().as_secs_f64()),
            "busy_s": s(self.occupato),
            "n_steps": self.n_passi,
            "step_busy_s": s(self.occupato_passi),
            "step_window_s": finestra,
            "step_utilization": occ,
            "by_rows": per_righe,
            "timings": istantanea(),
        })
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn timings_accumulate() {
        aggiungi(Fase::Attesa, Duration::from_millis(1500));
        let v = istantanea();
        assert!(v["network_wait_s"].as_f64().unwrap() >= 1.5);
    }

    #[test]
    fn timings_by_rows_and_utilization() {
        let mut t = StageTimings::default();
        assert_eq!(t.json(2, "middle")["step_utilization"], 0.0);
        for rows in [8u32, 8, 16] {
            let t0 = Instant::now();
            std::thread::sleep(Duration::from_millis(10));
            t.step(rows, Duration::from_millis(5), t0);
        }
        t.lavoro(Instant::now());
        let j = t.json(2, "middle");
        assert_eq!(j["index"], 2);
        assert_eq!(j["n_steps"], 3);
        let pr = j["by_rows"].as_array().unwrap();
        assert_eq!(pr.len(), 2);
        assert_eq!(pr[0]["rows"], 8);
        assert_eq!(pr[0]["n"], 2);
        assert!(pr[0]["mean_compute_ms"].as_f64().unwrap() >= 10.0);
        assert!((pr[0]["mean_wait_ms"].as_f64().unwrap() - 5.0).abs() < 1e-6);
        let occ = j["step_utilization"].as_f64().unwrap();
        assert!(occ > 0.9 && occ <= 1.0, "{occ}");
    }
}
