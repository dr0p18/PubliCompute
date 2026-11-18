//! A monotonic counter lets the coordinator tell useful work from heartbeats.
//! It advances for processed prompt chunks, tokens and completed operations,
//! and persists across leases. Phase and elapsed stall time let the coordinator
//! apply the deadline appropriate to the current work.

use std::{
    sync::{
        atomic::{AtomicU64, AtomicU8, Ordering},
        Mutex,
    },
    time::{Duration, Instant},
};

use publicompute_proto::node::LeasePhase;

/// Lease phases reported to the coordinator. Keep the phase conversion in sync with the protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Phase {
    Starting = 1,
    Prefill = 2,
    Decode = 3,
    BetweenRequests = 4,
    Finishing = 5,
}

impl Phase {
    fn from_u8(v: u8) -> Phase {
        match v {
            2 => Phase::Prefill,
            3 => Phase::Decode,
            4 => Phase::BetweenRequests,
            5 => Phase::Finishing,
            _ => Phase::Starting,
        }
    }

    pub fn to_proto(self) -> LeasePhase {
        match self {
            Phase::Starting => LeasePhase::Starting,
            Phase::Prefill => LeasePhase::Prefill,
            Phase::Decode => LeasePhase::Decode,
            Phase::BetweenRequests => LeasePhase::BetweenRequests,
            Phase::Finishing => LeasePhase::Finishing,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Snapshot {
    pub phase: Phase,
    pub seq: u64,
    pub prompt_tokens_processed: u64,
}

pub struct Progress {
    seq: AtomicU64,
    phase: AtomicU8,
    prompt_processed: AtomicU64,
    last: Mutex<Instant>,
}

impl Default for Progress {
    fn default() -> Self {
        Self {
            seq: AtomicU64::new(0),
            phase: AtomicU8::new(Phase::Starting as u8),
            prompt_processed: AtomicU64::new(0),
            last: Mutex::new(Instant::now()),
        }
    }
}

impl Progress {
    /// Record a completed unit of work and reset the stall clock.
    pub fn tick(&self) {
        self.seq.fetch_add(1, Ordering::Relaxed);
        *self.last.lock().unwrap() = Instant::now();
    }

    pub fn set_phase(&self, p: Phase) {
        self.phase.store(p as u8, Ordering::Relaxed);
    }

    pub fn set_prompt_processed(&self, n: u64) {
        self.prompt_processed.store(n, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> Snapshot {
        Snapshot {
            phase: Phase::from_u8(self.phase.load(Ordering::Relaxed)),
            seq: self.seq.load(Ordering::Relaxed),
            prompt_tokens_processed: self.prompt_processed.load(Ordering::Relaxed),
        }
    }

    pub fn since_last(&self) -> Duration {
        self.last.lock().unwrap().elapsed()
    }

    /// Start a new lease in STARTING with a fresh stall clock. Keep the progress
    /// counter monotonic across leases.
    pub fn begin_work(&self) {
        self.set_phase(Phase::Starting);
        self.set_prompt_processed(0);
        *self.last.lock().unwrap() = Instant::now();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seq_only_grows_with_ticks() {
        let p = Progress::default();
        let a = p.snapshot().seq;
        p.set_phase(Phase::Decode);
        assert_eq!(p.snapshot().seq, a, "a phase change does not count as progress");
        p.tick();
        assert_eq!(p.snapshot().seq, a + 1);
        assert_eq!(p.snapshot().phase, Phase::Decode);
    }
}
