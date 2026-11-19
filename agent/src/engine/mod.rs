//! Engine interface used by the session for generation, health checks and verification.

pub mod llama;
pub mod managed;
pub mod mock;
pub mod output;
pub mod progress;
pub mod stages;
pub mod stats;
pub mod supervisor;

use std::{fmt, time::Duration};

use anyhow::Result;
use publicompute_proto::node::{FinishReason, InferenceRequest, RequestFailureCode};

/// Separate silence deadlines from the request cap so a slow engine can keep
/// working as long as it reports progress. Startup flags supply the defaults.
#[derive(Debug, Clone)]
pub struct Limits {
    /// Maximum silence during prefill, including the wait for the first token.
    pub prefill_stall: Duration,
    /// Maximum silence after generation starts.
    pub decode_stall: Duration,
    /// Absolute request deadline.
    pub request_max: Duration,
    /// Deadline for short auxiliary engine calls.
    pub call_timeout: Duration,
    /// Deadline for a verification step consisting of one prefill and one token.
    pub step_timeout: Duration,
    /// Deadline for the one-token health probe.
    pub probe_timeout: Duration,
}

/// Engine fault classification shared by logs and counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaultCause {
    /// No progress before the phase deadline.
    Watchdog,
    /// Absolute request deadline expired.
    RequestMax,
    /// Engine connection failed or closed early.
    Connection,
    /// An individual engine call timed out.
    Timeout,
    /// Engine returned an HTTP 5xx error.
    ServerError,
    /// The inference probe failed.
    ProbeFailed,
    /// The coordinator revoked the lease for lack of progress.
    NoProgress,
    /// An abandoned request held the engine past the grace period.
    SettleTimeout,
    /// The managed engine exited or could not start.
    Process,
}

impl FaultCause {
    pub fn as_str(self) -> &'static str {
        match self {
            FaultCause::Watchdog => "watchdog",
            FaultCause::RequestMax => "request_timeout",
            FaultCause::Connection => "connection",
            FaultCause::Timeout => "timeout",
            FaultCause::ServerError => "http_5xx",
            FaultCause::ProbeFailed => "probe_failed",
            FaultCause::NoProgress => "no_progress",
            FaultCause::SettleTimeout => "settle_timeout",
            FaultCause::Process => "process",
        }
    }
}

/// Engine faults require lease release and recovery. A request error can come
/// from a healthy engine without changing its availability.
#[derive(Debug)]
pub enum EngineError {
    Fault { cause: FaultCause, detail: String },
    Request { code: RequestFailureCode, detail: String },
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EngineError::Fault { cause, detail } => write!(f, "engine fault ({}): {detail}", cause.as_str()),
            EngineError::Request { code, detail } => write!(f, "request failed ({code:?}): {detail}"),
        }
    }
}

impl std::error::Error for EngineError {}

pub fn fault(cause: FaultCause, detail: impl Into<String>) -> anyhow::Error {
    EngineError::Fault { cause, detail: detail.into() }.into()
}

pub fn request_error(code: RequestFailureCode, detail: impl Into<String>) -> anyhow::Error {
    EngineError::Request { code, detail: detail.into() }.into()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorClass {
    Fault(FaultCause),
    Request(RequestFailureCode),
    /// Failure outside the engine, such as transport, protocol or shutdown.
    Other,
}

pub fn classify(e: &anyhow::Error) -> ErrorClass {
    match e.downcast_ref::<EngineError>() {
        Some(EngineError::Fault { cause, .. }) => ErrorClass::Fault(*cause),
        Some(EngineError::Request { code, .. }) => ErrorClass::Request(*code),
        None => ErrorClass::Other,
    }
}

/// Probability data for one generated position.
#[derive(Debug, Clone, Default)]
pub struct TokenProbs {
    pub chosen_logprob: f32,
    /// Token IDs and log probabilities in descending probability order.
    pub top: Vec<(u32, f32)>,
}

#[derive(Debug, Clone)]
pub struct Generation {
    pub text: String,
    pub token_ids: Vec<u32>,
    pub finish_reason: FinishReason,
    pub prompt_tokens: u32,
    /// SHA-256 of the prompt token IDs concatenated as little-endian 32-bit integers.
    pub prompt_tokens_sha256: [u8; 32],
    /// One entry per generated token.
    pub probs: Vec<TokenProbs>,
    pub verification_incomplete: bool,
    pub prefill: Duration,
    pub generation: Duration,
}

pub enum Engine {
    Mock(mock::MockEngine),
    Llama(llama::LlamaEngine),
    /// Route chat generation through the head stage's service socket.
    Stages(stages::StadiEngine),
}

/// An independently recomputed output position.
#[derive(Debug, Clone)]
pub struct Recomputed {
    pub position: u32,
    /// Highest-probability token IDs and their log probabilities at this position.
    pub top: Vec<(u32, f32)>,
    /// Log probability of the inspected token when it is among those returned.
    pub chosen_logprob: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct Verification {
    pub prompt_tokens_sha256: [u8; 32],
    /// Independent prompt token count for checking the usage submitted with the result.
    pub prompt_tokens: u32,
    pub positions: Vec<Recomputed>,
}

impl Engine {
    /// Check that the engine responds to its health endpoint. Use the inference
    /// probe to verify that it can generate.
    pub async fn health(&self) -> Result<()> {
        match self {
            Engine::Mock(_) | Engine::Stages(_) => Ok(()),
            Engine::Llama(e) => e.health().await,
        }
    }

    /// Return the loaded weights path when the engine reports it.
    pub async fn loaded_model_path(&self) -> Result<Option<String>> {
        match self {
            Engine::Mock(_) | Engine::Stages(_) => Ok(None),
            Engine::Llama(e) => e.loaded_model_path().await.map(Some),
        }
    }

    /// Return the running engine's build identifier when it is available.
    pub async fn build_info(&self) -> Option<String> {
        match self {
            Engine::Mock(_) | Engine::Stages(_) => None,
            Engine::Llama(e) => e.build_info().await,
        }
    }

    /// Generate one token from a fixed prompt to verify inference.
    pub async fn probe(&self) -> Result<Duration> {
        match self {
            Engine::Mock(e) => e.probe().await,
            // Stage readiness comes from assignment status because there may be no stage to probe yet.
            Engine::Stages(_) => Ok(Duration::from_millis(1)),
            Engine::Llama(e) => e.probe().await,
        }
    }

    /// Wait for the engine to finish work left behind by abandoned requests.
    pub async fn wait_idle(&self, within: Duration) -> Result<()> {
        match self {
            Engine::Mock(_) | Engine::Stages(_) => Ok(()),
            Engine::Llama(e) => e.wait_idle(within).await,
        }
    }

    pub async fn generate(&self, req: &InferenceRequest, top_k: u32) -> Result<Generation> {
        match self {
            Engine::Mock(e) => e.generate(req, top_k).await,
            Engine::Llama(e) => e.generate(req, top_k).await,
            Engine::Stages(e) => e.generate(req, top_k).await,
        }
    }

    /// Generate a chat with lease-local progress, optional streamed deltas and
    /// cancellation. Engines with concurrent slots can serve more than one lease.
    pub async fn chat_lease(
        &self,
        req: &InferenceRequest,
        top_k: u32,
        progress: &progress::Progress,
        deltas: Option<stages::DeltaSender>,
        cancel: Option<tokio::sync::oneshot::Receiver<()>>,
    ) -> Result<Generation> {
        match self {
            Engine::Mock(e) => e.chat_lease(req, top_k, progress, deltas, cancel).await,
            Engine::Llama(e) => e.chat_lease(req, top_k, progress, deltas, cancel).await,
            Engine::Stages(e) => e.chat_lease(req, progress, deltas, cancel).await,
        }
    }

    /// Recompute the requested log probabilities independently. For each position,
    /// run one step on the prompt and the tokens preceding that position.
    pub async fn verify(&self, req: &InferenceRequest, output: &[u32], positions: &[u32], top_k: u32) -> Result<Verification> {
        if let Some(p) = positions.iter().find(|p| **p as usize >= output.len()) {
            anyhow::bail!("position {p} exceeds the output length ({} tokens)", output.len());
        }
        match self {
            Engine::Mock(e) => e.verify(req, output, positions, top_k).await,
            Engine::Llama(e) => e.verify(req, output, positions, top_k).await,
            Engine::Stages(_) => anyhow::bail!("verification requires the full model on one node"),
        }
    }
}

pub fn prompt_tokens_sha256(tokens: &[u32]) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    for t in tokens {
        h.update(t.to_le_bytes());
    }
    h.finalize().into()
}
