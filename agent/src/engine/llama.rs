//! The local HTTP backend applies the chat template, tokenizes the prompt
//! and sends token IDs for completion. Using those IDs for the prompt
//! fingerprint and requesting candidate probabilities ties the commitment
//! to the request that actually ran.
//!
//! Streaming events drive a watchdog that measures time since progress.
//! For non-streaming requests it polls decoded and prompt token counts
//! from the slots endpoint. A health response alone cannot establish that
//! generation works, so readiness also uses a one-token inference probe.
//!
//! Older engines can omit IDs for split UTF-8 characters. Streams can
//! also include IDs for text removed by a stop string. Tokenize the
//! visible text in these cases and keep probabilities at known positions.
//!
//! Older engines can omit probability records even in blocking mode.
//! Align the records to the emitted IDs and recover any missing record
//! in blocking mode with a one-token request on that prefix. The
//! one-token request biases sampling toward a complete character so the
//! server emits a record. The reported candidate scores come from before
//! this sampling adjustment.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::{bail, Context, Result};
use publicompute_proto::node::{ChatRole, FinishReason, InferenceRequest, RequestFailureCode};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::oneshot;
use tracing::{info, warn};

use super::{
    fault,
    progress::{Phase, Progress},
    prompt_tokens_sha256, request_error,
    stages::{Delta, DeltaSender},
    FaultCause, Generation, Limits, Recomputed, TokenProbs, Verification,
};

enum Streamed {
    Done { response: CompletionResp, probability_prefix: Option<usize> },
    /// Partial generation retained after the caller cancels.
    Cancelled(CompletionResp),
}

type Cancel = Option<oneshot::Receiver<()>>;

/// Wait for cancellation or loss of its sender. A missing receiver leaves the wait pending.
async fn cancelled(cancel: &mut Cancel) {
    match cancel.as_mut() {
        Some(rx) => {
            let _ = rx.await; // A dropped sender also ends the wait.
            *cancel = None;
        }
        None => std::future::pending::<()>().await,
    }
}

/// Poll often enough to detect progress before the stall deadline.
const SLOTS_POLL: Duration = Duration::from_secs(2);
/// Interval between slot checks while an abandoned request settles.
const IDLE_POLL: Duration = Duration::from_millis(250);
/// TCP connection deadline for the local engine.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
/// Tokenization after the final chunk must leave time to deliver the reply.
const STREAM_TOKENIZE_TIMEOUT: Duration = Duration::from_secs(1);
/// Fixed prompt for a probe that actually exercises inference.
const PROBE_TEXT: &str = "The capital of France is";

pub struct LlamaEngine {
    http: reqwest::Client,
    base: String,
    limits: Limits,
    progress: Arc<Progress>,
    /// Whether completions use the streaming watchdog.
    stream: bool,
    slots_poll: Duration,
    /// Count of streams completed with reconstructed IDs or neutral probabilities.
    stream_repairs: std::sync::atomic::AtomicU64,
    probe_ids: tokio::sync::OnceCell<Vec<u32>>,
}

#[derive(Deserialize, Default)]
struct TemplateResp {
    prompt: String,
}

#[derive(Deserialize)]
struct TokenizeResp {
    tokens: Vec<u32>,
}

#[derive(Deserialize, Default)]
struct CompletionResp {
    content: String,
    #[serde(default)]
    tokens: Vec<u32>,
    #[serde(default)]
    completion_probabilities: Vec<ProbEntry>,
    #[serde(default)]
    stop_type: String,
    #[serde(default)]
    timings: Option<Timings>,
}

#[derive(Deserialize)]
struct ProbEntry {
    id: u32,
    logprob: f32,
    #[serde(default)]
    top_logprobs: Vec<TopEntry>,
}

#[derive(Deserialize)]
struct TopEntry {
    id: u32,
    logprob: f32,
}

#[derive(Deserialize, Default)]
struct Timings {
    #[serde(default)]
    prompt_ms: f64,
    #[serde(default)]
    predicted_ms: f64,
}

#[derive(Deserialize)]
struct PropsResp {
    #[serde(default)]
    model_path: String,
    #[serde(default)]
    build_info: String,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct SlotsInfo {
    processing: bool,
    decoded: u64,
    prompt_processed: u64,
}

fn map_err(e: reqwest::Error, what: &str) -> anyhow::Error {
    if e.is_timeout() {
        fault(FaultCause::Timeout, format!("{what}: timeout ({e})"))
    } else {
        fault(FaultCause::Connection, format!("{what}: {e}"))
    }
}

/// Classify server errors as engine faults. Client errors such as an excessive
/// context length leave a healthy engine available.
async fn check_status(resp: reqwest::Response, what: &str) -> Result<reqwest::Response> {
    let st = resp.status();
    if st.is_success() {
        return Ok(resp);
    }
    let text = resp.text().await.unwrap_or_default();
    let short: String = text.chars().take(300).collect();
    if st.is_server_error() {
        return Err(fault(FaultCause::ServerError, format!("{what}: HTTP {st}: {short}")));
    }
    let lower = text.to_lowercase();
    let code = if lower.contains("context") || lower.contains("exceed") || lower.contains("too long") || lower.contains("too large") {
        RequestFailureCode::ContextTooLong
    } else {
        RequestFailureCode::EngineError
    };
    Err(request_error(code, format!("{what}: HTTP {st}: {short}")))
}

impl LlamaEngine {
    pub fn new(base: String, limits: Limits, progress: Arc<Progress>, stream: bool) -> Result<Self> {
        if !(base.starts_with("http://127.0.0.1") || base.starts_with("http://localhost") || base.starts_with("http://[::1]")) {
            bail!("llama-server must use a loopback address (127.0.0.1, localhost or ::1)");
        }
        // Disable connection reuse so a restart cannot leave a dead pooled connection.
        // Each request sets its own deadline. Initialize the TLS crypto provider
        // before building the HTTP client because its builder requires a provider
        // even for plain HTTP. Initialization is safe to repeat.
        let _ = rustls::crypto::ring::default_provider().install_default();
        let http = reqwest::Client::builder().connect_timeout(CONNECT_TIMEOUT).pool_max_idle_per_host(0).build()?;
        Ok(Self { http, base: base.trim_end_matches('/').to_string(), limits, progress, stream, slots_poll: SLOTS_POLL, stream_repairs: std::sync::atomic::AtomicU64::new(0), probe_ids: tokio::sync::OnceCell::new() })
    }

    #[cfg(test)]
    pub fn stream_repairs(&self) -> u64 {
        self.stream_repairs.load(std::sync::atomic::Ordering::Relaxed)
    }

    #[cfg(test)]
    pub fn with_slots_poll(mut self, d: Duration) -> Self {
        self.slots_poll = d;
        self
    }

    /// Return whether the engine reports that loading is complete.
    pub async fn health(&self) -> Result<()> {
        let resp = self
            .http
            .get(format!("{}/health", self.base))
            .timeout(self.limits.call_timeout)
            .send()
            .await
            .map_err(|e| map_err(e, "health"))?;
        if resp.status().is_success() {
            Ok(())
        } else {
            Err(fault(FaultCause::Connection, format!("health: HTTP {}", resp.status())))
        }
    }

    /// Read the weights path for comparison with the expected file.
    pub async fn loaded_model_path(&self) -> Result<String> {
        let p: PropsResp = self.get("/props", self.limits.call_timeout).await?;
        Ok(p.model_path)
    }

    /// Read the build identifier from the properties response. Return None when
    /// the field or response is unavailable.
    pub async fn build_info(&self) -> Option<String> {
        let p: PropsResp = self.get("/props", self.limits.call_timeout).await.ok()?;
        Some(p.build_info).filter(|b| !b.is_empty())
    }

    /// Generate one token from a fixed prompt at zero temperature to test actual
    /// inference. Treat any failure as a probe fault.
    pub async fn probe(&self) -> Result<Duration> {
        let run = async {
            let ids = self
                .probe_ids
                .get_or_try_init(|| async {
                    let t: TokenizeResp = self
                        .post("/tokenize", json!({"content": PROBE_TEXT, "add_special": true}), self.limits.call_timeout)
                        .await
                        .context("tokenize the probe prompt")?;
                    Ok::<_, anyhow::Error>(t.tokens)
                })
                .await?;
            let t0 = Instant::now();
            // Disable the prompt cache so the probe covers prompt processing.
            let v: Value = self
                .post(
                    "/completion",
                    json!({"prompt": ids, "n_predict": 1, "temperature": 0.0, "cache_prompt": false, "stream": false, "return_tokens": true}),
                    self.limits.probe_timeout,
                )
                .await
                .context("run probe completion")?;
            let produced = v["tokens"].as_array().is_some_and(|a| !a.is_empty()) || v["tokens_predicted"].as_u64().unwrap_or(0) >= 1;
            if !produced {
                bail!("probe produced no tokens");
            }
            Ok::<_, anyhow::Error>(t0.elapsed())
        };
        // Include tokenization in the outer deadline so it cannot extend the probe indefinitely.
        match tokio::time::timeout(self.limits.call_timeout + self.limits.probe_timeout, run).await {
            Ok(Ok(d)) => Ok(d),
            Ok(Err(e)) => Err(fault(FaultCause::ProbeFailed, format!("{e:#}"))),
            Err(_) => Err(fault(FaultCause::ProbeFailed, "probe timed out")),
        }
    }

    /// Query active slots. Return None for an unsupported endpoint.
    async fn slots(&self) -> Result<Option<SlotsInfo>> {
        let resp = self
            .http
            .get(format!("{}/slots", self.base))
            .timeout(self.limits.call_timeout)
            .send()
            .await
            .map_err(|e| map_err(e, "slots"))?;
        let st = resp.status();
        if st == reqwest::StatusCode::NOT_FOUND || st == reqwest::StatusCode::NOT_IMPLEMENTED {
            return Ok(None);
        }
        let resp = check_status(resp, "slots").await?;
        let v: Value = resp.json().await.map_err(|e| map_err(e, "slots"))?;
        let mut info = SlotsInfo::default();
        for s in v.as_array().map(Vec::as_slice).unwrap_or(&[]) {
            if s["is_processing"].as_bool().unwrap_or(false) {
                info.processing = true;
                info.prompt_processed += s["n_prompt_tokens_processed"].as_u64().unwrap_or(0);
                info.decoded += s["next_token"][0]["n_decoded"].as_u64().unwrap_or(0);
            }
        }
        Ok(Some(info))
    }

    /// Wait for abandoned work to finish. Fall back to a probe when slot status
    /// cannot be queried.
    pub async fn wait_idle(&self, within: Duration) -> Result<()> {
        let deadline = Instant::now() + within;
        loop {
            match self.slots().await {
                Ok(Some(s)) if !s.processing => return Ok(()),
                Ok(Some(_)) => {}
                Ok(None) | Err(_) => {
                    info!("/slots unavailable; probing the engine");
                    return self.probe().await.map(|_| ());
                }
            }
            if Instant::now() >= deadline {
                return Err(fault(FaultCause::SettleTimeout, format!("engine still busy {} s after revocation", within.as_secs())));
            }
            tokio::time::sleep(IDLE_POLL).await;
        }
    }

    /// Apply the same template and tokenization as generation.
    async fn prompt_tokens(&self, req: &InferenceRequest) -> Result<Vec<u32>> {
        let messages: Vec<_> = req
            .messages
            .iter()
            .map(|m| {
                let role = match m.role() {
                    ChatRole::System => "system",
                    ChatRole::Assistant => "assistant",
                    _ => "user",
                };
                json!({"role": role, "content": m.content})
            })
            .collect();
        let tpl: TemplateResp =
            self.post("/apply-template", json!({"messages": messages}), self.limits.call_timeout).await.context("apply-template")?;
        let tok: TokenizeResp = self
            .post("/tokenize", json!({"content": tpl.prompt, "add_special": true, "parse_special": true}), self.limits.call_timeout)
            .await
            .context("tokenize")?;
        self.progress.tick();
        Ok(tok.tokens)
    }

    /// Select a complete character so the server emits its probability record.
    async fn force_complete_bias(&self) -> Result<serde_json::Value> {
        let tok: TokenizeResp =
            self.post("/tokenize", json!({"content": "a"}), self.limits.call_timeout).await.context("tokenize \"a\"")?;
        match tok.tokens.as_slice() {
            [id] => Ok(json!([[id, 1000]])),
            other => bail!("expected \"a\" to be a single token, got {other:?}"),
        }
    }

    /// Read candidate probabilities for one token after the supplied prefix.
    /// A successful call records progress.
    async fn next_top(&self, ctx: &[u32], n_probs: u32, bias: &serde_json::Value) -> Result<Vec<(u32, f32)>> {
        let body = json!({
            "prompt": ctx, "n_predict": 1, "n_probs": n_probs, "temperature": 0.0,
            "return_tokens": true, "cache_prompt": true, "stream": false, "logit_bias": bias,
        });
        let r: CompletionResp = self.post("/completion", body, self.limits.step_timeout).await.context("request a one-token completion")?;
        self.progress.tick();
        let first = r.completion_probabilities.first().context("no probabilities returned")?;
        Ok(first.top_logprobs.iter().map(|e| (e.id, e.logprob)).collect())
    }

    /// Recover the chosen token's probability by requesting more candidates
    /// than the commitment keeps. Return an error if the chosen ID is still absent.
    async fn next_probs(&self, ctx: &[u32], chosen: u32, top_k: u32, bias: &serde_json::Value) -> Result<TokenProbs> {
        let mut top = self.next_top(ctx, top_k.max(REBUILD_PROBS), bias).await?;
        let chosen_logprob = top
            .iter()
            .find(|(id, _)| *id == chosen)
            .map(|(_, lp)| *lp)
            .with_context(|| format!("token {chosen} fuori dalle prime {} voci", top.len()))?;
        top.truncate(top_k as usize);
        Ok(TokenProbs { chosen_logprob, top })
    }

    /// Recompute the selected positions from the prompt and the generated prefix
    /// at each position. Reuse the cached prefix where the server allows it.
    pub async fn verify(&self, req: &InferenceRequest, output: &[u32], positions: &[u32], top_k: u32) -> Result<Verification> {
        self.progress.set_phase(Phase::Prefill);
        let prompt = self.prompt_tokens(req).await?;
        let bias = self.force_complete_bias().await?;
        let mut sorted = positions.to_vec();
        sorted.sort_unstable();
        let mut out = Vec::with_capacity(sorted.len());
        for p in sorted {
            let mut tokens = prompt.clone();
            tokens.extend_from_slice(&output[..p as usize]);
            let top = self.next_top(&tokens, top_k, &bias).await.context("request verification completion")?;
            let chosen = output[p as usize];
            let chosen_logprob = top.iter().find(|(id, _)| *id == chosen).map(|(_, lp)| *lp);
            out.push(Recomputed { position: p, top, chosen_logprob });
        }
        Ok(Verification { prompt_tokens_sha256: prompt_tokens_sha256(&prompt), prompt_tokens: prompt.len() as u32, positions: out })
    }

    pub async fn generate(&self, req: &InferenceRequest, top_k: u32) -> Result<Generation> {
        self.generate_with(req, top_k, &self.progress, None, None).await
    }

    /// Run a chat with its own progress counter, optional text deltas and
    /// cancellation. Also update the shared counter for supervision.
    pub async fn chat_lease(&self, req: &InferenceRequest, top_k: u32, progress: &Progress, deltas: Option<DeltaSender>, cancel: Cancel) -> Result<Generation> {
        self.generate_with(req, top_k, progress, deltas, cancel).await
    }

    /// Apply the update once per distinct progress counter.
    fn su_entrambi(&self, lease: &Progress, f: impl Fn(&Progress)) {
        f(lease);
        if !std::ptr::eq(lease, &*self.progress) {
            f(&self.progress);
        }
    }

    async fn generate_with(&self, req: &InferenceRequest, top_k: u32, progress: &Progress, deltas: Option<DeltaSender>, mut cancel: Cancel) -> Result<Generation> {
        self.su_entrambi(progress, |p| p.set_phase(Phase::Prefill));
        self.su_entrambi(progress, |p| p.set_prompt_processed(0));
        let prompt = self.prompt_tokens(req).await?;

        let s = req.sampling.clone().unwrap_or_default();
        let body = |stream: bool| {
            let mut b = json!({
                "prompt": prompt,
                "n_predict": if s.max_tokens > 0 { s.max_tokens as i64 } else { -1 },
                "temperature": s.temperature,
                "top_p": if s.top_p > 0.0 { s.top_p } else { 1.0 },
                "top_k": s.top_k,
                "seed": s.seed,
                "stop": s.stop,
                "n_probs": top_k,
                "return_tokens": true,
                "cache_prompt": true,
                "stream": stream,
            });
            if stream {
                b["return_progress"] = json!(true);
            }
            b
        };
        let interrupted = |r: CompletionResp| {
            // An incomplete reply needs no verification data, so give each partial token a neutral entry.
            let t = r.timings.unwrap_or_default();
            let probs = vec![TokenProbs::default(); r.tokens.len()];
            Generation {
                text: r.content,
                token_ids: r.tokens,
                finish_reason: FinishReason::Cancelled,
                prompt_tokens: prompt.len() as u32,
                cached_prompt_tokens: 0,
                slot_report: None,
                prompt_tokens_sha256: prompt_tokens_sha256(&prompt),
                probs,
                verification_incomplete: true,
                prefill: Duration::from_secs_f64(t.prompt_ms / 1000.0),
                generation: Duration::from_secs_f64(t.predicted_ms / 1000.0),
            }
        };
        let (r, probability_prefix) = if self.stream {
            match self.complete_stream(body(true), progress, deltas.as_ref(), &mut cancel).await? {
                Streamed::Done { response, probability_prefix } => (response, probability_prefix),
                Streamed::Cancelled(r) => return Ok(interrupted(r)),
            }
        } else {
            // Dropping the HTTP future lets the server release the abandoned slot.
            tokio::select! {
                r = self.complete_blocking(body(false), progress) => (r?, None),
                _ = cancelled(&mut cancel) => return Ok(interrupted(CompletionResp::default())),
            }
        };
        self.assemble(&prompt, r, top_k, progress, probability_prefix).await
    }

    /// Align probability records with the token IDs.
    async fn assemble(&self, prompt: &[u32], r: CompletionResp, top_k: u32, progress: &Progress, probability_prefix: Option<usize>) -> Result<Generation> {
        let mut probs = Vec::with_capacity(r.tokens.len());
        let mut neutral = 0;
        if top_k > 0 {
            if !self.stream && r.tokens.is_empty() && !r.completion_probabilities.is_empty() {
                bail!("unsupported probability entries without token IDs");
            }
            let prefix = probability_prefix.unwrap_or(r.tokens.len());
            let ids: Vec<u32> = r.completion_probabilities.iter().map(|p| p.id).collect();
            // An ambiguous record could attach scores to the wrong token.
            let slots = align_probs(&r.tokens[..prefix], &ids).unwrap_or_else(|| vec![None; prefix]);
            let mut bias = None;
            neutral = r.tokens.len() - prefix;
            for (i, slot) in slots.into_iter().enumerate() {
                match slot {
                    Some(j) if !self.stream || !r.completion_probabilities[j].top_logprobs.is_empty() => {
                        let p = &r.completion_probabilities[j];
                        probs.push(TokenProbs {
                            chosen_logprob: p.logprob,
                            top: p.top_logprobs.iter().map(|e| (e.id, e.logprob)).collect(),
                        });
                    }
                    _ if self.stream => {
                        neutral += 1;
                        probs.push(TokenProbs::default());
                    }
                    _ => {
                        self.su_entrambi(progress, |p| p.set_phase(Phase::Prefill));
                        if bias.is_none() {
                            bias = Some(self.force_complete_bias().await?);
                        }
                        let mut ctx = prompt.to_vec();
                        ctx.extend_from_slice(&r.tokens[..i]);
                        let p = self
                            .next_probs(&ctx, r.tokens[i], top_k, bias.as_ref().unwrap())
                            .await
                            .with_context(|| format!("reconstruct probability entry at position {i}"))?;
                        probs.push(p);
                    }
                }
            }
        }
        probs.resize(r.tokens.len(), TokenProbs::default());
        let verification_incomplete = probability_prefix.is_some() || neutral > 0;
        if verification_incomplete {
            self.stream_repairs.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            info!(neutral, "completed stream with incomplete verification data");
        }
        let finish = match r.stop_type.as_str() {
            "limit" => FinishReason::Length,
            "eos" | "word" => FinishReason::Stop,
            other => bail!("unknown stop_type: {other:?}"),
        };
        let t = r.timings.unwrap_or_default();
        Ok(Generation {
            text: r.content,
            token_ids: r.tokens,
            finish_reason: finish,
            prompt_tokens: prompt.len() as u32,
            cached_prompt_tokens: 0,
            slot_report: None,
            prompt_tokens_sha256: prompt_tokens_sha256(prompt),
            probs,
            verification_incomplete,
            prefill: Duration::from_secs_f64(t.prompt_ms / 1000.0),
            generation: Duration::from_secs_f64(t.predicted_ms / 1000.0),
        })
    }

    /// Stream text as it becomes available and reset the watchdog on valid
    /// progress. Recover missing IDs from the text after the final chunk.
    /// Cancellation closes the request and returns the partial response.
    async fn complete_stream(&self, body: Value, progress: &Progress, deltas: Option<&DeltaSender>, cancel: &mut Cancel) -> Result<Streamed> {
        let lim = &self.limits;
        let deadline = Instant::now() + lim.request_max;
        // The headers arrive with the first body event, so they share the prefill deadline.
        let send = self.http.post(format!("{}/completion", self.base)).json(&body).send();
        let resp = tokio::select! {
            r = tokio::time::timeout(lim.prefill_stall.min(lim.request_max), send) => match r {
                Err(_) => return Err(fault(FaultCause::Watchdog, format!("no response from the engine for {} s", lim.prefill_stall.as_secs()))),
                Ok(r) => r.map_err(|e| map_err(e, "completion"))?,
            },
            _ = cancelled(cancel) => return Ok(Streamed::Cancelled(CompletionResp::default())),
        };
        let mut resp = check_status(resp, "completion").await?;

        let mut buf: Vec<u8> = Vec::new();
        let mut out = CompletionResp::default();
        let (mut seen_token, mut stopped) = (false, false);
        let mut predicted = None;
        let mut last_predicted = None;
        let mut count_inconsistent = false;
        let mut stop_text_withheld = None;
        let mut visible_tokens = 0;
        let mut last_text_start = 0;
        let mut last_token_had_text = false;
        loop {
            let stall = if seen_token { lim.decode_stall } else { lim.prefill_stall };
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(fault(FaultCause::RequestMax, format!("request exceeded the {} s limit", lim.request_max.as_secs())));
            }
            let expected = tokio::select! {
                c = tokio::time::timeout(stall.min(remaining), resp.chunk()) => c,
                _ = cancelled(cancel) => return Ok(Streamed::Cancelled(out)),
            };
            let chunk = match expected {
                Err(_) if stall <= remaining => {
                    let what = if seen_token { "generazione" } else { "prefill" };
                    return Err(fault(FaultCause::Watchdog, format!("no chunk for {} s in {what}", stall.as_secs_f32())));
                }
                Err(_) => return Err(fault(FaultCause::RequestMax, format!("request exceeded the {} s limit", lim.request_max.as_secs()))),
                Ok(Err(e)) => return Err(map_err(e, "streaming")),
                Ok(Ok(c)) => c,
            };
            let Some(chunk) = chunk else { break };
            buf.extend_from_slice(&chunk);
            while let Some(pos) = buf.iter().position(|b| *b == b'\n') {
                let line: Vec<u8> = buf.drain(..=pos).collect();
                let line = String::from_utf8_lossy(&line);
                let line = line.trim();
                let Some(data) = line.strip_prefix("data:") else {
                    if line.starts_with("error:") {
                        return Err(fault(FaultCause::ServerError, format!("stream error: {line}")));
                    }
                    continue;
                };
                let data = data.trim();
                if data.is_empty() || data == "[DONE]" {
                    continue;
                }
                let v: Value = serde_json::from_str(data).map_err(|e| request_error(RequestFailureCode::EngineError, format!("invalid chunk: {e}")))?;
                if let Some(err) = v.get("error") {
                    let code = err["code"].as_u64().unwrap_or(500);
                    let msg = err["message"].as_str().unwrap_or("server error").to_string();
                    return Err(if (400..500).contains(&code) {
                        request_error(RequestFailureCode::EngineError, msg)
                    } else {
                        fault(FaultCause::ServerError, msg)
                    });
                }
                self.su_entrambi(progress, Progress::tick);
                if let Some(pp) = v.get("prompt_progress") {
                    // Ignore the placeholder token and probabilities attached to a prefill progress event.
                    self.su_entrambi(progress, |p| p.set_prompt_processed(pp["processed"].as_u64().unwrap_or(0)));
                    continue;
                }
                let index = out.tokens.len();
                let text = v["content"].as_str().unwrap_or_default();
                out.content.push_str(text);
                let tokens: Vec<u32> = v["tokens"].as_array().into_iter().flatten()
                    .filter_map(|t| u32::try_from(t.as_u64()?).ok()).collect();
                if !text.is_empty() || !tokens.is_empty() {
                    seen_token = true;
                    self.su_entrambi(progress, |p| p.set_phase(Phase::Decode));
                }
                out.tokens.extend(&tokens);
                let count = v["tokens_predicted"].as_u64();
                match count {
                    Some(count) => {
                        // The server can report the same count on several held records.
                        count_inconsistent |= count < out.tokens.len() as u64
                            || last_predicted.is_some_and(|last| count < last);
                        last_predicted = Some(count);
                    }
                    None => count_inconsistent |= v.get("tokens_predicted").is_some(),
                }
                let stop = v["stop"].as_bool().unwrap_or(false);
                if !stop || !text.is_empty() || !tokens.is_empty() {
                    last_token_had_text = !text.is_empty();
                    if last_token_had_text {
                        last_text_start = visible_tokens;
                        visible_tokens = out.tokens.len();
                    }
                }
                if let Some(d) = deltas {
                    if !text.is_empty() || !tokens.is_empty() {
                        d.send(Delta { index: index as u32, text: text.to_string(), token_ids: tokens, ..Default::default() })?;
                    }
                }
                if let Some(ps) = v.get("completion_probabilities").filter(|p| p.is_array()) {
                    let entries: Vec<ProbEntry> = serde_json::from_value(ps.clone())
                        .map_err(|e| request_error(RequestFailureCode::EngineError, format!("invalid probabilities: {e}")))?;
                    out.completion_probabilities.extend(entries);
                }
                if stop {
                    stopped = true;
                    out.stop_type = v["stop_type"].as_str().unwrap_or_default().to_string();
                    predicted = count;
                    stop_text_withheld = v.get("stop_text_withheld").map(|field| field.as_bool().unwrap_or(true));
                    out.timings = v.get("timings").and_then(|t| serde_json::from_value(t.clone()).ok());
                    break;
                }
            }
            if stopped {
                break;
            }
        }
        if !stopped {
            return Err(fault(FaultCause::Connection, "stream ended without a final chunk"));
        }
        drop(resp);
        let received = out.tokens.len();
        let trimmed = stop_text_withheld.unwrap_or_else(|| {
            out.stop_type == "word"
                || (visible_tokens < received && out.stop_type == "limit" && body["stop"].as_array().is_some_and(|s| !s.is_empty()))
        });
        let probability_prefix = if count_inconsistent || predicted != Some(received as u64) || trimmed {
            let safe_prefix = if trimmed {
                // Stop matching can cut through the last token or discard a run of held tokens.
                if last_token_had_text { last_text_start } else { visible_tokens }
            } else {
                received
            };
            let max_tokens = body["n_predict"].as_u64().and_then(|n| usize::try_from(n).ok()).unwrap_or(usize::MAX);
            Some(self.rebuild_stream_ids(&mut out, safe_prefix, max_tokens, predicted, trimmed, count_inconsistent).await)
        } else {
            None
        };
        Ok(Streamed::Done { response: out, probability_prefix })
    }

    async fn rebuild_stream_ids(&self, out: &mut CompletionResp, safe_prefix: usize, max_tokens: usize, predicted: Option<u64>, trimmed: bool, count_inconsistent: bool) -> usize {
        let received = out.tokens.len();
        let rebuilt = if out.content.is_empty() {
            Ok(TokenizeResp { tokens: Vec::new() })
        } else {
            let timeout = self.limits.call_timeout.min(STREAM_TOKENIZE_TIMEOUT);
            self.post("/tokenize", json!({"content": out.content, "add_special": false, "parse_special": false}), timeout).await
        };
        let tokenized = rebuilt.is_ok();
        let mut tokens = match rebuilt {
            Ok(r) => r.tokens,
            Err(err) => {
                info!(%err, "could not tokenize the streamed text; keeping available IDs");
                let keep = if trimmed { safe_prefix } else { received };
                out.tokens[..keep].to_vec()
            }
        };
        tokens.truncate(max_tokens);
        // After the first changed ID, the original scores used a different prefix.
        let mut prefix = if count_inconsistent {
            // A contradictory count leaves the positions of the sampled scores uncertain.
            0
        } else if tokenized || predicted == Some(received as u64) {
            tokens.iter().zip(&out.tokens).take(safe_prefix).take_while(|(a, b)| a == b).count()
        } else {
            // Without a matching count or a second tokenization, the gap could be anywhere.
            0
        };
        if prefix > 0 && predicted != Some(received as u64) && tokens.len() > received {
            // With repeated IDs a left-to-right match can hide a missing token.
            let mut end = tokens.len();
            for (i, id) in out.tokens.iter().enumerate().rev() {
                let Some(pos) = tokens[..end].iter().rposition(|t| t == id) else {
                    prefix = 0;
                    break;
                };
                if i != pos {
                    prefix = prefix.min(i);
                }
                end = pos;
            }
        }
        let ids: Vec<u32> = out.completion_probabilities.iter().map(|p| p.id).collect();
        if let Some(slots) = align_probs(&out.tokens, &ids) {
            out.completion_probabilities.truncate(slots[..prefix].iter().flatten().count());
        } else {
            out.completion_probabilities.clear();
        }
        out.tokens = tokens;
        info!(?predicted, received, retained = prefix, output = out.tokens.len(), stop = %out.stop_type, tokenized, count_inconsistent, "reconciled streamed token IDs");
        prefix
    }

    /// Poll slot progress while waiting for a non-streaming completion. Only the
    /// absolute request deadline applies if the endpoint is unavailable.
    async fn complete_blocking(&self, body: Value, progress: &Progress) -> Result<CompletionResp> {
        let lim = &self.limits;
        let post = self.post::<CompletionResp>("/completion", body, lim.request_max);
        tokio::pin!(post);
        let mut tick = tokio::time::interval(self.slots_poll);
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        tick.tick().await;
        let (mut marker, mut last_change, mut decoding) = (SlotsInfo::default(), Instant::now(), false);
        let mut slots_available = true;
        loop {
            tokio::select! {
                r = &mut post => {
                    return match r {
                        Ok(v) => { self.su_entrambi(progress, Progress::tick); Ok(v) }
                        Err(e) => Err(e),
                    };
                }
                _ = tick.tick(), if slots_available => {
                    match self.slots().await {
                        Ok(Some(s)) => {
                            if s.processing && (s.decoded, s.prompt_processed) != (marker.decoded, marker.prompt_processed) {
                                self.su_entrambi(progress, Progress::tick);
                                self.su_entrambi(progress, |p| p.set_prompt_processed(s.prompt_processed));
                                if s.decoded > 0 {
                                    decoding = true;
                                    self.su_entrambi(progress, |p| p.set_phase(Phase::Decode));
                                }
                                last_change = Instant::now();
                            }
                            marker = s;
                        }
                        Ok(None) => {
                            warn!("/slots unavailable; only the {} s deadline applies", lim.request_max.as_secs());
                            slots_available = false;
                        }
                        Err(_) => {} // A failed poll supplies no evidence of progress.
                    }
                    let stall = if decoding { lim.decode_stall } else { lim.prefill_stall };
                    if last_change.elapsed() > stall {
                        return Err(fault(FaultCause::Watchdog, format!("no progress for {} s during a non-streaming request", stall.as_secs_f32())));
                    }
                }
            }
        }
    }

    async fn post<T: for<'de> Deserialize<'de>>(&self, path: &str, body: serde_json::Value, timeout: Duration) -> Result<T> {
        let resp = self
            .http
            .post(format!("{}{}", self.base, path))
            .timeout(timeout)
            .json(&body)
            .send()
            .await
            .map_err(|e| map_err(e, path))?;
        let resp = check_status(resp, path).await?;
        let bytes = resp.bytes().await.map_err(|e| map_err(e, path))?;
        serde_json::from_slice(&bytes).map_err(|e| request_error(RequestFailureCode::EngineError, format!("{path}: invalid response: {e}")))
    }

    async fn get<T: for<'de> Deserialize<'de>>(&self, path: &str, timeout: Duration) -> Result<T> {
        let resp = self.http.get(format!("{}{}", self.base, path)).timeout(timeout).send().await.map_err(|e| map_err(e, path))?;
        let resp = check_status(resp, path).await?;
        let bytes = resp.bytes().await.map_err(|e| map_err(e, path))?;
        serde_json::from_slice(&bytes).map_err(|e| request_error(RequestFailureCode::EngineError, format!("{path}: invalid response: {e}")))
    }
}

/// Ask for extra candidates so a low-probability chosen token can still be found.
const REBUILD_PROBS: u32 = 200;

/// Match probability records as an ordered subsequence of generated IDs.
/// A missing entry has no index. If the earliest and latest matches differ,
/// more than one assignment is possible and the entire alignment is unusable.
fn align_probs(tokens: &[u32], ids: &[u32]) -> Option<Vec<Option<usize>>> {
    let mut left = vec![None; tokens.len()];
    let mut j = 0;
    for (i, t) in tokens.iter().enumerate() {
        if j < ids.len() && ids[j] == *t {
            left[i] = Some(j);
            j += 1;
        }
    }
    if j != ids.len() {
        return None;
    }
    let mut right = vec![None; tokens.len()];
    let mut j = ids.len();
    for i in (0..tokens.len()).rev() {
        if j > 0 && ids[j - 1] == tokens[i] {
            right[i] = Some(j - 1);
            j -= 1;
        }
    }
    (left == right).then_some(left)
}

#[cfg(test)]
mod tests {
    use super::align_probs;

    #[test]
    fn complete_probs_align_one_to_one() {
        let t = [5, 6, 7];
        assert_eq!(align_probs(&t, &t), Some(vec![Some(0), Some(1), Some(2)]));
    }

    #[test]
    fn missing_utf8_pieces_are_marked() {
        // This character spans two tokens, but only its second token has a record.
        let t = [13, 4891, 98, 243];
        assert_eq!(align_probs(&t, &[13, 98, 243]), Some(vec![Some(0), None, Some(1), Some(2)]));
    }

    #[test]
    fn ambiguous_or_impossible_alignment_is_refused() {
        // A single record for a repeated ID cannot be assigned to a unique position.
        assert_eq!(align_probs(&[7, 7], &[7]), None);
        assert_eq!(align_probs(&[1, 2], &[3]), None);
        assert_eq!(align_probs(&[1, 2], &[2, 1]), None);
    }
}

#[cfg(test)]
mod server_tests {
    use std::sync::atomic::Ordering;

    use publicompute_proto::node::{ChatMessage, SamplingParams};

    use super::*;
    use crate::{
        engine::{classify, Engine},
        fake_llama,
    };

    fn limits(prefill_ms: u64, decode_ms: u64) -> Limits {
        Limits {
            prefill_stall: Duration::from_millis(prefill_ms),
            decode_stall: Duration::from_millis(decode_ms),
            request_max: Duration::from_secs(60),
            call_timeout: Duration::from_secs(5),
            step_timeout: Duration::from_secs(5),
            probe_timeout: Duration::from_millis(prefill_ms),
        }
    }

    fn engine(f: &fake_llama::Fake, l: Limits, stream: bool) -> (LlamaEngine, Arc<Progress>) {
        let p = Arc::new(Progress::default());
        (LlamaEngine::new(f.url(), l, p.clone(), stream).unwrap().with_slots_poll(Duration::from_millis(100)), p)
    }

    fn request(text: &str, temperature: f32, seed: u64, max_tokens: u32) -> InferenceRequest {
        InferenceRequest {
            request_id: "r".into(),
            messages: vec![ChatMessage { role: ChatRole::User as i32, content: text.into() }],
            sampling: Some(SamplingParams { temperature, max_tokens, seed, ..Default::default() }),
        }
    }

    async fn fake() -> fake_llama::Fake {
        fake_llama::spawn(0, fake_llama::FakeState::new("/modelli/finto.gguf")).await.unwrap()
    }

    #[tokio::test]
    async fn missing_stream_probabilities_are_neutral() {
        let progress = Arc::new(Progress::default());
        let e = LlamaEngine::new("http://127.0.0.1:1".into(), limits(500, 500), progress.clone(), true).unwrap();
        let r = CompletionResp {
            content: "ab".into(),
            tokens: vec![10, 11],
            completion_probabilities: vec![ProbEntry { id: 10, logprob: -0.2, top_logprobs: vec![TopEntry { id: 10, logprob: -0.2 }] }],
            stop_type: "limit".into(),
            ..Default::default()
        };
        let g = e.assemble(&[1], r, 1, &progress, None).await.unwrap();
        assert_eq!(g.text, "ab");
        assert_eq!(g.finish_reason, FinishReason::Length);
        assert_eq!(g.probs[0].top, [(10, -0.2)]);
        assert_eq!(g.probs[1].chosen_logprob, 0.0);
        assert!(g.probs[1].top.is_empty());
        assert!(g.verification_incomplete);
        assert_eq!(e.stream_repairs(), 1);
    }

    fn assert_same(a: &Generation, b: &Generation) {
        assert_eq!(a.text, b.text);
        assert_eq!(a.token_ids, b.token_ids);
        assert_eq!(a.finish_reason, b.finish_reason);
        assert_eq!(a.prompt_tokens_sha256, b.prompt_tokens_sha256);
        assert_eq!(a.probs.len(), b.probs.len());
        for (x, y) in a.probs.iter().zip(&b.probs) {
            assert_eq!(x.chosen_logprob.to_bits(), y.chosen_logprob.to_bits());
            assert_eq!(x.top.len(), y.top.len());
            for (p, q) in x.top.iter().zip(&y.top) {
                assert_eq!(p.0, q.0);
                assert_eq!(p.1.to_bits(), q.1.to_bits());
            }
        }
    }

    /// Complete streams keep the same IDs and scores as blocking generation.
    #[tokio::test]
    async fn streaming_equals_non_streaming() {
        let f = fake().await;
        let (s, _) = engine(&f, limits(5000, 5000), true);
        let (n, _) = engine(&f, limits(5000, 5000), false);
        for (i, (temp, max)) in [(0.0, 0), (0.8, 0), (0.0, 5), (0.8, 8), (0.3, 20)].into_iter().enumerate() {
            let r = request(&format!("Ciao {i} 奕秋 🙂"), temp, 42 + i as u64, max);
            let a = s.generate(&r, 5).await.unwrap();
            let b = n.generate(&r, 5).await.unwrap();
            assert_same(&a, &b);
            assert!(!a.token_ids.is_empty());
            assert!(!a.verification_incomplete);
        }
        assert_eq!(f.state.completions.load(Ordering::SeqCst), 10);
        assert_eq!(s.stream_repairs(), 0);
    }

    async fn repaired_stream(f: &fake_llama::Fake, req: &InferenceRequest) -> (Generation, Vec<Delta>) {
        let (e, p) = engine(f, limits(5000, 5000), true);
        let (tx, mut rx, _) = super::super::stages::delta_channel();
        let g = e.chat_lease(req, 3, &p, Some(tx), None).await.unwrap();
        let mut deltas = Vec::new();
        while let Ok(delta) = rx.try_recv() {
            deltas.push(delta.item);
        }
        assert_eq!(deltas.iter().map(|d| d.text.as_str()).collect::<String>(), g.text);
        assert_eq!(f.state.completions.load(Ordering::SeqCst), 1);
        assert_eq!(e.stream_repairs(), 1);
        assert!(g.verification_incomplete);
        assert_eq!(g.probs.len(), g.token_ids.len());
        (g, deltas)
    }

    #[tokio::test]
    async fn split_utf8_stream_rebuilds_ids_without_completion_retry() {
        let f = fake().await;
        f.state.utf8_split.store(true, Ordering::SeqCst);
        let (g, deltas) = repaired_stream(&f, &request("x", 0.0, 1, 8)).await;
        assert_eq!(g.finish_reason, FinishReason::Length);
        assert_eq!(g.token_ids.len(), 8);
        assert_eq!(deltas[3].text, "è");
        assert_eq!(deltas[3].token_ids, [168]);
        assert_eq!(g.token_ids[3..5], [195, 168]);
        assert!(g.probs[..3].iter().all(|p| p.top.len() == 3));
        assert!(g.probs[3..].iter().all(|p| p.top.is_empty() && p.chosen_logprob == 0.0));
    }

    #[tokio::test]
    async fn repeated_id_cannot_hide_a_stream_gap() {
        let f = fake().await;
        f.state.utf8_split.store(true, Ordering::SeqCst);
        let (g, _) = repaired_stream(&f, &request("x", 0.0, 41, 8)).await;
        assert_eq!(g.token_ids[2..5], [195, 195, 168]);
        assert!(g.probs[..2].iter().all(|p| p.top.len() == 3));
        assert!(g.probs[2..].iter().all(|p| p.top.is_empty()));
    }

    #[tokio::test]
    async fn held_stop_word_stream_keeps_only_visible_text() {
        let f = fake().await;
        let mut req = request("x", 0.0, 1, 8);
        req.sampling.as_mut().unwrap().stop.push("END".into());
        let (g, deltas) = repaired_stream(&f, &req).await;
        assert_eq!(g.finish_reason, FinishReason::Stop);
        assert_eq!(deltas.len(), 5);
        assert_eq!(deltas[3].text, "");
        assert_eq!(deltas[4].text, "<350>");
        assert!(g.text.ends_with("<350>") && !g.text.contains("END"));
        assert_eq!(g.token_ids.len(), 4);
        assert_eq!(g.token_ids[3], 350);
        assert!(g.probs[3].top.is_empty());
    }

    #[tokio::test]
    async fn incorrect_stream_count_keeps_the_final_delta() {
        let f = fake().await;
        f.state.incorrect_stream_count.store(true, Ordering::SeqCst);
        let (g, deltas) = repaired_stream(&f, &request("x", 0.0, 1, 8)).await;
        assert_eq!(deltas.len(), 8);
        let tail = &deltas[7];
        assert!(!tail.text.is_empty());
        assert_eq!(g.text, deltas[..7].iter().map(|d| d.text.as_str()).collect::<String>() + &tail.text);
        assert_eq!(g.token_ids.len(), 8);
        assert_eq!(g.token_ids[7..], tail.token_ids);
        assert!(g.probs[..7].iter().all(|p| p.top.len() == 3));
        assert!(g.probs[7].top.is_empty());
    }

    #[tokio::test]
    async fn contradictory_stream_counts_discard_probabilities() {
        // With two IDs and a final count of two, 1 undercounts and 3 later decreases.
        for count in [1, 3] {
            let f = fake().await;
            f.state.second_stream_count.store(count, Ordering::SeqCst);
            let (g, deltas) = repaired_stream(&f, &request("x", 0.0, 1, 2)).await;
            assert_eq!(g.token_ids.len(), 2);
            assert!(g.token_ids.iter().copied().eq(deltas.into_iter().flat_map(|d| d.token_ids)));
            assert!(g.probs.iter().all(|p| p.top.is_empty() && p.chosen_logprob == 0.0));
        }
    }

    #[tokio::test]
    async fn failed_stream_tokenization_still_delivers_text() {
        let f = fake().await;
        f.state.utf8_split.store(true, Ordering::SeqCst);
        f.state.reject_output_tokenization.store(true, Ordering::SeqCst);
        let (g, _) = repaired_stream(&f, &request("x", 0.0, 1, 8)).await;
        assert!(g.text.contains('è'));
        assert_eq!(g.token_ids.len(), 7);
    }

    #[tokio::test]
    async fn blocking_completion_still_recovers_missing_probabilities() {
        let f = fake().await;
        f.state.utf8_split.store(true, Ordering::SeqCst);
        let (e, _) = engine(&f, limits(5000, 5000), false);
        let g = e.generate(&request("x", 0.0, 1, 8), 3).await.unwrap();
        assert_eq!(g.token_ids[3..5], [195, 168]);
        assert!(g.probs.iter().all(|p| p.top.len() == 3));
        assert!(!g.verification_incomplete);
        assert!(f.state.completions.load(Ordering::SeqCst) > 1);
    }

    #[tokio::test]
    async fn held_utf8_chunks_keep_each_probability() {
        let f = fake().await;
        f.state.utf8_hold.store(true, Ordering::SeqCst);
        let (stream, progress) = engine(&f, limits(5000, 5000), true);
        let (blocking, _) = engine(&f, limits(5000, 5000), false);
        // The limit can cut through a piece with safe text or between UTF-8 pieces.
        for (limit, count, finish) in [
            (2, 2, FinishReason::Length),
            (3, 3, FinishReason::Length),
            (5, 5, FinishReason::Length),
            (8, 8, FinishReason::Length),
            (0, 12, FinishReason::Stop),
        ] {
            let req = request("Perché? 🙂", 0.0, 1, limit);
            let (tx, mut rx, _) = super::super::stages::delta_channel();
            let g = stream.chat_lease(&req, 3, &progress, Some(tx), None).await.unwrap();
            assert_eq!(g.token_ids.len(), count);
            assert_eq!(g.probs.len(), count);
            assert_eq!(g.finish_reason, finish);
            assert!(!g.verification_incomplete);
            assert_eq!(g.text.contains('🙂'), limit >= 5 || limit == 0);
            let mut text = String::new();
            for (index, (&id, p)) in g.token_ids.iter().zip(&g.probs).enumerate() {
                let d = rx.try_recv().unwrap().item;
                assert_eq!(d.index, index as u32);
                assert_eq!(d.token_ids, [id]);
                if (1..4).contains(&index) {
                    assert_eq!(id, [240, 159, 153][index - 1]);
                    assert_eq!(d.text, if limit == 2 { "A" } else { "" });
                }
                if limit == 0 && index >= 9 {
                    assert_eq!(id, [240, 159, 151645][index - 9]);
                    assert!(d.text.is_empty());
                }
                text.push_str(&d.text);
                assert_eq!(p.top.len(), 3);
                assert_eq!(p.top[0], (id, p.chosen_logprob));
            }
            assert!(rx.try_recv().is_err());
            assert_eq!(text, g.text);
            let b = blocking.generate(&req, 3).await.unwrap();
            assert_same(&g, &b);
            assert!(!b.verification_incomplete);
        }
        assert_eq!(stream.stream_repairs(), 0);
        assert_eq!(f.state.completions.load(Ordering::SeqCst), 10);
    }

    #[tokio::test]
    async fn matching_count_still_repairs_text_hidden_by_stop() {
        let f = fake().await;
        f.state.utf8_hold.store(true, Ordering::SeqCst);
        let mut req = request("x", 0.0, 1, 8);
        req.sampling.as_mut().unwrap().stop.push("🙂".into());
        let (g, deltas) = repaired_stream(&f, &req).await;
        assert_eq!(g.text, deltas[0].text);
        assert_eq!(g.token_ids, deltas[0].token_ids);
        assert_eq!(deltas.len(), 5);
        for (i, (d, id)) in deltas[1..].iter().zip([240, 159, 153, 130]).enumerate() {
            assert_eq!(d.index, (i + 1) as u32);
            assert_eq!(d.token_ids, [id]);
            assert!(d.text.is_empty());
        }
        assert_eq!(g.finish_reason, FinishReason::Stop);
        assert_eq!(g.probs[0].top.len(), 3);
    }

    #[tokio::test]
    async fn unrelated_stop_keeps_truncated_utf8_ids_and_probabilities() {
        let f = fake().await;
        f.state.utf8_hold.store(true, Ordering::SeqCst);
        let mut req = request("x", 0.0, 1, 3);
        req.sampling.as_mut().unwrap().stop.push("END".into());
        let (stream, _) = engine(&f, limits(5000, 5000), true);
        let (blocking, _) = engine(&f, limits(5000, 5000), false);
        let g = stream.generate(&req, 3).await.unwrap();
        let b = blocking.generate(&req, 3).await.unwrap();
        assert_same(&g, &b);
        assert_eq!(g.finish_reason, FinishReason::Length);
        assert_eq!(g.token_ids[1..], [240, 159]);
        assert!(g.probs.iter().all(|p| p.top.len() == 3));
        assert!(!g.verification_incomplete);
        assert_eq!(stream.stream_repairs(), 0);
        assert_eq!(f.state.completions.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn zero_probabilities_need_no_completion_retry() {
        for stream in [true, false] {
            let progress = Arc::new(Progress::default());
            let e = LlamaEngine::new("http://127.0.0.1:1".into(), limits(500, 500), progress.clone(), stream).unwrap();
            let r = CompletionResp {
                tokens: vec![240, 159],
                stop_type: "limit".into(),
                ..Default::default()
            };
            let g = e.assemble(&[1], r, 0, &progress, None).await.unwrap();
            assert_eq!(g.token_ids, [240, 159]);
            assert_eq!(g.probs.len(), 2);
            assert!(g.probs.iter().all(|p| p.top.is_empty() && p.chosen_logprob == 0.0));
            assert!(!g.verification_incomplete);
        }
    }

    /// Repeated prefill events keep a long prefill below the silence deadline.
    #[tokio::test]
    async fn slow_prefill_that_advances_does_not_trip() {
        for stream in [true, false] {
            let f = fake().await;
            f.state.prefill_steps.store(10, Ordering::SeqCst);
            f.state.prefill_step_ms.store(200, Ordering::SeqCst); // The full prefill takes longer than the silence limit, but each step falls within it.
            let (e, p) = engine(&f, limits(800, 800), stream);
            let before = p.snapshot().seq;
            let t0 = Instant::now();
            e.generate(&request("prompt lungo", 0.0, 0, 0), 3).await.unwrap_or_else(|err| panic!("stream={stream}: {err:#}"));
            assert!(t0.elapsed() > Duration::from_millis(1800), "prefill should outlast the stall timeout");
            assert!(p.snapshot().seq > before + 10, "each chunk should advance progress");
        }
    }

    /// A decode that stops sending events must trigger the watchdog.
    #[tokio::test]
    async fn stalled_decode_trips_watchdog() {
        for stream in [true, false] {
            let f = fake().await;
            f.state.decode_stall_after.store(3, Ordering::SeqCst);
            let (e, _) = engine(&f, limits(800, 600), stream);
            let t0 = Instant::now();
            let err = e.generate(&request("x", 0.0, 0, 0), 3).await.unwrap_err();
            assert_eq!(classify(&err), crate::engine::ErrorClass::Fault(FaultCause::Watchdog), "stream={stream}: {err:#}");
            assert!(t0.elapsed() < Duration::from_secs(4), "watchdog fired too late: {:?}", t0.elapsed());
        }
    }

    /// Health can respond even when inference is stalled.
    #[tokio::test]
    async fn health_ok_but_probe_fails_when_completion_hangs() {
        let f = fake().await;
        f.state.hang_completion.store(true, Ordering::SeqCst);
        let (e, _) = engine(&f, limits(500, 500), true);
        e.health().await.unwrap();
        let err = e.probe().await.unwrap_err();
        assert_eq!(classify(&err), crate::engine::ErrorClass::Fault(FaultCause::ProbeFailed));
        f.state.hang_completion.store(false, Ordering::SeqCst);
        e.probe().await.unwrap();
    }

    /// A request rejected with a client error must leave the engine healthy.
    #[tokio::test]
    async fn context_too_long_is_a_request_failure() {
        let f = fake().await;
        let (e, _) = engine(&f, limits(2000, 2000), true);
        let err = e.generate(&request(&"a".repeat(6000), 0.0, 0, 0), 3).await.unwrap_err();
        assert_eq!(classify(&err), crate::engine::ErrorClass::Request(RequestFailureCode::ContextTooLong), "{err:#}");
    }

    #[tokio::test]
    async fn connection_refused_is_a_fault() {
        let f = fake().await;
        let (e, _) = engine(&f, limits(500, 500), true);
        drop(f);
        tokio::time::sleep(Duration::from_millis(100)).await;
        let err = e.generate(&request("x", 0.0, 0, 0), 3).await.unwrap_err();
        assert!(matches!(classify(&err), crate::engine::ErrorClass::Fault(_)), "{err:#}");
    }

    #[tokio::test]
    async fn wait_idle_waits_for_slots_and_times_out() {
        let f = fake().await;
        let (e, _) = engine(&f, limits(500, 500), true);
        f.state.busy.store(true, Ordering::SeqCst);
        let err = e.wait_idle(Duration::from_millis(600)).await.unwrap_err();
        assert_eq!(classify(&err), crate::engine::ErrorClass::Fault(FaultCause::SettleTimeout));
        let st = f.state.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(400)).await;
            st.busy.store(false, Ordering::SeqCst);
        });
        e.wait_idle(Duration::from_secs(3)).await.unwrap();
        // Probe when slot status is unsupported.
        f.state.slots_missing.store(true, Ordering::SeqCst);
        e.wait_idle(Duration::from_secs(3)).await.unwrap();
    }

    #[tokio::test]
    async fn verify_steps_count_as_progress() {
        let f = fake().await;
        let (e, p) = engine(&f, limits(2000, 2000), true);
        let before = p.snapshot().seq;
        let engine = Engine::Llama(e);
        let v = engine.verify(&request("x", 0.0, 0, 0), &[100, 101, 102, 103], &[1, 3], 3).await.unwrap();
        assert_eq!(v.positions.len(), 2);
        assert!(p.snapshot().seq >= before + 3);
        // The verification result must include the independently counted prompt usage.
        let g = engine.generate(&request("x", 0.0, 0, 0), 3).await.unwrap();
        assert!(v.prompt_tokens > 0 && v.prompt_tokens == g.prompt_tokens, "verification {} vs generation {}", v.prompt_tokens, g.prompt_tokens);
    }

    /// Parallel slots must let both chats finish with independent progress and text fragments.
    #[tokio::test]
    async fn two_simultaneous_chats_both_finish() {
        let f = fake().await;
        f.state.token_delay_ms.store(60, Ordering::SeqCst);
        let (e, comune) = engine(&f, limits(3000, 3000), true);
        let e = Arc::new(e);
        let (pa, pb) = (Arc::new(Progress::default()), Arc::new(Progress::default()));
        let t0 = Instant::now();
        let chat = |text: &'static str, p: Arc<Progress>| {
            let e = e.clone();
            let (tx, rx, _) = super::super::stages::delta_channel();
            let h = tokio::spawn(async move { e.chat_lease(&request(text, 0.0, 1, 0), 3, &p, Some(tx), None).await });
            (h, rx)
        };
        let ((ha, mut rxa), (hb, mut rxb)) = (chat("prima", pa.clone()), chat("seconda", pb.clone()));
        let (ga, gb) = (ha.await.unwrap().unwrap(), hb.await.unwrap().unwrap());
        // These delays make serialized completion exceed the expected bound.
        assert!(t0.elapsed() < Duration::from_millis(1300), "the two chats did not run concurrently: {:?}", t0.elapsed());
        for g in [&ga, &gb] {
            assert_eq!(g.token_ids.len(), 12);
            assert_eq!(g.finish_reason, FinishReason::Stop);
        }
        // Reassemble each chat from its own stream.
        for (g, rx) in [(&ga, &mut rxa), (&gb, &mut rxb)] {
            let mut text = String::new();
            while let Ok(d) = rx.try_recv() {
                text.push_str(&d.item.text);
            }
            assert_eq!(text, g.text);
        }
        // The shared counter must reflect progress from both leases.
        let (sa, sb, sc) = (pa.snapshot().seq, pb.snapshot().seq, comune.snapshot().seq);
        assert!(sa >= 12 && sb >= 12 && sc >= sa + sb, "seq lease {sa} e {sb}, comune {sc}");
    }

    /// Cancelling one chat must leave the other running.
    #[tokio::test]
    async fn cancelling_one_chat_preserves_the_other() {
        let f = fake().await;
        f.state.token_delay_ms.store(80, Ordering::SeqCst);
        let (e, _) = engine(&f, limits(3000, 3000), true);
        let e = Arc::new(e);
        let (cancel_tx, cancel_rx) = oneshot::channel();
        let (ea, eb) = (e.clone(), e.clone());
        let pa = Arc::new(Progress::default());
        let ha = tokio::spawn(async move { ea.chat_lease(&request("annullata", 0.0, 1, 0), 3, &pa, None, Some(cancel_rx)).await });
        let pb = Arc::new(Progress::default());
        let hb = tokio::spawn(async move { eb.chat_lease(&request("intera", 0.0, 2, 0), 3, &pb, None, None).await });
        tokio::time::sleep(Duration::from_millis(350)).await;
        cancel_tx.send(()).unwrap();
        let ga = ha.await.unwrap().unwrap();
        assert_eq!(ga.finish_reason, FinishReason::Cancelled);
        assert!(ga.token_ids.len() < 12, "cancelled mid-stream with {} tokens", ga.token_ids.len());
        assert_eq!(ga.probs.len(), ga.token_ids.len());
        let gb = hb.await.unwrap().unwrap();
        assert_eq!((gb.token_ids.len(), gb.finish_reason), (12, FinishReason::Stop));
    }
}

/// Process integration tests require a server executable and fixture weights
/// provided through PUBLICOMPUTE_LLAMA_BIN and PUBLICOMPUTE_LLAMA_MODEL. They are
/// ignored by a normal test run because they launch a real engine.
#[cfg(test)]
mod real_tests {
    use std::{path::PathBuf, sync::atomic::Ordering};

    use publicompute_proto::node::{ChatMessage, SamplingParams};

    use super::*;
    use crate::engine::{
        managed::ManagedCfg,
        stats::Stats,
        supervisor::{EngineState, LeaseOutcome, Supervisor, SupervisorCfg},
        Engine,
    };

    fn bin() -> PathBuf {
        std::env::var_os("PUBLICOMPUTE_LLAMA_BIN")
            .expect("Set PUBLICOMPUTE_LLAMA_BIN to the llama-server executable")
            .into()
    }

    fn model() -> PathBuf {
        std::env::var_os("PUBLICOMPUTE_LLAMA_MODEL")
            .expect("Set PUBLICOMPUTE_LLAMA_MODEL to the GGUF fixture")
            .into()
    }

    fn managed(port: u16, tag: &str) -> ManagedCfg {
        let lib = bin().parent().unwrap().to_string_lossy().into_owned();
        std::env::set_var("LD_LIBRARY_PATH", lib);
        ManagedCfg {
            bin: bin(),
            model_path: model(),
            port,
            gpu_index: None,
            extra_args: ["-t", "2", "-c", "2048", "-np", "1"].iter().map(|s| s.to_string()).collect(),
            log_path: std::env::temp_dir().join(format!("publicompute-real-{tag}.log")),
            stop_grace: Duration::from_secs(10),
            kill_wait: Duration::from_secs(15),
        }
    }

    fn req(text: &str, temperature: f32, seed: u64, max_tokens: u32, stop: &[&str]) -> InferenceRequest {
        InferenceRequest {
            request_id: "r".into(),
            messages: vec![ChatMessage { role: ChatRole::User as i32, content: text.into() }],
            sampling: Some(SamplingParams { temperature, max_tokens, seed, stop: stop.iter().map(|s| s.to_string()).collect(), ..Default::default() }),
        }
    }

    async fn wait_health(e: &LlamaEngine) {
        let t0 = Instant::now();
        while e.health().await.is_err() {
            assert!(t0.elapsed() < Duration::from_secs(60), "llama-server did not start");
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
    }

    fn lim(stall: u64) -> Limits {
        Limits {
            prefill_stall: Duration::from_secs(stall),
            decode_stall: Duration::from_secs(stall),
            request_max: Duration::from_secs(300),
            call_timeout: Duration::from_secs(30),
            step_timeout: Duration::from_secs(60),
            probe_timeout: Duration::from_secs(30),
        }
    }

    fn assert_same(a: &Generation, b: &Generation, what: &str) {
        assert_eq!(a.text, b.text, "{what}: testo");
        assert_eq!(a.token_ids, b.token_ids, "{what}: token");
        assert_eq!(a.finish_reason, b.finish_reason, "{what}: finish");
        assert_eq!(a.prompt_tokens_sha256, b.prompt_tokens_sha256, "{what}: prompt hash");
        assert_eq!(a.probs.len(), b.probs.len(), "{what}: number of entries");
        for (i, (x, y)) in a.probs.iter().zip(&b.probs).enumerate() {
            assert_eq!(x.chosen_logprob.to_bits(), y.chosen_logprob.to_bits(), "{what}: chosen log probability at {i}");
            assert_eq!(x.top.len(), y.top.len(), "{what}: lunghezza top in {i}");
            for (p, q) in x.top.iter().zip(&y.top) {
                assert_eq!((p.0, p.1.to_bits()), (q.0, q.1.to_bits()), "{what}: top in {i}");
            }
        }
    }

    /// Compare complete streams and check the text of reconstructed streams.
    #[tokio::test]
    #[ignore]
    async fn real_streaming_preserves_text_and_complete_scores() {
        let port = 18091;
        let cfg = managed(port, "equal");
        let _child = cfg.spawn(None).unwrap();
        let url = format!("http://127.0.0.1:{port}");
        let progress = Arc::new(Progress::default());
        let s = LlamaEngine::new(url.clone(), lim(60), progress.clone(), true).unwrap();
        let n = LlamaEngine::new(url, lim(60), progress, false).unwrap();
        wait_health(&s).await;

        let prompts = [
            "Qual è la capitale d'Italia? Rispondi con una parola.",
            "Conta da uno a cinque.",
            "Ripeti esattamente, senza altro testo: 奕秋 轩文 🙂🐢",
            "请用中文写一句关于春天的话。",
            "Scrivi tre emoji di animali: 🦊🐙🦉",
            "Perché è così difficile scrivere città, perché, più, già, caffè?",
            "Tradotto in giapponese: 'buongiorno a tutti'.",
            "Rispondi con una sola riga: cos'è un numero primo?",
            "List three colors, comma separated.",
            "Écris une phrase avec des accents : é è ê à ù ç.",
            "Ripeti: 🧑‍🚀🏳️‍🌈👨‍👩‍👧",
            "什么是人工智能？一句话回答。",
            "Dimmi un proverbio italiano.",
            "Complete: The quick brown fox",
            "Scrivi la parola 'ciao' in cinque lingue.",
            "Descrivi il mare in 10 parole.",
            "¿Cuál es la capital de España? Responde breve.",
            "Elenca i giorni della settimana.",
            "Quanto fa 17 per 23? Rispondi solo con il numero.",
            "Scrivi un haiku sulla neve.",
            "Ποια είναι η πρωτεύουσα της Ελλάδας;",
            "Rispondi con 'ok' e basta.",
        ];
        // Cached prefixes can change batch shape and rounding. At nonzero temperature
        // even small differences can change which token is sampled. Reset the prefix
        // before each comparison.
        let flush = req("Z", 0.0, 0, 1, &[]);
        let mut cases: Vec<(String, InferenceRequest)> = Vec::new();
        for (i, p) in prompts.iter().enumerate() {
            for (temp, seed) in [(0.0f32, 0u64), (0.8, 42)] {
                cases.push((format!("prompt {i} T={temp}"), req(p, temp, seed, 24, &[])));
            }
        }
        for stop in [&["o"][..], &["\n"][..], &["a", "e"][..]] {
            cases.push((format!("stop {stop:?}"), req("Conta da uno a cinque.", 0.0, 0, 24, stop)));
        }
        let (mut compared, mut repaired) = (0, 0);
        for (i, (what, r)) in cases.iter().enumerate() {
            n.generate(&flush, 1).await.unwrap();
            let (tx, mut rx, _) = super::super::stages::delta_channel();
            let streamed = s.chat_lease(r, 5, &s.progress, Some(tx), None).await.unwrap();
            let mut text = String::new();
            while let Ok(delta) = rx.try_recv() {
                text.push_str(&delta.item.text);
            }
            assert_eq!(text, streamed.text, "{what}: streamed text");
            assert_eq!(streamed.probs.len(), streamed.token_ids.len(), "{what}: probability positions");
            if streamed.verification_incomplete {
                repaired += 1;
            } else {
                n.generate(&flush, 1).await.unwrap();
                let baseline = n.generate(r, 5).await.unwrap();
                assert_same(&baseline, &streamed, &format!("{what} (case {i})"));
                compared += 1;
            }
        }
        println!("Compared {compared} complete streams and checked {repaired} reconstructed streams");
        assert!(compared + repaired >= 40);
    }

    /// Suspending a managed child during generation must trigger recovery.
    #[tokio::test]
    #[ignore]
    async fn real_managed_freeze_is_detected_and_recovered() {
        let port = 18092;
        let stall = 5;
        let progress = Arc::new(Progress::default());
        let url = format!("http://127.0.0.1:{port}");
        let engine = Arc::new(Engine::Llama(LlamaEngine::new(url, lim(stall), progress.clone(), true).unwrap()));
        let sup = Supervisor::new(
            engine.clone(),
            SupervisorCfg {
                managed: Some(managed(port, "freeze")),
                expected_model: Some(model()),
                probe_interval: Duration::from_secs(3600),
                settle: Duration::from_secs(30),
                max_restarts: 3,
                restart_window: Duration::from_secs(3600),
                backoff_base: Duration::from_secs(1),
                backoff_max: Duration::from_secs(5),
                recycle: None,
                start_timeout: Duration::from_secs(120),
                maintenance_tick: Duration::from_secs(1),
            },
            Arc::new(Stats::new(None)),
            progress.clone(),
        );
        let t0 = Instant::now();
        sup.start().await.unwrap();
        assert_eq!(sup.state(), EngineState::Ready);
        println!("Engine startup, including health check, model load and probe: {:?}", t0.elapsed());
        let old_pid = sup.child_pid_for_test().await;

        let guard = sup.try_begin().ok().unwrap();
        let eng = engine.clone();
        let gen = tokio::spawn(async move { eng.generate(&req("Scrivi una lunga storia su un drago.", 0.0, 0, 400, &[]), 5).await });
        // Confirm useful generation before suspending the process.
        let seq0 = progress.snapshot().seq;
        while progress.snapshot().seq < seq0 + 8 {
            tokio::time::sleep(Duration::from_millis(100)).await;
            assert!(!gen.is_finished(), "generation finished before the process was paused");
        }
        let frozen = Instant::now();
        unsafe { libc::kill(-(old_pid as i32), libc::SIGSTOP) };
        let err = gen.await.unwrap().unwrap_err();
        let trip = frozen.elapsed();
        println!("watchdog fired {trip:?} after the process paused (limit {stall}s): {err:#}");
        assert_eq!(crate::engine::classify(&err), crate::engine::ErrorClass::Fault(crate::engine::FaultCause::Watchdog));
        assert!(trip >= Duration::from_secs(stall - 1) && trip < Duration::from_secs(stall + 5));

        guard.finish(LeaseOutcome::Fault(crate::engine::FaultCause::Watchdog));
        let t1 = Instant::now();
        while sup.state() != EngineState::Ready {
            assert!(t1.elapsed() < Duration::from_secs(180), "recovery took too long, state {:?}", sup.state());
            assert_ne!(sup.state(), EngineState::Quarantined);
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        println!("Engine recovery including restart and probe: {:?}", t1.elapsed());
        let new_pid = sup.child_pid_for_test().await;
        assert_ne!(old_pid, new_pid);
        assert!(unsafe { libc::kill(old_pid as i32, 0) } != 0, "the replaced llama-server should have exited");
        assert_eq!(sup.stats.run().engine_restarts, 1);
        // Verify that the replacement can serve a normal request.
        let g = engine.generate(&req("Ciao", 0.0, 0, 8, &[]), 5).await.unwrap();
        assert!(!g.token_ids.is_empty());
        sup.shutdown().await;
        assert!(unsafe { libc::kill(new_pid as i32, 0) } != 0);
        let _ = Ordering::SeqCst;
    }
}
