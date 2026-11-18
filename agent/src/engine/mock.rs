//! Deterministic fixture that echoes the last user message and treats each byte
//! as a token. It exercises generation and verification without a hardware backend.

use std::{
    sync::{
        atomic::{AtomicU32, Ordering},
        Arc,
    },
    time::Duration,
};

use anyhow::Result;
use publicompute_proto::node::{ChatRole, FinishReason, InferenceRequest};
use tokio::sync::{mpsc, oneshot};

use super::{
    fault,
    progress::{Phase, Progress},
    prompt_tokens_sha256,
    stages::Delta,
    FaultCause, Generation, Limits, Recomputed, TokenProbs, Verification,
};

pub struct MockEngine {
    /// Per-token delay for exercising long leases and cancellation.
    pub per_token_delay: Duration,
    /// Fabricate a response and its matching commitment to exercise independent verification.
    pub cheat: bool,
    /// Stall this many consecutive generations after initial progress. The watchdog
    /// ends each after its decode stall deadline.
    pub hang: AtomicU32,
    pub limits: Limits,
    pub progress: Arc<Progress>,
}

impl MockEngine {
    pub async fn probe(&self) -> Result<Duration> {
        Ok(Duration::from_millis(1))
    }

    pub async fn generate(&self, req: &InferenceRequest, top_k: u32) -> Result<Generation> {
        self.generate_inner(req, top_k, self.cheat, &self.progress, None, None).await
    }

    /// Run a chat with independent progress, deltas and cancellation.
    pub async fn chat_lease(&self, req: &InferenceRequest, top_k: u32, progress: &Progress, deltas: Option<mpsc::UnboundedSender<Delta>>, cancel: Option<oneshot::Receiver<()>>) -> Result<Generation> {
        self.generate_inner(req, top_k, self.cheat, progress, deltas, cancel).await
    }

    /// Verification uses the genuine fixture output even when generation is configured to fabricate it.
    pub async fn verify(&self, req: &InferenceRequest, output: &[u32], positions: &[u32], top_k: u32) -> Result<Verification> {
        let g = self.generate_inner(req, top_k, false, &self.progress, None, None).await?;
        let positions = positions
            .iter()
            .map(|p| {
                let i = *p as usize;
                let top = g.probs.get(i).map(|tp| tp.top.clone()).unwrap_or_default();
                let chosen_logprob = top.iter().find(|(id, _)| *id == output[i]).map(|(_, lp)| *lp);
                Recomputed { position: *p, top, chosen_logprob }
            })
            .collect();
        Ok(Verification { prompt_tokens_sha256: g.prompt_tokens_sha256, prompt_tokens: g.prompt_tokens, positions })
    }

    async fn generate_inner(
        &self,
        req: &InferenceRequest,
        top_k: u32,
        cheat: bool,
        progress: &Progress,
        deltas: Option<mpsc::UnboundedSender<Delta>>,
        mut cancel: Option<oneshot::Receiver<()>>,
    ) -> Result<Generation> {
        let su_entrambi = |f: &dyn Fn(&Progress)| {
            f(progress);
            if !std::ptr::eq(progress, &*self.progress) {
                f(&self.progress);
            }
        };
        let prompt: Vec<u32> = req
            .messages
            .iter()
            .flat_map(|m| m.content.bytes().map(u32::from))
            .collect();
        let last_user = req
            .messages
            .iter()
            .rev()
            .find(|m| m.role() == ChatRole::User)
            .map(|m| m.content.as_str())
            .unwrap_or("");
        let mut out: Vec<u8> = if cheat {
            // Keep the fabricated response the same length to exercise verification beyond a length check.
            format!("eco: {last_user}").bytes().map(|b| if b.is_ascii_alphabetic() { b'z' } else { b }).collect()
        } else {
            format!("eco: {last_user}").into_bytes()
        };
        let max = req.sampling.as_ref().map(|s| s.max_tokens).unwrap_or(0) as usize;
        let mut finish = FinishReason::Stop;
        if max > 0 && out.len() > max {
            out.truncate(max);
            finish = FinishReason::Length;
        }
        su_entrambi(&|p| p.set_phase(Phase::Prefill));
        su_entrambi(&|p| p.tick());
        su_entrambi(&|p| p.set_phase(Phase::Decode));
        let hang = self.hang.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1)).is_ok();
        if hang {
            // Emit a few tokens before stalling so this exercises the decode deadline.
            su_entrambi(&|p| p.tick());
            tokio::time::sleep(self.limits.decode_stall).await;
            return Err(fault(FaultCause::Watchdog, format!("mock engine stalled: no progress for {} s", self.limits.decode_stall.as_secs_f32())));
        }
        // Buffer bytes until a complete UTF-8 character can be displayed.
        let (mut in_attesa, mut primo) = (Vec::<u8>::new(), 0usize);
        for i in 0..out.len() {
            if !self.per_token_delay.is_zero() {
                let pausa = tokio::time::sleep(self.per_token_delay);
                tokio::pin!(pausa);
                // Cancellation also applies when the receiver loses its sender.
                let fermata = match cancel.as_mut() {
                    Some(rx) => tokio::select! { _ = &mut pausa => false, _ = rx => true },
                    None => {
                        pausa.await;
                        false
                    }
                };
                if fermata {
                    out.truncate(i);
                    finish = FinishReason::Cancelled;
                    break;
                }
            } else if cancel.as_mut().is_some_and(|rx| !matches!(rx.try_recv(), Err(oneshot::error::TryRecvError::Empty))) {
                out.truncate(i);
                finish = FinishReason::Cancelled;
                break;
            }
            su_entrambi(&|p| p.tick());
            if let Some(d) = &deltas {
                in_attesa.push(out[i]);
                if let Ok(t) = std::str::from_utf8(&in_attesa) {
                    let _ = d.send(Delta { index: primo as u32, text: t.to_string(), token_ids: (primo..=i).map(|k| u32::from(out[k])).collect() });
                    in_attesa.clear();
                    primo = i + 1;
                }
            }
        }
        let probs = out
            .iter()
            .map(|b| TokenProbs {
                chosen_logprob: -0.05,
                top: (0..top_k).map(|i| (u32::from(*b) + i, -0.05 - i as f32)).collect(),
            })
            .collect();
        Ok(Generation {
            // Truncate only at a valid character boundary.
            text: String::from_utf8_lossy(&out).into_owned(),
            token_ids: out.iter().map(|b| u32::from(*b)).collect(),
            finish_reason: finish,
            prompt_tokens: prompt.len() as u32,
            prompt_tokens_sha256: prompt_tokens_sha256(&prompt),
            probs,
            verification_incomplete: false,
            prefill: Duration::from_millis(1),
            generation: self.per_token_delay * out.len() as u32,
        })
    }
}
