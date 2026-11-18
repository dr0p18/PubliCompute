//! The stage backend forwards one lease to the head's service socket. The
//! head receives the complete conversation and owns prompt formatting and
//! trimming. It also receives the sampling cutoffs and stop strings. Local
//! sampling uses the cutoffs there, while a remote tail receives temperature
//! and seed. The tail returns the chosen token's probability for the commitment.

use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Instant,
};

use anyhow::Result;
use publicompute_proto::{
    node::{ChatRole, FinishReason, InferenceRequest, RequestFailureCode, SamplingParams},
    service as pb,
};
use tokio::sync::{mpsc, oneshot};

use super::{
    fault,
    progress::{Phase, Progress},
    prompt_tokens_sha256, request_error, FaultCause, Generation, Limits, TokenProbs,
};
use crate::stages::{Stages, OutgoingLink};

/// Map the request's temperature, seed and sampling cutoffs to the head service protocol.
fn campionamento_testa(s: &SamplingParams) -> pb::Sampling {
    pb::Sampling { temperature: s.temperature, seed: s.seed, top_p: s.top_p, top_k: s.top_k }
}

/// Default generation limit when the request omits one.
const MAX_TOKEN_DEFAULT: u32 = 1024;

/// A streamed fragment and the token range it covers.
#[derive(Debug, Clone, PartialEq)]
pub struct Delta {
    /// Index of the first token covered by this fragment.
    pub index: u32,
    pub text: String,
    pub token_ids: Vec<u32>,
}

pub struct StadiEngine {
    pub stages: Arc<Stages>,
    /// Number of in-flight requests, bounded by the head's slot count.
    pub attive: AtomicUsize,
    pub limits: Limits,
    pub progress: Arc<Progress>,
}

/// Encode the complete conversation. Return None if it has no user message
/// or uses an unsupported role.
pub fn messaggi_testa(req: &InferenceRequest) -> Option<Vec<pb::Message>> {
    req.messages.iter().rev().find(|m| m.role() == ChatRole::User)?;
    req.messages
        .iter()
        .map(|m| {
            let role = match m.role() {
                ChatRole::System => "system",
                ChatRole::User => "user",
                ChatRole::Assistant => "assistant",
                _ => return None,
            };
            Some(pb::Message { role: role.into(), text: m.content.clone() })
        })
        .collect()
}

/// Publish RUNNING and READY only when the active count crosses zero.
struct Attiva<'a>(&'a StadiEngine);

impl<'a> Attiva<'a> {
    fn new(e: &'a StadiEngine) -> Self {
        if e.attive.fetch_add(1, Ordering::SeqCst) == 0 {
            e.stages.attivita(true);
        }
        Attiva(e)
    }
}

impl Drop for Attiva<'_> {
    fn drop(&mut self) {
        if self.0.attive.fetch_sub(1, Ordering::SeqCst) == 1 {
            self.0.stages.attivita(false);
        }
    }
}

impl StadiEngine {
    /// Readiness means the head service is connected.
    pub fn pronto(&self) -> bool {
        self.stages.head().is_some()
    }

    /// Apply a progress update to the lease and shared counters, deduplicating identical handles.
    fn su_entrambi(&self, lease: &Progress, f: impl Fn(&Progress)) {
        f(lease);
        if !std::ptr::eq(lease, &*self.progress) {
            f(&self.progress);
        }
    }

    pub async fn generate(&self, req: &InferenceRequest, _top_k: u32) -> Result<Generation> {
        self.chat(req, None, None).await
    }

    /// Run a chat using the shared engine progress counter.
    pub async fn chat(&self, req: &InferenceRequest, deltas: Option<mpsc::UnboundedSender<Delta>>, cancel: Option<oneshot::Receiver<()>>) -> Result<Generation> {
        self.chat_lease(req, &self.progress, deltas, cancel).await
    }

    /// Give this chat its own progress counter and optionally stream text as it
    /// arrives. Also update shared progress for supervision. Cancellation or
    /// loss of its sender sends a cancel message to the head and ends the lease.
    pub async fn chat_lease(&self, req: &InferenceRequest, progress: &Progress, deltas: Option<mpsc::UnboundedSender<Delta>>, mut cancel: Option<oneshot::Receiver<()>>) -> Result<Generation> {
        let Some(link) = self.stages.head() else {
            return Err(request_error(RequestFailureCode::EngineError, "no chain head is ready on this node"));
        };
        let Some(messages) = messaggi_testa(req) else {
            return Err(request_error(RequestFailureCode::EngineError, "request has no user messages"));
        };
        let text = messages.iter().rev().find(|m| m.role == "user").map(|m| m.text.clone()).unwrap_or_default();
        let _attiva = Attiva::new(self);
        let s = req.sampling.clone().unwrap_or_default();
        let id = link.nuovo_id();
        let t0 = Instant::now();
        self.su_entrambi(progress, |p| p.set_phase(Phase::Prefill));
        self.su_entrambi(progress, Progress::tick);
        let mut rx = link
            .request(pb::Request {
                id,
                prompt_token: vec![],
                text,
                messages,
                max_tokens: if s.max_tokens == 0 { MAX_TOKEN_DEFAULT } else { s.max_tokens },
                sampling: Some(campionamento_testa(&s)),
                stop: s.stop.clone(),
                class: pb::Class::Interactive as i32,
            })
            .await
            .map_err(|e| fault(FaultCause::Connection, format!("request to the head: {e:#}")))?;

        let (mut text, mut token_ids, mut probs) = (String::new(), Vec::<u32>::new(), Vec::<TokenProbs>::new());
        let (mut in_attesa, mut primo_indice) = (Vec::<u32>::new(), 0u32);
        let (mut primo_token, mut annullata) = (None::<Instant>, false);
        let done = loop {
            let limite = if primo_token.is_none() { self.limits.prefill_stall } else { self.limits.decode_stall };
            if t0.elapsed() > self.limits.request_max {
                link.cancel(id).await;
                link.dimentica(id);
                return Err(fault(FaultCause::RequestMax, format!("request exceeded {} s", self.limits.request_max.as_secs())));
            }
            tokio::select! {
                u = tokio::time::timeout(limite, rx.recv()) => match u {
                    Err(_) => {
                        link.cancel(id).await;
                        link.dimentica(id);
                        return Err(fault(FaultCause::Watchdog, format!("no token from the head for {} s", limite.as_secs())));
                    }
                    Ok(None) => {
                        link.dimentica(id);
                        return Err(fault(FaultCause::Connection, "connection to the head lost mid-response"));
                    }
                    Ok(Some(OutgoingLink::Token(t))) => {
                        if primo_token.is_none() {
                            primo_token = Some(Instant::now());
                            self.su_entrambi(progress, |p| p.set_phase(Phase::Decode));
                        }
                        self.su_entrambi(progress, Progress::tick);
                        token_ids.push(t.token as u32);
                        probs.push(TokenProbs { chosen_logprob: t.logprob, top: vec![] });
                        in_attesa.push(t.token as u32);
                        text.push_str(&t.text);
                        // Include tokens with no text in the next displayable fragment so token offsets stay aligned.
                        if !t.text.is_empty() || t.eog {
                            if let Some(d) = &deltas {
                                let _ = d.send(Delta { index: primo_indice, text: t.text, token_ids: std::mem::take(&mut in_attesa) });
                            }
                            in_attesa.clear();
                            primo_indice = token_ids.len() as u32;
                        }
                    }
                    Ok(Some(OutgoingLink::Done(f))) => break f,
                },
                r = async { cancel.as_mut().unwrap().await }, if cancel.is_some() && !annullata => {
                    // A dropped sender also means the caller no longer needs the response.
                    let _ = r;
                    annullata = true;
                    cancel = None;
                    link.cancel(id).await;
                }
            }
        };
        link.dimentica(id);
        let finish_reason = match done.reason.as_str() {
            "error" => {
                // The head includes this phrase in both context overflow messages.
                let code = if done.error.contains("the slot context") { RequestFailureCode::ContextTooLong } else { RequestFailureCode::EngineError };
                return Err(request_error(code, format!("head rejected the request: {}", done.error)));
            }
            "failed" => return Err(request_error(RequestFailureCode::EngineError, "conversation failed in the chain")),
            "eog" | "stop" => FinishReason::Stop,
            "max_token" => FinishReason::Length,
            "cancelled" => FinishReason::Cancelled,
            reason => return Err(request_error(RequestFailureCode::EngineError, format!("unknown head finish reason {reason:?}"))),
        };
        text.push_str(&done.remaining_text);
        if let Some(d) = &deltas {
            if !done.remaining_text.is_empty() || !in_attesa.is_empty() {
                let _ = d.send(Delta { index: primo_indice, text: done.remaining_text.clone(), token_ids: std::mem::take(&mut in_attesa) });
            }
        }
        let prompt: Vec<u32> = done.prompt_token.iter().map(|&t| t as u32).collect();
        let t_primo = primo_token.unwrap_or_else(Instant::now);
        Ok(Generation {
            text,
            token_ids,
            finish_reason,
            prompt_tokens: done.usage.map_or(prompt.len() as u32, |u| u.prompt_tokens),
            prompt_tokens_sha256: prompt_tokens_sha256(&prompt),
            probs,
            verification_incomplete: false,
            prefill: t_primo.saturating_duration_since(t0),
            generation: Instant::now().saturating_duration_since(t_primo),
        })
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn request_to_head_carries_top_p_top_k_and_stop() {
        let s = SamplingParams { temperature: 0.7, top_p: 0.9, top_k: 40, max_tokens: 64, seed: 5, stop: vec!["###".into(), "Fine".into()] };
        let c = campionamento_testa(&s);
        assert_eq!((c.temperature, c.seed, c.top_p, c.top_k), (0.7, 5, 0.9, 40));
        let r = pb::Request { sampling: Some(c), stop: s.stop.clone(), ..Default::default() };
        let letta = <pb::Request as prost::Message>::decode(prost::Message::encode_to_vec(&r).as_slice()).unwrap();
        assert_eq!(letta.stop, vec!["###".to_string(), "Fine".to_string()]);
        let c = letta.sampling.unwrap();
        assert_eq!((c.top_p, c.top_k), (0.9, 40));
        let c0 = campionamento_testa(&SamplingParams::default());
        assert_eq!((c0.top_p, c0.top_k), (0.0, 0));
    }
}
