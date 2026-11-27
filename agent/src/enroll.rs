//! Enrollment exchanges a certificate request and proof of work for the node certificate.

use std::path::Path;

use anyhow::{bail, Context, Result};
use publicompute_proto::{
    node::{enroll_client_message, enroll_server_message, EnrollBegin, EnrollClientMessage, EnrollProof, EnrollServerMessage, HardwareProfile},
    PROTOCOL_VERSION,
};
use tracing::info;

use crate::{
    framed::{self, FrameReader},
    identity::{self, Identity, PendingKey},
    transport::Transport,
};

/// Keep the enrollment token confined to the request. It must stay out of logs and errors.
pub async fn enroll(t: &Transport, data_dir: &Path, hw: HardwareProfile, owner_token: Option<&str>) -> Result<Identity> {
    let pending = PendingKey::generate()?;
    let conn = t.connect_anonymous().await.context("connect for enrollment")?;
    let (mut send, recv) = conn.open_bi().await?;
    let mut r = FrameReader::new(recv);

    let begin = begin_message(&pending.csr_der, hw, identity::hardware_fingerprint(data_dir)?, owner_token);
    framed::write(&mut send, &begin).await?;

    let challenge = match next(&mut r).await? {
        enroll_server_message::Body::Challenge(c) => c,
        other => bail!("expected EnrollChallenge, got {other:?}"),
    };
    info!(bits = challenge.pow_difficulty_bits, "solving proof of work");
    let (nonce, csr, bits) = (challenge.nonce.clone(), pending.csr_der.clone(), challenge.pow_difficulty_bits);
    // Run the proof-of-work search off the async executor so it can continue servicing I/O.
    let counter = tokio::task::spawn_blocking(move || identity::solve_pow(&nonce, &csr, bits)).await?;
    let proof = EnrollClientMessage {
        body: Some(enroll_client_message::Body::Proof(EnrollProof { pow_counter: counter, activated_secret: vec![] })),
    };
    framed::write(&mut send, &proof).await?;

    let result = match next(&mut r).await? {
        enroll_server_message::Body::Result(res) => res,
        other => bail!("expected EnrollResult, got {other:?}"),
    };
    send.finish().ok();
    conn.close(0u32.into(), b"");
    info!(node = %result.node_id, tpm_bound = result.tpm_bound, "node enrolled");
    identity::save(data_dir, pending, result)
}

fn begin_message(csr_der: &[u8], hw: HardwareProfile, fingerprint: Vec<u8>, owner_token: Option<&str>) -> EnrollClientMessage {
    EnrollClientMessage {
        body: Some(enroll_client_message::Body::Begin(EnrollBegin {
            protocol_version: Some(PROTOCOL_VERSION),
            agent_version: env!("CARGO_PKG_VERSION").to_string(),
            csr_der: csr_der.to_vec(),
            hardware: Some(hw),
            hardware_fingerprint: fingerprint,
            tpm: None,
            owner_token: owner_token.unwrap_or_default().to_string(),
        })),
    }
}

async fn next(r: &mut FrameReader) -> Result<enroll_server_message::Body> {
    let m: EnrollServerMessage = r.read().await?.context("coordinator closed the stream")?;
    match m.body {
        Some(enroll_server_message::Body::Error(e)) => {
            if let Some(rejection) = crate::session::IncompatibleRelease::from_protocol(&e) {
                return Err(rejection.into());
            }
            bail!("enrollment rejected: {} ({:?})", e.message, e.code())
        }
        Some(b) => Ok(b),
        None => bail!("empty message"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn owner_token_of(m: &EnrollClientMessage) -> String {
        match &m.body {
            Some(enroll_client_message::Body::Begin(b)) => b.owner_token.clone(),
            _ => panic!("expected EnrollBegin"),
        }
    }

    #[test]
    fn begin_carries_owner_token() {
        let m = begin_message(&[1], HardwareProfile::default(), vec![0; 32], Some("pc-own-prova"));
        assert_eq!(owner_token_of(&m), "pc-own-prova");
    }

    #[test]
    fn missing_owner_token_leaves_field_empty() {
        let m = begin_message(&[1], HardwareProfile::default(), vec![0; 32], None);
        assert_eq!(owner_token_of(&m), "");
    }
}
