//! The agent uses QUIC with TLS 1.3 and trusts only the configured coordinator
//! CA. An enrolled agent also presents its client certificate.

use std::{net::SocketAddr, sync::Arc, time::Duration};

use anyhow::{Context, Result};
use quinn::crypto::rustls::QuicClientConfig;
use rustls::pki_types::CertificateDer;

use crate::identity::Identity;

pub const ALPN: &[u8] = b"publicompute-node/0";

/// Application close codes shared with the coordinator. Keep them in sync with the peer.
pub const CLOSE_UNAUTHENTICATED: u32 = 2;
pub const CLOSE_BANNED: u32 = 3;
pub const CLOSE_REPLACED: u32 = 4;

pub struct Transport {
    endpoint: quinn::Endpoint,
    ca: CertificateDer<'static>,
    addr: SocketAddr,
    server_name: String,
}

impl Transport {
    pub fn new(addr: SocketAddr, server_name: String, ca_der: Vec<u8>) -> Result<Self> {
        let bind: SocketAddr = if addr.is_ipv6() { "[::]:0".parse()? } else { "0.0.0.0:0".parse()? };
        let endpoint = quinn::Endpoint::client(bind)?;
        Ok(Self { endpoint, ca: CertificateDer::from(ca_der), addr, server_name })
    }

    fn client_config(&self, id: Option<&Identity>) -> Result<quinn::ClientConfig> {
        let mut roots = rustls::RootCertStore::empty();
        roots.add(self.ca.clone()).context("invalid coordinator CA")?;
        let builder = rustls::ClientConfig::builder_with_provider(Arc::new(rustls::crypto::ring::default_provider()))
            .with_protocol_versions(&[&rustls::version::TLS13])?
            .with_root_certificates(roots);
        let mut tls = match id {
            Some(id) => builder.with_client_auth_cert(id.cert_chain(), id.private_key())?,
            None => builder.with_no_client_auth(),
        };
        tls.alpn_protocols = vec![ALPN.to_vec()];
        let mut cfg = quinn::ClientConfig::new(Arc::new(QuicClientConfig::try_from(tls)?));
        let mut t = quinn::TransportConfig::default();
        t.keep_alive_interval(Some(Duration::from_secs(10)));
        t.max_idle_timeout(Some(Duration::from_secs(30).try_into()?));
        cfg.transport_config(Arc::new(t));
        Ok(cfg)
    }

    /// Enrollment needs a connection before the agent has a client certificate.
    pub async fn connect_anonymous(&self) -> Result<quinn::Connection> {
        let cfg = self.client_config(None)?;
        Ok(self.endpoint.connect_with(cfg, self.addr, &self.server_name)?.await?)
    }

    pub async fn connect(&self, id: &Identity) -> Result<quinn::Connection> {
        let cfg = self.client_config(Some(id))?;
        Ok(self.endpoint.connect_with(cfg, self.addr, &self.server_name)?.await?)
    }
}

/// Read the application code when the coordinator closes the connection.
pub fn app_close_code(err: &quinn::ConnectionError) -> Option<u32> {
    match err {
        quinn::ConnectionError::ApplicationClosed(c) => u32::try_from(c.error_code.into_inner()).ok(),
        _ => None,
    }
}
