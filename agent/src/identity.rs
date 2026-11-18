//! The data directory holds the ECDSA P-256 signing key separately from the
//! serialized enrollment response. Both are needed to reconnect under the
//! enrolled identity. Keep the key private when writing it for the agent or
//! for a stage process.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use prost::Message;
use publicompute_proto::node::EnrollResult;
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
use sha2::{Digest, Sha256};

pub struct Identity {
    pub node_id: String,
    pub chain: Vec<Vec<u8>>,
    key_pkcs8: Vec<u8>,
}

impl Identity {
    pub fn cert_chain(&self) -> Vec<CertificateDer<'static>> {
        self.chain.iter().map(|c| CertificateDer::from(c.clone())).collect()
    }

    /// Export the private key as PKCS #8 DER for a stage process to use in TLS.
    pub fn key_der(&self) -> &[u8] {
        &self.key_pkcs8
    }

    pub fn private_key(&self) -> PrivateKeyDer<'static> {
        PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(self.key_pkcs8.clone()))
    }
}

/// A generated key awaiting its signed certificate.
pub struct PendingKey {
    key: rcgen::KeyPair,
    pub csr_der: Vec<u8>,
}

impl PendingKey {
    pub fn generate() -> Result<Self> {
        let key = rcgen::KeyPair::generate().context("generate the key")?;
        // The coordinator assigns the certificate subject, so the request can omit it.
        let params = rcgen::CertificateParams::new(Vec::<String>::new())?;
        let csr = params.serialize_request(&key).context("create the CSR")?;
        Ok(Self { csr_der: csr.der().to_vec(), key })
    }
}

fn key_path(dir: &Path) -> PathBuf {
    dir.join("key.pem")
}

fn identity_path(dir: &Path) -> PathBuf {
    dir.join("identity.bin")
}

pub fn load(dir: &Path) -> Result<Option<Identity>> {
    let (kp, ip) = (key_path(dir), identity_path(dir));
    if !kp.exists() || !ip.exists() {
        return Ok(None);
    }
    let key = rcgen::KeyPair::from_pem(&std::fs::read_to_string(&kp)?).context("read key.pem")?;
    let res = EnrollResult::decode(std::fs::read(&ip)?.as_slice()).context("read identity.bin")?;
    Ok(Some(Identity { node_id: res.node_id, chain: res.certificate_chain_der, key_pkcs8: key.serialize_der() }))
}

pub fn save(dir: &Path, pending: PendingKey, res: EnrollResult) -> Result<Identity> {
    std::fs::create_dir_all(dir)?;
    write_private(&key_path(dir), pending.key.serialize_pem().as_bytes())?;
    std::fs::write(identity_path(dir), res.encode_to_vec())?;
    Ok(Identity { node_id: res.node_id, chain: res.certificate_chain_der, key_pkcs8: pending.key.serialize_der() })
}

/// Remove local enrollment state so the next session can enroll again.
pub fn forget(dir: &Path) -> Result<()> {
    for p in [key_path(dir), identity_path(dir)] {
        if p.exists() {
            std::fs::remove_file(p)?;
        }
    }
    Ok(())
}

#[cfg(unix)]
fn write_private(path: &Path, data: &[u8]) -> Result<()> {
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;
    let mut f = std::fs::OpenOptions::new().write(true).create(true).truncate(true).mode(0o600).open(path)?;
    f.write_all(data)?;
    Ok(())
}

#[cfg(not(unix))]
fn write_private(path: &Path, data: &[u8]) -> Result<()> {
    use std::io::Write;
    // Restrict access to the Windows file before writing any key bytes.
    let mut f = std::fs::File::create(path)?;
    acl::riserva_all_utente(path)?;
    f.write_all(data)?;
    Ok(())
}

/// Build and apply an owner-only Windows ACL for private keys. Command
/// construction stays portable so its arguments can be tested on any platform.
#[cfg_attr(not(windows), allow(dead_code))]
pub(crate) mod acl {
    #[cfg(windows)]
    use anyhow::{bail, Context, Result};
    use std::ffi::OsString;
    use std::path::Path;

    /// Extract the account SID from the second column of the identity command's CSV output.
    pub fn sid_da_whoami(output: &str) -> Option<String> {
        let sid = output.lines().next()?.split(',').nth(1)?.trim().trim_matches('"').to_string();
        let valido = sid.starts_with("S-1-") && sid.len() > 4 && sid[2..].chars().all(|c| c.is_ascii_digit() || c == '-');
        valido.then_some(sid)
    }

    /// Disable inherited permissions and grant full control to this account SID.
    pub fn argomenti_icacls(path: &Path, sid: &str) -> Vec<OsString> {
        vec![path.as_os_str().to_owned(), "/inheritance:r".into(), "/grant:r".into(), format!("*{sid}:F").into()]
    }

    #[cfg(windows)]
    pub fn riserva_all_utente(path: &Path) -> Result<()> {
        use std::process::Command;
        let out = Command::new("whoami").args(["/user", "/fo", "csv", "/nh"]).output().context("whoami")?;
        let sid = sid_da_whoami(&String::from_utf8_lossy(&out.stdout)).context("whoami did not return the current user's SID")?;
        let r = Command::new("icacls").args(argomenti_icacls(path, &sid)).output().context("icacls")?;
        if !r.status.success() {
            bail!("icacls on {}: {}", path.display(), String::from_utf8_lossy(&r.stdout));
        }
        Ok(())
    }
}

/// Hash a fixed salt with a stable host identifier for enrollment. Treat
/// the fingerprint as a hint because it is easy to reproduce.
pub fn hardware_fingerprint(dir: &Path) -> Result<Vec<u8>> {
    const SALT: &[u8] = b"publicompute-hw-fingerprint-v1";
    let id = match std::fs::read_to_string("/etc/machine-id") {
        Ok(s) if !s.trim().is_empty() => s.trim().to_string(),
        // Use a persisted random identifier when the platform has no readable system identifier.
        _ => {
            let p = dir.join("machine-id");
            if let Ok(s) = std::fs::read_to_string(&p) {
                s
            } else {
                std::fs::create_dir_all(dir)?;
                let s = hex::encode(rand::random::<[u8; 16]>());
                std::fs::write(&p, &s)?;
                s
            }
        }
    };
    let mut h = Sha256::new();
    h.update(SALT);
    h.update(id.as_bytes());
    Ok(h.finalize().to_vec())
}

/// Find the enrollment nonce whose hash satisfies the coordinator's difficulty.
pub fn pow_digest(nonce: &[u8], csr: &[u8], counter: u64) -> [u8; 32] {
    let csr_hash = Sha256::digest(csr);
    let mut h = Sha256::new();
    h.update(nonce);
    h.update(csr_hash);
    h.update(counter.to_be_bytes());
    h.finalize().into()
}

fn leading_zero_bits(d: &[u8; 32]) -> u32 {
    let mut n = 0;
    for b in d {
        if *b == 0 {
            n += 8;
            continue;
        }
        return n + b.leading_zeros();
    }
    n
}

pub fn solve_pow(nonce: &[u8], csr: &[u8], difficulty: u32) -> u64 {
    (0u64..).find(|c| leading_zero_bits(&pow_digest(nonce, csr, *c)) >= difficulty).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Check that private-key access is granted only to the selected account.
    #[test]
    fn windows_acl_sid_and_icacls_arguments() {
        let sid = acl::sid_da_whoami("\"pc\\user\",\"S-1-5-21-111-222-333-1001\"\r\n");
        assert_eq!(sid.as_deref(), Some("S-1-5-21-111-222-333-1001"));
        assert_eq!(acl::sid_da_whoami("arbitrary error"), None);
        assert_eq!(acl::sid_da_whoami("\"a\",\"non-un-sid\""), None);
        let a = acl::argomenti_icacls(Path::new("C:\\dati\\key.pem"), "S-1-5-21-1-2-3-1001");
        let a: Vec<String> = a.iter().map(|x| x.to_string_lossy().into_owned()).collect();
        assert_eq!(a, ["C:\\dati\\key.pem", "/inheritance:r", "/grant:r", "*S-1-5-21-1-2-3-1001:F"]);
    }

    #[test]
    fn pow_matches_go_vector() {
        // Keep this proof-of-work vector in sync with the enrollment verifier.
        assert_eq!(solve_pow(&[1u8; 32], b"csr", 12), 248);
    }

    #[test]
    fn csr_is_valid_der() {
        let k = PendingKey::generate().unwrap();
        assert!(k.csr_der.len() > 50 && k.csr_der[0] == 0x30);
    }
}
