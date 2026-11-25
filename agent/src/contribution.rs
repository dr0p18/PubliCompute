//! Contribution preferences persist independently of the coordinator connection.
//! Startup flags supply defaults when the data directory has no preference file.
//! Each edit advances the revision. A newly created file uses the clock so the
//! coordinator can distinguish it from a previously observed preference. The
//! session publishes changes and retains the coordinator's status and summary
//! for the local client.

use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result};
use publicompute_proto::{
    common::ModelRef,
    node::{ContributionMode, ContributionPreference, ContributionStatus, ContributionSummary},
};
use serde::{Deserialize, Serialize};
use tokio::sync::watch;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PreferenceMode {
    /// Reserve the card for use with the selected weights.
    #[serde(rename = "uso")]
    Usage,
    /// Let the coordinator schedule the card wherever it is needed.
    #[serde(rename = "libera")]
    Free,
}

impl PreferenceMode {
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "usage" => Ok(PreferenceMode::Usage),
            "free" => Ok(PreferenceMode::Free),
            altro => anyhow::bail!("invalid mode \"{altro}\": use 'usage' or 'free'"),
        }
    }

    pub fn to_proto(self) -> ContributionMode {
        match self {
            PreferenceMode::Usage => ContributionMode::Use,
            PreferenceMode::Free => ContributionMode::Free,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelSelection {
    pub id: String,
    pub quant: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Preference {
    pub modo: PreferenceMode,
    #[serde(rename = "modello")]
    pub model: Option<ModelSelection>,
    #[serde(rename = "revisione")]
    pub revision: u64,
}

fn unix_millis() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

#[allow(dead_code)]
pub struct Contribution {
    file: PathBuf,
    tx: watch::Sender<Preference>,
    state: Mutex<Option<ContributionStatus>>,
    summary: Mutex<Option<ContributionSummary>>,
}

#[allow(dead_code)]
impl Contribution {
    /// Load the stored preference. If the file is missing, persist the supplied
    /// default with a clock-based revision. Return an error if an existing
    /// preference cannot be read or decoded.
    pub fn open(data_dir: &Path, initial: Preference) -> Result<Arc<Self>> {
        let file = data_dir.join("contributo.json");
        let pref = match std::fs::read(&file) {
            Ok(b) => serde_json::from_slice::<Preference>(&b).with_context(|| format!("cannot read {}", file.display()))?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                let p = Preference { revision: unix_millis(), ..initial };
                write(&file, &p)?;
                p
            }
            Err(e) => return Err(e).with_context(|| format!("read {}", file.display())),
        };
        Ok(Arc::new(Self { file, tx: watch::channel(pref).0, state: Mutex::new(None), summary: Mutex::new(None) }))
    }

    pub fn preference(&self) -> Preference {
        self.tx.borrow().clone()
    }

    /// Carry a saved USE preference over to the weights this process serves.
    pub fn align_model(&self, selected: ModelSelection) -> Result<()> {
        let saved = self.preference();
        if saved.modo == PreferenceMode::Usage && saved.model.as_ref() != Some(&selected) {
            let preference = self.imposta(PreferenceMode::Usage, Some(selected.clone()))?;
            tracing::info!(model = %selected.id, quantization = %selected.quant, revision = preference.revision,
                "updated the saved contribution model");
        }
        Ok(())
    }

    /// Subscribe to preference changes so the session can publish them.
    pub fn subscribe(&self) -> watch::Receiver<Preference> {
        self.tx.subscribe()
    }

    /// Persist the preference before notifying the session that its revision advanced.
    pub fn imposta(&self, modo: PreferenceMode, model: Option<ModelSelection>) -> Result<Preference> {
        let model = if modo == PreferenceMode::Free { None } else { model };
        let previous = self.tx.borrow().revision;
        let revision = previous.checked_add(1).context("contribution revision exhausted")?.max(unix_millis());
        let p = Preference { modo, model, revision };
        write(&self.file, &p)?;
        self.tx.send_replace(p.clone());
        Ok(p)
    }

    /// Encode the preference using the fingerprint from the weights inventory.
    /// An empty fingerprint lets the coordinator reject a selection the agent cannot serve.
    pub fn to_proto(&self, inventory: &[ModelRef]) -> ContributionPreference {
        preference_to_proto(&self.preference(), inventory)
    }

    /// Discard late replies for superseded preferences so they cannot replace
    /// the status for the current revision.
    pub fn receive_status(&self, s: ContributionStatus) -> bool {
        if s.revision < self.tx.borrow().revision {
            return false;
        }
        *self.state.lock().unwrap() = Some(s);
        true
    }

    pub fn receive_summary(&self, r: ContributionSummary) {
        *self.summary.lock().unwrap() = Some(r);
    }

    pub fn state(&self) -> Option<ContributionStatus> {
        self.state.lock().unwrap().clone()
    }

    pub fn summary(&self) -> Option<ContributionSummary> {
        self.summary.lock().unwrap().clone()
    }
}

pub fn preference_to_proto(p: &Preference, inventory: &[ModelRef]) -> ContributionPreference {
    let model = match (p.modo, &p.model) {
        (PreferenceMode::Usage, Some(m)) => Some(
            inventory
                .iter()
                .find(|r| r.model_id == m.id && r.quantization == m.quant)
                .cloned()
                .unwrap_or(ModelRef { model_id: m.id.clone(), quantization: m.quant.clone(), weights_sha256: vec![] }),
        ),
        _ => None,
    };
    ContributionPreference { mode: p.modo.to_proto() as i32, model, revision: p.revision }
}

/// Rename the completed temporary file so readers always see a complete preference.
fn write(file: &Path, p: &Preference) -> Result<()> {
    let tmp = file.with_extension("json.tmp");
    let dati = serde_json::to_vec_pretty(p)?;
    {
        use std::io::Write;
        let mut opts = std::fs::OpenOptions::new();
        opts.write(true).create(true).truncate(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            opts.mode(0o600);
        }
        let mut f = opts.open(&tmp).with_context(|| format!("write {}", tmp.display()))?;
        f.write_all(&dati)?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, file).with_context(|| format!("rename in {}", file.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("publicompute-contributo-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn usage(id: &str) -> Preference {
        Preference { modo: PreferenceMode::Usage, model: Some(ModelSelection { id: id.into(), quant: "Q4_K_M".into() }), revision: 0 }
    }

    #[test]
    fn initial_preferences_are_persisted() {
        let d = temp_dir("primo");
        let c = Contribution::open(&d, usage("qwen")).unwrap();
        assert_eq!(c.preference().modo, PreferenceMode::Usage);
        assert!(c.preference().revision > 0);
        let stored: serde_json::Value = serde_json::from_slice(&std::fs::read(d.join("contributo.json")).unwrap()).unwrap();
        assert_eq!(stored["modo"], "uso");
        c.imposta(PreferenceMode::Free, None).unwrap();
        // Startup defaults apply only when the preference file is missing.
        let c2 = Contribution::open(&d, usage("altro")).unwrap();
        assert_eq!(c2.preference().modo, PreferenceMode::Free);
        assert_eq!(c2.preference().revision, c.preference().revision);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn revision_advances_across_new_preference_files() {
        let d = temp_dir("rev");
        let c = Contribution::open(&d, usage("qwen")).unwrap();
        let mut ultima = c.preference().revision;
        for _ in 0..3 {
            let r = c.imposta(PreferenceMode::Usage, c.preference().model).unwrap().revision;
            assert!(r > ultima, "the revision should increase on every change");
            ultima = r;
        }
        // Recreating the directory must still advance the revision visible to the coordinator.
        std::thread::sleep(std::time::Duration::from_millis(10));
        std::fs::remove_file(d.join("contributo.json")).unwrap();
        let new = Contribution::open(&d, usage("qwen")).unwrap();
        assert!(new.preference().revision > ultima, "{} does not exceed {ultima}", new.preference().revision);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn aligns_saved_use_and_preserves_free() {
        let d = temp_dir("align");
        let c = Contribution::open(&d, usage("qwen-old")).unwrap();
        let old = c.preference();
        let medium = ModelSelection { id: "qwen-medium".into(), quant: "Q4_K_M".into() };
        c.align_model(medium.clone()).unwrap();
        let aligned = c.preference();
        assert_eq!(aligned.model.as_ref(), Some(&medium));
        assert!(aligned.revision > old.revision);

        let chosen = ModelSelection { quant: "Q8_0".into(), ..medium };
        c.align_model(chosen.clone()).unwrap();
        let reopened = Contribution::open(&d, usage("qwen-old")).unwrap();
        assert_eq!(reopened.preference().model.as_ref(), Some(&chosen));
        assert!(reopened.preference().revision > aligned.revision);
        let mut saved = reopened.preference();
        reopened.align_model(chosen.clone()).unwrap();
        assert_eq!(reopened.preference(), saved);

        saved = reopened.imposta(PreferenceMode::Free, None).unwrap();
        reopened.align_model(chosen).unwrap();
        assert_eq!(reopened.preference(), saved);
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn proto_uses_inventory_fingerprint() {
        let d = temp_dir("proto");
        let c = Contribution::open(&d, usage("qwen")).unwrap();
        let inv = [ModelRef { model_id: "qwen".into(), quantization: "Q4_K_M".into(), weights_sha256: vec![7; 32] }];
        let p = c.to_proto(&inv);
        assert_eq!(p.mode(), ContributionMode::Use);
        assert_eq!(p.model.unwrap().weights_sha256, vec![7; 32]);
        assert_eq!(p.revision, c.preference().revision);
        c.imposta(PreferenceMode::Free, None).unwrap();
        let p = c.to_proto(&inv);
        assert_eq!(p.mode(), ContributionMode::Free);
        assert!(p.model.is_none());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn keeps_status_for_current_revision() {
        let d = temp_dir("stato");
        let c = Contribution::open(&d, usage("qwen")).unwrap();
        let rev = c.preference().revision;
        assert!(c.state().is_none());
        let s = ContributionStatus { free_seat: true, revision: rev, in_chain: true, ..Default::default() };
        assert!(c.receive_status(s.clone()));
        assert_eq!(c.state(), Some(s));
        assert!(!c.receive_status(ContributionStatus { revision: rev - 1, free_seat: false, ..Default::default() }));
        assert!(c.state().unwrap().free_seat);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn keeps_latest_summary() {
        let d = temp_dir("riepilogo");
        let c = Contribution::open(&d, usage("qwen")).unwrap();
        assert!(c.summary().is_none());
        let r = ContributionSummary { days: 30, balance_mc: 12_345, free_chats: 4, ..Default::default() };
        c.receive_summary(r.clone());
        assert_eq!(c.summary(), Some(r));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn parses_command_line_mode() {
        assert_eq!(PreferenceMode::parse("usage").unwrap(), PreferenceMode::Usage);
        assert_eq!(PreferenceMode::parse("free").unwrap(), PreferenceMode::Free);
        assert!(PreferenceMode::parse("boh").is_err());
    }
}
