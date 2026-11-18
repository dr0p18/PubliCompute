//! Count discarded traffic and rejected link epochs for the head's summary.
//! These counters distinguish late messages from faults on the active chain.

use std::collections::BTreeMap;

/// Ignore a chain-wide confirmation whose epoch belongs to a superseded
/// control link. Zero leaves the epoch unspecified, so treat it as current.
pub fn conferma_catena_vecchia(epoca_conferma: u32, epoca_corrente: u64) -> bool {
    epoca_conferma != 0 && u64::from(epoca_conferma) < epoca_corrente
}

#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct Protezioni {
    /// `Token` for a conversation unknown to the head, by source stage (0 = unspecified).
    token_sconosciuti: BTreeMap<u32, u64>,
    /// Chain-wide `Confirmation` messages from a superseded control link, by source stage.
    conferme_vecchie: BTreeMap<u32, u64>,
}

impl Protezioni {
    pub fn token_sconosciuto(&mut self, stage: u32) {
        *self.token_sconosciuti.entry(stage).or_default() += 1;
    }

    pub fn conferma_vecchia(&mut self, stage: u32) {
        *self.conferme_vecchie.entry(stage).or_default() += 1;
    }

    pub fn token_sconosciuti_totale(&self) -> u64 {
        self.token_sconosciuti.values().sum()
    }

    pub fn conferme_vecchie_totale(&self) -> u64 {
        self.conferme_vecchie.values().sum()
    }

    pub fn json(&self) -> serde_json::Value {
        let per = |m: &BTreeMap<u32, u64>| -> serde_json::Value {
            m.iter().map(|(k, v)| (format!("stage_{k}"), serde_json::json!(v))).collect::<serde_json::Map<_, _>>().into()
        };
        serde_json::json!({
            "tokens_dropped_unknown_conversation": self.token_sconosciuti_totale(),
            "tokens_dropped_by_stage": per(&self.token_sconosciuti),
            "stale_chain_confirmations": self.conferme_vecchie_totale(),
            "stale_confirmations_by_stage": per(&self.conferme_vecchie),
        })
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn stale_epoch_must_be_explicit_and_lower() {
        assert!(conferma_catena_vecchia(1, 2));
        assert!(!conferma_catena_vecchia(2, 2));
        assert!(!conferma_catena_vecchia(3, 2));
        assert!(!conferma_catena_vecchia(0, 5), "missing epoch; assuming the current epoch");
    }

    #[test]
    fn counters_are_reported_by_stage_in_json() {
        let mut p = Protezioni::default();
        p.token_sconosciuto(3);
        p.token_sconosciuto(3);
        p.token_sconosciuto(0);
        p.conferma_vecchia(2);
        assert_eq!(p.token_sconosciuti_totale(), 3);
        let j = p.json();
        assert_eq!(j["tokens_dropped_unknown_conversation"], 3);
        assert_eq!(j["tokens_dropped_by_stage"]["stage_3"], 2);
        assert_eq!(j["stale_chain_confirmations"], 1);
        assert_eq!(j["stale_confirmations_by_stage"]["stage_2"], 1);
    }
}
