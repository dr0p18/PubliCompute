//! Expert routing traces use one JSONL record per conversation, step, and
//! layer. The step identifies the row whose input is the token at step - 1.
//! Sorting experts makes the comparison independent of their order in a
//! router tensor. Compare by key across stages and forced replay.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct RecEsperti {
    pub c: u64,
    pub s: u32,
    pub l: i32,
    pub e: Vec<i32>,
}

/// Map router output positions back to input rows. With separate sequences,
/// the engine forms a sub-batch from consecutive unused sequence IDs
/// and emits one router tensor per sub-batch.
pub fn ordine_sotto_lotti(seq: &[i32]) -> Vec<usize> {
    let mut usata = vec![false; seq.len()];
    let mut ordine = Vec::with_capacity(seq.len());
    while ordine.len() < seq.len() {
        let mut last: Option<i32> = None;
        for i in 0..seq.len() {
            if usata[i] {
                continue;
            }
            if match last {
                None => true,
                Some(u) => seq[i] == u + 1,
            } {
                usata[i] = true;
                ordine.push(i);
                last = Some(seq[i]);
            }
        }
    }
    ordine
}

pub fn write(path: &str, recs: &[RecEsperti]) -> Result<(), String> {
    let mut s = String::new();
    for r in recs {
        s.push_str(&serde_json::to_string(r).map_err(|e| e.to_string())?);
        s.push('\n');
    }
    std::fs::write(path, s).map_err(|e| format!("{path}: {e}"))
}

/// Read one or more comma-separated files and merge their rows.
pub fn read(elenco: &str) -> Result<Vec<RecEsperti>, String> {
    let mut v = Vec::new();
    for path in elenco.split(',').filter(|p| !p.is_empty()) {
        let s = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        for (i, row) in s.lines().enumerate().filter(|(_, r)| !r.trim().is_empty()) {
            v.push(serde_json::from_str(row).map_err(|e| format!("{path}, line {}: {e}", i + 1))?);
        }
    }
    Ok(v)
}

#[derive(Debug, Default, PartialEq)]
pub struct ConfrontoEsperti {
    /// Keys (c, s, l) present in both.
    pub confrontati: usize,
    /// Cases where the expert sets differ.
    pub diversi: usize,
    /// Total expert swaps (k minus the shared experts, summed across cases).
    pub esperti_scambiati: usize,
    pub esperti_totali: usize,
    pub solo_in_a: usize,
    pub solo_in_b: usize,
    /// layer -> (cases, mismatches)
    pub per_layer: BTreeMap<i32, (usize, usize)>,
}

pub fn compare(a: &[RecEsperti], b: &[RecEsperti]) -> ConfrontoEsperti {
    let mb: BTreeMap<(u64, u32, i32), &RecEsperti> = b.iter().map(|r| ((r.c, r.s, r.l), r)).collect();
    let mut c = ConfrontoEsperti::default();
    let mut viste = 0usize;
    for ra in a {
        let Some(rb) = mb.get(&(ra.c, ra.s, ra.l)) else {
            c.solo_in_a += 1;
            continue;
        };
        viste += 1;
        let (mut x, mut y) = (ra.e.clone(), rb.e.clone());
        x.sort_unstable();
        y.sort_unstable();
        let comuni = x.iter().filter(|e| y.binary_search(e).is_ok()).count();
        c.confrontati += 1;
        c.esperti_totali += x.len().max(y.len());
        let scambiati = x.len().max(y.len()) - comuni;
        c.esperti_scambiati += scambiati;
        let l = c.per_layer.entry(ra.l).or_default();
        l.0 += 1;
        if scambiati > 0 {
            c.diversi += 1;
            l.1 += 1;
        }
    }
    c.solo_in_b = mb.len().saturating_sub(viste);
    c
}

#[cfg(test)]
mod test {
    use super::*;

    fn r(c: u64, s: u32, l: i32, e: &[i32]) -> RecEsperti {
        RecEsperti { c, s, l, e: e.to_vec() }
    }

    #[test]
    fn sub_batch_order() {
        // Consecutive sequences: one sub-batch in batch order.
        assert_eq!(ordine_sotto_lotti(&[0, 1, 2, 3]), vec![0, 1, 2, 3]);
        // Gaps: [0], [2,3], [5]... Each sub-batch takes consecutive unused IDs.
        assert_eq!(ordine_sotto_lotti(&[0, 2, 3, 5]), vec![0, 1, 2, 3]);
        // Out of order: 2 starts the sub-batch, 3 continues it, and 0 and 1 come later.
        assert_eq!(ordine_sotto_lotti(&[2, 0, 3, 1]), vec![0, 2, 1, 3]);
        assert_eq!(ordine_sotto_lotti(&[]), Vec::<usize>::new());
    }

    #[test]
    fn comparison_counts_pairs_and_expert_changes() {
        let a = vec![r(1, 1, 0, &[3, 5, 1]), r(1, 1, 1, &[2, 4, 6]), r(1, 2, 0, &[1, 2, 3]), r(9, 1, 0, &[0])];
        let b = vec![r(1, 1, 0, &[1, 3, 5]), r(1, 1, 1, &[2, 4, 7]), r(1, 2, 0, &[7, 8, 9]), r(5, 1, 0, &[0])];
        let c = compare(&a, &b);
        assert_eq!(c.confrontati, 3);
        assert_eq!(c.diversi, 2);
        assert_eq!(c.esperti_scambiati, 1 + 3);
        assert_eq!(c.esperti_totali, 9);
        assert_eq!((c.solo_in_a, c.solo_in_b), (1, 1));
        assert_eq!(c.per_layer[&0], (2, 1));
        assert_eq!(c.per_layer[&1], (1, 1));
    }

    #[test]
    fn jsonl_round_trip_with_two_files() {
        let d = std::env::temp_dir();
        let (p1, p2) = (d.join(format!("esp1-{}.jsonl", std::process::id())), d.join(format!("esp2-{}.jsonl", std::process::id())));
        write(p1.to_str().unwrap(), &[r(1, 1, 0, &[1, 2])]).unwrap();
        write(p2.to_str().unwrap(), &[r(1, 1, 5, &[3, 4])]).unwrap();
        let v = read(&format!("{},{}", p1.display(), p2.display())).unwrap();
        std::fs::remove_file(p1).ok();
        std::fs::remove_file(p2).ok();
        assert_eq!(v, vec![r(1, 1, 0, &[1, 2]), r(1, 1, 5, &[3, 4])]);
    }
}
