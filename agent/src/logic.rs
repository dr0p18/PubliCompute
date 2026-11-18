//! Encode a result's per-token log probabilities for the coordinator to verify.
//!
//! The binary payload starts with the position count. Each position is prefixed
//! with its encoded byte length and contains the chosen token's index, the
//! candidate count, the token IDs and their scores. Counts, lengths, indices
//! and IDs are unsigned variable-length integers. Scores are signed 16-bit
//! little-endian integers in thousandths of a nat. The first score is absolute.
//! Each following score is stored as the previous score minus the current one.
//!
//! Keep each position even if its candidate list is empty. If the chosen token
//! is absent from the list, append it and point the index at that entry. Scores
//! below the representable floor saturate. A deterministic gzip header keeps
//! equal commitments byte-identical on the wire.

use std::io::Write;

use anyhow::{bail, Result};
use flate2::{write::GzEncoder, Compression, GzBuilder};

use crate::engine::TokenProbs;

pub const PROVISIONAL_FORMAT: u32 = 1;
pub const LOGIC_FORMAT: u32 = 2;

/// Floor for quantized scores. This range keeps each difference within a signed 16-bit integer.
pub const SATURATED: f64 = -32.767;

pub fn is_supported(version: u32) -> bool {
    version == PROVISIONAL_FORMAT || version == LOGIC_FORMAT
}

/// Align each chosen token ID with the log probabilities at the same index.
pub fn encode(format_version: u32, top_k: u32, probs: &[TokenProbs], token_ids: &[u32]) -> Result<Vec<u8>> {
    match format_version {
        PROVISIONAL_FORMAT => Ok(encode_v1(top_k, probs)),
        LOGIC_FORMAT => {
            if probs.len() != token_ids.len() {
                bail!("{} log probability positions for {} tokens", probs.len(), token_ids.len());
            }
            let positions: Vec<Position> = probs
                .iter()
                .zip(token_ids)
                .map(|(p, id)| Position {
                    chosen: (*id, f64::from(p.chosen_logprob)),
                    top: p.top.iter().take(top_k as usize).map(|(t, lp)| (*t, f64::from(*lp))).collect(),
                })
                .collect();
            gzip(&encode_v2_binary(&positions)?)
        }
        v => bail!("this agent does not support LOGIC format {v}"),
    }
}

fn encode_v1(top_k: u32, probs: &[TokenProbs]) -> Vec<u8> {
    let k = top_k.min(u8::MAX as u32) as usize;
    let mut out = Vec::with_capacity(probs.len() * (5 + 8 * k));
    for p in probs {
        out.extend_from_slice(&p.chosen_logprob.to_le_bytes());
        let top = &p.top[..p.top.len().min(k)];
        out.push(top.len() as u8);
        for (id, lp) in top {
            out.extend_from_slice(&id.to_le_bytes());
            out.extend_from_slice(&lp.to_le_bytes());
        }
    }
    out
}

/// One position with its chosen token and candidates in engine order.
#[derive(Debug, Clone)]
pub struct Position {
    pub chosen: (u32, f64),
    pub top: Vec<(u32, f64)>,
}

/// Return the binary payload before compression for wire-format checks.
pub fn encode_v2_binary(positions: &[Position]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    put_uvarint(&mut out, positions.len() as u64);
    for p in positions {
        let mut entries: Vec<(u32, f64)> = p.top.iter().map(|(id, lp)| (*id, saturate(*lp))).collect();
        let primary = match entries.iter().position(|(id, _)| *id == p.chosen.0) {
            Some(i) => i,
            None => {
                entries.push((p.chosen.0, saturate(p.chosen.1)));
                entries.len() - 1
            }
        };
        let chunk = pack_chunk(&entries, primary)?;
        put_uvarint(&mut out, chunk.len() as u64);
        out.extend_from_slice(&chunk);
    }
    Ok(out)
}

fn saturate(lp: f64) -> f64 {
    if lp.is_nan() {
        SATURATED
    } else {
        lp.max(SATURATED)
    }
}

/// Encode the candidate IDs and their quantized scores for one position.
fn pack_chunk(entries: &[(u32, f64)], primary: usize) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    put_uvarint(&mut out, primary as u64);
    put_uvarint(&mut out, entries.len() as u64);
    for (id, _) in entries {
        put_uvarint(&mut out, u64::from(*id));
    }
    let rounded: Vec<f64> = entries.iter().map(|(_, lp)| round12(*lp)).collect();
    for (i, r) in rounded.iter().enumerate() {
        let delta = if i == 0 { *r } else { round12(rounded[i - 1] - r) };
        // Round exact half-integer deltas to even so both ends quantize the same way.
        let millilog = (delta * 1000.0).round_ties_even();
        if !(-32768.0..=32767.0).contains(&millilog) {
            bail!("difference exceeds int16 range after clamping: {delta}");
        }
        out.extend_from_slice(&(millilog as i16).to_le_bytes());
    }
    Ok(out)
}

/// Round to 12 decimal places, with ties away from zero.
fn round12(value: f64) -> f64 {
    let scale = 1e12;
    let scaled = value * scale;
    let rounded = if scaled >= 0.0 { (scaled + 0.5).floor() } else { -(-scaled + 0.5).floor() };
    rounded / scale
}

fn put_uvarint(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let byte = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

/// Fix the gzip timestamp at zero so equal commitments produce equal bytes.
fn gzip(data: &[u8]) -> Result<Vec<u8>> {
    let mut enc: GzEncoder<Vec<u8>> = GzBuilder::new().mtime(0).write(Vec::new(), Compression::default());
    enc.write_all(data)?;
    Ok(enc.finish()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_v1() {
        let p = vec![TokenProbs { chosen_logprob: -0.5, top: vec![(7, -0.5), (9, -1.0), (3, -2.0)] }];
        let b = encode(1, 2, &p, &[7]).unwrap();
        assert_eq!(b.len(), 4 + 1 + 2 * 8);
        assert_eq!(&b[0..4], &(-0.5f32).to_le_bytes());
        assert_eq!(b[4], 2);
        assert_eq!(&b[5..9], &7u32.to_le_bytes());
        assert!(encode(3, 2, &p, &[7]).is_err());
    }

    #[test]
    fn v2_is_gzip_and_deterministic() {
        let p = vec![TokenProbs { chosen_logprob: -0.5, top: vec![(7, -0.5), (9, -1.0)] }];
        let a = encode(2, 10, &p, &[7]).unwrap();
        assert_eq!(&a[..2], &[0x1f, 0x8b]);
        assert_eq!(a, encode(2, 10, &p, &[7]).unwrap());
    }
}

/// Lock down the wire representation with shared compatibility vectors.
#[cfg(test)]
mod logic_compat {
    use super::*;
    use serde_json::Value;

    const VECTORS: &str = include_str!("../../proto/testdata/logic/vectors.json");
    const OURS_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../proto/testdata/logic/ours.json");

    /// Exercise scores and candidate lists at the limits of the wire format.
    const EXTENSIONS: &[&str] = &["scelto_fuori_topk", "sotto_limite_int16", "posizione_vuota"];

    fn positions(case: &Value) -> Vec<Position> {
        case["input"]["content"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| Position {
                chosen: (c["token_id"].as_u64().unwrap() as u32, c["logprob"].as_f64().unwrap()),
                top: c["top_logprobs"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|t| (t["token_id"].as_u64().unwrap() as u32, t["logprob"].as_f64().unwrap()))
                    .collect(),
            })
            .collect()
    }

    #[test]
    fn identical_bytes_to_logic() {
        let cases: Vec<Value> = serde_json::from_str(VECTORS).unwrap();
        let mut checked = 0;
        for c in &cases {
            let name = c["name"].as_str().unwrap();
            if EXTENSIONS.contains(&name) {
                continue;
            }
            let ours = hex::encode(encode_v2_binary(&positions(c)).unwrap());
            assert_eq!(ours, c["binary_hex"].as_str().unwrap(), "case {name}: bytes differ from LOGIC");
            checked += 1;
        }
        assert_eq!(checked, 3, "expected compatible cases: normale, arrotondamento_al_pari, id_grandi");
    }

    /// Use UPDATE_LOGIC_VECTORS=1 to refresh the vectors shared with the decoder tests.
    #[test]
    fn extension_cases() {
        let cases: Vec<Value> = serde_json::from_str(VECTORS).unwrap();
        let mut ours = Vec::new();
        for c in &cases {
            let name = c["name"].as_str().unwrap();
            let bin = encode_v2_binary(&positions(c)).unwrap_or_else(|e| panic!("case {name}: {e}"));
            let gz = gzip(&bin).unwrap();
            ours.push(serde_json::json!({"name": name, "binary_hex": hex::encode(&bin), "gzip_hex": hex::encode(&gz)}));
        }
        let text = serde_json::to_string_pretty(&ours).unwrap() + "\n";
        if std::env::var("UPDATE_LOGIC_VECTORS").as_deref() == Ok("1") {
            std::fs::write(OURS_PATH, &text).unwrap();
        }
        let saved = std::fs::read_to_string(OURS_PATH).expect("ours.json missing; run with UPDATE_LOGIC_VECTORS=1 to regenerate");
        assert_eq!(saved, text, "ours.json differs from encoder output; regenerate it");
    }
}
