//! The head retains one token prefix and recurrent checkpoint per parked slot.

use crate::slot::SlotKey;
use publicompute_proto::stage::{AckKind, Confirmation};
use std::cmp::Reverse;
use std::collections::{HashMap, VecDeque};
use std::time::Duration;
use zeroize::Zeroizing;

pub fn sequence(group: u32, local: u32) -> u32 {
    (group << 16) | local
}

pub struct Parked {
    pub key: SlotKey,
    pub owner: Zeroizing<Vec<u8>>,
    pub member: bool,
    pub member_key: Zeroizing<Vec<u8>>,
    pub tokens: Zeroizing<Vec<i32>>,
    pub checkpoint: Zeroizing<Vec<u8>>,
    pub since: Duration,
}

impl Parked {
    pub fn matched_position(&self, owner: &[u8], prompt: &[i32], save_at: Option<u32>) -> Option<u32> {
        let p = self.tokens.len() as u32;
        // Matching these actual token IDs proves the common prefix covers P.
        // Keep one token for logits and allow a checkpoint at or after P.
        (!owner.is_empty()
            && self.owner.as_slice() == owner
            && p > 0
            && self.tokens.len() < prompt.len()
            && prompt.starts_with(&self.tokens)
            && save_at.is_none_or(|next| next >= p))
        .then_some(p)
    }
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
enum Rank {
    Reuse(Reverse<u32>),
    Virgin(usize),
    Public(Duration),
    Member(Duration),
}

pub struct AdmissionRequest<'a> {
    pub owner: &'a [u8],
    pub member: bool,
    pub prompt: &'a [i32],
    pub save_at: Option<u32>,
}

pub fn choose(
    parked: &HashMap<u32, Parked>,
    free: &[VecDeque<u32>],
    occupancy: &[usize],
    request: AdmissionRequest<'_>,
    now: Duration,
    grace: Duration,
) -> Option<(usize, u32)> {
    free.iter()
        .enumerate()
        .flat_map(|(group, slots)| {
            slots.iter().filter_map(move |&local| {
                let rank = match parked.get(&sequence(group as u32, local)) {
                    None => Rank::Virgin(occupancy[group]),
                    Some(p) => match p.matched_position(request.owner, request.prompt, request.save_at) {
                        Some(position) => Rank::Reuse(Reverse(position)),
                        None if !p.member => Rank::Public(p.since),
                        None if request.member || p.member_key.is_empty() || now.saturating_sub(p.since) >= grace => Rank::Member(p.since),
                        None => return None,
                    },
                };
                Some((rank, group, local))
            })
        })
        .min_by(|a, b| a.0.cmp(&b.0))
        .map(|(_, group, local)| (group, local))
}

pub fn take(
    parked: &mut HashMap<u32, Parked>,
    slot: u32,
    member_key: &[u8],
    request: AdmissionRequest<'_>,
) -> Option<Parked> {
    // Taking a new slot ends the member's old grace while leaving its reusable state intact.
    if !member_key.is_empty() {
        for retained in parked.values_mut().filter(|p| p.member_key.as_slice() == member_key) {
            retained.member_key = Zeroizing::default();
        }
    }
    parked.remove(&slot).filter(|p| {
        p.matched_position(request.owner, request.prompt, request.save_at).is_some()
    })
}

pub fn block_len(offset: usize, total: usize, limit: usize, checkpoint: Option<u32>) -> usize {
    let end = checkpoint.map(|p| p as usize).filter(|&p| p > offset).unwrap_or(total);
    (end - offset).min(limit)
}

#[derive(Clone, Copy)]
pub struct OpenReply {
    pub key: SlotKey,
    pub epoch: u32,
    pub chain_epoch: u64,
    pub position: u32,
}

impl OpenReply {
    pub fn matches(self, c: &Confirmation) -> bool {
        self.epoch == c.epoch
            && self.chain_epoch == c.chain_epoch
            && self.key.seq == c.slot
            && self.key.generation == c.slot_generation
            && self.key.conversation == c.conversation_id
            && self.key.attempt == c.attempt_id
            && self.position == c.start_position
    }

    pub fn status(self, c: &Confirmation) -> Option<AckKind> {
        if !self.matches(c) {
            return None;
        }
        match c.kind() {
            AckKind::Open if c.ok => Some(AckKind::Open),
            AckKind::CacheMiss if !c.ok => Some(AckKind::CacheMiss),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::admission::Admission;

    fn parked(seq: u32, owner: u8, member: bool, since: u64, tokens: Vec<i32>) -> Parked {
        Parked {
            key: SlotKey {
                seq,
                generation: 9,
                conversation: 10,
                attempt: 1,
            },
            member,
            member_key: Zeroizing::new(if member { vec![owner] } else { Vec::new() }),
            owner: Zeroizing::new(vec![owner]),
            since: Duration::from_secs(since),
            tokens: Zeroizing::new(tokens),
            checkpoint: Zeroizing::new(vec![1]),
        }
    }

    #[test]
    fn only_the_same_owner_can_reuse_the_longest_verified_boundary() {
        let mut slots = HashMap::new();
        slots.insert(0, parked(0, 1, false, 0, vec![1, 2]));
        slots.insert(1, parked(1, 1, true, 2, vec![1, 2, 3]));
        let free = [VecDeque::from([0, 1, 2])];
        let select = |owner: &[u8], prompt: &[i32], save_at| {
            choose(
                &slots,
                &free,
                &[0],
                AdmissionRequest {
                    owner,
                    member: false,
                    prompt,
                    save_at,
                },
                Duration::ZERO,
                Duration::ZERO,
            )
        };
        assert_eq!(select(&[1], &[1, 2, 3, 4], Some(3)), Some((0, 1)));
        assert_eq!(select(&[1], &[1, 2, 3], None), Some((0, 0)));
        assert_eq!(select(&[1], &[1, 2, 3, 4], Some(2)), Some((0, 0)));
        assert_eq!(select(&[1], &[1, 4, 3], None), Some((0, 2)));
        assert_eq!(select(&[2], &[1, 2, 3, 4], None), Some((0, 2)));
    }

    #[test]
    fn eviction_uses_virgin_then_public_and_member_slots_in_lru_order() {
        let mut slots = HashMap::from([
            (0, parked(0, 1, true, 0, vec![1])),
            (1, parked(1, 2, false, 2, vec![1])),
            (2, parked(2, 3, false, 1, vec![1])),
            (3, parked(3, 4, true, 1, vec![1])),
        ]);
        let request = || AdmissionRequest {
            owner: &[5],
            member: true,
            prompt: &[1, 2],
            save_at: None,
        };
        let mut free = [VecDeque::from([0, 1, 2, 3, 4])];
        for slot in [4, 2, 1, 0, 3] {
            assert_eq!(
                choose(&slots, &free, &[0], request(), Duration::ZERO, Duration::ZERO),
                Some((0, slot))
            );
            free[0].retain(|&s| s != slot);
            slots.remove(&slot);
        }
    }

    #[test]
    fn public_is_rejected_at_59_seconds_and_can_evict_at_60_seconds() {
        let slots = HashMap::from([(0, parked(0, 1, true, 0, vec![1]))]);
        let choose_for = |free: &[VecDeque<u32>], occupancy: &[usize], member, now, grace| {
            let request = AdmissionRequest {
                owner: &[2],
                member,
                prompt: &[1, 2],
                save_at: None,
            };
            choose(
                &slots,
                free,
                occupancy,
                request,
                Duration::from_secs(now),
                Duration::from_secs(grace),
            )
        };
        let mut admission = Admission::new(vec![0, 0], 1, 1, 1);
        admission.arriva(0, false);
        assert_eq!(admission.occupati(), 0);
        assert_eq!(
            admission.prossima_con(|_, f, o| choose_for(f, o, false, 59, 60)),
            Err(0)
        );
        assert_eq!(admission.occupati(), 0);
        assert_eq!(
            admission.prossima_con(|_, f, o| choose_for(f, o, true, 59, 60)),
            Ok(Some((1, 0, 0)))
        );
        let free = [VecDeque::from([0])];
        assert_eq!(choose_for(&free, &[0], false, 60, 60), Some((0, 0)));
        assert_eq!(choose_for(&free, &[0], false, 0, 0), Some((0, 0)));
    }

    #[test]
    fn taking_a_new_slot_ends_only_that_members_grace_before_the_turn_finishes() {
        let mut slots = HashMap::from([
            (0, parked(0, 1, true, 0, vec![1])),
            (2, parked(2, 3, true, 0, vec![2])),
        ]);
        let request = || AdmissionRequest { owner: &[7], member: true, prompt: &[3, 4], save_at: None };
        let public = || AdmissionRequest { owner: &[8], member: false, prompt: &[5, 6], save_at: None };
        let now = Duration::from_secs(1);
        let grace = Duration::from_secs(60);
        assert_eq!(choose(&slots, &[VecDeque::from([0, 1, 2])], &[0], request(), now, grace), Some((0, 1)));
        let busy = [VecDeque::from([0, 2])];
        assert_eq!(choose(&slots, &busy, &[1], public(), now, grace), None);
        assert!(take(&mut slots, 1, &[1], request()).is_none());
        assert_eq!(choose(&slots, &busy, &[1], public(), now, grace), Some((0, 0)));
        assert!(slots[&0].matched_position(&[1], &[1, 2], None).is_some());
        assert!(!slots[&2].member_key.is_empty());
    }

    #[test]
    fn extra_slot_keeps_public_traffic_from_evicting_the_member() {
        let mut slots = HashMap::from([(0, parked(0, 1, true, 0, vec![1]))]);
        let free = [VecDeque::from([0, 1])];
        for request_id in 2..5 {
            let owner = [request_id];
            let request = AdmissionRequest {
                owner: &owner,
                member: false,
                prompt: &[1, 2],
                save_at: None,
            };
            let now = Duration::from_secs(request_id.into());
            assert_eq!(choose(&slots, &free, &[0], request, now, Duration::ZERO), Some((0, 1)));
            slots.insert(1, parked(1, request_id, false, now.as_secs(), vec![1]));
        }
    }

    #[test]
    fn acknowledgement_requires_the_whole_open_identity() {
        let key = parked(1, 1, true, 0, vec![1]).key;
        let open = OpenReply {
            key,
            epoch: 3,
            chain_epoch: 74,
            position: 50,
        };
        let ack = Confirmation {
            kind: AckKind::Open as i32,
            ok: true,
            epoch: 3,
            chain_epoch: 74,
            slot: key.seq,
            slot_generation: key.generation,
            attempt_id: key.attempt,
            conversation_id: key.conversation,
            start_position: 50,
            ..Default::default()
        };
        assert_eq!(open.status(&ack), Some(AckKind::Open));
        let miss = Confirmation {
            kind: AckKind::CacheMiss as i32,
            ok: false,
            ..ack.clone()
        };
        assert_eq!(open.status(&miss), Some(AckKind::CacheMiss));
        for stale in [
            Confirmation {
                epoch: 2,
                ..ack.clone()
            },
            Confirmation {
                chain_epoch: 73,
                ..ack.clone()
            },
            Confirmation { slot: 2, ..ack.clone() },
            Confirmation {
                slot_generation: 8,
                ..ack.clone()
            },
            Confirmation {
                conversation_id: 9,
                ..ack.clone()
            },
            Confirmation {
                attempt_id: 2,
                ..ack.clone()
            },
            Confirmation {
                start_position: 49,
                ..ack
            },
        ] {
            assert_eq!(open.status(&stale), None);
        }
        let replay = OpenReply {
            key: SlotKey {
                generation: 10,
                attempt: 2,
                ..key
            },
            position: 0,
            ..open
        };
        assert_eq!(replay.status(&miss), None);
    }

    #[test]
    fn prefill_splits_at_the_boundary_and_leaves_the_suffix() {
        let p = Some(13);
        assert_eq!(block_len(0, 18, 8, p), 8);
        assert_eq!(block_len(8, 18, 8, p), 5);
        assert_eq!(block_len(13, 18, 8, p), 5);
    }
}
