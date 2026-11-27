//! Admission state shared by the head and the single-process reference.
//! Arrival thresholds count completed generation steps. When there is
//! nothing left to advance that count, the head admits the next threshold
//! as a group so distant arrivals can still run.
//!
//! Each traffic class has a FIFO queue. Interactive requests take priority.
//! The slot reserve can block more batch admissions once work is active.
//! The head can suspend a batch, requeue it at the front, and replay its
//! confirmed prefix later.
//!
//! Choose the least occupied group with a free slot, then the slot that has
//! been free longest. Reusing a just-closed slot last gives stages time to
//! process its close before the next open.

use std::collections::VecDeque;

/// Split `slots` over at most `groups` contexts so the stage allocates
/// exactly `slots` sequences of context memory. The first groups take the
/// remainder; with fewer slots than groups, each group holds one slot.
/// Every stage and the head call this with the same values, so they agree
/// on the slot ids (group << 16 | local).
pub fn slots_per_group(slots: u32, groups: u32) -> Vec<u32> {
    let slots = slots.max(1);
    let g = groups.max(1).min(slots);
    (0..g).map(|i| slots / g + u32::from(i < slots % g)).collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum St {
    Futura,
    InCoda,
    Ammessa { gruppo: usize, locale: u32 },
    Finita,
}

/// Admission priority determines which queue can claim a free slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Class {
    #[default]
    Interactive,
    Batch,
}

/// Parse one class per prompt. Missing entries default to interactive.
pub fn parse_classes(text: &str, n: usize) -> Result<Vec<Class>, String> {
    let mut v = Vec::new();
    for p in text.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        v.push(match p {
            "i" | "interactive" => Class::Interactive,
            "b" | "batch" => Class::Batch,
            _ => return Err(format!("--classes: invalid class \"{p}\" (i/interactive, b/batch)")),
        });
    }
    if v.len() > n {
        return Err(format!("--classes has {} values for {n} conversations", v.len()));
    }
    v.resize(n, Class::Interactive);
    Ok(v)
}

#[derive(Debug)]
pub struct Admission {
    soglie: Vec<u64>,
    st: Vec<St>,
    classi: Vec<Class>,
    /// Suspended batches return to the front of the batch queue.
    tail: VecDeque<usize>,
    coda_b: VecDeque<usize>,
    riserva: usize,
    sospensioni: u32,
    liberi: Vec<VecDeque<u32>>,
    membri: Vec<usize>,
    slots: usize,
    occupati: usize,
}

/// Parse arrival thresholds in prompt order. Omitted thresholds are zero.
/// Reject a list with more thresholds than conversations.
pub fn parse_arrivals(text: &str, n: usize) -> Result<Vec<u64>, String> {
    let mut v = Vec::new();
    for p in text.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        v.push(p.parse::<u64>().map_err(|_| format!("--arrivals: expected a step count, got \"{p}\""))?);
    }
    if v.len() > n {
        return Err(format!("--arrivals has {} values for {n} conversations", v.len()));
    }
    v.resize(n, 0);
    Ok(v)
}

impl Admission {
    /// Create groups with local slots and a separate global limit on live conversations.
    pub fn new(soglie: Vec<u64>, groups: usize, per_gruppo: usize, slots: usize) -> Self {
        let n = soglie.len();
        Self::con_classi(soglie, vec![Class::Interactive; n], groups, per_gruppo, slots, 0)
    }

    /// Create an admission queue with an explicit class per conversation and an interactive reserve.
    pub fn con_classi(soglie: Vec<u64>, classi: Vec<Class>, groups: usize, per_gruppo: usize, slots: usize, riserva: usize) -> Self {
        Self::con_gruppi(soglie, classi, &vec![per_gruppo as u32; groups.max(1)], slots, riserva)
    }

    /// Like `con_classi`, with the local slot count of each group (see `slots_per_group`).
    pub fn con_gruppi(soglie: Vec<u64>, classi: Vec<Class>, per_gruppo: &[u32], slots: usize, riserva: usize) -> Self {
        assert!(!per_gruppo.is_empty(), "expected at least one group");
        let g = per_gruppo.len();
        let n = soglie.len();
        assert_eq!(classi.len(), n, "expected one class per conversation");
        Admission {
            soglie,
            st: vec![St::Futura; n],
            classi,
            coda_b: VecDeque::new(),
            riserva,
            sospensioni: 0,
            tail: VecDeque::new(),
            liberi: per_gruppo.iter().map(|&k| (0..k).collect()).collect(),
            membri: vec![0; g],
            slots,
            occupati: 0,
        }
    }

    pub fn iniziali(&self) -> usize {
        self.soglie.iter().filter(|&&s| s == 0).count()
    }

    pub fn soglia(&self, c: usize) -> u64 {
        self.soglie[c]
    }

    /// Enqueue eligible conversations by threshold and then prompt order. When
    /// idle and no arrival was selected, enqueue the next threshold so the run
    /// can advance. Return the indices that joined the queues.
    pub fn arriva(&mut self, passi: u64, fermo: bool) -> Vec<usize> {
        let mut v: Vec<usize> = (0..self.st.len()).filter(|&c| self.st[c] == St::Futura && self.soglie[c] <= passi).collect();
        if v.is_empty() && fermo && self.tail.is_empty() && self.coda_b.is_empty() {
            if let Some(min) = (0..self.st.len()).filter(|&c| self.st[c] == St::Futura).map(|c| self.soglie[c]).min() {
                v = (0..self.st.len()).filter(|&c| self.st[c] == St::Futura && self.soglie[c] == min).collect();
            }
        }
        v.sort_by_key(|&c| (self.soglie[c], c));
        for &c in &v {
            self.st[c] = St::InCoda;
            match self.classi[c] {
                Class::Interactive => self.tail.push_back(c),
                Class::Batch => self.coda_b.push_back(c),
            }
        }
        v
    }

    /// Return the conversation, group, and local slot assigned to the first
    /// eligible request. Leave the queues intact when admission is blocked.
    pub fn prossima(&mut self) -> Option<(usize, usize, u32)> {
        self.prossima_con(|_, liberi, membri| {
            let g = (0..liberi.len()).filter(|&g| !liberi[g].is_empty()).min_by_key(|&g| (membri[g], g))?;
            Some((g, *liberi[g].front()?))
        }).ok().flatten()
    }

    /// Select among free slots. If free slots exist but the selection
    /// declines all of them, return that request so it can be rejected.
    pub fn prossima_con(&mut self, choose: impl FnOnce(usize, &[VecDeque<u32>], &[usize]) -> Option<(usize, u32)>)
        -> Result<Option<(usize, usize, u32)>, usize>
    {
        let (c, batch) = match (self.tail.front(), self.coda_b.front()) {
            (Some(&c), _) => (c, false),
            (None, Some(&c)) => (c, true),
            (None, None) => return Ok(None),
        };
        if self.occupati >= self.slots || self.liberi.iter().all(VecDeque::is_empty) {
            return Ok(None);
        }
        // Let a batch start on an empty stage even when the reserve would otherwise block it.
        if batch && self.occupati > 0 && self.slots - self.occupati <= self.riserva {
            return Ok(None);
        }
        let Some((g, locale)) = choose(c, &self.liberi, &self.membri) else {
            self.cancel(c);
            return Err(c);
        };
        let pos = self.liberi[g].iter().position(|&s| s == locale).expect("selected slot must be free");
        self.liberi[g].remove(pos);
        if batch {
            self.coda_b.pop_front();
        } else {
            self.tail.pop_front();
        }
        self.st[c] = St::Ammessa { gruppo: g, locale };
        self.membri[g] += 1;
        self.occupati += 1;
        Ok(Some((c, g, locale)))
    }

    /// Return the slot to its group's free list. Repeated calls are harmless.
    pub fn libera(&mut self, c: usize) {
        if let St::Ammessa { gruppo, locale } = self.st[c] {
            self.liberi[gruppo].push_back(locale);
            self.membri[gruppo] -= 1;
            self.occupati -= 1;
            self.st[c] = St::Finita;
        }
    }

    /// Append an immediate arrival and return the index used by the head's
    /// conversation table. The next call to arriva moves it into the queue.
    pub fn aggiungi(&mut self, class: Class) -> usize {
        self.soglie.push(0);
        self.st.push(St::Futura);
        self.classi.push(class);
        self.soglie.len() - 1
    }

    /// Reset service request indices and queues without changing slot layout.
    /// Call only when every slot is free.
    pub fn azzera_conversazioni(&mut self) {
        debug_assert_eq!(self.occupati, 0);
        self.soglie = Vec::new();
        self.st = Vec::new();
        self.classi = Vec::new();
        self.tail.clear();
        self.coda_b.clear();
    }

    /// Cancel an arrival before admission. After a slot has been assigned,
    /// close it through the head and call libera instead.
    pub fn cancel(&mut self, c: usize) -> bool {
        match self.st[c] {
            St::Futura => {
                self.st[c] = St::Finita;
                true
            }
            St::InCoda => {
                self.tail.retain(|&x| x != c);
                self.coda_b.retain(|&x| x != c);
                self.st[c] = St::Finita;
                true
            }
            _ => false,
        }
    }

    pub fn pendenti(&self) -> bool {
        !self.tail.is_empty() || !self.coda_b.is_empty() || self.st.contains(&St::Futura)
    }

    pub fn in_coda(&self) -> usize {
        self.tail.len() + self.coda_b.len()
    }

    pub fn in_coda_interattive(&self) -> usize {
        self.tail.len()
    }

    pub fn in_coda_batch(&self) -> usize {
        self.coda_b.len()
    }

    pub fn class(&self, c: usize) -> Class {
        self.classi[c]
    }

    pub fn sospensioni(&self) -> u32 {
        self.sospensioni
    }

    pub fn senza_posto_per_interattiva(&self) -> bool {
        !self.tail.is_empty() && (self.occupati >= self.slots || self.liberi.iter().all(|l| l.is_empty()))
    }

    /// When interactive work is waiting for a slot, choose the eligible batch
    /// with the lowest replay cost. The larger index wins a tie. A missing cost
    /// excludes a candidate, such as a conversation still being admitted.
    pub fn da_sospendere(&self, costo: &dyn Fn(usize) -> Option<u64>) -> Option<usize> {
        if !self.senza_posto_per_interattiva() {
            return None;
        }
        (0..self.st.len())
            .filter(|&c| self.classi[c] == Class::Batch && matches!(self.st[c], St::Ammessa { .. }))
            .filter_map(|c| costo(c).map(|k| (k, std::cmp::Reverse(c))))
            .min()
            .map(|(_, std::cmp::Reverse(c))| c)
    }

    /// Free an admitted batch's slot and requeue it ahead of waiting batches.
    /// Return false when the conversation cannot be suspended.
    pub fn suspend(&mut self, c: usize) -> bool {
        if self.classi[c] != Class::Batch {
            return false;
        }
        let St::Ammessa { gruppo, locale } = self.st[c] else { return false };
        self.liberi[gruppo].push_back(locale);
        self.membri[gruppo] -= 1;
        self.occupati -= 1;
        self.st[c] = St::InCoda;
        self.coda_b.push_front(c);
        self.sospensioni += 1;
        true
    }

    pub fn ammessa(&self, c: usize) -> Option<(usize, u32)> {
        match self.st[c] {
            St::Ammessa { gruppo, locale } => Some((gruppo, locale)),
            _ => None,
        }
    }

    pub fn membri(&self, g: usize) -> usize {
        self.membri[g]
    }

    pub fn occupati(&self) -> usize {
        self.occupati
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn service_adds_and_cancels_requests() {
        let mut a = Admission::new(Vec::new(), 2, 2, 4);
        assert!(!a.pendenti());
        assert_eq!(a.prossima(), None);
        let c0 = a.aggiungi(Class::Interactive);
        let c1 = a.aggiungi(Class::Batch);
        let c2 = a.aggiungi(Class::Interactive);
        assert_eq!((c0, c1, c2), (0, 1, 2));
        assert!(a.pendenti());
        assert_eq!(a.arriva(0, false), vec![0, 1, 2]);
        assert!(a.cancel(c2));
        assert_eq!(a.prossima().map(|x| x.0), Some(0));
        assert_eq!(a.prossima().map(|x| x.0), Some(1));
        assert_eq!(a.prossima(), None);
        assert!(!a.pendenti());
        assert!(!a.cancel(c0));
        a.libera(c0);
        assert_eq!(a.occupati(), 1);
        let c3 = a.aggiungi(Class::Interactive);
        assert_eq!(c3, 3);
        assert_eq!(a.arriva(7, false), vec![3]);
        assert_eq!(a.prossima().map(|x| x.0), Some(3));
        let c4 = a.aggiungi(Class::Interactive);
        assert!(a.cancel(c4));
        assert!(a.arriva(100, false).is_empty());
        assert!(!a.pendenti());
    }

    #[test]
    fn arrival_parser_fills_defaults_and_rejects_invalid_values() {
        assert_eq!(parse_arrivals("0,0,16", 5).unwrap(), vec![0, 0, 16, 0, 0]);
        assert_eq!(parse_arrivals(" 3 , 4 ", 2).unwrap(), vec![3, 4]);
        assert_eq!(parse_arrivals("", 2).unwrap(), vec![0, 0]);
        assert!(parse_arrivals("1,2,3", 2).is_err());
        assert!(parse_arrivals("1,x", 2).is_err());
        assert!(parse_arrivals("-1", 2).is_err());
    }

    /// Simultaneous arrivals fill groups in round-robin order when capacity suffices.
    #[test]
    fn zero_arrivals_keep_original_distribution() {
        for (n, g) in [(8usize, 2usize), (4, 4), (5, 3), (3, 4), (1, 2)] {
            let g_eff = g.min(n);
            let per = n.div_ceil(g);
            let mut a = Admission::new(vec![0; n], g_eff, per, n);
            assert_eq!(a.arriva(0, false), (0..n).collect::<Vec<_>>());
            for i in 0..n {
                assert_eq!(a.prossima(), Some((i, i % g_eff, (i / g_eff) as u32)), "n={n} g={g} i={i}");
            }
            assert_eq!(a.prossima(), None);
            assert!(!a.pendenti());
        }
    }

    #[test]
    fn admits_in_order_after_threshold() {
        let mut a = Admission::new(vec![0, 20, 20, 40, 5], 2, 4, 8);
        assert_eq!(a.arriva(0, false), vec![0]);
        assert_eq!(a.arriva(4, false), Vec::<usize>::new());
        assert_eq!(a.arriva(19, false), vec![4]);
        assert_eq!(a.arriva(20, false), vec![1, 2]);
        assert_eq!(a.arriva(100, false), vec![3]);
        assert!(a.pendenti());
        let ord: Vec<usize> = std::iter::from_fn(|| a.prossima()).map(|x| x.0).collect();
        assert_eq!(ord, vec![0, 4, 1, 2, 3]);
        assert!(!a.pendenti());
    }

    #[test]
    fn picks_smallest_group_then_lowest_index() {
        let mut a = Admission::new(vec![0; 6], 3, 4, 12);
        a.arriva(0, false);
        let g: Vec<usize> = (0..3).map(|_| a.prossima().unwrap().1).collect();
        assert_eq!(g, vec![0, 1, 2]);
        a.libera(0);
        assert_eq!(a.prossima().unwrap(), (3, 0, 1)); // The untouched slot beats the one just closed.
        a.libera(1);
        assert_eq!(a.prossima().unwrap().1, 1);
        assert_eq!((a.membri(0), a.membri(1), a.membri(2)), (1, 1, 1));
    }

    #[test]
    fn released_slot_goes_to_back_of_queue() {
        let mut a = Admission::new(vec![0; 3], 1, 2, 2);
        a.arriva(0, false);
        assert_eq!(a.prossima(), Some((0, 0, 0)));
        assert_eq!(a.prossima(), Some((1, 0, 1)));
        assert_eq!(a.prossima(), None);
        assert_eq!(a.in_coda(), 1);
        a.libera(0);
        a.libera(0);
        assert_eq!(a.occupati(), 1);
        assert_eq!(a.prossima(), Some((2, 0, 0)));
    }

    #[test]
    fn slot_limit_and_strict_fifo_queue() {
        let mut a = Admission::new(vec![0, 0, 0, 0, 0], 2, 2, 2);
        a.arriva(0, false);
        assert_eq!(a.prossima().map(|x| x.0), Some(0));
        assert_eq!(a.prossima().map(|x| x.0), Some(1));
        assert_eq!(a.prossima(), None);
        a.libera(1);
        assert_eq!(a.prossima().map(|x| x.0), Some(2));
        assert_eq!(a.prossima(), None);
        a.libera(0);
        a.libera(2);
        assert_eq!(a.prossima().map(|x| x.0), Some(3));
        assert_eq!(a.prossima().map(|x| x.0), Some(4));
        assert!(a.occupati() <= 2);
    }

    #[test]
    fn idle_head_advances_to_earliest_arrival() {
        let mut a = Admission::new(vec![0, 50, 50, 90], 2, 2, 4);
        a.arriva(0, false);
        a.prossima();
        assert_eq!(a.arriva(3, true), vec![1, 2]);
        assert_eq!(a.arriva(3, true), Vec::<usize>::new());
        a.prossima();
        a.prossima();
        assert_eq!(a.arriva(3, true), vec![3]);
        let mut b = Admission::new(vec![7], 1, 1, 1);
        assert_eq!(b.arriva(0, false), Vec::<usize>::new());
        assert!(b.pendenti());
    }

    #[test]
    fn class_parser_fills_defaults_and_rejects_invalid_values() {
        assert_eq!(parse_classes("i,b", 4).unwrap(), vec![Class::Interactive, Class::Batch, Class::Interactive, Class::Interactive]);
        assert_eq!(parse_classes("batch, interactive", 2).unwrap(), vec![Class::Batch, Class::Interactive]);
        assert_eq!(parse_classes("", 2).unwrap(), vec![Class::Interactive; 2]);
        assert!(parse_classes("i,b,b", 2).is_err());
        assert!(parse_classes("x", 2).is_err());
    }

    #[test]
    fn interactive_priority_and_reserved_slots() {
        use Class::*;
        let mut a = Admission::con_classi(vec![0; 4], vec![Batch, Batch, Interactive, Interactive], 1, 3, 3, 1);
        a.arriva(0, false);
        assert_eq!((a.in_coda_interattive(), a.in_coda_batch()), (2, 2));
        assert_eq!(a.prossima().map(|x| x.0), Some(2));
        assert_eq!(a.prossima().map(|x| x.0), Some(3));
        assert_eq!(a.prossima(), None);
        a.libera(2);
        assert_eq!(a.prossima().map(|x| x.0), Some(0));
        assert_eq!(a.prossima(), None);
        let mut b = Admission::con_classi(vec![0, 5], vec![Batch, Interactive], 1, 2, 2, 1);
        b.arriva(0, false);
        assert_eq!(b.prossima().map(|x| x.0), Some(0));
        b.arriva(5, false);
        assert_eq!(b.prossima().map(|x| x.0), Some(1));
        let mut c = Admission::con_classi(vec![0], vec![Batch], 1, 1, 0, 0);
        c.arriva(0, false);
        assert!(c.pendenti() && c.prossima().is_none());
    }

    #[test]
    fn slots_split_exactly_over_groups() {
        assert_eq!(slots_per_group(1, 2), vec![1]);
        assert_eq!(slots_per_group(2, 2), vec![1, 1]);
        assert_eq!(slots_per_group(3, 2), vec![2, 1]);
        assert_eq!(slots_per_group(4, 2), vec![2, 2]);
        assert_eq!(slots_per_group(5, 2), vec![3, 2]);
        assert_eq!(slots_per_group(4, 0), vec![4]);
        assert_eq!(slots_per_group(0, 2), vec![1]);
        for s in 1..=17 {
            for g in 1..=5 {
                assert_eq!(slots_per_group(s, g).iter().sum::<u32>(), s);
            }
        }
    }

    #[test]
    fn uneven_groups_never_hand_out_missing_slots() {
        let mut a = Admission::con_gruppi(vec![0; 3], vec![Class::Interactive; 3], &slots_per_group(3, 2), 3, 0);
        a.arriva(0, false);
        let mut posti: Vec<(usize, u32)> = std::iter::from_fn(|| a.prossima()).map(|x| (x.1, x.2)).collect();
        posti.sort();
        assert_eq!(posti, vec![(0, 0), (0, 1), (1, 0)]);
    }

    #[test]
    fn interactive_request_preempts_cheapest_batch_and_requeues_it_first() {
        use Class::*;
        let mut a = Admission::con_classi(vec![0, 0, 0, 10], vec![Batch, Batch, Interactive, Interactive], 2, 2, 3, 0);
        a.arriva(0, false);
        let amm: Vec<usize> = std::iter::from_fn(|| a.prossima()).map(|x| x.0).collect();
        assert_eq!(amm, vec![2, 0, 1]);
        assert!(a.da_sospendere(&|_| Some(1)).is_none());
        a.arriva(10, false);
        assert!(a.senza_posto_per_interattiva());
        assert_eq!(a.da_sospendere(&|c| Some(if c == 1 { 10 } else { 500 })), Some(1));
        assert_eq!(a.da_sospendere(&|_| Some(7)), Some(1));
        assert_eq!(a.da_sospendere(&|c| if c == 1 { None } else { Some(5) }), Some(0));
        assert_eq!(a.da_sospendere(&|_| None), None);
        assert!(!a.suspend(2));
        assert!(a.suspend(1));
        assert_eq!((a.occupati(), a.sospensioni(), a.in_coda_batch()), (2, 1, 1));
        assert!(!a.suspend(1));
        assert_eq!(a.prossima().map(|x| x.0), Some(3));
        assert_eq!(a.prossima(), None);
        a.libera(3);
        assert_eq!(a.prossima().map(|x| x.0), Some(1));
    }

    // A discrete-time model of admission and first-token latency under mixed load.

    use crate::conversation::{Action, Config, Conversation, TokenOutcome, Status};
    use std::time::Duration;

    /// The synthetic step cost has a fixed component and a per-row component.
    /// Prefill time grows with prompt length, and each step includes a network round trip.
    const FISSO_MS: f64 = 8.0;
    const RIGA_MS: f64 = 1.9;
    const RETE_MS: f64 = 60.0;
    const PREFILL_TOK_S: f64 = 230.0;
    const BLOCCO: usize = 64;

    struct Conv {
        class: Class,
        prompt: usize,
        max: usize,
        m: Option<Conversation>,
        tok: Vec<u32>,
        arrivo_ms: f64,
        primo_ms: Option<f64>,
        fine_ms: Option<f64>,
        da_inviare: usize,
        inviati: usize,
    }

    struct SimulationResult {
        ttft_interattive: Vec<f64>,
        token_batch: usize,
        token_interattive: usize,
        durata_ms: f64,
        sospensioni: u32,
        finite: usize,
    }

    /// Each arrival lists time in milliseconds, class, prompt length, and output length. The model uses one group.
    fn simula(arrivi: &[(f64, Class, usize, usize)], slots: usize, riserva: usize) -> SimulationResult {
        let n = arrivi.len();
        // Use distinct step thresholds as indices and release them according to the synthetic clock.
        let ordine: Vec<usize> = {
            let mut o: Vec<usize> = (0..n).collect();
            o.sort_by(|&a, &b| arrivi[a].0.partial_cmp(&arrivi[b].0).unwrap().then(a.cmp(&b)));
            o
        };
        let soglie: Vec<u64> = (0..n).map(|i| ordine.iter().position(|&x| x == i).unwrap() as u64).collect();
        let classi: Vec<Class> = arrivi.iter().map(|a| a.1).collect();
        let mut amm = Admission::con_classi(soglie, classi, 1, slots, slots, riserva);
        let mut cv: Vec<Conv> = arrivi
            .iter()
            .map(|&(t, class, prompt, max)| Conv { class, prompt, max, m: None, tok: Vec::new(), arrivo_ms: t, primo_ms: None, fine_ms: None, da_inviare: 0, inviati: 0 })
            .collect();
        let mut t = 0.0f64;
        let mut corrente: Option<usize> = None;
        let mut attesa_primo: Vec<usize> = Vec::new();
        let mut arrivati = 0u64;
        let mut guard = 0;
        loop {
            guard += 1;
            assert!(guard < 200_000, "simulation did not terminate");
            while (arrivati as usize) < n && arrivi[ordine[arrivati as usize]].0 <= t {
                arrivati += 1;
            }
            if arrivati > 0 {
                amm.arriva(arrivati - 1, false);
            }
            // Process preemption before admission so a waiting interactive request can
            // use the released slot in this round.
            if amm.in_coda_interattive() > 0 {
                if let Some(b) = corrente {
                    if cv[b].class == Class::Batch {
                        let az = cv[b].m.as_mut().unwrap().suspend(Duration::from_millis(t as u64)).unwrap();
                        assert!(matches!(az[0], Action::Close { .. }));
                        assert!(amm.suspend(b));
                        corrente = None;
                    }
                }
                if corrente.is_none() {
                    let costo = |c: usize| if cv[c].m.as_ref().is_some_and(|m| m.state() == Status::Active) && !attesa_primo.contains(&c) { Some((cv[c].prompt + cv[c].tok.len()) as u64) } else { None };
                    if let Some(b) = amm.da_sospendere(&costo) {
                        cv[b].m.as_mut().unwrap().suspend(Duration::from_millis(t as u64)).unwrap();
                        assert!(amm.suspend(b));
                    }
                }
            }
            if corrente.is_none() {
                if let Some((c, _, _)) = amm.prossima() {
                    let adesso = Duration::from_millis(t as u64);
                    match cv[c].m.take() {
                        Some(mut m) if m.sospesa_volontaria() => {
                            m.resume(adesso).unwrap();
                            cv[c].m = Some(m);
                        }
                        _ => cv[c].m = Some(Conversation::new(c as u64 + 1, cv[c].prompt as u32, Config::default(), adesso).0),
                    }
                    cv[c].da_inviare = cv[c].prompt + cv[c].tok.len();
                    cv[c].inviati = 0;
                    corrente = Some(c);
                }
            }
            let mut costo_giro = 0.0;
            if let Some(c) = corrente {
                let b = BLOCCO.min(cv[c].da_inviare - cv[c].inviati);
                cv[c].inviati += b;
                costo_giro += b as f64 / PREFILL_TOK_S * 1000.0;
                if cv[c].inviati >= cv[c].da_inviare {
                    corrente = None;
                    attesa_primo.push(c);
                }
            }
            // Newly admitted conversations first generate in the following round.
            let attivi: Vec<usize> = (0..n).filter(|&c| cv[c].m.as_ref().is_some_and(|m| m.state() == Status::Active) && cv[c].fine_ms.is_none()).collect();
            let rows = attivi.len() as f64;
            let durata = (RETE_MS).max(FISSO_MS + RIGA_MS * rows) + costo_giro;
            t += durata;
            let adesso = Duration::from_millis(t as u64);
            for c in attivi {
                let tent = cv[c].m.as_ref().unwrap().tentativo();
                let step = cv[c].tok.len() as u32;
                let v = (c as u32 * 1000 + step) % 997;
                let esito = cv[c].m.as_mut().unwrap().token(tent, step, v, adesso);
                assert!(matches!(esito, TokenOutcome::Registrato { .. }), "{esito:?}");
                cv[c].tok.push(v);
            }
            for c in std::mem::take(&mut attesa_primo) {
                let (tent, step) = (cv[c].m.as_ref().unwrap().tentativo(), cv[c].tok.len() as u32);
                let v = (c as u32 * 1000 + step) % 997;
                let esito = cv[c].m.as_mut().unwrap().token(tent, step, v, adesso);
                assert!(matches!(esito, TokenOutcome::Registrato { .. }), "{esito:?}");
                cv[c].tok.push(v);
                cv[c].primo_ms.get_or_insert(t);
            }
            for c in 0..n {
                if cv[c].fine_ms.is_none() && cv[c].tok.len() >= cv[c].max && cv[c].m.as_ref().is_some_and(|m| m.state() == Status::Active) {
                    cv[c].fine_ms = Some(t);
                    cv[c].m.as_mut().unwrap().close_by_user(adesso).unwrap();
                    amm.libera(c);
                }
            }
            if cv.iter().all(|c| c.fine_ms.is_some()) {
                break;
            }
            if corrente.is_none() && attesa_primo.is_empty() && amm.in_coda() == 0 && !cv.iter().any(|c| c.m.as_ref().is_some_and(|m| m.state() == Status::Active) && c.fine_ms.is_none()) {
                // When idle, skip straight to the next arrival rather than ticking through empty rounds.
                let prox = cv.iter().filter(|c| c.m.is_none()).map(|c| c.arrivo_ms).fold(f64::INFINITY, f64::min);
                if prox.is_finite() && prox > t {
                    t = prox;
                }
            }
        }
        // Validate that each transcript has consecutive steps and the expected token count.
        for (i, c) in cv.iter().enumerate() {
            let tr = c.m.as_ref().unwrap().transcript();
            assert_eq!(tr.len(), c.tok.len(), "conv {i}");
            for (j, r) in tr.iter().enumerate() {
                assert_eq!((r.token_seq, r.step, r.token), (j as u64 + 1, j as u32, c.tok[j]), "conv {i} passo {j}");
            }
            assert_eq!(c.tok.len(), c.max, "conv {i} completa");
        }
        SimulationResult {
            ttft_interattive: cv.iter().filter(|c| c.class == Class::Interactive).map(|c| c.primo_ms.unwrap() - c.arrivo_ms).collect(),
            token_batch: cv.iter().filter(|c| c.class == Class::Batch).map(|c| c.tok.len()).sum(),
            token_interattive: cv.iter().filter(|c| c.class == Class::Interactive).map(|c| c.tok.len()).sum(),
            durata_ms: t,
            sospensioni: cv.iter().map(|c| c.m.as_ref().unwrap().sospensioni()).sum(),
            finite: cv.iter().filter(|c| c.fine_ms.is_some()).count(),
        }
    }

    fn mediana(v: &[f64]) -> f64 {
        let mut s = v.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        s[s.len() / 2]
    }

    /// The interactive wave exceeds the four free slots, forcing preemption.
    /// The latency bounds allow the cost of a prefill block and a generation round.
    #[test]
    fn backfill_batches_do_not_increase_interactive_ttft() {
        use Class::*;
        let inter: Vec<(f64, Class, usize, usize)> = (0..8).map(|i| (30_000.0 + 700.0 * i as f64, Interactive, 200, 80)).collect();
        let solo = simula(&inter, 10, 0);
        let mut misto: Vec<(f64, Class, usize, usize)> = (0..6).map(|i| (i as f64, Batch, 600, 400)).collect();
        misto.extend(inter.iter().copied());
        let con = simula(&misto, 10, 0);
        let (m0, m1) = (mediana(&solo.ttft_interattive), mediana(&con.ttft_interattive));
        let (x0, x1) = (solo.ttft_interattive.iter().cloned().fold(0.0, f64::max), con.ttft_interattive.iter().cloned().fold(0.0, f64::max));
        eprintln!("Interactive TTFT: alone median {m0:.0} ms, max {x0:.0} ms; with batch median {m1:.0} ms, max {x1:.0} ms; suspensions {}", con.sospensioni);
        assert!(m1 <= m0 * 1.3, "mediana {m1} contro {m0}");
        assert!(x1 <= x0 + 1000.0, "massimo {x1} contro {x0}");
        assert!(con.sospensioni > 0, "load should force suspension");
        assert_eq!(con.finite, 14);
        assert_eq!(con.token_batch, 6 * 400);
        assert_eq!(con.token_interattive, solo.token_interattive);
    }

    /// A large enough reserve lets interactive requests enter without a batch suspension.
    #[test]
    fn batches_use_free_slots_without_preemption() {
        use Class::*;
        let inter: Vec<(f64, Class, usize, usize)> = (0..4).map(|i| (30_000.0 + 900.0 * i as f64, Interactive, 200, 60)).collect();
        let mut misto: Vec<(f64, Class, usize, usize)> = (0..3).map(|i| (i as f64, Batch, 400, 300)).collect();
        misto.extend(inter.iter().copied());
        let con = simula(&misto, 8, 4);
        assert_eq!(con.sospensioni, 0);
        let solo = simula(&inter, 8, 0);
        assert!(mediana(&con.ttft_interattive) <= mediana(&solo.ttft_interattive) * 1.3);
    }

    #[test]
    fn initial_arrivals_count_zero_thresholds() {
        assert_eq!(Admission::new(vec![0, 0, 3, 0], 2, 2, 4).iniziali(), 3);
    }
}
