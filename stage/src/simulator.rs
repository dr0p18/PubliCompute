//! Drive the head and stage state machines with a seeded clock and message
//! queues. Inject loss, duplication, delayed delivery, stalled work, and
//! restarts while checking slot ownership, transcript ordering, and eventual
//! termination. Delivery stays ordered within a stream. Different streams
//! can race, and worker epochs keep pre-restart traffic recognizable.

use crate::admission::Admission;
use crate::conversation::{Action, Config, Conversation, TokenOutcome, Status};
use crate::slot::{OpenOutcome, WorkOutcome, ManualClock, Slots, Deadlines};
use std::collections::HashMap;
use std::time::Duration;

const S: Duration = Duration::from_secs(1);
const MS: Duration = Duration::from_millis(1);
const SC: Deadlines = Deadlines { riserva: Duration::from_secs(30), inattivo: Duration::from_secs(120) };
/// Max time a stage can stall (long computation, disk), explicitly allowed beyond expiry.
const BLOCCO_MAX: Duration = Duration::from_secs(40);
/// No expiry grace for idle stages: cleanup runs at every step with an empty queue.
const TOLLERANZA: Duration = Duration::ZERO;
/// Deadline for unconfirmed work before the head suspends it.
const T_LAVORO: Duration = Duration::from_secs(20);

pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        let mut r = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03);
        for _ in 0..4 {
            r.next();
        }
        r
    }
    pub fn next(&mut self) -> u64 {
        // xorshift64*
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// Integer in `[0, n)`.
    pub fn sotto(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }
    /// True with probability `per_mille / 1000`.
    pub fn per_mille(&mut self, per_mille: u64) -> bool {
        self.sotto(1000) < per_mille
    }
    pub fn tra(&mut self, a: u64, b: u64) -> u64 {
        a + self.sotto(b - a + 1)
    }
}

/// Deterministic token for `step`: one token per step, regardless of which attempt computes it.
fn valore(conv: u64, step: u32) -> u32 {
    (conv.wrapping_mul(1_000_003).wrapping_add(step as u64 * 7919) % 50_000) as u32
}

#[derive(Clone, Copy, Debug)]
enum Kind {
    Open { pos: u32 },
    PrefillBlock { pos: u32, n: u32, step: u32 },
    Step { pos: u32, step: u32 },
    Close,
}

impl Kind {
    fn stream(&self) -> usize {
        match self {
            Kind::Open { .. } | Kind::PrefillBlock { .. } => 0,
            _ => 1,
        }
    }
}

#[derive(Clone, Debug)]
struct Msg {
    conv: u64,
    tent: u32,
    kind: Kind,
    epoche: Vec<u32>,
}

#[derive(Clone, Debug)]
enum Payload {
    Stage(usize, Msg),
    Token { conv: u64, tent: u32, step: u32, token: u32 },
    Error { conv: u64, tent: u32 },
    VedeCrash,
}

struct InVolo {
    arrivo: Duration,
    ordine: u64,
    payload: Payload,
}

/// Fault rate (per thousand per event).
#[derive(Clone, Copy)]
struct Caos {
    perdita: u64,
    doppione: u64,
    ritardo_lungo: u64,
    crash: u64,
    block: u64,
    cancel: u64,
    spurio: u64,
}

impl Caos {
    fn da_seme(r: &mut Rng) -> Self {
        match r.sotto(4) {
            0 => Caos { perdita: 0, doppione: 20, ritardo_lungo: 10, crash: 0, block: 0, cancel: 2, spurio: 20 },
            1 => Caos { perdita: 20, doppione: 60, ritardo_lungo: 30, crash: 8, block: 20, cancel: 5, spurio: 50 },
            2 => Caos { perdita: 60, doppione: 100, ritardo_lungo: 60, crash: 25, block: 40, cancel: 10, spurio: 80 },
            _ => Caos { perdita: 150, doppione: 150, ritardo_lungo: 100, crash: 60, block: 80, cancel: 20, spurio: 100 },
        }
    }
    fn nessuno() -> Self {
        Caos { perdita: 0, doppione: 0, ritardo_lungo: 0, crash: 0, block: 0, cancel: 0, spurio: 0 }
    }
}

struct Stage {
    slots: Slots<ManualClock>,
    epoch: u32,
    vivo: bool,
    fermo_fino: Duration,
    /// Expected slot expiry, tracked separately by the simulator from the machine under test.
    scadenza: HashMap<u64, Duration>,
    /// Independent oracle for the highest closed attempt.
    fil_sim: HashMap<u64, u32>,
    /// Sequence owners reconstructed independently by the simulator: sequence -> conversation.
    seq_sim: HashMap<u32, u64>,
}

/// Use fewer sequences than conversations so the simulator exercises slot reuse.
const N_SEQ: u64 = 3;

fn seq_di(conv: u64) -> u32 {
    (conv % N_SEQ) as u32
}

struct ConvSim {
    c: Conversation,
    /// Position the head sends in the next `Step` / `PrefillBlock` for this attempt.
    pos_testa: u32,
    /// Tokens still to produce for the current response.
    da_produrre: u32,
    turni_rimasti: u32,
    utente_torna: Duration,
    ultimo_progresso: Duration,
    /// Attempt the head sees as open, so we can check that a terminal conversation sends no more `Open`.
    terminale_da: Option<Duration>,
    /// Tracked by the simulator: the conversation was suspended while idle and awaits a
    /// user turn (lazy recovery: no `Open` until the new turn).
    attesa_turno: bool,
}

#[derive(Default, Debug, Clone, Copy)]
pub struct Statistiche {
    pub chiuse: u32,
    pub annullate: u32,
    pub fallite: u32,
    pub recoveries: u32,
    /// Conversations suspended while idle and woken by a new turn.
    pub dormienti: u32,
    pub svegliate: u32,
    pub token_registrati: u64,
    pub scartati_epoca: u64,
    pub scartati_filigrana: u64,
    pub posti_scaduti: u64,
    pub messages: u64,
    /// Errors injected into a single conversation.
    pub errori_conv: u64,
    /// `Open` rejected because another live conversation owned the sequence.
    pub sequenze_occupate: u64,
}

/// Drive the same admission rules as the head and check their invariants
/// while conversations fail and recover.
struct AmmSim {
    a: Admission,
    /// For each simulator conversation index: rule index, group and local slot.
    mappa: Vec<(usize, usize, u32)>,
    /// Admission order, which must match arrival order.
    ammesse: Vec<usize>,
    /// Observed arrival order (output of `arriva`).
    arrivate: Vec<usize>,
    slots: usize,
}

pub struct Sim {
    seed: u64,
    rng: Rng,
    now: ManualClock,
    cfg: Config,
    stages: Vec<Stage>,
    conv: Vec<ConvSim>,
    volo: Vec<InVolo>,
    ordine: u64,
    ultimo_arrivo: HashMap<(usize, usize), Duration>,
    ripristini: Vec<(Duration, usize)>,
    caos: Caos,
    traccia: Vec<String>,
    pub stat: Statistiche,
    amm: Option<AmmSim>,
}

impl Sim {
    pub fn new(seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let n_stadi = 2 + rng.sotto(2) as usize;
        let caos = Caos::da_seme(&mut rng);
        let now = ManualClock::default();
        let stages = (0..n_stadi)
            .map(|_| Stage {
                slots: Slots::new(now.clone(), SC),
                epoch: 1,
                vivo: true,
                fermo_fino: Duration::ZERO,
                scadenza: HashMap::new(),
                fil_sim: HashMap::new(),
                seq_sim: HashMap::new(),
            })
            .collect();
        Sim {
            seed,
            rng,
            now,
            cfg: Config::default(),
            stages,
            conv: Vec::new(),
            volo: Vec::new(),
            ordine: 0,
            ultimo_arrivo: HashMap::new(),
            ripristini: Vec::new(),
            caos,
            traccia: Vec::new(),
            stat: Statistiche::default(),
            amm: None,
        }
    }

    /// Like `new`, with conversations arriving by the admission rule. Random thresholds in recorded
    /// tokens (simulator step count), 2-3 groups, 1 to 4 slots per group and a total cap.
    pub fn con_arrivi(seed: u64) -> Self {
        let mut s = Sim::new(seed);
        let n = s.rng.tra(4, 10) as usize;
        let mut soglie: Vec<u64> = (0..n).map(|_| if s.rng.per_mille(300) { 0 } else { s.rng.tra(1, 60) }).collect();
        soglie.sort_unstable();
        let groups = s.rng.tra(1, 3) as usize;
        let per = s.rng.tra(1, 3) as usize;
        let slots = s.rng.tra(1, (groups * per) as u64) as usize;
        s.amm = Some(AmmSim { a: Admission::new(soglie, groups, per, slots), mappa: Vec::new(), ammesse: Vec::new(), arrivate: Vec::new(), slots });
        s
    }

    /// Arrivals and admissions this step, with checks for the rule's invariants.
    fn arrivi(&mut self) {
        let Some(mut am) = self.amm.take() else { return };
        let passi = self.stat.token_registrati;
        let fermo = self.conv.iter().all(|c| c.c.state().terminale());
        let nuovi = am.a.arriva(passi, fermo);
        for &c in &nuovi {
            if am.a.soglia(c) > passi && !fermo {
                self.viola(&format!("early arrival: conversation {c} became eligible at {}, current step {passi}", am.a.soglia(c)));
            }
        }
        am.arrivate.extend(nuovi);
        while let Some((c, g, locale)) = am.a.prossima() {
            if am.mappa.iter().any(|&(_, gg, ll)| gg == g && ll == locale && self.conv_viva_di(&am, gg, ll)) {
                self.viola(&format!("admission assigned slot ({g},{locale}) to a second live conversation"));
            }
            am.ammesse.push(c);
            am.mappa.push((c, g, locale));
            self.nuova_conversazione();
        }
        // Invariants: FIFO (admitted is a prefix of arrivals), slot cap and no slot shared by live conversations.
        if am.ammesse[..] != am.arrivate[..am.ammesse.len()] {
            self.viola("admission queue violated FIFO");
        }
        if am.a.occupati() > am.slots {
            self.viola("more live conversations than slots");
        }
        let vive: Vec<(usize, u32)> = am.mappa.iter().enumerate().filter(|(i, _)| !self.conv[*i].c.state().terminale()).map(|(_, &(_, g, l))| (g, l)).collect();
        for (i, a) in vive.iter().enumerate() {
            if vive[i + 1..].contains(a) {
                self.viola(&format!("two live conversations share slot {a:?}"));
            }
        }
        self.amm = Some(am);
    }

    fn conv_viva_di(&self, am: &AmmSim, g: usize, l: u32) -> bool {
        am.mappa.iter().enumerate().any(|(i, &(_, gg, ll))| gg == g && ll == l && !self.conv[i].c.state().terminale())
    }

    /// A terminal conversation frees its slot in the admission rule.
    fn libera_terminali(&mut self) {
        let Some(am) = self.amm.as_mut() else { return };
        for (i, &(c, _, _)) in am.mappa.iter().enumerate() {
            if self.conv[i].c.state().terminale() {
                am.a.libera(c);
            }
        }
    }

    fn adesso(&self) -> Duration {
        use crate::slot::Clock;
        self.now.adesso()
    }

    fn log(&mut self, s: String) {
        let t = self.adesso();
        self.traccia.push(format!("[{:>8.3}s] {}", t.as_secs_f64(), s));
    }

    fn viola(&self, cosa: &str) -> ! {
        let da = self.traccia.len().saturating_sub(40);
        let mut s = format!("\nINVARIANT VIOLATION (seed {}): {}\nstages: {}, conversations: {}\n--- recent trace ---\n",
            self.seed, cosa, self.stages.len(), self.conv.len());
        for r in &self.traccia[da..] {
            s.push_str(r);
            s.push('\n');
        }
        panic!("{s}");
    }

    // Simulated network delivery.

    fn ritardo(&mut self) -> Duration {
        if self.rng.per_mille(self.caos.ritardo_lungo) {
            Duration::from_millis(self.rng.tra(3_000, 35_000))
        } else {
            Duration::from_millis(self.rng.tra(1, 300))
        }
    }

    /// Put a message in flight: loss, delay, order within the stream and duplication.
    fn send(&mut self, payload: Payload) {
        self.stat.messages += 1;
        if self.rng.per_mille(self.caos.perdita) {
            self.log(format!("DROPPED {payload:?}"));
            return;
        }
        let stream = match &payload {
            Payload::Stage(k, m) => (*k, m.kind.stream()),
            _ => (usize::MAX, 2),
        };
        let mut arrivo = self.adesso() + self.ritardo();
        let ult = self.ultimo_arrivo.entry(stream).or_insert(Duration::ZERO);
        if arrivo <= *ult {
            arrivo = *ult + MS;
        }
        *ult = arrivo;
        let copie = if self.rng.per_mille(self.caos.doppione) { 2 } else { 1 };
        for i in 0..copie {
            let a = if i == 0 { arrivo } else { arrivo + Duration::from_millis(self.rng.tra(1, 5_000)) };
            self.ordine += 1;
            self.volo.push(InVolo { arrivo: a, ordine: self.ordine, payload: payload.clone() });
        }
    }

    fn epoche(&self) -> Vec<u32> {
        self.stages.iter().map(|s| s.epoch).collect()
    }

    fn manda(&mut self, k: usize, conv: u64, tent: u32, kind: Kind) {
        let m = Msg { conv, tent, kind, epoche: self.epoche() };
        self.send(Payload::Stage(k, m));
    }

    // Head event handling.

    fn applica(&mut self, i: usize, azioni: Vec<Action>) {
        let id = self.conv[i].c.id();
        for a in azioni {
            self.log(format!("head conversation {id}: {a:?} (state {:?})", self.conv[i].c.state()));
            match a {
                Action::Open { tentativo, start_position } => {
                    if self.conv[i].terminale_da.is_some() {
                        self.viola("terminal conversation reopened with Open");
                    }
                    if self.conv[i].attesa_turno || self.conv[i].c.dormant() {
                        self.viola("a conversation suspended from idle emitted Open without a user turn");
                    }
                    let n = self.conv[i].c.prefisso_len();
                    let step = self.conv[i].c.expected_step();
                    self.manda(0, id, tentativo, Kind::Open { pos: start_position });
                    self.manda(0, id, tentativo, Kind::PrefillBlock { pos: start_position, n, step });
                    self.conv[i].pos_testa = start_position + n;
                    self.conv[i].ultimo_progresso = self.adesso();
                }
                Action::Close { tentativo } => self.manda(0, id, tentativo, Kind::Close),
                Action::ErroreClient(_) => {}
            }
        }
        let c = &mut self.conv[i];
        if c.c.state().terminale() && c.terminale_da.is_none() {
            c.terminale_da = Some(self.now.adesso_pub());
            match c.c.state() {
                Status::Closed => self.stat.chiuse += 1,
                Status::Cancelled => self.stat.annullate += 1,
                _ => self.stat.fallite += 1,
            }
        }
    }

    /// `fault` with the simulator's lazy recovery bookkeeping.
    fn guasto_sim(&mut self, i: usize, tent: u32) -> Option<Vec<Action>> {
        let adesso = self.adesso();
        let before = self.conv[i].c.state();
        let r = self.conv[i].c.fault(tent, adesso);
        if r.is_ok() && before == Status::Waiting {
            self.conv[i].attesa_turno = true;
            self.stat.dormienti += 1;
            if !self.conv[i].c.dormant() {
                self.viola("fault while waiting for a user turn left the conversation awake");
            }
        }
        if r.is_ok() && before != Status::Waiting && self.conv[i].c.dormant() {
            self.viola("conversation went to sleep without waiting for a user turn");
        }
        r.ok()
    }

    fn nuova_conversazione(&mut self) {
        let id = self.conv.len() as u64 + 1;
        let prompt = self.rng.tra(1, 2_000) as u32;
        let (c, az) = Conversation::new(id, prompt, self.cfg, self.adesso());
        let da_produrre = self.rng.tra(1, 8) as u32;
        let turni_rimasti = self.rng.sotto(3) as u32;
        self.conv.push(ConvSim {
            c,
            pos_testa: 0,
            da_produrre,
            turni_rimasti,
            utente_torna: Duration::ZERO,
            ultimo_progresso: self.adesso(),
            terminale_da: None,
            attesa_turno: false,
        });
        let i = self.conv.len() - 1;
        self.log(format!("new conversation {id}, prompt {prompt}"));
        self.applica(i, az);
    }

    fn testa_riceve(&mut self, p: Payload) {
        match p {
            Payload::Token { conv, tent, step, token } => {
                let i = conv as usize - 1;
                let adesso = self.adesso();
                let before = self.conv[i].c.state();
                let attivo_prima = self.conv[i].c.tentativo();
                let e = self.conv[i].c.token(tent, step, token, adesso);
                self.log(format!("head conversation {conv}: Token t{tent} step {step} -> {e:?} ({before:?})"));
                if let TokenOutcome::Registrato { .. } = e {
                    if tent != attivo_prima || before.terminale() {
                        self.viola("recorded a token from an inactive attempt or a terminal conversation");
                    }
                    self.stat.token_registrati += 1;
                    self.conv[i].ultimo_progresso = adesso;
                    if self.conv[i].da_produrre > 0 {
                        self.conv[i].da_produrre -= 1;
                    }
                    if self.conv[i].da_produrre == 0 {
                        let az = self.conv[i].c.finish_response(adesso).expect("end response while active");
                        self.applica(i, az);
                        self.conv[i].utente_torna = adesso + Duration::from_millis(self.rng.tra(500, 60_000));
                    } else {
                        let pos = self.conv[i].pos_testa;
                        let step = self.conv[i].c.expected_step();
                        self.conv[i].pos_testa += 1;
                        self.manda(0, conv, tent, Kind::Step { pos, step });
                    }
                }
            }
            Payload::Error { conv, tent } => {
                let i = conv as usize - 1;
                // Invariant: an error in one conversation leaves the others' attempts, states and recoveries alone.
                let before: Vec<(u32, Status, u32)> = self.conv.iter().map(|c| (c.c.tentativo(), c.c.state(), c.c.recoveries())).collect();
                let r = self.guasto_sim(i, tent);
                self.log(format!("head conversation {conv}: Error t{tent} -> {:?}", r.as_ref().map(|_| ())));
                if let Some(az) = r {
                    self.applica(i, az);
                }
                for (j, c) in self.conv.iter().enumerate() {
                    if j != i && before[j] != (c.c.tentativo(), c.c.state(), c.c.recoveries()) {
                        self.viola(&format!("fault in conversation {conv} changed attempt, state or recoveries for conversation {}", c.c.id()));
                    }
                }
            }
            Payload::VedeCrash => {
                for i in 0..self.conv.len() {
                    let t = self.conv[i].c.tentativo();
                    if let Some(az) = self.guasto_sim(i, t) {
                        self.applica(i, az);
                    }
                }
            }
            Payload::Stage(..) => unreachable!(),
        }
    }

    // Stage event handling.

    fn forward(&mut self, k: usize, m: &Msg) {
        if k + 1 < self.stages.len() {
            self.send(Payload::Stage(k + 1, m.clone()));
        }
    }

    fn tail_token(&mut self, k: usize, m: &Msg, step: u32) {
        if k + 1 == self.stages.len() {
            let token = valore(m.conv, step);
            self.send(Payload::Token { conv: m.conv, tent: m.tent, step, token });
        }
    }

    fn stadio_riceve(&mut self, k: usize, m: Msg) {
        if !self.stages[k].vivo {
            self.log(format!("stage {k} is down; dropped {:?} for conversation {}", m.kind, m.conv));
            return;
        }
        if m.epoche[k] != self.stages[k].epoch {
            self.stat.scartati_epoca += 1;
            self.log(format!("stage {k}: epoch {} != {}; rejected {:?} for conversation {}, attempt {}", m.epoche[k], self.stages[k].epoch, m.kind, m.conv, m.tent));
            return;
        }
        let adesso = self.adesso();
        let fil_macchina = self.stages[k].slots.watermark(m.conv);
        let fil = self.stages[k].fil_sim.get(&m.conv).copied().unwrap_or(0);
        if fil_macchina != fil {
            self.viola(&format!("stage {k}, conversation {}: watermark {fil_macchina}, expected {fil}", m.conv));
        }
        let seq = seq_di(m.conv);
        match m.kind {
            Kind::Open { pos } => {
                let e = self.stages[k].slots.open(m.conv, m.tent, seq, pos);
                self.log(format!("stage {k}: Open t{} for conversation {} (watermark {fil}) -> {e:?}", m.tent, m.conv));
                match e {
                    OpenOutcome::Riservato { libera } => {
                        if let Some(&altra) = self.stages[k].seq_sim.get(&seq) {
                            if altra != m.conv {
                                self.viola(&format!("stage {k}: Open for conversation {} took sequence {seq} from live conversation {altra}", m.conv));
                            }
                        }
                        if let Some(l) = libera {
                            self.stages[k].seq_sim.remove(&l.seq);
                            let f = self.stages[k].fil_sim.entry(m.conv).or_insert(0);
                            *f = (*f).max(l.tentativo);
                        }
                        if m.tent <= fil {
                            self.viola("Open reserved a slot at or below the attempt watermark");
                        }
                        self.stages[k].seq_sim.insert(seq, m.conv);
                        self.stages[k].scadenza.insert(m.conv, adesso + SC.riserva);
                        self.forward(k, &m);
                    }
                    OpenOutcome::Duplicato => self.forward(k, &m),
                    OpenOutcome::Scartato => self.stat.scartati_filigrana += 1,
                    // Use `open` without eviction to exercise sequence collisions between live conversations.
                    OpenOutcome::Sfrattato { .. } => self.viola("unexpected eviction with evictions disabled"),
                    OpenOutcome::Occupato { da } => {
                        if self.stages[k].seq_sim.get(&seq) != Some(&da) || da == m.conv || m.tent <= fil {
                            self.viola(&format!("stage {k}: Open for conversation {} rejected on sequence {seq} with unexpected owner {da}", m.conv));
                        }
                        self.stat.sequenze_occupate += 1;
                        self.send(Payload::Error { conv: m.conv, tent: m.tent });
                    }
                }
            }
            Kind::PrefillBlock { pos, n, step } => {
                let e = self.stages[k].slots.block(m.conv, m.tent, pos, n);
                self.log(format!("stage {k}: Block t{} for conversation {} at {pos} (watermark {fil}) -> {e:?}", m.tent, m.conv));
                self.lavoro(k, &m, e, fil, step);
            }
            Kind::Step { pos, step } => {
                let e = self.stages[k].slots.step(m.conv, m.tent, pos);
                self.log(format!("stage {k}: Step t{} for conversation {} at {pos} (watermark {fil}) -> {e:?}", m.tent, m.conv));
                self.lavoro(k, &m, e, fil, step);
            }
            Kind::Close => {
                let e = self.stages[k].slots.close(m.conv, m.tent);
                let f = self.stages[k].fil_sim.entry(m.conv).or_insert(0);
                *f = (*f).max(m.tent);
                self.log(format!("stage {k}: Close t{} for conversation {} -> {e:?}", m.tent, m.conv));
                if let Some(l) = e.libera {
                    self.libera_seq(k, m.conv, l.seq);
                }
                if self.stages[k].slots.slot(m.conv).is_none() {
                    self.stages[k].scadenza.remove(&m.conv);
                }
                self.forward(k, &m);
            }
        }
    }

    fn lavoro(&mut self, k: usize, m: &Msg, e: WorkOutcome, fil: u32, step: u32) {
        match e {
            WorkOutcome::Compute => {
                if m.tent <= fil {
                    self.viola("stage decoded a message at or below the attempt watermark");
                }
                let adesso = self.adesso();
                self.stages[k].scadenza.insert(m.conv, adesso + SC.inattivo);
                self.forward(k, m);
                self.tail_token(k, m, step);
            }
            WorkOutcome::Duplicate => {}
            WorkOutcome::Discarded => self.stat.scartati_filigrana += 1,
            WorkOutcome::NoActiveSlot | WorkOutcome::UnexpectedPosition { .. } => {
                if m.tent <= fil {
                    self.viola("stage sent an error for an attempt at or below the watermark");
                }
                self.send(Payload::Error { conv: m.conv, tent: m.tent });
            }
        }
    }

    /// Ownership invariant: any sequence cleared by a stage on behalf of `conv` belongs to `conv`.
    fn libera_seq(&mut self, k: usize, conv: u64, seq: u32) {
        match self.stages[k].seq_sim.get(&seq) {
            Some(&c) if c == conv => {
                self.stages[k].seq_sim.remove(&seq);
            }
            altro => self.viola(&format!("stage {k}: clearing sequence {seq} for conversation {conv} with unexpected owner {altro:?}")),
        }
    }

    fn pulizia_stadio(&mut self, k: usize) {
        let adesso = self.adesso();
        if !self.stages[k].vivo || adesso < self.stages[k].fermo_fino {
            return;
        }
        for s in self.stages[k].slots.scadenze() {
            self.stat.posti_scaduti += 1;
            self.stages[k].scadenza.remove(&s.conversation_id);
            let f = self.stages[k].fil_sim.entry(s.conversation_id).or_insert(0);
            *f = (*f).max(s.liberato.tentativo);
            self.log(format!("stage {k}: expired slot for conversation {}, attempt {}", s.conversation_id, s.liberato.tentativo));
            self.libera_seq(k, s.conversation_id, s.liberato.seq);
            self.send(Payload::Error { conv: s.conversation_id, tent: s.liberato.tentativo });
        }
    }

    fn crash(&mut self, k: usize) {
        let adesso = self.adesso();
        let s = &mut self.stages[k];
        s.vivo = false;
        s.epoch += 1;
        s.slots = Slots::new(self.now.clone(), SC);
        s.scadenza.clear();
        s.fil_sim.clear();
        s.seq_sim.clear();
        let at = adesso + Duration::from_millis(self.rng.tra(500, 30_000));
        self.ripristini.push((at, k));
        self.log(format!("stage {k} crashed; new epoch {}", self.stages[k].epoch));
        self.send(Payload::VedeCrash);
    }

    // Advance the simulated clock and deliver work.

    fn consegna(&mut self) {
        let adesso = self.adesso();
        let mut dovuti: Vec<InVolo> = Vec::new();
        let mut resto = Vec::new();
        for m in self.volo.drain(..) {
            let ferma = match &m.payload {
                Payload::Stage(k, _) => adesso < self.stages[*k].fermo_fino,
                _ => false,
            };
            if m.arrivo <= adesso && !ferma {
                dovuti.push(m);
            } else {
                resto.push(m);
            }
        }
        self.volo = resto;
        dovuti.sort_by_key(|m| (m.arrivo, m.ordine));
        for m in dovuti {
            match m.payload {
                Payload::Stage(k, msg) => self.stadio_riceve(k, msg),
                p => self.testa_riceve(p),
            }
        }
    }

    fn eventi_utente_e_testa(&mut self, caos_attivo: bool, chiudi_subito: bool) {
        let adesso = self.adesso();
        for i in 0..self.conv.len() {
            let id = self.conv[i].c.id();
            let state = self.conv[i].c.state();
            match state {
                Status::Waiting if adesso >= self.conv[i].utente_torna || chiudi_subito => {
                    if self.conv[i].turni_rimasti > 0 && !chiudi_subito && self.rng.per_mille(700) {
                        self.conv[i].turni_rimasti -= 1;
                        let n = self.rng.tra(1, 60) as u32;
                        let pos = self.conv[i].pos_testa;
                        let step = self.conv[i].c.expected_step();
                        let t = self.conv[i].c.tentativo();
                        self.conv[i].c.next_turn(n, adesso).expect("new turn from an idle conversation");
                        self.conv[i].pos_testa += n;
                        self.conv[i].da_produrre = self.rng.tra(1, 8) as u32;
                        self.conv[i].ultimo_progresso = adesso;
                        self.log(format!("user starts a new turn in conversation {id} ({n} tokens)"));
                        self.manda(0, id, t, Kind::PrefillBlock { pos, n, step });
                    } else {
                        let az = self.conv[i].c.close_by_user(adesso).expect("close");
                        self.log(format!("user closes conversation {id}"));
                        self.applica(i, az);
                    }
                }
                Status::Suspended if self.conv[i].c.dormant() => {
                    // Lazy recovery: no rebuild or expiry until the user returns.
                    if self.conv[i].c.chain_ready(adesso).is_ok() {
                        self.viola("sleeping conversation accepted chain recovery");
                    }
                    if !self.conv[i].c.tempo(adesso).unwrap_or_default().is_empty() {
                        self.viola("sleeping conversation exceeded the recovery deadline");
                    }
                    if adesso >= self.conv[i].utente_torna || chiudi_subito {
                        if self.conv[i].turni_rimasti > 0 && !chiudi_subito && self.rng.per_mille(700) {
                            self.conv[i].turni_rimasti -= 1;
                            let n = self.rng.tra(1, 60) as u32;
                            self.conv[i].c.next_turn(n, adesso).expect("new turn for a sleeping conversation");
                            self.conv[i].attesa_turno = false;
                            self.stat.svegliate += 1;
                            self.conv[i].da_produrre = self.rng.tra(1, 8) as u32;
                            self.conv[i].ultimo_progresso = adesso;
                            self.log(format!("user starts a new turn ({n} tokens) in suspended conversation {id}"));
                        } else {
                            let az = self.conv[i].c.close_by_user(adesso).expect("close");
                            self.log(format!("user closes dormant conversation {id}"));
                            self.applica(i, az);
                        }
                    }
                }
                Status::Suspended => {
                    let az = self.conv[i].c.tempo(adesso).unwrap_or_default();
                    if !az.is_empty() {
                        self.applica(i, az);
                    } else if self.stages.iter().all(|s| s.vivo) && self.rng.per_mille(400) {
                        let az = self.conv[i].c.chain_ready(adesso).expect("chain recovered while suspended");
                        if self.conv[i].c.state() == Status::Rebuilding {
                            self.stat.recoveries += 1;
                            self.conv[i].da_produrre = self.conv[i].da_produrre.max(1);
                        }
                        self.applica(i, az);
                    }
                }
                Status::Rebuilding => {
                    if let Ok(az) = self.conv[i].c.tempo(adesso) {
                        self.applica(i, az);
                    }
                }
                _ => {}
            }
            // Unconfirmed work past its deadline.
            let c = &self.conv[i];
            if matches!(c.c.state(), Status::Opening | Status::Active | Status::Rebuilding)
                && adesso.saturating_sub(c.ultimo_progresso) >= T_LAVORO
            {
                let t = c.c.tentativo();
                if let Some(az) = self.guasto_sim(i, t) {
                    self.log(format!("head conversation {id}: work not confirmed within {T_LAVORO:?}"));
                    self.applica(i, az);
                }
            }
            if !caos_attivo {
                continue;
            }
            if !self.conv[i].c.state().terminale() && self.rng.per_mille(self.caos.cancel) {
                let az = self.conv[i].c.cancel(adesso).expect("annulla");
                self.log(format!("user or coordinator cancels conversation {id}"));
                self.applica(i, az);
            }
            if self.rng.per_mille(self.caos.spurio) {
                self.evento_spurio(i);
            }
        }
    }

    /// Events that should leave state alone: reject them for a terminal conversation. For a live conversation,
    /// reject a token with the wrong attempt.
    fn evento_spurio(&mut self, i: usize) {
        let adesso = self.adesso();
        let id = self.conv[i].c.id();
        let (state, tent, n) = (self.conv[i].c.state(), self.conv[i].c.tentativo(), self.conv[i].c.transcript().len());
        if state.terminale() {
            let r = match self.rng.sotto(7) {
                0 => self.conv[i].c.next_turn(3, adesso),
                1 => self.conv[i].c.chain_ready(adesso),
                2 => self.conv[i].c.fault(tent, adesso),
                3 => self.conv[i].c.finish_response(adesso),
                4 => self.conv[i].c.tempo(adesso + 1000 * S),
                5 => self.conv[i].c.close_by_user(adesso),
                _ => self.conv[i].c.cancel(adesso),
            };
            if r.is_ok() {
                self.viola("terminal conversation accepted an event and reopened");
            }
            let e = self.conv[i].c.token(tent, n as u32, 1, adesso);
            if e != TokenOutcome::Scartato {
                self.viola("terminal conversation accepted a token");
            }
            if self.conv[i].c.state() != state {
                self.viola("terminal conversation changed state");
            }
        } else {
            let altro = if self.rng.per_mille(500) && tent > 1 { tent - 1 } else { tent + 1 };
            let e = self.conv[i].c.token(altro, n as u32, 1, adesso);
            self.log(format!("unexpected token for conversation {id}, attempt {altro} -> {e:?}"));
            if e != TokenOutcome::Scartato {
                self.viola("transcript accepted a token from an inactive attempt");
            }
        }
    }

    fn eventi_stadi(&mut self) {
        let adesso = self.adesso();
        let mut pronti = Vec::new();
        self.ripristini.retain(|&(q, k)| {
            if q <= adesso {
                pronti.push(k);
                false
            } else {
                true
            }
        });
        for k in pronti {
            self.stages[k].vivo = true;
            self.log(format!("stage {k} recovered at epoch {}", self.stages[k].epoch));
        }
    }

    fn disturbi(&mut self) {
        let n = self.stages.len();
        // Single-conversation error (e.g. slot rejected or expired at one stage). Send an error `Confirmation` for the
        // current attempt of one live conversation and leave the others alone.
        if self.rng.per_mille(15) && !self.conv.is_empty() {
            let i = self.rng.sotto(self.conv.len() as u64) as usize;
            if !self.conv[i].c.state().terminale() {
                let (conv, tent) = (self.conv[i].c.id(), self.conv[i].c.tentativo());
                self.log(format!("injected error in conversation {conv}, attempt {tent}"));
                self.send(Payload::Error { conv, tent });
                self.stat.errori_conv += 1;
            }
        }
        if self.rng.per_mille(self.caos.crash) {
            let k = self.rng.sotto(n as u64) as usize;
            if self.stages[k].vivo {
                self.crash(k);
            }
        }
        if self.rng.per_mille(self.caos.block) {
            let k = self.rng.sotto(n as u64) as usize;
            let d = Duration::from_millis(self.rng.tra(1_000, BLOCCO_MAX.as_millis() as u64));
            let fino = self.adesso() + d;
            if self.stages[k].vivo && self.adesso() >= self.stages[k].fermo_fino {
                self.stages[k].fermo_fino = fino;
                self.log(format!("stage {k} busy for {d:?}"));
            }
        }
    }

    fn controlla(&mut self) {
        let adesso = self.adesso();
        for k in 0..self.stages.len() {
            let s = &self.stages[k];
            if !s.vivo {
                continue;
            }
            // A busy stage skips cleanup, so expiry can be delayed by at most the stall duration.
            let tol = if adesso < s.fermo_fino { BLOCCO_MAX } else { TOLLERANZA };
            for (&conv, &scad) in s.scadenza.iter() {
                if s.slots.slot(conv).is_none() {
                    self.viola(&format!("stage {k}, conversation {conv}: deadline without a slot"));
                }
                if adesso > scad + tol {
                    self.viola(&format!(
                        "stage {k}, conversation {conv}: slot still occupied after deadline (late by {:?}, tolerance {tol:?})",
                        adesso - scad
                    ));
                }
            }
            for (&seq, &conv) in s.seq_sim.iter() {
                if s.slots.proprietario(seq).map(|x| x.0) != Some(conv) {
                    self.viola(&format!("stage {k}: sequence {seq} should belong to conversation {conv}, got {:?}", s.slots.proprietario(seq)));
                }
            }
            for conv in 1..=self.conv.len() as u64 {
                if s.slots.slot(conv).is_some() && !s.scadenza.contains_key(&conv) {
                    self.viola(&format!("stage {k}, conversation {conv}: slot without a deadline"));
                }
            }
        }
        for cs in &self.conv {
            let tr = cs.c.transcript();
            for (j, r) in tr.iter().enumerate() {
                if r.token_seq != j as u64 + 1 || r.step != j as u32 || r.token != valore(cs.c.id(), r.step) {
                    self.viola(&format!("conversation {}: gap or duplicate at transcript position {j}: {r:?}", cs.c.id()));
                }
                if j > 0 && r.tentativo < tr[j - 1].tentativo {
                    self.viola(&format!("conversation {}: non-increasing attempts in the transcript", cs.c.id()));
                }
            }
            if cs.c.recoveries() > self.cfg.k_riprese + 1 {
                self.viola(&format!("conversation {}: {} recoveries exceed the limit", cs.c.id(), cs.c.recoveries()));
            }
            if cs.c.recoveries() > self.cfg.k_riprese && cs.c.state() != Status::Failed {
                self.viola(&format!("conversation {}: exceeded recovery limit without failing (state {:?})", cs.c.id(), cs.c.state()));
            }
            if let Some(_t) = cs.terminale_da {
                if !cs.c.state().terminale() {
                    self.viola(&format!("conversation {} reopened after termination", cs.c.id()));
                }
            }
        }
    }

    fn step(&mut self, caos_attivo: bool, chiudi_subito: bool) {
        let dt = match self.rng.sotto(100) {
            0..=69 => Duration::from_millis(self.rng.tra(1, 500)),
            70..=89 => Duration::from_millis(self.rng.tra(500, 5_000)),
            90..=97 => Duration::from_millis(self.rng.tra(5_000, 40_000)),
            _ => Duration::from_millis(self.rng.tra(40_000, 150_000)),
        };
        // Use small time jumps while draining. A few seconds suffice for everything to finish.
        let dt = if chiudi_subito { dt.min(5 * S) } else { dt };
        self.now.avanza(dt);
        self.eventi_stadi();
        self.consegna();
        self.eventi_utente_e_testa(caos_attivo, chiudi_subito);
        // Cleanup on an empty queue precedes new faults, so a stall starts only after the stage has cleaned up.
        for k in 0..self.stages.len() {
            self.pulizia_stadio(k);
        }
        if caos_attivo {
            self.disturbi();
            // An occasional new conversation, up to 5 (the rule decides when using admission).
            if self.amm.is_none() && self.conv.len() < 5 && self.rng.per_mille(80) {
                self.nuova_conversazione();
            }
        }
        self.libera_terminali();
        self.arrivi();
        self.controlla();
    }

    /// Run `eventi` steps with faults, then drain without faults. Every conversation must end in a terminal state.
    pub fn run(&mut self, eventi: usize) {
        if self.amm.is_none() {
            self.nuova_conversazione();
        } else {
            self.arrivi();
        }
        for _ in 0..eventi {
            self.step(true, false);
        }
        // Drain with no new losses, duplicates, crashes or stalls. Users close immediately.
        self.caos = Caos::nessuno();
        for _ in 0..4_000 {
            if self.conv.iter().all(|c| c.c.state().terminale()) && !self.amm.as_ref().is_some_and(|a| a.a.pendenti()) {
                break;
            }
            self.step(false, true);
        }
        if self.amm.as_ref().is_some_and(|a| a.a.pendenti()) {
            self.viola("conversations still queued or pending after draining");
        }
        for cs in &self.conv {
            if !cs.c.state().terminale() {
                self.viola(&format!("conversation {} still in {:?} after draining", cs.c.id(), cs.c.state()));
            }
        }
        // Once idle, all slots must also free up by the longest deadline. With arrivals,
        // the last conversation may fail late in the drain. Allow more steps, but exit as soon as all slots are free.
        let massimo = if self.amm.is_some() { 400 } else { 40 };
        for _ in 0..massimo {
            if self.amm.is_some() && self.stages.iter().all(|s| !s.vivo || s.slots.posti_in_uso() == 0) {
                break;
            }
            self.step(false, true);
        }
        for (k, s) in self.stages.iter().enumerate() {
            if s.vivo && s.slots.posti_in_uso() > 0 {
                self.viola(&format!("stage {k} still holds {} slots after draining", s.slots.posti_in_uso()));
            }
        }
    }
}

trait AdessoPub {
    fn adesso_pub(&self) -> Duration;
}
impl AdessoPub for ManualClock {
    fn adesso_pub(&self) -> Duration {
        use crate::slot::Clock;
        self.adesso()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn corri(da: u64, a: u64, eventi: usize) -> Statistiche {
        let mut tot = Statistiche::default();
        for seed in da..a {
            let mut s = Sim::new(seed);
            s.run(eventi);
            let x = s.stat;
            tot.chiuse += x.chiuse;
            tot.annullate += x.annullate;
            tot.fallite += x.fallite;
            tot.recoveries += x.recoveries;
            tot.dormienti += x.dormienti;
            tot.svegliate += x.svegliate;
            tot.token_registrati += x.token_registrati;
            tot.scartati_epoca += x.scartati_epoca;
            tot.scartati_filigrana += x.scartati_filigrana;
            tot.posti_scaduti += x.posti_scaduti;
            tot.messages += x.messages;
            tot.errori_conv += x.errori_conv;
            tot.sequenze_occupate += x.sequenze_occupate;
        }
        tot
    }

    #[test]
    fn rng_is_deterministic() {
        let (mut a, mut b) = (Rng::new(5), Rng::new(5));
        assert!((0..50).all(|_| a.next() == b.next()));
        assert_ne!(Rng::new(5).next(), Rng::new(6).next());
    }

    /// Include staggered admission in a run with delivery faults. Also check
    /// FIFO order, total live capacity, and exclusive slot ownership.
    #[test]
    fn simulation_with_arrivals_and_faults() {
        let semi: u64 = std::env::var("SEMI").ok().and_then(|v| v.parse().ok()).unwrap_or(1_500);
        let (mut tot, mut ammesse, mut con_coda) = (Statistiche::default(), 0usize, 0u32);
        for seed in 0..semi {
            let mut s = Sim::con_arrivi(seed);
            s.run(300);
            let am = s.amm.as_ref().unwrap();
            ammesse += am.ammesse.len();
            assert_eq!(am.ammesse.len(), am.arrivate.len(), "seed {seed}: arrived conversations were not admitted");
            if am.slots < am.arrivate.len() {
                con_coda += 1;
            }
            tot.chiuse += s.stat.chiuse;
            tot.annullate += s.stat.annullate;
            tot.fallite += s.stat.fallite;
            tot.recoveries += s.stat.recoveries;
            tot.token_registrati += s.stat.token_registrati;
        }
        eprintln!("arrivals: {tot:?}, admitted: {ammesse}, seeds with slot contention: {con_coda}");
        assert!(ammesse as u64 > semi * 3, "too few admitted conversations: {ammesse}");
        assert!(con_coda > 0, "test did not exercise slot contention");
        assert!(tot.recoveries > 0 && tot.token_registrati > 5_000, "{tot:?}");
    }

    #[test]
    fn randomized_simulation_with_2000_seeds() {
        // Use SEMI=n for more seeds (e.g. 50000). Default is 2000.
        let semi = std::env::var("SEMI").ok().and_then(|v| v.parse().ok()).unwrap_or(2_000);
        let t = corri(0, semi, 300);
        eprintln!("simulation: {t:?}");
        // The simulation must exercise the cases. A zero counter means the test did not cover that case.
        assert!(t.chiuse > 0 && t.annullate > 0 && t.fallite > 0, "{t:?}");
        assert!(t.recoveries > 0 && t.scartati_epoca > 0 && t.scartati_filigrana > 0 && t.posti_scaduti > 0, "{t:?}");
        assert!(t.token_registrati > 10_000, "{t:?}");
        assert!(t.errori_conv > 100, "test did not exercise enough single-conversation faults: {t:?}");
        assert!(t.sequenze_occupate > 0, "test did not exercise busy slot rejection: {t:?}");
        assert!(t.dormienti > 0 && t.svegliate > 0, "test did not exercise lazy recovery: {t:?}");
    }
}
