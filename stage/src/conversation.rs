//! The head owns the authoritative transcript. It records only the expected
//! logical step from the active attempt. Each stored token gets a sequence
//! number starting at one, which lets clients consume a contiguous stream
//! even when stages replay a prefix.
//!
//! Fault recovery closes the current attempt and opens a new one when the
//! chain is ready. Waiting for the chain and replaying the prefix each get
//! a fresh deadline. A fault between turns defers both recovery and its
//! timer until another turn arrives. Time is supplied by the caller.

use std::time::Duration;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Opening,
    Active,
    Waiting,
    Suspended,
    Rebuilding,
    Closed,
    Cancelled,
    Failed,
}

impl Status {
    pub fn terminale(self) -> bool {
        matches!(self, Status::Closed | Status::Cancelled | Status::Failed)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Config {
    /// Maximum rebuilds after faults. The next rebuild request fails the conversation.
    pub k_riprese: u32,
    /// Deadline interval for waiting on the chain and for rebuilding the prefix.
    pub t_ripresa: Duration,
}

impl Default for Config {
    fn default() -> Self {
        Self { k_riprese: 3, t_ripresa: Duration::from_secs(120) }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Motivo {
    RecoveryLimitReached,
    RecoveryTimedOut,
}

/// What the caller (the head) must do after a transition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    /// Open the given attempt, then prefill its confirmed prefix.
    Open { tentativo: u32, start_position: u32 },
    /// `Close` for attempt t to all stages.
    Close { tentativo: u32 },
    /// Explicit error to the client.
    ErroreClient(Motivo),
}

/// Event ignored, state unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rejection {
    /// The conversation is terminal, so no event can reopen it.
    Terminal(Status),
    /// The event is invalid in this state.
    WrongState(Status),
    /// Event for an attempt other than the current one.
    StaleAttempt,
}

pub type Outcome = Result<Vec<Action>, Rejection>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Registry {
    /// Increases from 1 without gaps.
    pub token_seq: u64,
    /// `conversation_step` = number of tokens recorded before this one (starts at 0).
    pub step: u32,
    pub tentativo: u32,
    pub token: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokenOutcome {
    Registrato { token_seq: u64 },
    /// Step already recorded. Ignore the duplicate.
    Duplicato,
    /// Inactive attempt (old, suspended, terminal) or a state that rejects tokens.
    Scartato,
    /// Step beyond the expected one. Do not record it.
    Inatteso { expected: u32 },
}

pub struct Conversation {
    id: u64,
    cfg: Config,
    state: Status,
    tentativo: u32,
    recoveries: u32,
    /// Prompt tokens, including user turns, excluding response tokens.
    prompt_len: u32,
    transcript: Vec<Registry>,
    /// Time of entry into the current state (for `T_ripresa`).
    da: Duration,
    /// Suspended while awaiting a user turn: wait for a new user turn before rebuilding (lazy recovery).
    attende_turno: bool,
    /// A preempted batch resumes through riprendi. It uses neither the fault
    /// recovery budget nor the fault recovery timer.
    volontaria: bool,
    /// Times the head chose to suspend the conversation.
    sospensioni: u32,
}

impl Conversation {
    /// Create the conversation on attempt one and return an open action.
    /// The caller sends the prompt after opening the slot.
    pub fn new(id: u64, prompt_len: u32, cfg: Config, adesso: Duration) -> (Self, Vec<Action>) {
        let c = Self {
            id,
            cfg,
            state: Status::Opening,
            tentativo: 1,
            recoveries: 0,
            prompt_len,
            transcript: Vec::new(),
            da: adesso,
            attende_turno: false,
            volontaria: false,
            sospensioni: 0,
        };
        (c, vec![Action::Open { tentativo: 1, start_position: 0 }])
    }

    pub fn id(&self) -> u64 {
        self.id
    }
    pub fn state(&self) -> Status {
        self.state
    }
    /// Current attempt (in `SOSPESA`: the last, already closed).
    pub fn tentativo(&self) -> u32 {
        self.tentativo
    }
    pub fn recoveries(&self) -> u32 {
        self.recoveries
    }
    /// `SOSPESA` waiting for the next user turn before rebuilding (lazy recovery).
    pub fn dormant(&self) -> bool {
        self.state == Status::Suspended && self.attende_turno
    }
    /// `SOSPESA` by the head's choice (see `suspend`).
    pub fn sospesa_volontaria(&self) -> bool {
        self.state == Status::Suspended && self.volontaria
    }
    pub fn sospensioni(&self) -> u32 {
        self.sospensioni
    }
    /// Drop transcript allocations after terminal service output has been sent.
    pub fn clear_transcript(&mut self) {
        if self.state.terminale() {
            self.transcript = Vec::new();
        }
    }
    pub fn transcript(&self) -> &[Registry] {
        &self.transcript
    }
    /// Confirmed input for a rebuild: all prompt and recorded response tokens.
    pub fn prefisso_len(&self) -> u32 {
        self.prompt_len + self.transcript.len() as u32
    }
    /// `conversation_step` of the next token to record.
    pub fn expected_step(&self) -> u32 {
        self.transcript.len() as u32
    }

    fn vietato(&self, ammessi: bool) -> Result<(), Rejection> {
        if self.state.terminale() {
            Err(Rejection::Terminal(self.state))
        } else if !ammessi {
            Err(Rejection::WrongState(self.state))
        } else {
            Ok(())
        }
    }

    fn accept_incoming(&mut self, state: Status, adesso: Duration) {
        self.state = state;
        self.da = adesso;
    }

    fn terminate(&mut self, state: Status, adesso: Duration, reason: Option<Motivo>) -> Vec<Action> {
        self.accept_incoming(state, adesso);
        let mut v = vec![Action::Close { tentativo: self.tentativo }];
        if let Some(m) = reason {
            v.push(Action::ErroreClient(m));
        }
        v
    }

    /// Record only the expected token of the active attempt. The first accepted
    /// token promotes an opening or rebuilding conversation to active.
    pub fn token(&mut self, tentativo: u32, step: u32, token: u32, adesso: Duration) -> TokenOutcome {
        let accept = matches!(self.state, Status::Opening | Status::Active | Status::Rebuilding);
        if !accept || tentativo != self.tentativo {
            return TokenOutcome::Scartato;
        }
        let expected = self.expected_step();
        if step < expected {
            return TokenOutcome::Duplicato;
        }
        if step > expected {
            return TokenOutcome::Inatteso { expected };
        }
        let token_seq = self.transcript.len() as u64 + 1;
        self.transcript.push(Registry { token_seq, step, tentativo, token });
        if self.state != Status::Active {
            self.accept_incoming(Status::Active, adesso);
        }
        TokenOutcome::Registrato { token_seq }
    }

    /// Finish this response while keeping context available for another turn.
    pub fn finish_response(&mut self, adesso: Duration) -> Outcome {
        self.vietato(self.state == Status::Active)?;
        self.accept_incoming(Status::Waiting, adesso);
        Ok(vec![])
    }

    /// Append a new turn. While awaiting a user turn, prefill it in the existing slot. If
    /// suspended between turns, wake the conversation and start its recovery
    /// timer. The next rebuild includes the appended turn.
    pub fn next_turn(&mut self, n_token: u32, adesso: Duration) -> Outcome {
        self.vietato(self.state == Status::Waiting || self.dormant())?;
        self.prompt_len += n_token;
        if self.state == Status::Suspended {
            self.attende_turno = false;
            self.da = adesso;
        } else {
            self.accept_incoming(Status::Active, adesso);
        }
        Ok(vec![])
    }

    /// Suspend on a fault in the current attempt. Tell the caller to close that
    /// attempt at every stage so late tokens cannot enter the transcript.
    pub fn fault(&mut self, tentativo: u32, adesso: Duration) -> Outcome {
        self.vietato(!matches!(self.state, Status::Suspended))?;
        if tentativo != self.tentativo {
            return Err(Rejection::StaleAttempt);
        }
        // There is no pending response between turns, so defer recovery until the next turn.
        self.attende_turno = self.state == Status::Waiting;
        self.accept_incoming(Status::Suspended, adesso);
        Ok(vec![Action::Close { tentativo: self.tentativo }])
    }

    /// Preempt a live batch to release its slot. Keep the confirmed transcript
    /// and the remaining fault recovery budget. Resume through riprendi.
    pub fn suspend(&mut self, adesso: Duration) -> Outcome {
        // Admission may be interrupted during prefill. Resuming in a new slot
        // requires replay from position zero.
        self.vietato(matches!(self.state, Status::Active | Status::Opening | Status::Rebuilding))?;
        self.volontaria = true;
        self.attende_turno = false;
        self.sospensioni += 1;
        self.accept_incoming(Status::Suspended, adesso);
        Ok(vec![Action::Close { tentativo: self.tentativo }])
    }

    /// Reopen a preempted conversation on the next attempt and replay its
    /// confirmed prefix without spending a fault recovery.
    pub fn resume(&mut self, adesso: Duration) -> Outcome {
        self.vietato(self.sospesa_volontaria())?;
        self.volontaria = false;
        self.tentativo += 1;
        self.accept_incoming(Status::Rebuilding, adesso);
        Ok(vec![Action::Open { tentativo: self.tentativo, start_position: 0 }])
    }

    /// Start another attempt once the chain is complete. Fail if the recovery
    /// limit is exhausted. A conversation suspended between turns must wake first.
    pub fn chain_ready(&mut self, adesso: Duration) -> Outcome {
        self.vietato(self.state == Status::Suspended && !self.attende_turno && !self.volontaria)?;
        self.recoveries += 1;
        if self.recoveries > self.cfg.k_riprese {
            return Ok(self.terminate(Status::Failed, adesso, Some(Motivo::RecoveryLimitReached)));
        }
        self.tentativo += 1;
        self.accept_incoming(Status::Rebuilding, adesso);
        Ok(vec![Action::Open { tentativo: self.tentativo, start_position: 0 }])
    }

    /// Fail an overdue wait or rebuild. Time spent asleep or preempted is exempt.
    pub fn tempo(&mut self, adesso: Duration) -> Outcome {
        self.vietato(matches!(self.state, Status::Suspended | Status::Rebuilding))?;
        if self.attende_turno || self.volontaria {
            return Ok(vec![]);
        }
        if adesso.saturating_sub(self.da) >= self.cfg.t_ripresa {
            return Ok(self.terminate(Status::Failed, adesso, Some(Motivo::RecoveryTimedOut)));
        }
        Ok(vec![])
    }

    /// User closes: any nonterminal state → closed, `Close` for attempt t to all stages.
    pub fn close_by_user(&mut self, adesso: Duration) -> Outcome {
        self.vietato(true)?;
        Ok(self.terminate(Status::Closed, adesso, None))
    }

    /// Cancel (client, coordinator): any nonterminal state → `ANNULLATA`. Unrecorded tokens are never
    /// in the transcript, so they incur no charge.
    pub fn cancel(&mut self, adesso: Duration) -> Outcome {
        self.vietato(true)?;
        Ok(self.terminate(Status::Cancelled, adesso, None))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    const S: Duration = Duration::from_secs(1);
    const T0: Duration = Duration::ZERO;

    fn new() -> Conversation {
        Conversation::new(1, 100, Config::default(), T0).0
    }

    fn attiva() -> Conversation {
        let mut c = new();
        assert_eq!(c.token(1, 0, 11, T0), TokenOutcome::Registrato { token_seq: 1 });
        c
    }

    #[test]
    fn voluntary_suspension_preserves_transcript_and_recovery_budget() {
        let mut c = attiva();
        let n0 = c.transcript().len();
        let az = c.suspend(S).unwrap();
        assert_eq!(az, vec![Action::Close { tentativo: 1 }]);
        assert!(c.sospesa_volontaria() && c.state() == Status::Suspended);
        // A voluntarily closed attempt can still have tokens in flight.
        assert_eq!(c.token(1, c.expected_step(), 5, S), TokenOutcome::Scartato);
        assert!(c.fault(1, S).is_err());
        assert!(c.chain_ready(S).is_err());
        // Does not expire.
        assert!(c.tempo(Duration::from_secs(10_000)).unwrap().is_empty());
        let az = c.resume(2 * S).unwrap();
        assert_eq!(az, vec![Action::Open { tentativo: 2, start_position: 0 }]);
        assert_eq!((c.state(), c.recoveries(), c.sospensioni(), c.tentativo()), (Status::Rebuilding, 0, 1, 2));
        assert_eq!(c.transcript().len(), n0);
        // First token after recovery continues without gaps.
        let expected = c.expected_step();
        assert!(matches!(c.token(2, expected, 7, 3 * S), TokenOutcome::Registrato { .. }));
        assert_eq!(c.state(), Status::Active);
        // Voluntary suspension can repeat without exhausting the fault recovery limit.
        for i in 0..10u32 {
            c.suspend(S * (4 + i)).unwrap();
            c.resume(S * (4 + i)).unwrap();
            let a = c.expected_step();
            assert!(matches!(c.token(c.tentativo(), a, 1, S), TokenOutcome::Registrato { .. }));
        }
        assert_eq!(c.recoveries(), 0);
        assert!(c.suspend(S).is_ok());
        // Reject riprendi when the head did not choose to suspend this conversation.
        let mut d = attiva();
        assert!(d.resume(S).is_err());
        d.fault(1, S).unwrap();
        assert!(d.resume(S).is_err());
        // Can suspend during prefill to interrupt admission. Refuse while awaiting a user turn or after termination.
        let mut e = new();
        assert!(e.suspend(S).is_ok() && e.resume(S).is_ok());
        assert!(in_attesa().suspend(S).is_err());
    }

    fn in_attesa() -> Conversation {
        let mut c = attiva();
        c.finish_response(T0).unwrap();
        c
    }

    fn sospesa() -> Conversation {
        let mut c = attiva();
        c.fault(1, T0).unwrap();
        c
    }

    fn rebuild() -> Conversation {
        let mut c = sospesa();
        c.chain_ready(T0).unwrap();
        c
    }

    // Transition coverage for the head state machine.

    #[test]
    fn assignment_opens_first_attempt() {
        let (c, az) = Conversation::new(9, 50, Config::default(), T0);
        assert_eq!(c.state(), Status::Opening);
        assert_eq!(c.tentativo(), 1);
        assert_eq!(az, vec![Action::Open { tentativo: 1, start_position: 0 }]);
        assert_eq!(c.prefisso_len(), 50);
    }

    #[test]
    fn first_token_activates_opening_attempt() {
        let mut c = new();
        assert_eq!(c.token(1, 0, 11, T0), TokenOutcome::Registrato { token_seq: 1 });
        assert_eq!(c.state(), Status::Active);
        assert_eq!(c.transcript()[0], Registry { token_seq: 1, step: 0, tentativo: 1, token: 11 });
    }

    #[test]
    fn finished_response_waits_for_next_turn() {
        let mut c = attiva();
        assert_eq!(c.finish_response(T0), Ok(vec![]));
        assert_eq!(c.state(), Status::Waiting);
    }

    #[test]
    fn new_turn_activates_waiting_conversation() {
        let mut c = in_attesa();
        assert_eq!(c.next_turn(7, T0), Ok(vec![]));
        assert_eq!(c.state(), Status::Active);
        assert_eq!(c.prefisso_len(), 100 + 7 + 1);
        // Same attempt: the turn uses the existing slot.
        assert_eq!(c.tentativo(), 1);
    }

    #[test]
    fn fault_suspends_conversation_and_closes_attempt() {
        for mut c in [new(), attiva(), in_attesa()] {
            assert_eq!(c.fault(1, 5 * S), Ok(vec![Action::Close { tentativo: 1 }]));
            assert_eq!(c.state(), Status::Suspended);
            // Drop tokens from the closed attempt.
            assert_eq!(c.token(1, c.expected_step(), 99, 6 * S), TokenOutcome::Scartato);
        }
    }

    #[test]
    fn chain_rebuild_uses_new_attempt() {
        let mut c = sospesa();
        let az = c.chain_ready(S).unwrap();
        assert_eq!(c.state(), Status::Rebuilding);
        assert_eq!(c.tentativo(), 2);
        assert_eq!(c.recoveries(), 1);
        assert_eq!(az, vec![Action::Open { tentativo: 2, start_position: 0 }]);
        // The confirmed prefix includes the token already recorded.
        assert_eq!(c.prefisso_len(), 101);
    }

    #[test]
    fn new_attempt_resumes_from_recorded_step() {
        let mut c = rebuild();
        // Drop a token from the superseded attempt.
        assert_eq!(c.token(1, 1, 5, T0), TokenOutcome::Scartato);
        // The step is the number of recorded tokens.
        assert_eq!(c.token(2, 1, 12, T0), TokenOutcome::Registrato { token_seq: 2 });
        assert_eq!(c.state(), Status::Active);
        assert_eq!(c.transcript()[1], Registry { token_seq: 2, step: 1, tentativo: 2, token: 12 });
    }

    #[test]
    fn recovery_limit_returns_explicit_failure() {
        let mut c = attiva();
        for _ in 0..3 {
            c.fault(c.tentativo(), T0).unwrap();
            c.chain_ready(T0).unwrap();
            assert_eq!(c.state(), Status::Rebuilding);
        }
        c.fault(c.tentativo(), T0).unwrap();
        let az = c.chain_ready(T0).unwrap();
        assert_eq!(c.state(), Status::Failed);
        assert_eq!(az, vec![Action::Close { tentativo: 4 }, Action::ErroreClient(Motivo::RecoveryLimitReached)]);
    }

    #[test]
    fn recovery_deadline_returns_failure() {
        let mut c = sospesa();
        assert_eq!(c.tempo(119 * S), Ok(vec![]));
        let az = c.tempo(120 * S).unwrap();
        assert_eq!(c.state(), Status::Failed);
        assert_eq!(az, vec![Action::Close { tentativo: 1 }, Action::ErroreClient(Motivo::RecoveryTimedOut)]);
        // Also in RICOSTRUZIONE, counting from entry.
        let mut c = sospesa();
        c.chain_ready(100 * S).unwrap();
        assert_eq!(c.tempo(219 * S), Ok(vec![]));
        assert!(c.tempo(220 * S).is_ok());
        assert_eq!(c.state(), Status::Failed);
    }

    #[test]
    fn user_can_close_any_nonterminal_conversation() {
        for mut c in [new(), attiva(), in_attesa(), sospesa(), rebuild()] {
            let t = c.tentativo();
            assert_eq!(c.close_by_user(S), Ok(vec![Action::Close { tentativo: t }]));
            assert_eq!(c.state(), Status::Closed);
        }
    }

    #[test]
    fn cancel_works_in_any_nonterminal_state() {
        for mut c in [new(), attiva(), in_attesa(), sospesa(), rebuild()] {
            let t = c.tentativo();
            assert_eq!(c.cancel(S), Ok(vec![Action::Close { tentativo: t }]));
            assert_eq!(c.state(), Status::Cancelled);
        }
        // Unrecorded tokens are absent, so the transcript is unchanged.
        let mut c = attiva();
        let before = c.transcript().to_vec();
        c.cancel(S).unwrap();
        assert_eq!(c.transcript(), &before[..]);
    }

    // Recovery deferred until the next turn.

    #[test]
    fn lazy_recovery_waits_after_fault_between_turns() {
        let mut c = in_attesa();
        assert_eq!(c.fault(1, 5 * S), Ok(vec![Action::Close { tentativo: 1 }]));
        assert_eq!((c.state(), c.dormant()), (Status::Suspended, true));
        // No recovery before a turn: reject chain completion and ignore timeout.
        assert_eq!(c.chain_ready(6 * S), Err(Rejection::WrongState(Status::Suspended)));
        assert_eq!(c.tempo(10_000 * S), Ok(vec![]));
        assert_eq!((c.state(), c.tentativo(), c.recoveries()), (Status::Suspended, 1, 0));
    }

    #[test]
    fn new_turn_wakes_dormant_conversation_then_rebuilds() {
        let mut c = in_attesa();
        c.fault(1, 5 * S).unwrap();
        assert_eq!(c.next_turn(7, 500 * S), Ok(vec![]));
        assert_eq!((c.state(), c.dormant()), (Status::Suspended, false));
        // T_ripresa starts on the new turn.
        assert_eq!(c.tempo(500 * S + 119 * S), Ok(vec![]));
        let az = c.chain_ready(501 * S).unwrap();
        assert_eq!(az, vec![Action::Open { tentativo: 2, start_position: 0 }]);
        assert_eq!((c.state(), c.recoveries()), (Status::Rebuilding, 1));
        // The confirmed prefix includes the prompt, recorded tokens and new turn.
        assert_eq!(c.prefisso_len(), 100 + 1 + 7);
        assert_eq!(c.token(2, 1, 12, 502 * S), TokenOutcome::Registrato { token_seq: 2 });
        assert_eq!(c.state(), Status::Active);
    }

    #[test]
    fn lazy_recovery_deadline_applies_after_new_turn() {
        let mut c = in_attesa();
        c.fault(1, 5 * S).unwrap();
        c.next_turn(3, 500 * S).unwrap();
        let az = c.tempo(500 * S + 120 * S).unwrap();
        assert_eq!(c.state(), Status::Failed);
        assert_eq!(az, vec![Action::Close { tentativo: 1 }, Action::ErroreClient(Motivo::RecoveryTimedOut)]);
    }

    #[test]
    fn lazy_recovery_only_between_turns() {
        // Fault in ATTIVA: no sleep, so recovery can start immediately and the turn is rejected.
        let mut c = sospesa();
        assert!(!c.dormant());
        assert_eq!(c.next_turn(3, T0), Err(Rejection::WrongState(Status::Suspended)));
        assert!(c.chain_ready(T0).is_ok());
        // Sleeping conversations can still be closed or canceled.
        for f in [Conversation::close_by_user, Conversation::cancel] {
            let mut c = in_attesa();
            c.fault(1, T0).unwrap();
            assert!(f(&mut c, S).is_ok() && c.state().terminale());
        }
        // Reject a second turn once the conversation is awake.
        let mut c = in_attesa();
        c.fault(1, T0).unwrap();
        c.next_turn(3, S).unwrap();
        assert_eq!(c.next_turn(3, S), Err(Rejection::WrongState(Status::Suspended)));
    }

    // Recovery faults and transcript ordering.

    #[test]
    fn fault_during_rebuild_suspends_again() {
        let mut c = rebuild();
        assert_eq!(c.fault(2, S), Ok(vec![Action::Close { tentativo: 2 }]));
        assert_eq!(c.state(), Status::Suspended);
        c.chain_ready(S).unwrap();
        assert_eq!(c.tentativo(), 3);
        assert_eq!(c.recoveries(), 2);
    }

    #[test]
    fn ignores_fault_from_old_attempt_or_suspended_conversation() {
        let mut c = rebuild();
        assert_eq!(c.fault(1, S), Err(Rejection::StaleAttempt));
        assert_eq!(c.state(), Status::Rebuilding);
        let mut c = sospesa();
        assert_eq!(c.fault(1, S), Err(Rejection::WrongState(Status::Suspended)));
    }

    #[test]
    fn transcript_has_no_duplicates_or_gaps() {
        let mut c = attiva();
        assert_eq!(c.token(1, 0, 11, T0), TokenOutcome::Duplicato);
        assert_eq!(c.token(1, 5, 3, T0), TokenOutcome::Inatteso { expected: 1 });
        assert_eq!(c.token(2, 1, 3, T0), TokenOutcome::Scartato);
        assert_eq!(c.token(0, 1, 3, T0), TokenOutcome::Scartato);
        assert_eq!(c.token(1, 1, 22, T0), TokenOutcome::Registrato { token_seq: 2 });
        assert_eq!(c.transcript().len(), 2);
        // Waiting for a user turn rejects tokens, including late arrivals.
        c.finish_response(T0).unwrap();
        assert_eq!(c.token(1, 2, 33, T0), TokenOutcome::Scartato);
        assert_eq!(c.transcript().len(), 2);
    }

    #[test]
    fn invalid_events_preserve_state() {
        let mut c = new();
        assert_eq!(c.finish_response(T0), Err(Rejection::WrongState(Status::Opening)));
        assert_eq!(c.next_turn(3, T0), Err(Rejection::WrongState(Status::Opening)));
        assert_eq!(c.chain_ready(T0), Err(Rejection::WrongState(Status::Opening)));
        assert_eq!(c.tempo(T0), Err(Rejection::WrongState(Status::Opening)));
        assert_eq!(c.state(), Status::Opening);
        let mut c = attiva();
        assert_eq!(c.next_turn(3, T0), Err(Rejection::WrongState(Status::Active)));
        assert_eq!(c.prefisso_len(), 101);
    }

    #[test]
    fn terminal_conversation_cannot_reopen() {
        let finali: Vec<Box<dyn Fn() -> Conversation>> = vec![
            Box::new(|| {
                let mut c = attiva();
                c.close_by_user(T0).unwrap();
                c
            }),
            Box::new(|| {
                let mut c = in_attesa();
                c.cancel(T0).unwrap();
                c
            }),
            Box::new(|| {
                let mut c = attiva();
                for _ in 0..4 {
                    c.fault(c.tentativo(), T0).unwrap();
                    c.chain_ready(T0).unwrap();
                    if c.state().terminale() {
                        break;
                    }
                }
                c
            }),
        ];
        for f in finali {
            let mut c = f();
            let state = c.state();
            assert!(state.terminale());
            let (t, n) = (c.tentativo(), c.transcript().len());
            let rif = Err(Rejection::Terminal(state));
            assert_eq!(c.finish_response(T0), rif);
            assert_eq!(c.next_turn(3, T0), rif);
            assert_eq!(c.fault(t, T0), rif);
            assert_eq!(c.chain_ready(T0), rif);
            assert_eq!(c.tempo(1000 * S), rif);
            assert_eq!(c.close_by_user(T0), rif);
            assert_eq!(c.cancel(T0), rif);
            for tent in [t, t + 1] {
                assert_eq!(c.token(tent, n as u32, 1, T0), TokenOutcome::Scartato);
            }
            assert_eq!((c.state(), c.tentativo(), c.transcript().len()), (state, t, n));
        }
    }
}
