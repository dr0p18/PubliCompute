//! A stage evaluates a contiguous layer range in an inference chain. The
//! head schedules conversations and records tokens, intermediate stages
//! forward activations, and the tail samples. Engine-independent modules
//! implement the protocol and scheduling rules without native libraries.

pub mod admission;
pub mod draft;
pub mod sampling;
pub mod seed_check;
pub mod conversation;
pub mod experts;
pub mod slot;
pub mod safeguards;
pub mod network;
pub mod service;
#[cfg(test)]
mod simulator;
pub mod calibration;
pub mod timings;
pub mod tls;
pub mod output;

#[cfg(feature = "llama")]
pub mod engine;
#[cfg(feature = "llama")]
pub mod stage;
#[cfg(feature = "llama")]
pub mod head;
