//! Raw native bindings. The stage crate owns the safe wrappers and the
//! lifetimes of objects allocated through this interface.
#![allow(non_upper_case_globals, non_camel_case_types, non_snake_case, clippy::all)]
#![allow(unnecessary_transmutes)]

include!(concat!(env!("OUT_DIR"), "/llama_bindings.rs"));
