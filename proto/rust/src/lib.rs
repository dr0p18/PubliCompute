//! Generated message types and length-delimited stream framing. Nested Rust
//! modules follow the protobuf package names. The re-exports give callers
//! a shorter path to the current message types.

pub mod wire;

#[allow(clippy::all)]
pub mod publicompute {
    pub mod common {
        pub mod v1 {
            include!(concat!(env!("OUT_DIR"), "/publicompute.common.v1.rs"));
        }
    }
    pub mod stage {
        pub mod v1 {
            include!(concat!(env!("OUT_DIR"), "/publicompute.stage.v1.rs"));
        }
        pub mod service {
            pub mod v1 {
                include!(concat!(env!("OUT_DIR"), "/publicompute.stage.service.v1.rs"));
            }
        }
    }
    pub mod node {
        pub mod v1 {
            include!(concat!(env!("OUT_DIR"), "/publicompute.node.v1.rs"));
        }
    }
}

pub use publicompute::common::v1 as common;
pub use publicompute::node::v1 as node;
pub use publicompute::stage::service::v1 as service;
pub use publicompute::stage::v1 as stage;

/// Version advertised when opening a control session.
pub const PROTOCOL_VERSION: common::ProtocolVersion = common::ProtocolVersion { major: 0, minor: 2 };
