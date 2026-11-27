//! The control-stream fixture from proto/wire checks that both implementations
//! agree on framing and message decoding.

use bytes::BytesMut;
use publicompute_proto::{common, node, wire};

const GOLDEN: &[u8] = include_bytes!("../../testdata/control_stream.bin");

#[test]
fn decodes_go_golden_stream() {
    // Seven-byte reads force the decoder to wait for partial frames.
    let mut buf = BytesMut::new();
    let mut chunks = GOLDEN.chunks(7);
    let mut next = |buf: &mut BytesMut| -> bool {
        match chunks.next() {
            Some(c) => {
                buf.extend_from_slice(c);
                true
            }
            None => false,
        }
    };

    let hello: node::NodeMessage = loop {
        if let Some(m) = wire::try_decode_frame(&mut buf).unwrap() {
            break m;
        }
        assert!(next(&mut buf), "stream ended before Hello");
    };
    assert_eq!(hello.seq, 1);
    let Some(node::node_message::Body::Hello(h)) = hello.body else { panic!("expected Hello") };
    // The fixture represents an older peer and stays fixed across releases.
    assert_eq!(h.agent_version, "0.1.1");
    let hw = h.hardware.unwrap();
    assert!(hw.network.is_none());
    assert_eq!(hw.gpus[0].vram_bytes, 24 << 30);
    assert_eq!(hw.gpus[0].vendor(), node::GpuVendor::Nvidia);
    assert_eq!(hw.policy.unwrap().electricity_price_per_kwh.unwrap().micros, 300_000);

    let offer: node::CoordinatorMessage = loop {
        if let Some(m) = wire::try_decode_frame(&mut buf).unwrap() {
            break m;
        }
        assert!(next(&mut buf), "stream ended before WorkOffer");
    };
    let Some(node::coordinator_message::Body::WorkOffer(o)) = offer.body else { panic!("expected WorkOffer") };
    assert_eq!(o.lease.unwrap().lease_id, "lease-1");
    let Some(node::work_offer::Task::Batch(b)) = o.task else { panic!("expected BatchTask") };
    assert_eq!(b.requests[0].messages[0].content, "Ciao!");
    assert_eq!(b.requests[0].sampling.as_ref().unwrap().seed, 42);

    let mut read = || loop {
        if let Some(m) = wire::try_decode_frame::<node::CoordinatorMessage>(&mut buf).unwrap() {
            break m.body.unwrap();
        }
        assert!(next(&mut buf), "stream ended before parking fixture");
    };
    let node::coordinator_message::Body::StageAssign(assign) = read() else { panic!("expected StageAssign") };
    assert_eq!((assign.park_ttl_s, assign.park_grace_s), (900, 60));
    let node::coordinator_message::Body::WorkOffer(offer) = read() else { panic!("expected WorkOffer") };
    let Some(node::work_offer::Task::Chat(chat)) = offer.task else { panic!("expected ChatTask") };
    assert_eq!(chat.affinity_owner, [0x1f; 16]);
    assert!(chat.member_seat);
    assert!(buf.is_empty());
}

#[test]
fn round_trip_and_size_limit() {
    let msg = node::NodeMessage {
        seq: 7,
        body: Some(node::node_message::Body::Heartbeat(node::Heartbeat::default())),
    };
    let mut buf = BytesMut::new();
    wire::encode_frame(&msg, &mut buf).unwrap();
    let back: node::NodeMessage = wire::try_decode_frame(&mut buf).unwrap().unwrap();
    assert_eq!(back, msg);

    // Supply only the length to check that the limit applies before payload arrival.
    let mut big = BytesMut::new();
    prost::encoding::encode_varint(32 << 20, &mut big);
    assert!(matches!(
        wire::try_decode_frame::<node::NodeMessage>(&mut big),
        Err(wire::FrameError::TooLarge(_))
    ));
    let _ = publicompute_proto::PROTOCOL_VERSION;
    let _ = common::TrustTier::Attested;
}

#[test]
fn chain_messages_round_trip() {
    let assign = node::CoordinatorMessage {
        seq: 5,
        body: Some(node::coordinator_message::Body::StageAssign(node::StageAssign {
            chain_id: "catena-1".into(),
            stage_index: 2,
            n_stages: 3,
            model: Some(common::ModelRef {
                model_id: "qwen".into(),
                quantization: "IQ3_XXS".into(),
                weights_sha256: vec![0xcd; 32],
            }),
            layer_start: 16,
            layer_end: -1,
            slots: 4,
            ctx_per_slot: 8192,
            groups: 2,
            n_batch: 512,
            state_format: "fp16".into(),
            listen_addr: "0.0.0.0:7001".into(),
            next_addr: "192.0.2.3:7001".into(),
            head_token_addr: "192.0.2.1:7002".into(),
            capability: vec![1, 2, 3],
            assign_epoch: 9,
            park_ttl_s: 900,
            park_grace_s: 60,
            engine_profile: Some(node::StageEngineProfile {
                cache_type_k: "q8_0".into(),
                cache_type_v: "q8_0".into(),
                flash_attn: "on".into(),
                draft_source: "mtp".into(),
                draft_max_tokens: Some(1),
            }),
            ..Default::default()
        })),
    };
    let status = node::NodeMessage {
        seq: 6,
        body: Some(node::node_message::Body::StageStatus(node::StageStatus {
            chain_id: "catena-1".into(),
            assign_epoch: 9,
            state: node::stage_status::State::Ready as i32,
            error: String::new(),
            port: 7001,
            load_seconds: 12.5,
        })),
    };
    let mut buf = BytesMut::new();
    wire::encode_frame(&assign, &mut buf).unwrap();
    wire::encode_frame(&status, &mut buf).unwrap();
    let a: node::CoordinatorMessage = wire::try_decode_frame(&mut buf).unwrap().unwrap();
    let s: node::NodeMessage = wire::try_decode_frame(&mut buf).unwrap().unwrap();
    assert_eq!(a, assign);
    assert_eq!(s, status);

    // An explicit Some(0) must survive framing so receivers distinguish zero from an omitted field.
    let zero_profile = node::StageEngineProfile {
        cache_type_k: "f16".into(),
        cache_type_v: "f16".into(),
        flash_attn: "auto".into(),
        draft_source: "none".into(),
        draft_max_tokens: Some(0),
    };
    wire::encode_frame(&zero_profile, &mut buf).unwrap();
    let decoded: node::StageEngineProfile = wire::try_decode_frame(&mut buf).unwrap().unwrap();
    assert_eq!(decoded, zero_profile);
    assert_eq!(decoded.draft_max_tokens, Some(0));

    let omitted_depth = node::StageEngineProfile { draft_max_tokens: None, ..zero_profile };
    wire::encode_frame(&omitted_depth, &mut buf).unwrap();
    let decoded_omitted: node::StageEngineProfile = wire::try_decode_frame(&mut buf).unwrap().unwrap();
    assert_eq!(decoded_omitted.draft_max_tokens, None);
}

#[test]
fn parking_round_trip() {
    use publicompute_proto::{service, stage};
    use stage::message::Body;

    let frames = [
        Body::Open(stage::Open {
            conversation_id: 17, attempt_id: 1, slot: 3, slot_generation: 22, start_position: 48,
            reuse_slot: true, restore_checkpoint: true,
            expected_prev_conv_id: 16, expected_prev_attempt_id: 2, expected_prev_slot_generation: 21,
            ..Default::default()
        }),
        Body::Block(stage::PrefillBlock { slot: 3, slot_generation: 22, save_checkpoint: true, ..Default::default() }),
        Body::Close(stage::Close { slot: 3, slot_generation: 22, keep_parked: true, ..Default::default() }),
        Body::Confirmation(stage::Confirmation {
            kind: stage::AckKind::CacheMiss as i32, epoch: 7, chain_epoch: 237, slot: 3, slot_generation: 22, start_position: 48, ..Default::default()
        }),
        Body::Reset(stage::Reset { chain_epoch: 237 }),
    ];
    for body in frames {
        check_round_trip(stage::Message { body: Some(body) });
    }
    check_round_trip(service::Request { id: 16, owner_key: vec![1; 16], member_seat: true, ..Default::default() });
    check_round_trip(service::Usage { prompt_tokens: 55, generated_tokens: 10, cached_prompt_tokens: 48 });
    check_round_trip(node::Usage { prompt_tokens: 55, completion_tokens: 10, cached_prompt_tokens: 48 });
}

fn check_round_trip<T: prost::Message + Default + PartialEq>(message: T) {
    let mut bytes = BytesMut::new();
    wire::encode_frame(&message, &mut bytes).unwrap();
    assert_eq!(wire::try_decode_frame::<T>(&mut bytes).unwrap(), Some(message));
    assert!(bytes.is_empty());
}

/// A reduced Hello schema for checking protobuf handling of omitted and unknown fields.
#[derive(Clone, PartialEq, prost::Message)]
struct MinimalHello {
    #[prost(message, optional, tag = "1")]
    protocol_version: Option<common::ProtocolVersion>,
    #[prost(string, tag = "2")]
    agent_version: String,
    #[prost(string, tag = "6")]
    region: String,
    #[prost(string, tag = "8")]
    iroh_node_id: String,
}

#[test]
fn hello_ignores_unknown_fields_and_defaults_missing_fields() {
    use prost::Message;
    let hello = node::Hello {
        protocol_version: Some(publicompute_proto::PROTOCOL_VERSION),
        agent_version: env!("CARGO_PKG_VERSION").into(),
        hardware: Some(node::HardwareProfile { usable_ram_bytes: 12 << 30, models_free_disk_bytes: 300 << 30, ..Default::default() }),
        region: "EU-IT".into(),
        iroh_node_id: "ab".repeat(32),
        contribution: Some(node::ContributionPreference {
            mode: node::ContributionMode::Use as i32,
            model: Some(common::ModelRef { model_id: "qwen".into(), quantization: "Q4_K_M".into(), weights_sha256: vec![1; 32] }),
            revision: 99,
        }),
        features: vec![node::AgentFeature::Contribution as i32],
        ..Default::default()
    };
    let bytes = hello.encode_to_vec();
    // Decoding into the reduced schema must skip fields outside its definition.
    let minimal = MinimalHello::decode(bytes.as_slice()).expect("the reduced schema must decode Hello");
    assert_eq!(minimal.protocol_version.unwrap(), publicompute_proto::PROTOCOL_VERSION);
    assert_eq!(minimal.agent_version, env!("CARGO_PKG_VERSION"));
    assert_eq!(minimal.region, "EU-IT");
    assert_eq!(minimal.iroh_node_id.len(), 64);
    // Fields absent from the encoded message decode to their protobuf defaults.
    let minimal_message = MinimalHello { agent_version: env!("CARGO_PKG_VERSION").into(), ..Default::default() };
    let new = node::Hello::decode(minimal_message.encode_to_vec().as_slice()).unwrap();
    assert!(new.contribution.is_none());
    assert!(new.features.is_empty());
    // A missing version decodes as None.
    let welcome = node::Welcome::decode(&b""[..]).unwrap();
    assert!(welcome.protocol_version.is_none());
    assert!(!welcome.peer_probes);
}
