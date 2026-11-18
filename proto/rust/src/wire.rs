//! A frame starts with a varint payload length followed by one protobuf
//! message. Keep the limit and framing in sync with proto/wire.

use bytes::{Buf, BufMut, BytesMut};
use prost::Message;

/// Limit on the declared payload, excluding the length prefix.
pub const MAX_FRAME_SIZE: usize = 16 << 20;

#[derive(Debug)]
pub enum FrameError {
    TooLarge(usize),
    BadLength,
    Decode(prost::DecodeError),
}

impl std::fmt::Display for FrameError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooLarge(n) => write!(f, "frame of {n} bytes exceeds the {MAX_FRAME_SIZE}-byte limit"),
            Self::BadLength => write!(f, "invalid length prefix"),
            Self::Decode(e) => write!(f, "could not decode message: {e}"),
        }
    }
}

impl std::error::Error for FrameError {}

/// Append a length-delimited message. Oversized payloads are rejected before
/// any bytes are written to buf.
pub fn encode_frame<M: Message>(msg: &M, buf: &mut impl BufMut) -> Result<(), FrameError> {
    let len = msg.encoded_len();
    if len > MAX_FRAME_SIZE {
        return Err(FrameError::TooLarge(len));
    }
    msg.encode_length_delimited(buf).expect("BufMut with unlimited capacity");
    Ok(())
}

/// Decode one frame from the front of buf. On Ok(None), keep the accumulated
/// bytes and call again after reading more. A complete frame is consumed
/// before its payload is decoded.
pub fn try_decode_frame<M: Message + Default>(buf: &mut BytesMut) -> Result<Option<M>, FrameError> {
    let mut peek = &buf[..];
    let len = match prost::encoding::decode_varint(&mut peek) {
        Ok(len) => len,
        // A uint64 varint can occupy up to ten bytes. A shorter failed read may
        // still become valid when the rest of the prefix arrives.
        Err(_) if buf.len() < 10 => return Ok(None),
        Err(_) => return Err(FrameError::BadLength),
    };
    let len = usize::try_from(len).map_err(|_| FrameError::BadLength)?;
    if len > MAX_FRAME_SIZE {
        return Err(FrameError::TooLarge(len));
    }
    let header = buf.len() - peek.len();
    if peek.len() < len {
        buf.reserve(header + len - buf.len());
        return Ok(None);
    }
    buf.advance(header);
    let frame = buf.split_to(len).freeze();
    M::decode(frame).map(Some).map_err(FrameError::Decode)
}
