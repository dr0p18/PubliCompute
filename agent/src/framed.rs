//! Length-prefixed protobuf messages let consecutive frames share one QUIC stream.

use anyhow::{bail, Result};
use bytes::BytesMut;
use prost::Message;
use publicompute_proto::wire;

pub struct FrameReader {
    recv: quinn::RecvStream,
    buf: BytesMut,
}

impl FrameReader {
    pub fn new(recv: quinn::RecvStream) -> Self {
        Self { recv, buf: BytesMut::with_capacity(16 << 10) }
    }

    /// Read the next frame. Return None at clean EOF and an error on a truncated frame.
    pub async fn read<M: Message + Default>(&mut self) -> Result<Option<M>> {
        let mut tmp = [0u8; 16 << 10];
        loop {
            if let Some(m) = wire::try_decode_frame(&mut self.buf)? {
                return Ok(Some(m));
            }
            match self.recv.read(&mut tmp).await? {
                Some(n) => self.buf.extend_from_slice(&tmp[..n]),
                None if self.buf.is_empty() => return Ok(None),
                None => bail!("stream closed mid-frame"),
            }
        }
    }
}

pub async fn write<M: Message>(send: &mut quinn::SendStream, m: &M) -> Result<()> {
    let mut buf = Vec::with_capacity(m.encoded_len() + 10);
    wire::encode_frame(m, &mut buf)?;
    send.write_all(&buf).await?;
    Ok(())
}
