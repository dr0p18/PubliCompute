//! Length-prefixed protobuf messages let consecutive frames share one QUIC stream.

use anyhow::{bail, Context, Result};
use bytes::BytesMut;
use prost::Message;
use publicompute_proto::wire;
use std::time::Duration;

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

pub struct ChatWriter {
    send: quinn::SendStream,
    finished: bool,
}

#[derive(Debug)]
pub struct ChatStreamError;

impl std::fmt::Display for ChatStreamError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("chat work stream failed")
    }
}

impl std::error::Error for ChatStreamError {}

impl ChatWriter {
    pub fn new(send: quinn::SendStream) -> Self {
        Self { send, finished: false }
    }

    pub async fn write<M: Message>(&mut self, message: &M) -> Result<()> {
        self.write_with_timeout(message, Duration::from_secs(30)).await
    }

    pub(crate) async fn write_with_timeout<M: Message>(&mut self, message: &M, timeout: Duration) -> Result<()> {
        tokio::time::timeout(timeout, write(&mut self.send, message))
            .await
            .context("chat frame write timed out")
            .and_then(|result| result)
            .context(ChatStreamError)
    }

    pub fn finish(&mut self) -> Result<()> {
        self.send.finish().context(ChatStreamError)?;
        self.finished = true;
        Ok(())
    }
}

impl Drop for ChatWriter {
    fn drop(&mut self) {
        if !self.finished {
            let _ = self.send.reset(1u32.into());
        }
    }
}
