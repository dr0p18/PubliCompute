use std::{fmt, sync::Arc};
use tokio::sync::{mpsc, watch, OwnedSemaphorePermit, Semaphore};

pub trait OutputSize {
    fn bytes(&self) -> usize;
}

#[derive(Debug)]
pub struct OutputStalled;

impl fmt::Display for OutputStalled {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("chat output queue is full or closed")
    }
}

impl std::error::Error for OutputStalled {}

#[derive(Debug)]
pub struct Queued<T> {
    pub item: T,
    _permit: OwnedSemaphorePermit,
}

pub struct OutputSender<T> {
    items: mpsc::Sender<Queued<T>>,
    budget: Arc<Semaphore>,
    saturated: watch::Sender<bool>,
}

impl<T: OutputSize> OutputSender<T> {
    pub fn fail(&self) {
        self.saturated.send_replace(true);
    }

    pub fn send(&self, item: T) -> Result<(), OutputStalled> {
        if *self.saturated.borrow() || self.items.is_closed() {
            return Err(OutputStalled);
        }
        let permit = u32::try_from(item.bytes().max(1)).ok().and_then(|bytes| self.budget.clone().try_acquire_many_owned(bytes).ok());
        if let Some(permit) = permit {
            if self.items.try_send(Queued { item, _permit: permit }).is_ok() {
                return Ok(());
            }
        }
        self.fail();
        Err(OutputStalled)
    }
}

pub fn channel<T>(items: usize, bytes: usize) -> (OutputSender<T>, mpsc::Receiver<Queued<T>>, watch::Receiver<bool>) {
    let (tx, rx) = mpsc::channel(items);
    let (saturated, status) = watch::channel(false);
    (OutputSender { items: tx, budget: Arc::new(Semaphore::new(bytes)), saturated }, rx, status)
}

pub async fn wait_for_saturation(status: &mut watch::Receiver<bool>) {
    if status.wait_for(|value| *value).await.is_err() {
        std::future::pending::<()>().await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    impl OutputSize for usize {
        fn bytes(&self) -> usize {
            *self
        }
    }

    #[tokio::test]
    async fn output_queue_counts_bytes_during_write() {
        let (send, mut recv, mut full) = channel(4, 10);
        send.send(8).unwrap();
        let in_flight = recv.recv().await.unwrap();
        assert!(send.send(3).is_err());
        wait_for_saturation(&mut full).await;
        drop(in_flight);
        assert!(send.send(1).is_err());
    }

    #[test]
    fn output_queue_counts_items_even_when_small() {
        let (send, _recv, full) = channel(2, 1024);
        send.send(1).unwrap();
        send.send(1).unwrap();
        assert!(send.send(1).is_err());
        assert!(*full.borrow());
    }
}
