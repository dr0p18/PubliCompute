use std::{
    fmt,
    io::SeekFrom,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::Duration,
};

use anyhow::{bail, ensure, Context, Result};
use publicompute_proto::node::{InstalledModel, ModelStatus};
use reqwest::{header, StatusCode};
use serde::Serialize;
use tokio::{
    io::{AsyncSeekExt, AsyncWriteExt},
    sync::watch,
    time::Instant,
};

use crate::{
    contribution::Contribution,
    control::Live,
    models::{self, Model, Store},
};

const DRAIN_TIMEOUT: Duration = Duration::from_secs(600);
const RETRY_SECONDS: [u64; 4] = [10, 30, 60, 300];

#[derive(Debug)]
struct RetryDownload;

impl fmt::Display for RetryDownload {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("transient download error")
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Phase {
    Downloading,
    Verifying,
    Switching(Instant),
    Failed,
    Committing,
    Committed,
}

#[derive(Clone, Debug)]
pub struct Change {
    ticket: u64,
    model: &'static Model,
    phase: Phase,
    bytes: u64,
    error: Option<String>,
}

#[derive(Serialize, Clone, PartialEq)]
pub struct DownloadView {
    id: String,
    state: &'static str,
    bytes: u64,
    total_bytes: u64,
    error: Option<String>,
}

pub struct Changes {
    state: watch::Sender<Option<Change>>,
    next: AtomicU64,
    // Serialize claims with local commands that persist a preference or pause.
    pub commands: Arc<tokio::sync::Mutex<()>>,
}

impl Default for Changes {
    fn default() -> Self {
        Self {
            state: watch::channel(None).0,
            next: AtomicU64::new(0),
            commands: Default::default(),
        }
    }
}

impl Change {
    pub fn deadline(&self) -> Option<Instant> {
        match self.phase {
            Phase::Switching(deadline) => Some(deadline),
            _ => None,
        }
    }
}

impl Changes {
    pub fn subscribe(&self) -> watch::Receiver<Option<Change>> {
        self.state.subscribe()
    }

    pub fn switching(&self) -> Option<Change> {
        self.state.borrow().as_ref().filter(|c| c.deadline().is_some()).cloned()
    }

    pub async fn wait_switch(&self) -> Change {
        let mut state = self.subscribe();
        let state = state
            .wait_for(|s| s.as_ref().is_some_and(|c| c.deadline().is_some()))
            .await
            .expect("model changes sender lives with the receiver");
        state.as_ref().unwrap().clone()
    }

    pub async fn wait_deadline(&self) -> Change {
        let mut changes = self.subscribe();
        loop {
            let pending = changes.borrow_and_update().clone();
            tokio::select! {
                biased;
                _ = changes.changed() => {}
                change = async {
                    if let Some((change, deadline)) = pending.and_then(|c| c.deadline().map(|d| (c, d))) {
                        tokio::time::sleep_until(deadline).await;
                        change
                    } else {
                        std::future::pending().await
                    }
                } => return change,
            }
        }
    }

    #[cfg(test)]
    pub fn expire_switch(&self) {
        let change = self.switching().expect("expected a switch to drain");
        self.update(change.ticket, |c| {
            c.phase = Phase::Switching(Instant::now());
            true
        });
    }

    pub fn committed(&self) -> bool {
        self.state
            .borrow()
            .as_ref()
            .is_some_and(|c| c.phase == Phase::Committed)
    }

    pub fn view(&self) -> Option<DownloadView> {
        self.state.borrow().as_ref().map(|c| DownloadView {
            id: c.model.id.clone(),
            state: match c.phase {
                Phase::Downloading => "downloading",
                Phase::Verifying => "verifying",
                Phase::Failed => "failed",
                _ => "switching",
            },
            bytes: c.bytes,
            total_bytes: c.model.bytes,
            error: c.error.clone(),
        })
    }

    pub fn inventory(&self) -> Option<InstalledModel> {
        let state = self.state.borrow();
        let change = state.as_ref()?;
        let status = match change.phase {
            Phase::Downloading => ModelStatus::Downloading,
            Phase::Failed => ModelStatus::Failed,
            _ => ModelStatus::Verifying,
        };
        Some(InstalledModel {
            model: Some(change.model.reference()),
            status: status as i32,
            size_bytes: change.model.bytes,
            downloaded_bytes: change.bytes,
            ..Default::default()
        })
    }

    // Local callers hold commands until their write and cancellation are both complete.
    pub fn can_cancel(&self) -> Result<bool> {
        match self.state.borrow().as_ref().map(|c| c.phase) {
            Some(Phase::Committing | Phase::Committed) => bail!("the agent is restarting to change models"),
            Some(Phase::Failed) | None => Ok(false),
            Some(_) => Ok(true),
        }
    }

    pub fn cancel(&self) {
        self.state.send_if_modified(|state| {
            if state.as_ref().is_some_and(|c| !matches!(c.phase, Phase::Failed | Phase::Committing | Phase::Committed)) {
                *state = None;
                return true;
            }
            false
        });
    }

    fn current(&self, ticket: u64) -> bool {
        self.state.borrow().as_ref().is_some_and(|c| c.ticket == ticket)
    }

    fn update(&self, ticket: u64, change: impl FnOnce(&mut Change) -> bool) -> bool {
        self.state
            .send_if_modified(|state| state.as_mut().filter(|c| c.ticket == ticket).is_some_and(change))
    }

    pub fn request(
        &self,
        live: &Arc<Live>,
        store: &Arc<Store>,
        model: &'static Model,
        reply: Option<tokio::sync::oneshot::Receiver<()>>,
    ) -> Result<(bool, u64)> {
        let (ticket, installed, missing) = self.begin(live, store, model)?;
        let (live, store) = (live.clone(), store.clone());
        tokio::spawn(async move {
            // A switch of already verified weights can otherwise close the socket first.
            if let Some(reply) = reply {
                let _ = reply.await;
            }
            prepare(live, store, ticket, model, models::download_client).await;
        });
        Ok((installed, missing))
    }

    fn begin(&self, live: &Live, store: &Store, model: &'static Model) -> Result<(u64, bool, u64)> {
        let mut missing = model.bytes;
        let plan = (|| -> Result<bool> {
            let Some(bytes) = store.download_size(model)? else {
                missing = 0;
                return Ok(true);
            };
            missing = bytes;
            store.space(missing)?;
            Ok(false)
        })();
        let ticket = self.next.fetch_add(1, Ordering::Relaxed);
        let change = Change {
            ticket,
            model,
            bytes: model.bytes - missing,
            phase: if plan.is_err() {
                Phase::Failed
            } else if missing == 0 {
                Phase::Verifying
            } else {
                Phase::Downloading
            },
            error: plan.as_ref().err().map(|err| format!("{err:#}")),
        };
        ensure!(
            self.state.send_if_modified(|state| {
                if state
                    .as_ref()
                    .is_some_and(|c| matches!(c.phase, Phase::Committing | Phase::Committed))
                {
                    return false;
                }
                *state = Some(change);
                true
            }),
            "the agent is restarting to change models"
        );
        live.notify();
        Ok((ticket, plan?, missing))
    }

    pub async fn claim(&self, change: &Change) -> bool {
        let _lock = self.commands.lock().await;
        self.update(change.ticket, |state| {
            if state.deadline().is_none() {
                return false;
            }
            state.phase = Phase::Committing;
            true
        })
    }

    pub async fn commit(&self, change: Change, store: Arc<Store>, contribution: &Contribution) -> bool {
        let revision = Some(contribution.preference().revision);
        let result = tokio::task::spawn_blocking(move || store.save_active(change.model, revision))
            .await
            .unwrap_or_else(|err| Err(err.into()));
        match result {
            Ok(()) => {
                self.update(change.ticket, |c| {
                    c.phase = Phase::Committed;
                    true
                });
                true
            }
            Err(err) => {
                self.fail(change.ticket, err);
                false
            }
        }
    }

    fn fail(&self, ticket: u64, err: anyhow::Error) -> bool {
        self.update(ticket, |c| {
            tracing::warn!(model = %c.model.id, error = %format_args!("{err:#}"), "model change failed");
            c.phase = Phase::Failed;
            c.error = Some(format!("{err:#}"));
            true
        })
    }
}

async fn prepare(
    live: Arc<Live>,
    store: Arc<Store>,
    ticket: u64,
    model: &'static Model,
    client: impl Fn() -> Result<reqwest::Client>,
) {
    // Wait for the previous worker to close its file before inspecting the partial again.
    let _worker = store.worker.lock().await;
    let mut state = live.changes.subscribe();
    let mut retry = 0;
    let result = loop {
        if !live.changes.current(ticket) {
            return;
        }
        let result = fetch_weights(&live, &store, ticket, model, &client).await;
        let Err(err) = &result else { break result };
        if !err.is::<RetryDownload>() {
            break result;
        }
        let delay = RETRY_SECONDS[retry];
        retry = (retry + 1).min(RETRY_SECONDS.len() - 1);
        if !live.changes.update(ticket, |c| {
            c.error = Some(format!("retrying in {delay} s: {err:#}"));
            true
        }) {
            return;
        }
        tracing::warn!(model = %model.id, error = %format_args!("{err:#}"), retry_in_seconds = delay, "will retry model download");
        live.notify();
        tokio::select! {
            _ = state.wait_for(|s| s.as_ref().is_none_or(|c| c.ticket != ticket)) => return,
            _ = tokio::time::sleep(Duration::from_secs(delay)) => {}
        }
    };
    let changed = match result {
        Ok(()) => live.changes.update(ticket, |c| {
            c.bytes = model.bytes;
            c.error = None;
            c.phase = Phase::Switching(Instant::now() + DRAIN_TIMEOUT);
            true
        }),
        Err(err) => live.changes.fail(ticket, err),
    };
    if changed {
        live.notify();
    }
}

async fn fetch_weights(
    live: &Arc<Live>,
    store: &Arc<Store>,
    ticket: u64,
    model: &'static Model,
    client: &impl Fn() -> Result<reqwest::Client>,
) -> Result<()> {
    if store.download_candidate(model)? {
        let (live, store) = (live.clone(), store.clone());
        let result = tokio::task::spawn_blocking(move || store.verify(model, || live.changes.current(ticket))).await?;
        if result.is_ok() {
            return Ok(());
        }
    }
    let mut state = live.changes.subscribe();
    let replaced = state.wait_for(|s| s.as_ref().is_none_or(|c| c.ticket != ticket));
    tokio::pin!(replaced);
    ensure!(live.changes.current(ticket), "model request replaced");
    let mut part = tokio::fs::File::from_std(models::open_file(&store.partial(model), true)?);
    let mut offset = part.metadata().await?.len();
    if offset > model.bytes {
        part.set_len(0).await?;
        offset = 0;
    }
    if offset < model.bytes {
        store.space(model.bytes - offset)?;
        report(live, ticket, Phase::Downloading, offset);
        let client = client()?;
        let url = reqwest::Url::parse(&model.url)?;
        ensure!(models::allowed_url(&url), "model URL not allowed");
        let mut request = client.get(url).header(header::ACCEPT_ENCODING, "identity");
        if offset > 0 {
            request = request.header(header::RANGE, format!("bytes={offset}-"));
        }
        let mut response = tokio::select! {
            biased;
            _ = &mut replaced => bail!("model request replaced"),
            response = request.send() => response.map_err(|err| request_error(err, "request model weights"))?,
        };
        let start = response_start(response.status(), response.headers(), offset, model.bytes)?;
        if start == 0 && offset > 0 {
            // A server can ignore Range. The old prefix is no longer part of this response.
            part.set_len(0).await?;
            store.space(model.bytes)?;
            offset = 0;
            report(live, ticket, Phase::Downloading, offset);
        }
        part.seek(SeekFrom::Start(start)).await?;
        loop {
            let chunk = tokio::select! {
                biased;
                _ = &mut replaced => bail!("model request replaced"),
                chunk = response.chunk() => chunk.map_err(|err| request_error(err, "read model weights"))?,
            };
            let Some(chunk) = chunk else { break };
            let next = offset.saturating_add(chunk.len() as u64);
            ensure!(next <= model.bytes, "model response exceeds the expected size");
            part.write_all(&chunk).await.context("write model partial")?;
            // Complete the write before exposing its size or handing the file to another worker.
            part.flush().await?;
            offset = next;
            report(live, ticket, Phase::Downloading, offset);
        }
    }
    if offset != model.bytes {
        return Err(anyhow::anyhow!("incomplete model download ({offset}/{} bytes)", model.bytes).context(RetryDownload));
    }
    part.sync_all().await?;
    drop(part);
    report(live, ticket, Phase::Verifying, offset);
    let (live, store) = (live.clone(), store.clone());
    tokio::task::spawn_blocking(move || store.promote(model, || live.changes.current(ticket))).await?
}

fn request_error(err: reqwest::Error, action: &'static str) -> anyhow::Error {
    let transport = err.is_timeout() || err.is_request() || err.is_body() || err.is_decode() || err.is_connect();
    let err = anyhow::Error::new(err.without_url()).context(action);
    let tls = err.chain().any(|cause| {
        cause.is::<rustls::Error>() || cause.downcast_ref::<std::io::Error>()
            .and_then(std::io::Error::get_ref).is_some_and(|err| err.is::<rustls::Error>())
    });
    if transport && !tls {
        err.context(RetryDownload)
    } else {
        err
    }
}

fn report(live: &Live, ticket: u64, phase: Phase, bytes: u64) {
    // Wake the network and status followers at coarse intervals during a fast transfer.
    let changed = live.changes.update(ticket, |c| {
        let changed = c.error.take().is_some() || c.phase != phase || c.bytes >> 24 != bytes >> 24 || bytes == c.model.bytes;
        c.bytes = bytes;
        c.phase = phase;
        changed
    });
    if changed {
        live.notify();
    }
}

fn response_start(status: StatusCode, headers: &header::HeaderMap, offset: u64, total: u64) -> Result<u64> {
    let start = match status {
        StatusCode::OK => 0,
        StatusCode::PARTIAL_CONTENT if offset > 0 && offset < total => {
            let expected = format!("bytes {offset}-{}/{total}", total - 1);
            ensure!(
                headers.get(header::CONTENT_RANGE).and_then(|v| v.to_str().ok()) == Some(&expected),
                "invalid Content-Range for the model download"
            );
            offset
        }
        StatusCode::RANGE_NOT_SATISFIABLE if offset == total => return Ok(total),
        _ => {
            let err = anyhow::anyhow!("unexpected model download HTTP status {status}");
            return Err(if status.is_server_error() || matches!(status, StatusCode::REQUEST_TIMEOUT | StatusCode::TOO_MANY_REQUESTS) {
                err.context(RetryDownload)
            } else {
                err
            });
        }
    };
    if let Some(encoding) = headers.get(header::CONTENT_ENCODING) {
        ensure!(encoding == "identity", "unexpected content encoding for model weights");
    }
    if let Some(length) = headers.get(header::CONTENT_LENGTH) {
        let length: u64 = length.to_str()?.parse()?;
        ensure!(length == total - start, "incorrect model response length");
    }
    Ok(start)
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::models::tests::{fixture_model, fixture_store};
    use std::fs;
    use tokio::{
        io::{AsyncBufReadExt, BufReader},
        net::TcpListener,
        task::JoinHandle,
        time::timeout,
    };

    const WEIGHTS: &[u8] = b"abcdefgh";

    async fn https_fixture(
        replies: &[&'static str],
        stall: bool,
    ) -> (reqwest::Client, String, watch::Receiver<String>, JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let cert = rcgen::generate_simple_self_signed(vec!["huggingface.co".into()]).unwrap();
        let key = rustls::pki_types::PrivatePkcs8KeyDer::from(cert.signing_key.serialize_der());
        let tls = rustls::ServerConfig::builder_with_provider(Arc::new(rustls::crypto::ring::default_provider()))
            .with_safe_default_protocol_versions()
            .unwrap()
            .with_no_client_auth()
            .with_single_cert(vec![cert.cert.der().clone()], key.into())
            .unwrap();
        let mut roots = rustls::RootCertStore::empty();
        roots.add(cert.cert.der().clone()).unwrap();
        let client = models::download_client_builder(roots)
            .unwrap()
            .no_proxy()
            .resolve("huggingface.co", addr)
            .build()
            .unwrap();
        let (headers, received) = watch::channel(String::new());
        let replies = replies.to_vec();
        let task = tokio::spawn(async move {
            let acceptor = tokio_rustls::TlsAcceptor::from(Arc::new(tls));
            for reply in replies {
                let (stream, _) = listener.accept().await.unwrap();
                let mut stream = BufReader::new(acceptor.accept(stream).await.unwrap());
                let mut request = String::new();
                loop {
                    let end = request.len();
                    if stream.read_line(&mut request).await.unwrap() == 0 || &request[end..] == "\r\n" {
                        break;
                    }
                }
                headers.send_replace(request);
                stream.get_mut().write_all(reply.as_bytes()).await.unwrap();
                stream.get_mut().flush().await.unwrap();
                if stall {
                    std::future::pending::<()>().await;
                }
                stream.get_mut().shutdown().await.unwrap();
            }
        });
        (
            client,
            format!("https://huggingface.co:{}/weights", addr.port()),
            received,
            task,
        )
    }

    async fn download(
        name: &str,
        prefix: &[u8],
        response: &'static str,
    ) -> (Arc<Live>, Arc<Store>, &'static Model, String) {
        let (client, url, received, server) = https_fixture(&[response], false).await;
        let mut model = fixture_model(name, WEIGHTS);
        model.url = url;
        let model = Box::leak(Box::new(model));
        let store = fixture_store(name);
        fs::write(store.partial(model), prefix).unwrap();
        let live = Live::new();
        let (ticket, _, _) = live.changes.begin(&live, &store, model).unwrap();
        timeout(
            Duration::from_secs(5),
            prepare(live.clone(), store.clone(), ticket, model, || Ok(client.clone())),
        )
        .await
        .unwrap();
        server.await.unwrap();
        let request = received.borrow().to_ascii_lowercase();
        (live, store, model, request)
    }

    #[tokio::test]
    async fn resumed_download_checks_range_and_hash_before_installing() {
        let (live, store, model, request) = download("https-resume", b"abc",
            "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes 3-7/8\r\nContent-Length: 5\r\nConnection: close\r\n\r\ndefgh").await;
        assert!(request.contains("range: bytes=3-\r\n"));
        assert!(live.changes.switching().is_some());
        assert_eq!(fs::read(store.path(model)).unwrap(), WEIGHTS);
        assert!(!store.partial(model).exists());
    }

    #[tokio::test]
    async fn interrupted_download_retries_from_the_saved_offset() {
        let (client, url, received, server) = https_fixture(&[
            "HTTP/1.1 200 OK\r\nContent-Length: 8\r\nConnection: close\r\n\r\nabc",
            "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes 3-7/8\r\nContent-Length: 5\r\nConnection: close\r\n\r\ndefgh",
        ], false).await;
        let mut model = fixture_model("retry", WEIGHTS);
        model.url = url;
        let model = Box::leak(Box::new(model));
        let store = fixture_store("https-retry");
        let live = Live::new();
        let mut state = live.changes.subscribe();
        let (ticket, _, _) = live.changes.begin(&live, &store, model).unwrap();
        let worker = tokio::spawn(prepare(live.clone(), store.clone(), ticket, model, move || Ok(client.clone())));
        timeout(Duration::from_secs(5), state.wait_for(|state| {
            state.as_ref().is_some_and(|change| change.error.is_some())
        })).await.unwrap().unwrap();
        let status = live.changes.view().unwrap();
        assert_eq!(status.state, "downloading");
        assert!(status.error.unwrap().contains("retrying in 10 s"));
        assert_eq!(fs::read(store.partial(model)).unwrap(), b"abc");

        timeout(Duration::from_secs(15), worker).await.unwrap().unwrap();
        server.await.unwrap();
        assert!(received.borrow().to_ascii_lowercase().contains("range: bytes=3-\r\n"));
        assert!(live.changes.switching().is_some());
        assert!(live.changes.view().unwrap().error.is_none());
        assert_eq!(fs::read(store.path(model)).unwrap(), WEIGHTS);
        assert!(!store.partial(model).exists());
    }

    #[tokio::test]
    async fn incorrect_partial_response_keeps_the_original_prefix() {
        let (live, store, model, _) = download("https-wrong-range", b"abc",
            "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes 2-7/8\r\nContent-Length: 5\r\nConnection: close\r\n\r\ndefgh").await;
        let status = live.changes.view().unwrap();
        assert_eq!(status.state, "failed");
        assert!(status.error.unwrap().contains("Content-Range"));
        assert_eq!(fs::read(store.partial(model)).unwrap(), b"abc");
        assert!(!store.path(model).exists());
    }

    #[tokio::test]
    async fn full_response_replaces_a_partial_when_the_server_ignores_range() {
        let (live, store, model, request) = download(
            "https-full",
            b"xxx",
            "HTTP/1.1 200 OK\r\nContent-Length: 8\r\nConnection: close\r\n\r\nabcdefgh",
        )
        .await;
        assert!(request.contains("range: bytes=3-\r\n"));
        assert!(live.changes.switching().is_some());
        assert_eq!(fs::read(store.path(model)).unwrap(), WEIGHTS);
    }

    #[tokio::test]
    async fn corrupt_download_is_removed_and_reported() {
        let (live, store, model, _) = download(
            "https-corrupt",
            b"",
            "HTTP/1.1 200 OK\r\nContent-Length: 8\r\nConnection: close\r\n\r\n12345678",
        )
        .await;
        let status = live.changes.view().unwrap();
        assert_eq!(status.state, "failed");
        assert!(status.error.unwrap().contains("SHA-256 mismatch"));
        assert!(!store.partial(model).exists());
        assert!(!store.path(model).exists());
    }

    #[tokio::test]
    async fn rejects_redirect_to_another_host() {
        let (live, store, model, _) = download("https-bad-redirect", b"abc",
            "HTTP/1.1 302 Found\r\nLocation: https://huggingface.co.invalid/weights\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").await;
        let status = live.changes.view().unwrap();
        assert_eq!(status.state, "failed");
        assert!(status.error.unwrap().contains("model redirect not allowed"));
        assert_eq!(fs::read(store.partial(model)).unwrap(), b"abc");
    }

    #[test]
    fn redirects_require_https_and_a_weight_host() {
        for url in [
            "https://huggingface.co/x",
            "https://hub.huggingface.co/x",
            "https://cdn.x.hf.co/x",
        ] {
            assert!(models::allowed_url(&reqwest::Url::parse(url).unwrap()), "{url}");
        }
        for url in [
            "http://huggingface.co/x",
            "https://hf.co/x",
            "https://evil-huggingface.co/x",
            "https://huggingface.co.evil/x",
        ] {
            assert!(!models::allowed_url(&reqwest::Url::parse(url).unwrap()), "{url}");
        }
    }

    #[tokio::test]
    async fn response_cannot_append_more_than_the_manifest_size() {
        let (live, store, model, _) = download("https-oversize", b"",
            "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n8\r\nabcdefgh\r\n1\r\ni\r\n0\r\n\r\n").await;
        assert_eq!(live.changes.view().unwrap().state, "failed");
        assert_eq!(fs::read(store.partial(model)).unwrap(), WEIGHTS);
        assert!(!store.path(model).exists());
    }

    #[tokio::test]
    async fn full_partial_is_verified_without_network() {
        let headers = header::HeaderMap::new();
        let status = StatusCode::RANGE_NOT_SATISFIABLE;
        assert_eq!(response_start(status, &headers, 8, 8).unwrap(), 8);
        assert!(response_start(status, &headers, 7, 8).is_err());

        let store = fixture_store("full-partial");
        let model = Box::leak(Box::new(fixture_model("full", WEIGHTS)));
        fs::write(store.partial(model), WEIGHTS).unwrap();
        let live = Live::new();
        let (ticket, _, missing) = live.changes.begin(&live, &store, model).unwrap();
        assert_eq!(missing, 0);
        prepare(live.clone(), store.clone(), ticket, model, || {
            panic!("full partial contacted the network")
        })
        .await;
        assert!(live.changes.switching().is_some());
        assert_eq!(fs::read(store.path(model)).unwrap(), WEIGHTS);
    }

    #[tokio::test]
    async fn rejected_final_weights_can_be_downloaded_and_replaced() {
        use std::os::unix::fs::symlink;

        for is_link in [false, true] {
            let name = if is_link { "replace-symlink" } else { "replace-bad-weights" };
            let store = fixture_store(name);
            let (client, url, _, server) = https_fixture(
                &["HTTP/1.1 200 OK\r\nContent-Length: 8\r\nConnection: close\r\n\r\nabcdefgh"],
                false,
            ).await;
            let mut model = fixture_model(name, WEIGHTS);
            model.url = url;
            let model = Box::leak(Box::new(model));
            let path = store.path(model);
            let target = path.with_extension("target");
            if is_link {
                fs::write(&target, b"original").unwrap();
                symlink(&target, &path).unwrap();
            } else {
                fs::write(&path, b"bad-file").unwrap();
            }
            assert!(store.verify(model, || true).is_err());
            let live = Live::new();
            let (ticket, installed, missing) = live.changes.begin(&live, &store, model).unwrap();
            assert_eq!((installed, missing), (false, model.bytes));
            timeout(Duration::from_secs(5), prepare(live.clone(), store.clone(), ticket, model, || Ok(client.clone()))).await.unwrap();
            assert!(live.changes.switching().is_some());
            assert!(fs::symlink_metadata(&path).unwrap().is_file());
            assert_eq!(fs::read(path).unwrap(), WEIGHTS);
            if is_link {
                assert_eq!(fs::read(target).unwrap(), b"original");
            }
            server.await.unwrap();
        }
    }

    #[tokio::test]
    async fn a_new_use_stops_an_active_transfer_and_keeps_its_partial() {
        let (client, url, _, server) = https_fixture(
            &["HTTP/1.1 200 OK\r\nContent-Length: 8\r\nConnection: close\r\n\r\nabc"],
            true,
        )
        .await;
        let mut first = fixture_model("first", WEIGHTS);
        first.url = url;
        let first = Box::leak(Box::new(first));
        let second = Box::leak(Box::new(fixture_model("second", WEIGHTS)));
        let store = fixture_store("replace-transfer");
        let live = Live::new();
        let (ticket, _, _) = live.changes.begin(&live, &store, first).unwrap();
        let old_worker = tokio::spawn(prepare(live.clone(), store.clone(), ticket, first, move || Ok(client.clone())));
        timeout(Duration::from_secs(5), async {
            while fs::metadata(store.partial(first)).map_or(0, |m| m.len()) < 3 {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();

        fs::write(store.path(second), WEIGHTS).unwrap();
        assert_eq!(live.changes.request(&live, &store, second, None).unwrap(), (true, 0));
        let ready = timeout(Duration::from_secs(5), live.changes.wait_switch())
            .await
            .unwrap();
        assert_eq!(ready.model.id, "second");
        old_worker.await.unwrap();
        assert_eq!(fs::read(store.partial(first)).unwrap(), b"abc");
        assert!(!store.path(first).exists());
        assert!(!live.changes.committed());
        server.abort();
    }
}
