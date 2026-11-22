//! Measure the host's line without sending bulk traffic to the coordinator.

use std::{
    fs,
    sync::{Arc, Mutex},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use prost_types::Timestamp;
use publicompute_proto::node::{LatencySample, NetworkInfo};
use reqwest::{header, Client, Response, StatusCode};
use serde::{Deserialize, Serialize};
use tokio::time::{self, Instant};

const SPEED_URL: &str = "https://publicompute.com/api/speed";
const DIRECTION_TIME: Duration = Duration::from_secs(5);
const BYTE_LIMIT: usize = 75_000_000;
const GET_LIMIT: usize = 25 * 1024 * 1024;
const POST_BLOCK: usize = 8 * 1024 * 1024;
const FIRST_POST_BLOCK: usize = 512 * 1024;
const NETWORK_FILE: &str = "network.json";
const WEEK: u64 = 7 * 24 * 3600;
const JITTER: u64 = 30 * 60;
const RETRY: u64 = 3600;
pub const RTT_SAMPLES: usize = 10;

#[cfg(unix)]
type StateDir = fs::File;
#[cfg(not(unix))]
type StateDir = std::path::PathBuf;

pub fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[derive(Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Reading {
    pub download_bits_per_sec: u64,
    pub upload_bits_per_sec: u64,
    pub bandwidth_measured_at: Option<u64>,
    pub coordinator_rtt_micros: Option<u32>,
    pub coordinator_rtt_measured_at: Option<u64>,
}

impl Reading {
    fn to_proto(&self) -> NetworkInfo {
        let stamp = |seconds: u64| Timestamp {
            seconds: seconds.min(i64::MAX as u64) as i64,
            nanos: 0,
        };
        let latencies = self
            .coordinator_rtt_micros
            .zip(self.coordinator_rtt_measured_at)
            .map(|(rtt_micros, at)| LatencySample {
                target_id: "coordinator".into(),
                rtt_micros,
                measured_at: Some(stamp(at)),
            })
            .into_iter()
            .collect();
        NetworkInfo {
            download_bits_per_sec: self.download_bits_per_sec,
            upload_bits_per_sec: self.upload_bits_per_sec,
            bandwidth_measured_at: self.bandwidth_measured_at.map(stamp),
            latencies,
        }
    }
}

#[derive(Clone, Default, Serialize, Deserialize)]
#[serde(default)]
struct Saved {
    #[serde(flatten)]
    reading: Reading,
    // Keep a server-imposed delay across reconnects and process restarts.
    retry_at: u64,
    weekly_jitter: u64,
}

#[derive(Default)]
pub struct Store {
    dir: Option<Arc<StateDir>>,
    state: Arc<Mutex<State>>,
}

#[derive(Default)]
struct State {
    saved: Saved,
    startup: Option<bool>,
    dirty: bool,
    writing: bool,
}

impl Store {
    pub fn open(dir: Arc<StateDir>) -> Self {
        let saved = match read_state(&dir) {
            Ok(bytes) => serde_json::from_slice(&bytes).unwrap_or_else(|err| {
                tracing::warn!(%err, "could not decode network measurements");
                Saved::default()
            }),
            Err(err) => {
                if err.kind() != std::io::ErrorKind::NotFound {
                    tracing::warn!(%err, "could not read network measurements");
                }
                Saved::default()
            }
        };
        Self {
            dir: Some(dir),
            state: Arc::new(Mutex::new(State {
                saved,
                ..State::default()
            })),
        }
    }

    pub fn enabled(&self) -> bool {
        self.dir.is_some()
    }

    pub fn reading(&self) -> Option<Reading> {
        let state = self.state.lock().unwrap();
        let saved = &state.saved;
        (saved.reading != Reading::default()).then(|| saved.reading.clone())
    }

    pub fn report(&self) -> Option<NetworkInfo> {
        self.reading().map(|r| r.to_proto())
    }

    pub fn begin_session(&self, now: u64) {
        let mut state = self.state.lock().unwrap();
        if state.startup.is_none() {
            state.startup = Some(state.saved.due(now, true));
        }
    }

    pub fn due(&self, now: u64) -> bool {
        let state = self.state.lock().unwrap();
        self.enabled() && state.saved.due(now, state.startup.unwrap_or(false))
    }

    pub fn record_bandwidth(&self, result: Bandwidth, now: u64) {
        let mut state = self.state.lock().unwrap();
        state.startup = Some(false);
        let saved = &mut state.saved;
        match result {
            Bandwidth::Limited(delay) => {
                // Round the current second up so the stored deadline cannot shorten the delay.
                saved.retry_at = now.saturating_add(delay).saturating_add(1);
            }
            Bandwidth::Measured { down, up, complete } => {
                // An intermittent failure should not erase a valid measurement.
                if complete
                    || saved
                        .reading
                        .bandwidth_measured_at
                        .is_none_or(|at| now.saturating_sub(at) >= WEEK + 86400)
                {
                    saved.reading.download_bits_per_sec = down;
                    saved.reading.upload_bits_per_sec = up;
                    saved.reading.bandwidth_measured_at = complete.then_some(now);
                }
                saved.retry_at = if complete {
                    0
                } else {
                    now.saturating_add(RETRY)
                };
                saved.weekly_jitter = rand::random_range(0..=JITTER);
            }
        }
        self.schedule_save(&mut state);
    }

    pub fn record_rtt(&self, micros: u32, now: u64) {
        let mut state = self.state.lock().unwrap();
        state.saved.reading.coordinator_rtt_micros = Some(micros);
        state.saved.reading.coordinator_rtt_measured_at = Some(now);
        self.schedule_save(&mut state);
    }

    fn schedule_save(&self, state: &mut State) {
        let Some(dir) = &self.dir else { return };
        state.dirty = true;
        if state.writing {
            return;
        }
        state.writing = true;
        let dir = dir.clone();
        let shared = self.state.clone();
        tokio::task::spawn_blocking(move || loop {
            let saved = {
                let mut state = shared.lock().unwrap();
                if !state.dirty {
                    state.writing = false;
                    break;
                }
                state.dirty = false;
                state.saved.clone()
            };
            if let Err(err) = save(&dir, &saved) {
                tracing::warn!(%err, "could not save network measurements");
            }
        });
    }
}

#[cfg(unix)]
fn read_state(dir: &StateDir) -> std::io::Result<Vec<u8>> {
    use rustix::fs::{openat, Mode, OFlags};
    use std::io::{Error, ErrorKind, Read};

    let flags = OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC;
    let file = fs::File::from(openat(dir, NETWORK_FILE, flags, Mode::empty())?);
    if !file.metadata()?.is_file() {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "network state is not a regular file",
        ));
    }
    let mut data = Vec::new();
    file.take(4097).read_to_end(&mut data)?;
    if data.len() > 4096 {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "network state exceeds 4 KiB",
        ));
    }
    Ok(data)
}

#[cfg(not(unix))]
fn read_state(dir: &StateDir) -> std::io::Result<Vec<u8>> {
    fs::read(dir.join(NETWORK_FILE))
}

#[cfg(unix)]
fn save(dir: &StateDir, saved: &Saved) -> std::io::Result<()> {
    let bytes = serde_json::to_vec(saved).map_err(std::io::Error::other)?;
    crate::control::pause_file::replace_state_file(dir, NETWORK_FILE, &bytes)?;
    dir.sync_all()
}

#[cfg(not(unix))]
fn save(dir: &StateDir, saved: &Saved) -> std::io::Result<()> {
    use std::io::Write;
    let tmp = dir.join(format!(
        ".{NETWORK_FILE}-{:016x}.tmp",
        rand::random::<u64>()
    ));
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&tmp)?;
    let result = (|| {
        file.write_all(&serde_json::to_vec(saved).map_err(std::io::Error::other)?)?;
        file.sync_all()?;
        drop(file);
        fs::rename(&tmp, dir.join(NETWORK_FILE))
    })();
    if result.is_err() {
        let _ = fs::remove_file(tmp);
    }
    result
}

impl Saved {
    fn due(&self, now: u64, startup: bool) -> bool {
        if now < self.retry_at {
            return false;
        }
        let jitter = if startup {
            0
        } else {
            self.weekly_jitter.min(JITTER)
        };
        self.reading
            .bandwidth_measured_at
            .is_none_or(|at| at > now || now - at >= WEEK + jitter)
    }
}

pub enum Bandwidth {
    Limited(u64),
    Measured { down: u64, up: u64, complete: bool },
}

impl Bandwidth {
    fn near_threshold(&self) -> bool {
        match *self {
            Self::Measured {
                down,
                up,
                complete: true,
            } => {
                (80_000_000..=120_000_000).contains(&down)
                    || (80_000_000..=120_000_000).contains(&up)
            }
            _ => false,
        }
    }
}

pub async fn measure() -> Bandwidth {
    let _ = rustls::crypto::ring::default_provider().install_default();
    let Ok(http) = Client::builder()
        // Keep transfers on the speed endpoint and account for each body once.
        .https_only(true)
        .redirect(reqwest::redirect::Policy::none())
        .retry(reqwest::retry::never())
        .no_proxy()
        .build()
        .inspect_err(|err| tracing::warn!(%err, "could not prepare bandwidth measurement"))
    else {
        return Bandwidth::Measured {
            down: 0,
            up: 0,
            complete: false,
        };
    };
    measure_with(&http, SPEED_URL, DIRECTION_TIME, BYTE_LIMIT).await
}

async fn measure_with(http: &Client, url: &str, time: Duration, limit: usize) -> Bandwidth {
    let first = measure_once(http, url, time, limit).await;
    if !first.near_threshold() {
        return first;
    }
    let Bandwidth::Measured { down, up, .. } = first else {
        return first;
    };
    match measure_once(http, url, time, limit).await {
        Bandwidth::Measured {
            down: second_down,
            up: second_up,
            complete: true,
        } => Bandwidth::Measured {
            down: down.min(second_down),
            up: up.min(second_up),
            complete: true,
        },
        limited @ Bandwidth::Limited(_) => limited,
        _ => first,
    }
}

async fn measure_once(http: &Client, url: &str, time: Duration, limit: usize) -> Bandwidth {
    let down = direction(http, url, time, limit, Direction::Down).await;
    let (down, down_ok) = match down {
        Ok(sample) => sample,
        Err(delay) => return Bandwidth::Limited(delay),
    };
    match direction(http, url, time, limit, Direction::Up).await {
        Ok((up, up_ok)) => Bandwidth::Measured {
            down,
            up,
            complete: down_ok && up_ok,
        },
        Err(delay) => Bandwidth::Limited(delay),
    }
}

enum TransferError {
    Limited(u64),
    Failed,
}

impl From<reqwest::Error> for TransferError {
    fn from(err: reqwest::Error) -> Self {
        tracing::debug!(%err, "network transfer failed");
        Self::Failed
    }
}

type Transfer = Result<(), TransferError>;

enum Direction {
    Down,
    Up,
}

async fn direction(
    http: &Client,
    url: &str,
    budget: Duration,
    limit: usize,
    direction: Direction,
) -> Result<(u64, bool), u64> {
    let start = Instant::now();
    let deadline = start + budget;
    let (sample, result) = match direction {
        Direction::Down => {
            let mut bytes = 0;
            let result = time::timeout_at(deadline, download(http, url, limit, &mut bytes)).await;
            (
                TransferSample {
                    bytes,
                    elapsed: start.elapsed().min(budget),
                },
                result,
            )
        }
        Direction::Up => {
            let mut sample = TransferSample::default();
            let result =
                time::timeout_at(deadline, upload(http, url, limit, deadline, &mut sample)).await;
            (sample, result)
        }
    };
    if let Ok(Err(TransferError::Limited(delay))) = result {
        return Err(delay);
    }
    Ok((
        sample.bits_per_sec(),
        sample.bytes != 0 && !matches!(result, Ok(Err(TransferError::Failed))),
    ))
}

fn retry_after(value: &str, now: SystemTime) -> Option<u64> {
    let value = value.trim();
    value.parse::<u64>().ok().or_else(|| {
        let until = httpdate::parse_http_date(value).ok()?;
        let delay = until.duration_since(now).unwrap_or_default();
        Some(delay.as_secs() + u64::from(delay.subsec_nanos() != 0))
    })
}

fn checked(response: Response) -> Result<Response, TransferError> {
    if response.status() == StatusCode::TOO_MANY_REQUESTS {
        let delay = response
            .headers()
            .get(header::RETRY_AFTER)
            .and_then(|h| h.to_str().ok())
            .and_then(|s| retry_after(s, SystemTime::now()))
            .unwrap_or(RETRY);
        return Err(TransferError::Limited(delay.max(1)));
    }
    if !response.status().is_success() {
        tracing::debug!(status = %response.status(), "speed endpoint rejected the request");
        return Err(TransferError::Failed);
    }
    Ok(response)
}

async fn download(http: &Client, url: &str, limit: usize, bytes: &mut usize) -> Transfer {
    while *bytes < limit {
        let wanted = (limit - *bytes).min(GET_LIMIT);
        let nonce: u64 = rand::random();
        let mut response = checked(
            http.get(format!("{url}?bytes={wanted}&nonce={nonce}"))
                .header(header::ACCEPT_ENCODING, "identity")
                .send()
                .await?,
        )?;
        let end = *bytes + wanted;
        while *bytes < end {
            let chunk = response.chunk().await?.ok_or(TransferError::Failed)?;
            *bytes += chunk.len().min(end - *bytes);
        }
    }
    Ok(())
}

#[derive(Deserialize)]
struct Acknowledgement {
    bytes: usize,
}

#[derive(Default)]
struct TransferSample {
    bytes: usize,
    elapsed: Duration,
}

impl TransferSample {
    fn bits_per_sec(&self) -> u64 {
        (self.bytes as u128 * 8_000_000_000)
            .checked_div(self.elapsed.as_nanos())
            .unwrap_or(0)
            .min(u128::from(u64::MAX)) as u64
    }

    fn bytes_per_sec(&self) -> f64 {
        if self.elapsed.is_zero() {
            0.0
        } else {
            self.bytes as f64 / self.elapsed.as_secs_f64()
        }
    }

    fn post_size(&self, remaining: Duration, remaining_bytes: usize) -> usize {
        let size = if self.bytes == 0 {
            FIRST_POST_BLOCK
        } else if remaining < Duration::from_millis(100) {
            0
        } else {
            // Leave time for the acknowledgement and for variation in the link rate.
            (self.bytes_per_sec() * remaining.as_secs_f64() * 0.8) as usize
        };
        size.min(POST_BLOCK).min(remaining_bytes)
    }
}

async fn upload(
    http: &Client,
    url: &str,
    limit: usize,
    deadline: Instant,
    sample: &mut TransferSample,
) -> Transfer {
    let payload = bytes::Bytes::from(vec![0; POST_BLOCK.min(limit)]);
    let mut attempted = 0;
    while attempted < limit {
        let start = Instant::now();
        let size = sample.post_size(deadline.saturating_duration_since(start), limit - attempted);
        if size == 0 || start >= deadline {
            break;
        }
        attempted += size;
        let mut response = checked(
            http.post(url)
                .header(header::CONTENT_TYPE, "application/octet-stream")
                .body(payload.slice(..size))
                .send()
                .await?,
        )?;
        // A truncated request or an incomplete acknowledgement contributes no bytes.
        let mut ack = Vec::new();
        while let Some(chunk) = response.chunk().await? {
            if ack.len() + chunk.len() > 256 {
                return Err(TransferError::Failed);
            }
            ack.extend_from_slice(&chunk);
        }
        let ack: Acknowledgement =
            serde_json::from_slice(&ack).map_err(|_| TransferError::Failed)?;
        if ack.bytes != size {
            return Err(TransferError::Failed);
        }
        sample.bytes += ack.bytes;
        sample.elapsed += start.elapsed();
    }
    Ok(())
}

pub fn median_and_p95(samples: &mut [u32; RTT_SAMPLES]) -> (u32, u32) {
    samples.sort_unstable();
    (
        ((u64::from(samples[4]) + u64::from(samples[5])) / 2) as u32,
        samples[9],
    )
}

pub fn micros(elapsed: Duration) -> u32 {
    elapsed.as_micros().clamp(1, u128::from(u32::MAX)) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use tokio::{
        io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader},
        net::{TcpListener, TcpStream},
    };

    fn open(dir: &Path) -> Store {
        #[cfg(unix)]
        let dir = fs::File::open(dir).unwrap();
        #[cfg(not(unix))]
        let dir = dir.to_owned();
        Store::open(Arc::new(dir))
    }

    async fn loopback() -> (TcpListener, String, Client) {
        let _ = rustls::crypto::ring::default_provider().install_default();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/api/speed", listener.local_addr().unwrap());
        (listener, url, Client::builder().no_proxy().build().unwrap())
    }

    async fn headers(listener: &TcpListener) -> (BufReader<TcpStream>, String) {
        let (stream, _) = listener.accept().await.unwrap();
        let mut stream = BufReader::new(stream);
        let mut headers = String::new();
        loop {
            let mut line = String::new();
            assert!(stream.read_line(&mut line).await.unwrap() > 0);
            if line == "\r\n" {
                return (stream, headers);
            }
            headers.push_str(&line);
        }
    }

    async fn response(stream: &mut BufReader<TcpStream>, status: &str, headers: &str) {
        stream
            .get_mut()
            .write_all(
                format!("HTTP/1.1 {status}\r\nConnection: close\r\n{headers}\r\n").as_bytes(),
            )
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn transfer_caps_bytes_and_counts_acknowledgements() {
        let (server, url, http) = loopback().await;
        let fixture = tokio::spawn(async move {
            let mut urls = Vec::new();
            for size in [GET_LIMIT, 23] {
                let (mut stream, head) = headers(&server).await;
                assert!(head.starts_with(&format!("GET /api/speed?bytes={size}&nonce=")));
                assert!(head.contains("accept-encoding: identity"));
                urls.push(
                    head.lines()
                        .next()
                        .unwrap()
                        .split("&nonce=")
                        .nth(1)
                        .unwrap()
                        .to_owned(),
                );
                response(
                    &mut stream,
                    "200 OK",
                    &format!("Content-Length: {size}\r\n"),
                )
                .await;
                stream.get_mut().write_all(&vec![0; size]).await.unwrap();
            }
            assert_ne!(urls[0], urls[1]);
            for size in [
                FIRST_POST_BLOCK,
                POST_BLOCK,
                POST_BLOCK - FIRST_POST_BLOCK - 10,
            ] {
                let (mut stream, head) = headers(&server).await;
                assert!(head.starts_with("POST /api/speed "));
                assert!(head.contains(&format!("content-length: {size}")));
                stream.read_exact(&mut vec![0; size]).await.unwrap();
                let ack = format!(r#"{{"bytes":{size}}}"#);
                response(
                    &mut stream,
                    "200 OK",
                    &format!("Content-Length: {}\r\n", ack.len()),
                )
                .await;
                stream.get_mut().write_all(ack.as_bytes()).await.unwrap();
            }
        });
        let mut down = 0;
        let mut up = TransferSample::default();
        assert!(download(&http, &url, GET_LIMIT + 23, &mut down)
            .await
            .is_ok());
        let deadline = Instant::now() + Duration::from_secs(10);
        assert!(upload(&http, &url, POST_BLOCK * 2 - 10, deadline, &mut up)
            .await
            .is_ok());
        assert_eq!(down, GET_LIMIT + 23);
        assert_eq!(up.bytes, POST_BLOCK * 2 - 10);
        fixture.await.unwrap();
    }

    #[tokio::test]
    async fn timeout_keeps_read_bytes_but_interrupted_upload_has_no_acknowledgement() {
        let (server, url, http) = loopback().await;
        let fixture = tokio::spawn(async move {
            let (mut stream, _) = headers(&server).await;
            response(&mut stream, "200 OK", "Content-Length: 16384\r\n").await;
            stream.get_mut().write_all(&[0; 8192]).await.unwrap();
            let (mut upload, _) = headers(&server).await;
            upload.read_exact(&mut [0; 512]).await.unwrap();
            drop(upload);
        });
        let timeout = Duration::from_millis(250);
        let start = Instant::now();
        let result = measure_once(&http, &url, timeout, 16384).await;
        assert!(start.elapsed() >= timeout);
        assert!(matches!(
            result,
            Bandwidth::Measured {
                down: 262_144,
                up: 0,
                complete: false
            }
        ));
        fixture.await.unwrap();
    }

    #[tokio::test]
    async fn rate_limit_stops_the_measurement_and_honors_retry_after() {
        let (server, url, http) = loopback().await;
        let fixture = tokio::spawn(async move {
            let (mut stream, _) = headers(&server).await;
            response(
                &mut stream,
                "429 Too Many Requests",
                "Content-Length: 0\r\nRetry-After: 3600\r\n",
            )
            .await;
            assert!(time::timeout(Duration::from_millis(50), server.accept())
                .await
                .is_err());
        });
        assert!(matches!(
            measure_with(&http, &url, DIRECTION_TIME, 16).await,
            Bandwidth::Limited(3600)
        ));
        fixture.await.unwrap();
        let date = UNIX_EPOCH + Duration::from_secs(1800);
        assert_eq!(
            retry_after(&httpdate::fmt_http_date(date), UNIX_EPOCH),
            Some(1800)
        );
    }

    #[test]
    fn upload_window_measures_a_hundred_and_ten_megabits_within_five_seconds() {
        for bits_per_sec in [100_000_000.0, 10_000_000.0] {
            let mut sample = TransferSample::default();
            loop {
                let remaining = DIRECTION_TIME - sample.elapsed;
                let size = sample.post_size(remaining, BYTE_LIMIT - sample.bytes);
                if size == 0 {
                    break;
                }
                let elapsed = Duration::from_secs_f64(size as f64 * 8.0 / bits_per_sec);
                assert!(
                    elapsed <= remaining,
                    "block cannot finish before the deadline"
                );
                sample.bytes += size;
                sample.elapsed += elapsed;
            }
            assert!(sample.bytes <= BYTE_LIMIT);
            assert!(sample.elapsed >= DIRECTION_TIME - Duration::from_millis(100));
            assert_eq!(sample.bits_per_sec(), bits_per_sec as u64);
        }
    }

    async fn wait_for_save(store: &Store) {
        time::timeout(Duration::from_secs(3), async {
            while store.state.lock().unwrap().writing {
                time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn state_survives_restart_and_checks_freshness_and_retry_delay() {
        let dir =
            std::env::temp_dir().join(format!("publicompute-network-{}", rand::random::<u64>()));
        fs::create_dir(&dir).unwrap();
        let now = WEEK * 100;
        let store = open(&dir);
        store.begin_session(now);
        assert!(store.due(now));
        store.record_bandwidth(
            Bandwidth::Measured {
                down: 110_000_000,
                up: 100_000_000,
                complete: true,
            },
            now,
        );
        store.record_rtt(12_000, now + 1);
        wait_for_save(&store).await;
        let reopened = open(&dir);
        let info = reopened.report().unwrap();
        assert_eq!(info.bandwidth_measured_at.unwrap().seconds, now as i64);
        assert_eq!(info.latencies[0].target_id, "coordinator");
        assert_eq!(info.latencies[0].rtt_micros, 12_000);
        reopened.begin_session(now + WEEK);
        assert!(!reopened.due(now + WEEK - 1));
        assert!(reopened.due(now + WEEK));
        assert!(reopened.due(now + WEEK + JITTER));
        reopened.record_bandwidth(Bandwidth::Limited(3600), now + WEEK);
        wait_for_save(&reopened).await;
        let reopened = open(&dir);
        reopened.begin_session(now + WEEK + 3600);
        assert!(!reopened.due(now + WEEK + 3600));
        assert!(reopened.due(now + WEEK + 3601));
        assert_eq!(
            reopened.report().unwrap().download_bits_per_sec,
            110_000_000
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn reconnect_uses_the_process_weekly_deadline() {
        let now = WEEK * 100;
        let dir = crate::stages::tests::tempdir("network-jitter");
        let saved = Saved {
            weekly_jitter: 90,
            reading: Reading {
                bandwidth_measured_at: Some(now),
                ..Reading::default()
            },
            ..Saved::default()
        };
        fs::write(dir.join(NETWORK_FILE), serde_json::to_vec(&saved).unwrap()).unwrap();
        let store = open(&dir);
        store.begin_session(now + WEEK - 1);
        store.begin_session(now + WEEK + 1);
        assert!(!store.due(now + WEEK + 89));
        assert!(store.due(now + WEEK + 90));
        drop(store);
        fs::remove_dir_all(dir).unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn state_writer_keeps_latest_retry_and_replaces_files_privately() {
        use std::os::unix::fs::{symlink, PermissionsExt};

        let dir = crate::stages::tests::tempdir("net-state");
        let target = dir.join("untouched");
        fs::write(&target, b"keep").unwrap();
        symlink(&target, dir.join("network.json.tmp")).unwrap();
        let store = open(&dir);
        store.record_rtt(1200, 100);
        store.record_bandwidth(Bandwidth::Limited(3600), 100);
        store.record_rtt(1500, 101);
        assert!(!store.due(3700));
        wait_for_save(&store).await;
        assert_eq!(fs::read(&target).unwrap(), b"keep");
        let mode = fs::metadata(dir.join(NETWORK_FILE))
            .unwrap()
            .permissions()
            .mode();
        assert_eq!(mode & 0o777, 0o600);
        let reopened = open(&dir);
        assert!(!reopened.due(3700));
        assert!(reopened.due(3701));
        assert_eq!(
            reopened.reading().unwrap().coordinator_rtt_micros,
            Some(1500)
        );
        fs::remove_dir_all(dir).unwrap();
    }
}
