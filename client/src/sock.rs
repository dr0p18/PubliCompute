//! The control socket carries one JSON line in each direction for a one-shot
//! command. A follow request leaves the read side open for status changes.

use std::{
    io::{self, BufRead, BufReader, ErrorKind, Read, Write},
    os::fd::AsFd,
    os::unix::net::UnixStream,
    path::{Path, PathBuf},
    thread,
    time::{Duration, Instant},
};

use rustix::{
    event::{poll, PollFd, PollFlags, Timespec},
    io::Errno,
    net::{self, AddressFamily, SocketAddrUnix, SocketType},
};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::status::Status;

#[derive(Deserialize, Clone, Debug, PartialEq)]
pub struct LocalModel {
    pub id: String,
    pub quant: String,
    #[serde(default)]
    pub loaded: bool,
}

pub const START: &str = "Agent not running. Run sudo systemctl start publicompute-agent";
const INSTALL: &str = "Agent not installed. Run curl -fsSL https://publicompute.com/install.sh | sh";
const PERMISSION: &str = "Permission denied. Join the publicompute-agent group, then log out and back in";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(3);
const MAX_REPLY_LINE: usize = 64 * 1024;
pub const FOLLOW_TIMEOUT: Duration = Duration::from_secs(75);

#[derive(Clone)]
pub enum Socket {
    Explicit(PathBuf),
    Default,
}

impl Socket {
    fn connect(&self, deadline: Instant) -> Result<UnixStream, String> {
        const RUNTIME_SOCKET: &str = "/run/publicompute-agent/agent.sock";
        let installed = Path::new("/etc/systemd/system/publicompute-agent.service").exists();
        let path = match self {
            Self::Explicit(path) => path.clone(),
            Self::Default => {
                let runtime = Path::new(RUNTIME_SOCKET);
                match connect_until(runtime, deadline) {
                    Ok(stream) => return Ok(stream),
                    Err(e) if is_unavailable(&e) => {}
                    Err(e) => return Err(connect_error(runtime, &e, installed)),
                }
                let old = Path::new("/var/lib/publicompute");
                let dir = if old.exists() { old } else { Path::new("data/agent") };
                dir.join("agent.sock")
            }
        };
        connect_until(&path, deadline).map_err(|e| connect_error(&path, &e, installed))
    }
}

fn connect_until(path: &Path, deadline: Instant) -> io::Result<UnixStream> {
    let addr = SocketAddrUnix::new(path)?;
    loop {
        remaining(deadline)?;
        #[cfg(any(target_os = "linux", target_os = "android"))]
        let socket = UnixStream::from(net::socket_with(AddressFamily::UNIX, SocketType::STREAM,
            net::SocketFlags::NONBLOCK | net::SocketFlags::CLOEXEC, None)?);
        #[cfg(not(any(target_os = "linux", target_os = "android")))]
        let socket = {
            let fd = net::socket(AddressFamily::UNIX, SocketType::STREAM, None)?;
            rustix::io::fcntl_setfd(&fd, rustix::io::FdFlags::CLOEXEC)?;
            let stream = UnixStream::from(fd);
            stream.set_nonblocking(true)?;
            stream
        };
        match net::connect(&socket, &addr) {
            Ok(()) => return Ok(socket),
            Err(Errno::INPROGRESS) => {
                wait_for(&socket, PollFlags::OUT, deadline)?;
                net::sockopt::socket_error(&socket)??;
                return Ok(socket);
            }
            Err(Errno::AGAIN) => {
                // A full Unix listen queue can leave the socket unconnected.
                thread::sleep(remaining(deadline)?.min(Duration::from_millis(10)));
                continue;
            }
            Err(Errno::INTR) => continue,
            Err(err) => return Err(err.into()),
        }
    }
}

fn remaining(deadline: Instant) -> io::Result<Duration> {
    let left = deadline.saturating_duration_since(Instant::now());
    if left.is_zero() {
        return Err(ErrorKind::TimedOut.into());
    }
    Ok(left)
}

fn wait_for(fd: &impl AsFd, flags: PollFlags, deadline: Instant) -> io::Result<()> {
    let mut fds = [PollFd::new(fd, flags)];
    loop {
        let timeout = Timespec::try_from(remaining(deadline)?).map_err(io::Error::other)?;
        match poll(&mut fds, Some(&timeout)) {
            Ok(0) | Err(Errno::INTR) => continue,
            Ok(_) => return Ok(()),
            Err(err) => return Err(err.into()),
        }
    }
}

fn is_unavailable(e: &io::Error) -> bool {
    matches!(e.kind(), ErrorKind::NotFound | ErrorKind::ConnectionRefused)
}

pub(crate) fn connect_error(socket: &Path, e: &io::Error, installed: bool) -> String {
    match e.kind() {
        ErrorKind::PermissionDenied => PERMISSION.into(),
        ErrorKind::NotFound | ErrorKind::ConnectionRefused => if installed { START } else { INSTALL }.into(),
        _ => format!("cannot connect to {}: {e}", socket.display()),
    }
}

/// Exchange one request and reply. Return the agent's message as an error when it refuses.
pub fn request(socket: &Socket, req: &Value) -> Result<Value, String> {
    let deadline = Instant::now() + REQUEST_TIMEOUT;
    let mut s = socket.connect(deadline)?;
    write_request(&mut s, req, deadline).map_err(|e| e.to_string())?;
    let mut line = Vec::new();
    if !read_reply(&mut BufReader::new(s), &mut line, deadline).map_err(|e| e.to_string())? {
        return Err("the agent closed the connection".into());
    }
    let v: Value = serde_json::from_slice(&line).map_err(|e| format!("bad reply from the agent: {e}"))?;
    reply_ok(v)
}

fn write_request(stream: &mut (impl Write + AsFd), req: &Value, deadline: Instant) -> io::Result<()> {
    let line = format!("{req}\n");
    let mut bytes = line.as_bytes();
    while !bytes.is_empty() {
        remaining(deadline)?;
        match stream.write(bytes) {
            Ok(0) => return Err(ErrorKind::WriteZero.into()),
            Ok(n) => bytes = &bytes[n..],
            Err(err) if err.kind() == ErrorKind::WouldBlock => wait_for(stream, PollFlags::OUT, deadline)?,
            Err(err) if err.kind() == ErrorKind::Interrupted => continue,
            Err(err) => return Err(err),
        }
    }
    Ok(())
}

fn read_reply(reader: &mut BufReader<impl Read + AsFd>, line: &mut Vec<u8>, deadline: Instant) -> io::Result<bool> {
    line.clear();
    loop {
        // Bytes alone don't prove the agent is making progress toward a reply.
        remaining(deadline)?;
        let bytes = match reader.fill_buf() {
            Ok(bytes) if bytes.is_empty() => return Ok(false),
            Ok(bytes) => bytes,
            Err(err) if err.kind() == ErrorKind::WouldBlock => {
                wait_for(reader.get_ref(), PollFlags::IN, deadline)?;
                continue;
            }
            Err(err) if err.kind() == ErrorKind::Interrupted => continue,
            Err(err) => return Err(err),
        };
        let end = bytes.iter().position(|byte| *byte == b'\n');
        let len = end.unwrap_or(bytes.len());
        if line.len() + len > MAX_REPLY_LINE {
            return Err(io::Error::new(ErrorKind::InvalidData, "agent reply exceeds 64 KiB"));
        }
        line.extend_from_slice(&bytes[..len]);
        reader.consume(len + usize::from(end.is_some()));
        if end.is_some() {
            return Ok(true);
        }
    }
}

pub fn reply_ok(v: Value) -> Result<Value, String> {
    if v["ok"].as_bool() == Some(true) {
        return Ok(v);
    }
    if v["error"] == "durability_uncertain" {
        if let Some(paused) = v["paused"].as_bool() {
            return Err(format!("pause state applied (paused={paused}), but durability is uncertain"));
        }
    }
    Err(v["text"].as_str().or(v["error"].as_str()).unwrap_or("the agent refused the request").to_string())
}

pub fn get_status(socket: &Socket) -> Result<Status, String> {
    let v = request(socket, &json!({"op": "status"}))?;
    serde_json::from_value(v["status"].clone()).map_err(|e| format!("unexpected status from the agent: {e}"))
}

pub fn get_models(socket: &Socket) -> Result<Vec<LocalModel>, String> {
    let v = request(socket, &json!({"op": "models"}))?;
    serde_json::from_value(v["models"].clone()).map_err(|e| format!("unexpected model list: {e}"))
}

pub fn set_preference(socket: &Socket, model: Option<&str>, quant: Option<&str>) -> Result<(), String> {
    let req = match model {
        Some(m) => json!({"op": "preference", "mode": "use", "model": m, "quant": quant}),
        None => json!({"op": "preference", "mode": "free"}),
    };
    request(socket, &req).map(|_| ())
}

pub fn set_pause(socket: &Socket, active: bool) -> Result<(), String> {
    request(socket, &json!({"op": "pause", "active": active})).map(|_| ())
}

pub fn follow(socket: &Socket, silence: Duration, on_status: impl FnMut(Status) -> bool) -> Result<(), String> {
    let deadline = Instant::now() + REQUEST_TIMEOUT;
    let mut s = socket.connect(deadline)?;
    write_request(&mut s, &json!({"op": "follow"}), deadline).map_err(|_| "cannot write to the agent".to_string())?;
    follow_replies(s, silence, on_status)
}

fn follow_replies(stream: impl Read + AsFd, silence: Duration, mut on_status: impl FnMut(Status) -> bool) -> Result<(), String> {
    let mut reader = BufReader::new(stream);
    let mut line = Vec::new();
    let mut deadline = Instant::now() + silence;
    loop {
        match read_reply(&mut reader, &mut line, deadline) {
            Ok(true) => {}
            Ok(false) => return Err("the agent closed the connection".into()),
            Err(err) if matches!(err.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => {
                return Err("the agent stopped sending status updates".into());
            }
            Err(err) => return Err(err.to_string()),
        }
        let Ok(v) = serde_json::from_slice::<Value>(&line) else { continue };
        match v["ok"].as_bool() {
            Some(true) => {}
            Some(false) => return reply_ok(v).map(|_| ()),
            None => continue,
        }
        if let Ok(st) = serde_json::from_value(v["status"].clone()) {
            deadline = Instant::now() + silence;
            if !on_status(st) {
                return Ok(());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{fs::File, os::fd::OwnedFd};

    #[test]
    fn uncertain_pause_durability_reports_the_effective_state_on_one_line() {
        for paused in [true, false] {
            let reply = json!({"ok": false, "error": "durability_uncertain", "paused": paused});
            assert_eq!(reply_ok(reply).unwrap_err(), format!("pause state applied (paused={paused}), but durability is uncertain"));
        }
    }

    fn socket_pair() -> (File, File) {
        let (client, peer) = UnixStream::pair().unwrap();
        client.set_nonblocking(true).unwrap();
        (File::from(OwnedFd::from(client)), File::from(OwnedFd::from(peer)))
    }

    #[test]
    fn write_request_uses_the_remaining_deadline_under_backpressure() {
        let (mut client, _peer) = socket_pair();
        let request = json!({"op": "preference", "model": "x".repeat(1024 * 1024)});
        let started = Instant::now();
        let budget = Duration::from_millis(100);
        let err = write_request(&mut client, &request, started + budget).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::TimedOut);
        assert!(started.elapsed() >= budget && started.elapsed() < Duration::from_secs(2));
    }

    #[test]
    fn incomplete_snapshots_do_not_publish_or_reset_the_watchdog() {
        let (client, mut peer) = socket_pair();
        let invalid = [
            json!({"ok": true, "status": {}}),
            json!({"status": {"connected": true, "paused": false}}),
            json!({"ok": true, "status": {"connected": true}}),
            json!({"ok": true, "status": {"paused": false}}),
        ];
        let server = thread::spawn(move || {
            for reply in invalid.iter().cycle().take(12) {
                if writeln!(peer, "{reply}").is_err() {
                    return;
                }
                thread::sleep(Duration::from_millis(30));
            }
        });

        let result = follow_replies(client, Duration::from_millis(150), |_| panic!("published an incomplete snapshot"));
        assert_eq!(result.unwrap_err(), "the agent stopped sending status updates");
        server.join().unwrap();
    }

    #[test]
    fn follow_caps_reply_lines_even_without_a_newline() {
        let (client, mut peer) = socket_pair();
        let server = thread::spawn(move || {
            let mut reply = br#"{"ok":true,"status":{"connected":true,"paused":false}}"#.to_vec();
            reply.resize(MAX_REPLY_LINE, b' ');
            reply.push(b'\n');
            peer.write_all(&reply).unwrap();
            peer.write_all(&vec![b' '; MAX_REPLY_LINE + 1]).unwrap();
        });

        let mut snapshots = 0;
        let result = follow_replies(client, Duration::from_secs(2), |_| {
            snapshots += 1;
            true
        });
        assert_eq!(snapshots, 1);
        assert_eq!(result.unwrap_err(), "agent reply exceeds 64 KiB");
        server.join().unwrap();
    }
}
