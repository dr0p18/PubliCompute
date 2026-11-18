//! The control socket carries one JSON line in each direction for a one-shot
//! command. A follow request leaves the read side open for status changes.

use std::{
    io::{self, BufRead, BufReader, ErrorKind, Write},
    os::unix::net::UnixStream,
    path::{Path, PathBuf},
    time::Duration,
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

#[derive(Clone)]
pub enum Socket {
    Explicit(PathBuf),
    Default,
}

impl Socket {
    fn connect(&self) -> Result<UnixStream, String> {
        const RUNTIME_SOCKET: &str = "/run/publicompute-agent/agent.sock";
        let installed = Path::new("/etc/systemd/system/publicompute-agent.service").exists();
        let path = match self {
            Self::Explicit(path) => path.clone(),
            Self::Default => {
                let runtime = Path::new(RUNTIME_SOCKET);
                match UnixStream::connect(runtime) {
                    Ok(stream) => return Ok(stream),
                    Err(e) if is_unavailable(&e) => {}
                    Err(e) => return Err(connect_error(runtime, &e, installed)),
                }
                let old = Path::new("/var/lib/publicompute");
                let dir = if old.exists() { old } else { Path::new("data/agent") };
                dir.join("agent.sock")
            }
        };
        UnixStream::connect(&path).map_err(|e| connect_error(&path, &e, installed))
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
    let mut s = socket.connect()?;
    // Bound one-shot operations so an unresponsive agent cannot hold the CLI open.
    let t = Some(Duration::from_secs(3));
    let _ = s.set_read_timeout(t);
    let _ = s.set_write_timeout(t);
    writeln!(s, "{req}").map_err(|e| e.to_string())?;
    let mut line = String::new();
    BufReader::new(s).read_line(&mut line).map_err(|e| e.to_string())?;
    let v: Value = serde_json::from_str(&line).map_err(|e| format!("bad reply from the agent: {e}"))?;
    reply_ok(v)
}

pub fn reply_ok(v: Value) -> Result<Value, String> {
    if v["ok"].as_bool() == Some(true) {
        return Ok(v);
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

/// Deliver each status in a follow stream to the callback. Surface the reason
/// the stream ended so the caller can display it and reconnect.
pub fn follow(socket: &Socket, mut on_status: impl FnMut(Status) -> bool) -> String {
    let mut s = match socket.connect() {
        Ok(s) => s,
        Err(e) => return e,
    };
    if writeln!(s, "{}", json!({"op": "follow"})).is_err() {
        return "cannot write to the agent".into();
    }
    for line in BufReader::new(s).lines() {
        let Ok(line) = line else { break };
        let Ok(v) = serde_json::from_str::<Value>(&line) else { continue };
        if let Ok(st) = serde_json::from_value(v["status"].clone()) {
            if !on_status(st) {
                return "closed".into();
            }
        }
    }
    "the agent closed the connection".into()
}
