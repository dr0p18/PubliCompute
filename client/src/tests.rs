use std::{
    io::{BufRead, BufReader, Write},
    os::unix::net::UnixListener,
    path::PathBuf,
    sync::mpsc,
    thread,
};

use serde_json::{json, Value};

use crate::{command, command_with_uninstaller, render, run_uninstaller, sock::{self, Socket}, status::*};

/// Fixture shaped like the agent's JSON status reply.
fn sample_json() -> Value {
    json!({
        "version": "0.1.0", "connected": true, "paused": false,
        "preference": {"mode": "use", "model": "qwen3.5-9b", "quant": "Q4_K_M", "revision": 3},
        "engine": "ready", "loaded_model": "qwen3.5-9b", "active_leases": 1, "chain_id": null,
        "gpu": {"name": "NVIDIA GeForce RTX 2080 SUPER", "vram_used_bytes": 6442450944u64,
                "vram_total_bytes": 8589934592u64, "temperature_celsius": 62},
        "contribution": {"effective_mode": "use", "serving": "qwen3.5-9b", "in_chain": false, "free_seat": true,
                         "reason": "FREE_SEAT_REASON_OK", "stable_until": null, "revision": 3},
        "summary": {"days": 30, "free_value_mc": 1240, "free_chats": 86, "earned_mc": 310,
                    "service_seconds": 255600, "work_seconds": 32400, "balance_mc": 500}
    })
}

fn sample() -> Status {
    serde_json::from_value(sample_json()).unwrap()
}

fn with_reason(reason: &str, free_seat: bool) -> Status {
    let mut s = sample();
    let c = s.contribution.as_mut().unwrap();
    c.reason = format!("FREE_SEAT_REASON_{reason}");
    c.free_seat = free_seat;
    s
}

/// Measure characters after removing terminal styling sequences.
fn visible(line: &str) -> usize {
    let mut n = 0;
    let mut chars = line.chars();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            for e in chars.by_ref() {
                if e == 'm' {
                    break;
                }
            }
        } else {
            n += 1;
        }
    }
    n
}

#[test]
fn parses_status_reply() {
    let s = sample();
    assert!(s.connected && !s.paused);
    assert_eq!(s.preference.model.as_deref(), Some("qwen3.5-9b"));
    assert_eq!(s.gpu.as_ref().unwrap().temperature_celsius, 62);
    assert_eq!(s.summary.as_ref().unwrap().free_chats, 86);
    assert!(s.chain_id.is_none());
}

#[test]
fn render_never_exceeds_width() {
    let mut s = sample();
    s.contribution.as_mut().unwrap().stable_until = Some(i64::MAX);
    s.chain_id = Some("a-very-long-chain-identifier-that-goes-on".into());
    for w in [40u16, 80, 120] {
        for color in [false, true] {
            let lines = render::render(&s, w, 40, 0, "Use which model?  [1] qwen3.5-9b Q4_K_M  [2] another-model Q8_0", color);
            assert!(lines.len() > 8);
            for l in &lines {
                assert!(visible(l) <= w as usize, "width {w}: {l:?}");
            }
            for l in render::render_down(w, "cannot connect to /var/lib/publicompute/agent.sock: No such file", color) {
                assert!(visible(&l) <= w as usize, "down, width {w}: {l:?}");
            }
            for l in render::render_logo(w, color) {
                assert!(visible(&l) <= w as usize);
            }
        }
    }
}

#[test]
fn render_fits_height_and_keeps_keys() {
    let lines = render::render(&sample(), 80, 6, 0, "", false);
    assert_eq!(lines.len(), 6);
    assert_eq!(lines.last().unwrap(), render::KEYS);
}

#[test]
fn render_shows_summary_and_disclaimer() {
    let text = render::render(&sample(), 80, 40, 0, "", false).join("\n");
    assert!(text.contains("Last 30 days"));
    assert!(text.contains("1,240 mc  (86 chats)"));
    assert!(text.contains("credits earned"));
    assert!(text.contains("71 h   (of which working 9 h)"), "{text}");
    assert!(text.contains("No earnings guarantee: credits depend on demand."));
}

#[test]
fn render_down_text() {
    let text = render::render_down(120, "", false).join("\n");
    assert!(text.contains(sock::START));
}

fn line_of(s: &Status) -> String {
    status_line(s).0
}

#[test]
fn line_not_connected() {
    let mut s = sample();
    s.connected = false;
    assert_eq!(status_line(&s), ("Not connected to the coordinator — retrying".into(), Tone::Warn));
}

#[test]
fn line_quarantined_beats_pause() {
    let mut s = sample();
    s.engine = "quarantined".into();
    s.paused = true;
    assert_eq!(line_of(&s), "Engine stopped: the agent needs a restart");
}

#[test]
fn line_paused() {
    let mut s = sample();
    s.paused = true;
    assert_eq!(line_of(&s), "Paused — resume with [P]");
}

#[test]
fn line_busy_hot_and_other() {
    let mut s = with_reason("BUSY", false);
    assert_eq!(line_of(&s), "Card busy with another program — the free seat is suspended");
    s.gpu.as_mut().unwrap().temperature_celsius = 88;
    assert_eq!(line_of(&s), "Card hot (88 °C) — it resumes by itself");
}

#[test]
fn line_free_seat() {
    let mut s = with_reason("OK", true);
    assert_eq!(status_line(&s), ("Free seat active on qwen3.5-9b".into(), Tone::Good));
    s.contribution.as_mut().unwrap().in_chain = true;
    assert_eq!(line_of(&s), "Free seat active on qwen3.5-9b (in chain)");
}

#[test]
fn line_warming_up() {
    assert_eq!(line_of(&with_reason("WARMING_UP", false)), "Starting qwen3.5-9b — free seat shortly");
}

#[test]
fn line_mode_free_chosen_and_automatic() {
    let mut s = with_reason("MODE_FREE", false);
    s.preference.mode = "free".into();
    assert_eq!(line_of(&s), "FARM mode — serving qwen3.5-9b");
    s.preference.mode = "use".into();
    assert_eq!(line_of(&s), "FARM mode (automatic) — back to use at the next chat on qwen3.5-9b");
}

#[test]
fn line_not_placed() {
    let mut s = with_reason("NOT_PLACED", false);
    s.preference.model = Some("qwen3.6-35b-a3b".into());
    assert_eq!(
        line_of(&s),
        "Serving qwen3.5-9b. qwen3.6-35b-a3b is paid in credits: the card was not placed in one of its chains"
    );
}

#[test]
fn line_unverified_quality_account_missing() {
    assert_eq!(line_of(&with_reason("UNVERIFIED", false)), "Serving, checks in progress — free seat after the first checks");
    assert_eq!(line_of(&with_reason("BELOW_QUALITY", false)), "Serving below the speed threshold — no free seat on qwen3.5-9b");
    assert_eq!(line_of(&with_reason("NO_ACCOUNT", false)), "Card without a linked user — link your user on the site");
    assert_eq!(line_of(&with_reason("MODEL_MISSING", false)), "qwen3.5-9b is not installed on this PC");
}

#[test]
fn line_no_contribution_status() {
    let mut s = sample();
    s.contribution = None;
    assert_eq!(line_of(&s), "Serving qwen3.5-9b");
}

/// Render a full-width screen so the test output can be inspected as one frame.
#[test]
fn capture_80() {
    let mut s = sample();
    s.contribution.as_mut().unwrap().stable_until = Some(1_800_000_000);
    for l in render::render(&s, 80, 30, 0, "", false) {
        println!("|{l}");
    }
}


fn temp_socket(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("publicompute-test-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("agent.sock");
    let _ = std::fs::remove_file(&path);
    path
}

/// Reply to one socket connection and return the request line for assertions.
fn fake_agent(path: &PathBuf, reply: Value) -> mpsc::Receiver<Value> {
    let listener = UnixListener::bind(path).unwrap();
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let (mut s, _) = listener.accept().unwrap();
        let mut line = String::new();
        BufReader::new(s.try_clone().unwrap()).read_line(&mut line).unwrap();
        let _ = tx.send(serde_json::from_str(&line).unwrap());
        writeln!(s, "{reply}").unwrap();
    });
    rx
}

#[test]
fn command_status_and_json() {
    let p = temp_socket("status");
    let rx = fake_agent(&p, json!({"ok": true, "status": sample_json()}));
    let out = command(&Socket::Explicit(p.clone()), &["status".into()]).unwrap();
    assert_eq!(rx.recv().unwrap(), json!({"op": "status"}));
    assert_eq!(out, "Free seat active on qwen3.5-9b");

    let p = temp_socket("status-json");
    fake_agent(&p, json!({"ok": true, "status": sample_json()}));
    let out = command(&Socket::Explicit(p.clone()), &["status".into(), "--json".into()]).unwrap();
    let v: Value = serde_json::from_str(&out).unwrap();
    assert_eq!(v["gpu"]["temperature_celsius"], 62);
}

#[test]
fn command_use_sends_preference() {
    let p = temp_socket("use");
    let rx = fake_agent(&p, json!({"ok": true, "revision": 4}));
    let out = command(&Socket::Explicit(p.clone()), &["use".into(), "qwen3.5-9b".into(), "--quant".into(), "Q4_K_M".into()]).unwrap();
    assert_eq!(out, "using qwen3.5-9b");
    assert_eq!(rx.recv().unwrap(), json!({"op": "preference", "mode": "use", "model": "qwen3.5-9b", "quant": "Q4_K_M"}));
}

#[test]
fn command_use_shows_agent_refusal() {
    let p = temp_socket("refuse");
    fake_agent(&p, json!({"ok": false, "error": "not_installed", "text": "model x is not installed on this machine"}));
    let err = command(&Socket::Explicit(p.clone()), &["use".into(), "x".into()]).unwrap_err();
    assert_eq!(err, "model x is not installed on this machine");
}

#[test]
fn command_farm() {
    let p = temp_socket("farm");
    let rx = fake_agent(&p, json!({"ok": true, "revision": 5}));
    assert_eq!(command(&Socket::Explicit(p.clone()), &["farm".into()]).unwrap(), "FARM mode: set");
    assert_eq!(rx.recv().unwrap(), json!({"op": "preference", "mode": "free"}));
}

/// Serve one reply per connection, in order: pause and start ask for the
/// status before changing it.
fn fake_agent_seq(path: &PathBuf, replies: Vec<Value>) -> mpsc::Receiver<Value> {
    let listener = UnixListener::bind(path).unwrap();
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        for reply in replies {
            let (mut s, _) = listener.accept().unwrap();
            let mut line = String::new();
            BufReader::new(s.try_clone().unwrap()).read_line(&mut line).unwrap();
            let _ = tx.send(serde_json::from_str(&line).unwrap());
            writeln!(s, "{reply}").unwrap();
        }
    });
    rx
}

fn status_with_pause(paused: bool) -> Value {
    let mut v = sample_json();
    v["paused"] = json!(paused);
    json!({"ok": true, "status": v})
}

#[test]
fn command_pause_and_start() {
    let p = temp_socket("pause");
    let rx = fake_agent_seq(&p, vec![status_with_pause(false), json!({"ok": true})]);
    assert_eq!(command(&Socket::Explicit(p.clone()), &["pause".into()]).unwrap(), "paused");
    assert_eq!(rx.recv().unwrap(), json!({"op": "status"}));
    assert_eq!(rx.recv().unwrap(), json!({"op": "pause", "active": true}));

    let p = temp_socket("start");
    let rx = fake_agent_seq(&p, vec![status_with_pause(true), json!({"ok": true})]);
    assert_eq!(command(&Socket::Explicit(p.clone()), &["start".into()]).unwrap(), "started");
    assert_eq!(rx.recv().unwrap(), json!({"op": "status"}));
    assert_eq!(rx.recv().unwrap(), json!({"op": "pause", "active": false}));
}

#[test]
fn command_pause_and_start_already_there() {
    let p = temp_socket("pause-again");
    let rx = fake_agent_seq(&p, vec![status_with_pause(true)]);
    let out = command(&Socket::Explicit(p.clone()), &["pause".into()]).unwrap();
    assert!(out.starts_with("already paused") && out.contains("publicompute start"));
    assert_eq!(rx.recv().unwrap(), json!({"op": "status"}));
    assert!(rx.recv_timeout(std::time::Duration::from_millis(200)).is_err());

    let p = temp_socket("start-again");
    fake_agent_seq(&p, vec![status_with_pause(false)]);
    let out = command(&Socket::Explicit(p.clone()), &["start".into()]).unwrap();
    assert!(out.starts_with("already running") && out.contains("publicompute pause"));
}

#[test]
fn messages_without_agent() {
    use std::io::ErrorKind;
    let path = PathBuf::from("/run/publicompute-agent/agent.sock");
    for kind in [ErrorKind::NotFound, ErrorKind::ConnectionRefused] {
        assert!(sock::connect_error(&path, &kind.into(), false).contains("curl -fsSL https://publicompute.com/install.sh | sh"));
        assert!(sock::connect_error(&path, &kind.into(), true).contains("sudo systemctl start publicompute-agent"));
    }
    let permission = sock::connect_error(&path, &ErrorKind::PermissionDenied.into(), true);
    assert!(permission.contains("publicompute-agent group") && permission.contains("log out and back in"));
    assert!(command(&Socket::Explicit(temp_socket("none")), &["status".into()]).is_err());
}

#[test]
fn gpu_lost_message_at_40_80_120_columns() {
    let mut value = sample_json();
    value["gpu_lost"] = true.into();
    value["connected"] = false.into();
    let s: Status = serde_json::from_value(value).unwrap();
    assert_eq!(status_line(&s), (GPU_LOST.into(), Tone::Warn));
    for width in [40, 80, 120] {
        let lines = render::render(&s, width, 40, 0, "", false);
        assert!(lines.iter().all(|line| visible(line) <= width as usize));
        let text = lines.iter().flat_map(|line| line.split_whitespace()).collect::<Vec<_>>().join(" ");
        assert!(text.contains(GPU_LOST), "{text}");
    }
}

#[test]
fn uninstall_without_agent_still_runs_script() {
    use std::{fs, process::Command};
    let socket = temp_socket("uninstall");
    let dir = socket.parent().unwrap();
    let script = dir.join("uninstall.sh");
    fs::write(&script, "test \"$1\" = --keep-models && printf done > \"$2\"\n").unwrap();
    let marker = dir.join("ran");
    let mut script_command = Command::new("sh");
    script_command.arg(&script);
    let args = ["uninstall".into(), "--keep-models".into()];
    command_with_uninstaller(&Socket::Explicit(socket.clone()), &args, |args| {
        let mut args = args.to_vec();
        args.push(marker.to_string_lossy().into_owned());
        run_uninstaller(&mut script_command, &args)
    })
    .unwrap();
    assert_eq!(fs::read_to_string(&marker).unwrap(), "done");
    fs::remove_dir_all(dir).unwrap();
}
