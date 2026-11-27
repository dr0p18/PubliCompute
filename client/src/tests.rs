use std::{
    io::{BufRead, BufReader, Read, Write},
    os::unix::net::{UnixListener, UnixStream},
    path::PathBuf,
    sync::mpsc,
    thread,
    time::Duration,
};

use serde_json::{json, Value};
use unicode_width::UnicodeWidthStr;

use crate::{command, command_with_uninstaller, follow_updates, render, run_uninstaller, sock::{self, Socket}, status::*, Msg};

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

#[test]
fn tui_clears_the_switch_notice_only_after_the_requested_weights_are_loaded() {
    let mut status = sample_json();
    let mut pending = Some(sock::LocalModel {
        id: "qwen3.6-35b-a3b".into(), quant: "UD-Q4_K_XL".into(), weights_sha256: Some("cc".repeat(32)), loaded: false,
    });
    let mut notice = "switching to qwen3.6-35b-a3b".to_string();
    crate::clear_finished_switch(&serde_json::from_value(status.clone()).unwrap(), &mut pending, &mut notice);
    assert!(!notice.is_empty());

    status["loaded_model"] = json!("qwen3.6-35b-a3b");
    status["loaded_quant"] = json!("Q4_K_M");
    crate::clear_finished_switch(&serde_json::from_value(status.clone()).unwrap(), &mut pending, &mut notice);
    assert!(!notice.is_empty());

    status["loaded_quant"] = json!("UD-Q4_K_XL");
    status["weights_sha256"] = json!("bb".repeat(32));
    crate::clear_finished_switch(&serde_json::from_value(status.clone()).unwrap(), &mut pending, &mut notice);
    assert!(!notice.is_empty());

    status["weights_sha256"] = json!("cc".repeat(32));
    crate::clear_finished_switch(&serde_json::from_value(status).unwrap(), &mut pending, &mut notice);
    assert!(notice.is_empty());
    assert!(pending.is_none());
}

fn with_reason(reason: &str, free_seat: bool) -> Status {
    let mut s = sample();
    let c = s.contribution.as_mut().unwrap();
    c.reason = format!("FREE_SEAT_REASON_{reason}");
    c.free_seat = free_seat;
    s
}

/// Measure cells after removing terminal styling sequences.
fn visible(line: &str) -> usize {
    let mut plain = String::new();
    let mut chars = line.chars();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            for e in chars.by_ref() {
                if e == 'm' {
                    break;
                }
            }
        } else {
            plain.push(c);
        }
    }
    plain.width()
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
fn render_adapts_to_30_60_and_100_columns() {
    let mut s = sample();
    s.gpu.as_mut().unwrap().name = "測試測試測試測試測試測試測試測試 GPU".into();
    s.contribution.as_mut().unwrap().stable_until = Some(i64::MAX);
    s.chain_id = Some("a-very-long-chain-identifier-that-goes-on".into());
    for w in [30u16, 60, 100] {
        for color in [false, true] {
            let dashboard = render::render(&s, w, 40, 0, "Use which model? [1] medium Q4_K_M [2] deepseek Q8_0", color);
            for key in ["[U]", "[F]", "[P]", "[C]", "[Q]"] {
                assert!(dashboard.iter().any(|line| line.contains(key)));
            }
            let frames = [
                dashboard,
                render::render_down(w, "cannot connect to /var/lib/publicompute/agent.sock: No such file", color),
                render::render_logo(w, color),
                render::render_intro(w, 40, color, Duration::from_millis(540)),
                render::render_intro(w, 40, color, Duration::from_millis(2400)),
                render::render_empty_chat("medium", w, color, None),
                render::render_empty_chat("medium", w, color, Some(Duration::from_secs(7))),
            ];
            for line in frames.iter().flatten() {
                assert!(visible(line) <= w as usize, "width {w}: {line:?}");
                assert!(color || !line.contains('\x1b'), "{line:?}");
            }
        }
        let logo = render::render_logo(w, false);
        match w {
            30 => assert_eq!(logo, ["PubliCompute"]),
            60 => {
                assert_eq!(logo.len(), 6);
                assert!(logo[0].starts_with("█▀▀█    █▀▀█"));
                assert!(logo.iter().any(|line| line.contains("PubliCompute")));
            }
            100 => {
                assert_eq!(logo.len(), 6);
                assert!(logo[0].starts_with("█▀▀█    █▀▀█"));
                assert!(logo[1].width() > 60);
                assert!(!logo.iter().any(|line| line.contains("PubliCompute")));
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn render_uses_signal_for_indicators_and_keeps_warnings_visible() {
    let signal = "\x1b[38;2;198;255;61m";
    let mut s = sample();
    let active = render::render(&s, 60, 24, 0, "", true).join("\n");
    assert_eq!(active.matches(signal).count(), 1);
    assert!(active.contains(&format!("{signal}ONLINE\x1b[0m")));
    s.paused = true;
    let paused = render::render(&s, 60, 24, 0, "! command failed", true).join("\n");
    assert!(!paused.contains(signal));
    assert!(paused.contains("\x1b[1;38;2;255;255;255m  ! command failed"));
    assert!(render::chat_prompt(true).contains(signal));
}

#[test]
fn empty_chat_eyes_follow_the_site_cycle() {
    let frame = |ms| render::chat_eyes(60, true, Some(Duration::from_millis(ms)));
    assert!(frame(0).iter().all(String::is_empty));
    assert_ne!(frame(2000), frame(3000));
    assert_ne!(frame(4000), frame(5000));
    assert_eq!(frame(2000), frame(9000));
    assert_eq!(frame(0), frame(13_000));
    assert!(!frame(5000).join("").contains("198;255;61"));
    assert!(frame(7000).join("").contains("198;255;61"));
    assert_eq!(render::chat_eyes(60, true, None), frame(3000)[5..9]);
}

#[test]
fn render_fits_height_and_keeps_keys() {
    let lines = render::render(&sample(), 80, 6, 0, "", false);
    assert_eq!(lines.len(), 6);
    assert_eq!(lines.last().unwrap(), "[U] use a model  [F] free card  [P] pause/resume  [C] chat link  [Q] quit");
    let notice = "downloading medium (20.8 GiB remaining), see publicompute status";
    let lines = render::render(&sample(), 80, 6, 0, notice, false);
    assert_eq!(lines.len(), 6);
    assert!(lines.iter().any(|line| line.trim() == notice));
}

#[test]
fn render_shows_usage_without_credits() {
    let text = render::render(&sample(), 80, 40, 0, "", false).join("\n");
    assert!(text.contains("LAST 30 DAYS"));
    assert!(text.contains("Free chats 86"));
    assert!(text.contains("In service 71 h"));
    assert!(text.contains("Working    9 h"));
    assert!(!text.contains("credits"));
    assert!(!text.contains(" mc"));
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
    assert_eq!(line_of(&s), "Free card — serving qwen3.5-9b");
    s.contribution.as_mut().unwrap().serving = None;
    assert_eq!(line_of(&s), "Free card — waiting for work");
    s.preference.mode = "use".into();
    assert_eq!(line_of(&s), "Free card (automatic) — back to use at the next chat on qwen3.5-9b");
}

#[test]
fn line_not_placed() {
    let mut s = with_reason("NOT_PLACED", false);
    s.preference.model = Some("qwen3.6-35b-a3b".into());
    assert_eq!(
        line_of(&s),
        "Serving qwen3.5-9b. The card was not placed in a chain of qwen3.6-35b-a3b"
    );
    s.contribution.as_mut().unwrap().detail =
        "this model is waiting for memory measurements before it can run in a chain".into();
    assert_eq!(
        status_line(&s),
        ("This model is waiting for memory measurements before it can run in a chain".into(), Tone::Plain)
    );
}

#[test]
fn line_new_reasons_prefer_coordinator_detail() {
    let mut s = with_reason("WAITING_PEERS", false);
    s.contribution.as_mut().unwrap().detail =
        "waiting for peers: 1 card(s) online in your region, a chain needs at least 2".into();
    assert_eq!(line_of(&s), "Waiting for peers: 1 card(s) online in your region, a chain needs at least 2");
    assert_eq!(line_of(&with_reason("WAITING_PEERS", false)), "Waiting for other cards to form a chain of qwen3.5-9b");
    assert_eq!(line_of(&with_reason("CHAIN_FULL", false)), "The chains of qwen3.5-9b are full — waiting for a seat");
    assert_eq!(
        line_of(&with_reason("NETWORK_BELOW_STANDARD", false)),
        "Network below the chain standard (100 Mbit/s each way, 25 ms)"
    );
    assert_eq!(line_of(&with_reason("VRAM_TOO_SMALL", false)), "This card cannot hold a stage of qwen3.5-9b");
    assert_eq!(
        line_of(&with_reason("NAT_BLOCKED", false)),
        "The router blocks incoming connections — the card cannot join a chain"
    );
}

#[test]
fn line_unknown_reason_uses_detail() {
    let mut s = with_reason("SOMETHING_NEW", false);
    assert_eq!(line_of(&s), "Serving qwen3.5-9b");
    s.contribution.as_mut().unwrap().detail = "a newer coordinator explains it".into();
    assert_eq!(line_of(&s), "A newer coordinator explains it");
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

#[test]
#[ignore = "prints the preview on demand"]
fn capture_ui_preview() {
    let mut active = sample();
    active.loaded_model = Some("qwen3.6-35b-a3b".into());
    active.contribution.as_mut().unwrap().serving = active.loaded_model.clone();
    active.contribution.as_mut().unwrap().in_chain = true;
    active.chain_id = Some("chain-012a".into());
    let mut free = active.clone();
    free.preference.mode = "free".into();
    free.loaded_model = None;
    free.active_leases = 0;
    free.chain_id = None;
    free.gpu.as_mut().unwrap().vram_used_bytes = 0;
    let contribution = free.contribution.as_mut().unwrap();
    contribution.serving = None;
    contribution.free_seat = false;
    contribution.in_chain = false;
    contribution.reason = "FREE_SEAT_REASON_MODE_FREE".into();
    let mut paused = active.clone();
    paused.paused = true;
    for width in [100, 60] {
        for (name, frame) in [
            ("SERVING", render::render(&active, width, 24, 0, "", false)),
            ("FREE CARD", render::render(&free, width, 24, 0, "", false)),
            ("PAUSED", render::render(&paused, width, 24, 0, "", false)),
            ("EMPTY CHAT", render::render_empty_chat("qwen3.6-35b-a3b", width, false, None)),
            ("LOGO", render::render_logo(width, false)),
        ] {
            println!("# {width} COLUMNS / {name}\n\n{}\n", frame.join("\n"));
        }
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

fn accept_request(listener: &UnixListener, op: &str) -> UnixStream {
    let (stream, _) = listener.accept().unwrap();
    stream.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
    let mut line = String::new();
    BufReader::new(&stream).read_line(&mut line).unwrap();
    assert_eq!(serde_json::from_str::<Value>(&line).unwrap()["op"], op);
    stream
}

#[test]
fn silent_follow_falls_back_to_status_and_reports_down_if_also_silent() {
    let path = temp_socket("silent-follow");
    let listener = UnixListener::bind(&path).unwrap();
    let server = thread::spawn(move || {
        let _follow = accept_request(&listener, "follow");
        let mut status = accept_request(&listener, "status");
        assert_eq!(status.read(&mut [0]).unwrap(), 0);
    });

    let mut events = Vec::new();
    assert!(follow_updates(&Socket::Explicit(path), Duration::from_millis(200), |event| {
        events.push(event);
        true
    }));
    assert!(matches!(events.as_slice(), [Msg::Down(why)] if why.contains("stopped sending")));
    server.join().unwrap();
}

#[test]
fn quiet_healthy_agent_keeps_panel_status_during_reconnect() {
    let path = temp_socket("quiet-healthy");
    let listener = UnixListener::bind(&path).unwrap();
    let server = thread::spawn(move || {
        let reply = json!({"ok": true, "status": sample_json()});
        let mut follow = accept_request(&listener, "follow");
        writeln!(follow, "{reply}").unwrap();
        let mut status = accept_request(&listener, "status");
        writeln!(status, "{reply}").unwrap();
        let mut next = accept_request(&listener, "follow");
        writeln!(next, "{reply}").unwrap();
    });

    let socket = Socket::Explicit(path);
    let silence = Duration::from_millis(200);
    let mut snapshots = Vec::new();
    let mut emit = |event| {
        let Msg::Status(status) = event else { panic!("healthy agent reported down") };
        snapshots.push(status);
        snapshots.len() < 3
    };
    assert!(follow_updates(&socket, silence, &mut emit));
    assert!(!follow_updates(&socket, silence, &mut emit));
    assert_eq!(snapshots.len(), 3);
    assert!(snapshots.iter().all(|status| status.connected && !status.paused));
    server.join().unwrap();
}

#[test]
fn follow_watchdog_resets_for_snapshots_but_not_for_invalid_lines() {
    let path = temp_socket("watchdog");
    let listener = UnixListener::bind(&path).unwrap();
    let (closed, wait) = mpsc::channel();
    let server = thread::spawn(move || {
        let reply = json!({"ok": true, "status": sample_json()});
        let mut follow = accept_request(&listener, "follow");
        for _ in 0..3 {
            writeln!(follow, "{reply}").unwrap();
            thread::sleep(Duration::from_millis(250));
        }
        let invalid = [
            json!({"ok": true, "status": {}}),
            json!({"status": sample_json()}),
            json!({"ok": true, "status": {"connected": true}}),
            json!({"ok": true, "status": {"paused": false}}),
        ];
        for reply in invalid.iter().cycle().take(8) {
            if writeln!(follow, "{reply}").is_err() {
                return;
            }
            if wait.recv_timeout(Duration::from_millis(100)).is_ok() {
                return;
            }
        }
        panic!("invalid lines kept the follower open");
    });

    let mut snapshots = 0;
    let result = sock::follow(&Socket::Explicit(path), Duration::from_millis(400), |_| {
        snapshots += 1;
        true
    });
    let _ = closed.send(());
    assert_eq!(snapshots, 3);
    assert!(result.unwrap_err().contains("stopped sending"));
    server.join().unwrap();
}

#[cfg(target_os = "linux")]
fn queued_listener(path: &std::path::Path) -> (UnixListener, UnixStream) {
    use rustix::net::{self, AddressFamily, SocketAddrUnix, SocketType};

    let listener = net::socket(AddressFamily::UNIX, SocketType::STREAM, None).unwrap();
    net::bind(&listener, &SocketAddrUnix::new(path).unwrap()).unwrap();
    net::listen(&listener, 0).unwrap();
    let queued = UnixStream::connect(path).unwrap();
    (UnixListener::from(listener), queued)
}

#[cfg(target_os = "linux")]
#[test]
fn status_times_out_if_the_listen_queue_stays_full() {
    let path = temp_socket("full-queue");
    let (_listener, _queued) = queued_listener(&path);
    let (done, wait) = mpsc::channel();
    let client = thread::spawn(move || {
        done.send(sock::get_status(&Socket::Explicit(path))).unwrap();
    });
    let err = wait.recv_timeout(Duration::from_secs(5)).unwrap().unwrap_err();
    assert!(err.contains("cannot connect") && err.contains("timed out"), "{err}");
    client.join().unwrap();
}

#[cfg(target_os = "linux")]
#[test]
fn status_read_uses_only_the_time_left_after_connecting() {
    use rustix::event::{poll, PollFd, PollFlags, Timespec};

    let path = temp_socket("shared-budget");
    let (listener, _queued) = queued_listener(&path);
    let server = thread::spawn(move || {
        thread::sleep(Duration::from_secs(2));
        drop(listener.accept().unwrap());
        let mut fds = [PollFd::new(&listener, PollFlags::IN)];
        let timeout = Timespec::try_from(Duration::from_secs(3)).unwrap();
        assert_ne!(poll(&mut fds, Some(&timeout)).unwrap(), 0, "status did not connect");
        let mut status = accept_request(&listener, "status");
        assert_eq!(status.read(&mut [0]).unwrap(), 0);
    });

    let start = std::time::Instant::now();
    assert!(sock::get_status(&Socket::Explicit(path)).is_err());
    let elapsed = start.elapsed();
    assert!(elapsed >= Duration::from_secs(2) && elapsed < Duration::from_millis(4500), "{elapsed:?}");
    server.join().unwrap();
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
fn blocked_agent_displays_update_reason_in_status_and_dashboard() {
    let mut status = sample_json();
    status["connected"] = json!(false);
    status["engine"] = json!("blocked");
    status["gpu_lost"] = json!(true);
    status["update_required"] = json!("the coordinator requires agent version 0.1.7");
    let expected = "Update required: the coordinator requires agent version 0.1.7 — run the installer again";
    for (name, args) in [("blocked-default", vec![]), ("blocked-status", vec!["status".into()])] {
        let p = temp_socket(name);
        let rx = fake_agent(&p, json!({"ok": true, "status": status}));
        assert_eq!(command(&Socket::Explicit(p), &args).unwrap(), format!("! {expected}"));
        assert_eq!(rx.recv().unwrap(), json!({"op":"status"}));
    }
    let p = temp_socket("blocked-start");
    fake_agent(&p, json!({"ok": true, "status": status}));
    assert_eq!(command(&Socket::Explicit(p), &["start".into()]).unwrap_err(), expected);
    let status = serde_json::from_value(status).unwrap();
    let rows = render::render(&status, 100, 30, 0, "", false);
    assert!(rows.iter().any(|row| row.trim() == format!("! {expected}")), "{rows:?}");
}

#[test]
fn command_use_sends_preference() {
    let p = temp_socket("use");
    let rx = fake_agent(&p, json!({"ok": true, "revision": 4}));
    let out = command(&Socket::Explicit(p.clone()), &["use".into(), "qwen3.5-9b".into(), "--quant".into(), "Q4_K_M".into()]).unwrap();
    assert_eq!(out, "using qwen3.5-9b");
    assert_eq!(rx.recv().unwrap(), json!({"op": "preference", "mode": "use", "model": "qwen3.5-9b", "quant": "Q4_K_M"}));

    let p = temp_socket("use-variant");
    let rx = fake_agent(&p, json!({"ok": true, "revision": 5}));
    let sha = "cc".repeat(32);
    let args = ["use", "qwen3.6-35b-a3b", "--weights-sha256", &sha].map(str::to_owned);
    command(&Socket::Explicit(p), &args).unwrap();
    assert_eq!(rx.recv().unwrap(), json!({
        "op": "preference", "mode": "use", "model": "qwen3.6-35b-a3b", "quant": null, "weights_sha256": sha,
    }));
}

#[test]
fn command_use_shows_agent_refusal() {
    let p = temp_socket("refuse");
    fake_agent(&p, json!({"ok": false, "error": "not_installed", "text": "model x is not installed on this machine"}));
    let err = command(&Socket::Explicit(p.clone()), &["use".into(), "x".into()]).unwrap_err();
    assert_eq!(err, "model x is not installed on this machine");
}

#[test]
fn command_free_replaces_farm() {
    let p = temp_socket("free");
    let rx = fake_agent(&p, json!({"ok": true, "revision": 5}));
    assert_eq!(command(&Socket::Explicit(p.clone()), &["free".into()]).unwrap(), "Free card: set");
    assert_eq!(rx.recv().unwrap(), json!({"op": "preference", "mode": "free"}));
    let err = command(&Socket::Explicit(p), &["farm".into()]).unwrap_err();
    assert!(err.starts_with("unknown command"));
    assert!(err.contains(" | free | "));
    assert!(!err.contains("farm"));
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
    let rx = fake_agent_seq(&p, vec![status_with_pause(true), json!({"ok": true})]);
    let out = command(&Socket::Explicit(p.clone()), &["pause".into()]).unwrap();
    assert!(out.starts_with("already paused") && out.contains("publicompute start"));
    assert_eq!(rx.recv().unwrap(), json!({"op": "status"}));
    assert_eq!(rx.recv().unwrap(), json!({"op": "pause", "active": true}));

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

#[test]
fn status_displays_network_and_checks_both_rates_and_age() {
    let mut status = sample();
    let original = status_line(&status).0;
    assert_eq!(status_text(&status, 1_000_000), original);

    let now = 1_000_000;
    status.network = Some(Network {
        download_bits_per_sec: 105_000_000,
        upload_bits_per_sec: 100_000_000,
        bandwidth_measured_at: Some(now - 8 * 86400),
        coordinator_rtt_micros: Some(14_200),
    });
    let measured = status_text(&status, now);
    assert!(measured.ends_with("network: 105.0/100.0 Mbit/s, 14.2 ms"));
    let warning = "chains: below the network standard (100 Mbit/s up and down)";
    assert!(status_text(&status, now + 1).ends_with(warning));

    status.network.as_mut().unwrap().download_bits_per_sec = 99_999_999;
    assert!(status_text(&status, now).ends_with(warning));
    status.network.as_mut().unwrap().download_bits_per_sec = 100_000_000;
    status.network.as_mut().unwrap().upload_bits_per_sec = 99_999_999;
    assert!(status_text(&status, now).ends_with(warning));
}

#[test]
fn use_prints_the_agent_message() {
    for (state, text) in [
        ("downloading", "downloading medium (20.8 GiB remaining), see publicompute status"),
        ("switching", "switching to medium"),
    ] {
        let path = temp_socket(state);
        fake_agent(&path, json!({"ok": true, "state": state, "text": text}));
        assert_eq!(command(&Socket::Explicit(path), &["use".into(), "medium".into()]).unwrap(), text);
    }
}

#[test]
fn version_accepts_an_agent_version_without_other_status_fields() {
    for alias in ["--version", "version"] {
        let path = temp_socket(alias);
        let received = fake_agent(&path, json!({"ok": true, "status": {"version": "0.8.1-beta"}}));
        assert_eq!(command(&Socket::Explicit(path), &[alias.into()]).unwrap(),
            concat!("publicompute ", env!("CARGO_PKG_VERSION"), "\npublicompute-agent 0.8.1-beta"));
        assert_eq!(received.recv().unwrap(), json!({"op": "status"}));
    }
}

#[test]
fn version_succeeds_without_an_agent() {
    let path = temp_socket("version-offline");
    assert_eq!(command(&Socket::Explicit(path), &["--version".into()]).unwrap(),
        concat!("publicompute ", env!("CARGO_PKG_VERSION")));
}

#[test]
fn status_displays_model_progress_and_failure() {
    let mut status = sample_json();
    status["download"] = json!({"id":"medium", "state":"downloading", "bytes":1073741824_u64, "total_bytes":2147483648_u64});
    let s: Status = serde_json::from_value(status.clone()).unwrap();
    assert_eq!(s.download_line().as_deref(), Some("download: medium 1.0/2.0 GiB"));
    assert!(render::render(&s, 80, 30, 0, "", false).iter().any(|line| line.contains("download: medium 1.0/2.0 GiB")));
    status["download"]["state"] = json!("failed");
    status["download"]["error"] = json!("SHA-256 mismatch");
    let s: Status = serde_json::from_value(status).unwrap();
    assert_eq!(s.download_line().as_deref(), Some("! download: medium failed: SHA-256 mismatch"));
}
