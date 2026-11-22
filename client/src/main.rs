//! An interactive terminal opens a live view when invoked without arguments.
//! Agent commands return a status line and chat streams the answer directly.

mod chat;
mod render;
mod sock;
mod status;

use std::{
    io::{IsTerminal, Write},
    path::{Path, PathBuf},
    process::Command,
    sync::mpsc,
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use crossterm::{
    cursor,
    event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers},
    execute,
    terminal::{self, EnterAlternateScreen, LeaveAlternateScreen},
};

use sock::Socket;
use status::Status;

const CHAT_URL: &str = "https://publicompute.com/chat";
const USAGE: &str = "usage: publicompute [--data-dir DIR | --socket FILE] [--version | version | status [--json] | use <model> [--quant Q] | farm | pause | start | uninstall [--keep-models | --remove-models] | login | logout | chat [--model ID] [--api URL] [\"question\"]]";
const UNINSTALLER: &str = "/opt/publicompute-agent/uninstall.sh";

fn main() {
    std::process::exit(run(std::env::args().skip(1).collect()));
}

fn run(args: Vec<String>) -> i32 {
    let (socket, rest) = match parse_args(args) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("{e}\n{USAGE}");
            return 1;
        }
    };
    if rest.is_empty() && std::io::stdout().is_terminal() && std::io::stdin().is_terminal() {
        return match interactive(&socket) {
            Ok(()) => 0,
            Err(e) => {
                eprintln!("error: {e}");
                1
            }
        };
    }
    let result = match rest.first().map(String::as_str) {
        Some("login") if rest.len() == 1 => chat::login(),
        Some("logout") if rest.len() == 1 => chat::logout(),
        Some("chat") => chat::run(&socket, &rest[1..]),
        _ => command(&socket, &rest).map(|out| {
            if !out.is_empty() {
                println!("{out}");
            }
        }),
    };
    match result {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("error: {e}");
            1
        }
    }
}

/// Extract socket overrides from any position and return the remaining command arguments.
fn parse_args(args: Vec<String>) -> Result<(Socket, Vec<String>), String> {
    let mut socket = None;
    let mut dir = None;
    let mut rest = Vec::new();
    let mut it = args.into_iter();
    while let Some(a) = it.next() {
        if a == "--" && rest.first().is_some_and(|command| command == "chat") {
            rest.push(a);
            rest.extend(it);
            break;
        }
        match a.as_str() {
            "--data-dir" => dir = Some(PathBuf::from(it.next().ok_or("--data-dir needs a value")?)),
            "--socket" => socket = Some(PathBuf::from(it.next().ok_or("--socket needs a value")?)),
            _ => rest.push(a),
        }
    }
    let socket = socket.or_else(|| dir.map(|p| p.join("agent.sock"))).map_or(Socket::Default, Socket::Explicit);
    Ok((socket, rest))
}

/// Dispatch a one-shot command and return its printable response or error.
fn command(socket: &Socket, args: &[String]) -> Result<String, String> {
    command_with_uninstaller(socket, args, uninstall)
}

fn command_with_uninstaller(
    socket: &Socket,
    args: &[String],
    uninstall: impl FnOnce(&[String]) -> Result<(), String>,
) -> Result<String, String> {
    let a: Vec<&str> = args.iter().map(String::as_str).collect();
    match a.as_slice() {
        ["--help"] | ["-h"] => Ok(USAGE.into()),
        ["--version"] | ["version"] => Ok(version(socket)),
        [] | ["status"] => {
            let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs();
            Ok(status::status_text(&sock::get_status(socket)?, now))
        }
        ["status", "--json"] => {
            let v = sock::request(socket, &serde_json::json!({"op": "status"}))?;
            Ok(v["status"].to_string())
        }
        ["farm"] => {
            sock::set_preference(socket, None, None)?;
            Ok("FARM mode: set".into())
        }
        ["use", model] => use_model(socket, model, None),
        ["use", model, "--quant", q] => use_model(socket, model, Some(q)),
        ["pause"] => switch_pause(socket, true),
        ["start"] => switch_pause(socket, false),
        ["uninstall"] | ["uninstall", "--keep-models" | "--remove-models"] => {
            uninstall(&args[1..])?;
            Ok(String::new())
        }
        _ => Err(format!("unknown command\n{USAGE}")),
    }
}

/// Pause or start the card. Asking first lets us tell the user when the card is
/// already in that state, instead of silently doing nothing.
fn switch_pause(socket: &Socket, pause: bool) -> Result<String, String> {
    if sock::get_status(socket)?.paused == pause {
        if pause {
            sock::set_pause(socket, true)?;
        }
        let notice = if pause {
            "already paused: run `publicompute start` to rejoin the network"
        } else {
            "already running: run `publicompute pause` to leave the network"
        };
        return Ok(notice.into());
    }
    sock::set_pause(socket, pause)?;
    Ok(if pause { "paused" } else { "started" }.into())
}

fn uninstall(args: &[String]) -> Result<(), String> {
    let script = Path::new(UNINSTALLER);
    if !script.is_file() {
        return Err(sock::connect_error(script, &std::io::ErrorKind::NotFound.into(), false));
    }
    let root = Command::new("id").arg("-u").output().is_ok_and(|o| o.status.success() && o.stdout == b"0\n");
    let mut command = if root {
        Command::new(script)
    } else {
        let mut sudo = Command::new("sudo");
        sudo.arg("--").arg(script);
        sudo
    };
    run_uninstaller(&mut command, args)
}

fn run_uninstaller(command: &mut Command, args: &[String]) -> Result<(), String> {
    let status = command.args(args).status().map_err(|e| format!("cannot run uninstaller: {e}"))?;
    if !status.success() {
        return Err(format!("uninstaller exited with {status}"));
    }
    Ok(())
}

fn version(socket: &Socket) -> String {
    let mut text = concat!("publicompute ", env!("CARGO_PKG_VERSION")).to_string();
    if let Ok(reply) = sock::request(socket, &serde_json::json!({"op": "status"})) {
        if let Some(version) = reply["status"]["version"].as_str() {
            text.push_str(&format!("\npublicompute-agent {version}"));
        }
    }
    text
}

fn use_model(socket: &Socket, model: &str, quant: Option<&str>) -> Result<String, String> {
    let reply = sock::set_preference(socket, Some(model), quant)?;
    Ok(use_reply_text(&reply, model))
}

fn use_reply_text(reply: &serde_json::Value, model: &str) -> String {
    // Older agents acknowledge successful preferences without a message.
    reply["text"].as_str().map_or_else(|| format!("using {model}"), str::to_owned)
}

fn clear_finished_switch(status: &Status, pending: &mut Option<(String, String)>, notice: &mut String) {
    if let Some((id, quant)) = pending {
        if status.download.is_none() && status.loaded_model.as_ref() == Some(id)
            && status.loaded_quant.as_ref().is_none_or(|loaded| loaded == quant)
        {
            *pending = None;
            notice.clear();
        }
    }
}

enum Msg {
    Status(Status),
    Down(String),
    Key(KeyEvent),
    Resize,
}

/// Restore the terminal on return and during panic unwinding. In raw mode
/// Ctrl-C arrives as a key event and follows the normal exit path.
struct TermGuard;

impl TermGuard {
    fn enter() -> std::io::Result<Self> {
        terminal::enable_raw_mode()?;
        let g = TermGuard;
        execute!(std::io::stdout(), EnterAlternateScreen, cursor::Hide)?;
        Ok(g)
    }
}

impl Drop for TermGuard {
    fn drop(&mut self) {
        let _ = execute!(std::io::stdout(), cursor::Show, LeaveAlternateScreen);
        let _ = terminal::disable_raw_mode();
    }
}

fn now_unix() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0)
}

fn draw(lines: &[String]) {
    let mut out = std::io::stdout().lock();
    // Clear one row at a time to avoid flicker, then erase any rows left below the new frame.
    let _ = write!(out, "\x1b[H");
    for l in lines {
        let _ = write!(out, "{l}\x1b[K\r\n");
    }
    let _ = write!(out, "\x1b[J");
    let _ = out.flush();
}

fn follow_updates(socket: &Socket, silence: Duration, mut emit: impl FnMut(Msg) -> bool) -> bool {
    match sock::follow(socket, silence, |status| emit(Msg::Status(status))) {
        Ok(()) => false,
        // A quiet subscription can still belong to an agent that answers commands.
        Err(why) => match sock::get_status(socket) {
            Ok(status) => emit(Msg::Status(status)),
            Err(_) => emit(Msg::Down(why)),
        },
    }
}

fn interactive(socket: &Socket) -> Result<(), String> {
    let color = std::env::var_os("NO_COLOR").is_none();
    let _guard = TermGuard::enter().map_err(|e| e.to_string())?;

    let (tx, rx) = mpsc::channel::<Msg>();

    // Keep retrying the subscription so the display recovers when the agent comes back.
    let t = tx.clone();
    let sock_path = socket.clone();
    thread::spawn(move || loop {
        if !follow_updates(&sock_path, sock::FOLLOW_TIMEOUT, |msg| t.send(msg).is_ok()) {
            return;
        }
        thread::sleep(Duration::from_secs(2));
    });
    // The key reader can remain blocked until process exit. It needs no join on shutdown.
    let t = tx;
    thread::spawn(move || loop {
        let msg = match event::read() {
            Ok(Event::Key(k)) if k.kind == KeyEventKind::Press => Msg::Key(k),
            Ok(Event::Resize(..)) => Msg::Resize,
            Ok(_) => continue,
            Err(_) => return,
        };
        if t.send(msg).is_err() {
            return;
        }
    });

    // Key events queue while the startup logo is displayed.
    let (w, _) = terminal::size().unwrap_or((80, 24));
    draw(&render::render_logo(w, color));
    thread::sleep(Duration::from_millis(700));

    let mut status: Option<Status> = None;
    let mut down = String::new();
    let mut notice = String::new();
    let mut pending_switch = None;
    let mut picking: Vec<sock::LocalModel> = Vec::new();

    loop {
        let (w, h) = terminal::size().unwrap_or((80, 24));
        match &status {
            Some(s) => draw(&render::render(s, w, h, now_unix(), &notice, color)),
            // Wait for an initial status or connection failure before replacing the startup view.
            None if down.is_empty() => {}
            None => draw(&render::render_down(w, &down, color)),
        }

        match rx.recv() {
            Ok(Msg::Status(s)) => {
                clear_finished_switch(&s, &mut pending_switch, &mut notice);
                status = Some(s);
            }
            Ok(Msg::Down(why)) => {
                status = None;
                down = why;
            }
            Ok(Msg::Resize) => {}
            Ok(Msg::Key(k)) => {
                if k.modifiers.contains(KeyModifiers::CONTROL) && k.code == KeyCode::Char('c') {
                    return Ok(());
                }
                if !picking.is_empty() {
                    notice = match k.code {
                        KeyCode::Char(c) => match c.to_digit(10).filter(|d| *d >= 1 && *d as usize <= picking.len()) {
                            Some(d) => {
                                let m = &picking[d as usize - 1];
                                match sock::set_preference(socket, Some(&m.id), Some(&m.quant)) {
                                    Ok(reply) => {
                                        if matches!(reply["state"].as_str(), Some("switching" | "downloading")) {
                                            pending_switch = Some((m.id.clone(), m.quant.clone()));
                                        }
                                        use_reply_text(&reply, &m.id)
                                    }
                                    Err(e) => e,
                                }
                            }
                            None => String::new(),
                        },
                        _ => String::new(),
                    };
                    picking.clear();
                    continue;
                }
                notice.clear();
                pending_switch = None;
                match k.code {
                    KeyCode::Char('q') | KeyCode::Char('Q') | KeyCode::Esc => return Ok(()),
                    KeyCode::Char('u') | KeyCode::Char('U') if status.is_some() => match sock::get_models(socket) {
                        Ok(m) if m.is_empty() => notice = "no models installed".into(),
                        Ok(m) => {
                            let list: Vec<String> =
                                m.iter().enumerate().map(|(i, x)| format!("[{}] {} {}", i + 1, x.id, x.quant)).collect();
                            notice = format!("Use which model?  {}  (any other key cancels)", list.join("  "));
                            picking = m;
                        }
                        Err(e) => notice = e,
                    },
                    KeyCode::Char('f') | KeyCode::Char('F') if status.is_some() => {
                        notice = match sock::set_preference(socket, None, None) {
                            Ok(_) => "FARM mode: set".into(),
                            Err(e) => e,
                        };
                    }
                    KeyCode::Char('p') | KeyCode::Char('P') => {
                        if let Some(s) = &status {
                            if let Err(e) = sock::set_pause(socket, !s.paused) {
                                notice = e;
                            }
                        }
                    }
                    KeyCode::Char('c') | KeyCode::Char('C') => notice = format!("Chat: {CHAT_URL}"),
                    _ => {}
                }
            }
            Err(_) => return Ok(()),
        }
    }
}

#[cfg(test)]
mod tests;
