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
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
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
const USAGE: &str = "usage: publicompute [--data-dir DIR | --socket FILE] [--version | version | status [--json] | use <model> [--quant Q] | free | pause | start | uninstall [--keep-models | --remove-models] | login | logout | chat [--model ID] [--api URL] [\"question\"]]";
const UNINSTALLER: &str = "/opt/publicompute-agent/uninstall.sh";

fn main() {
    std::process::exit(run(std::env::args().skip(1).collect()));
}

fn run(args: Vec<String>) -> i32 {
    let (socket, rest) = match parse_args(args) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("! {e}\n{USAGE}");
            return 1;
        }
    };
    if rest.is_empty() && std::io::stdout().is_terminal() && std::io::stdin().is_terminal()
        && std::env::var("TERM").as_deref() != Ok("dumb")
    {
        return match interactive(&socket) {
            Ok(()) => 0,
            Err(e) => {
                eprintln!("! {e}");
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
            eprintln!("! {e}");
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
        ["free"] => free_card(socket),
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

fn free_card(socket: &Socket) -> Result<String, String> {
    sock::set_preference(socket, None, None)?;
    Ok("Free card: set".into())
}

fn use_reply_text(reply: &serde_json::Value, model: &str) -> String {
    // Older agents acknowledge successful preferences without a message.
    reply["text"].as_str().map_or_else(|| format!("using {model}"), str::to_owned)
}

fn clear_finished_switch(status: &Status, pending: &mut Option<(String, String)>, notice: &mut String) {
    if let Some((id, quant)) = pending {
        // The switch is done once the weights are verified: in a chain the
        // stage loads only when the scheduler assigns one.
        let ready = |model: &Option<String>, q: &Option<String>| {
            model.as_ref() == Some(id) && q.as_ref().is_none_or(|got| got == quant)
        };
        if status.download.is_none()
            && (ready(&status.loaded_model, &status.loaded_quant) || ready(&status.weights_model, &status.weights_quant))
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

fn draw(lines: &[String], height: u16) -> std::io::Result<()> {
    let mut out = std::io::stdout().lock();
    // Clear before writing so a row that fills the screen keeps its last cell.
    write!(out, "\x1b[H")?;
    for (i, line) in lines.iter().take(height.into()).enumerate() {
        if i > 0 {
            write!(out, "\r\n")?;
        }
        write!(out, "\x1b[2K{line}")?;
    }
    if lines.len() < usize::from(height) {
        write!(out, "\r\n\x1b[J")?;
    }
    out.flush()
}

fn quit_key(key: KeyEvent) -> bool {
    matches!(key.code, KeyCode::Char('q' | 'Q') | KeyCode::Esc)
        || (key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL))
}

fn intro(color: bool) -> std::io::Result<bool> {
    let start = Instant::now();
    let mut size = (0, 0);
    while let Some(left) = render::INTRO.checked_sub(start.elapsed()) {
        let (width, height) = terminal::size().unwrap_or((80, 24));
        if (width, height) != size {
            // After a resize the terminal reflows the old frame: start from a blank screen.
            write!(std::io::stdout(), "\x1b[2J")?;
            size = (width, height);
        }
        let logo_width = if height < 6 { width.min(39) } else { width };
        draw(&render::render_intro(logo_width, height, color, start.elapsed()), height)?;
        if event::poll(left.min(Duration::from_millis(45)))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    return Ok(quit_key(key));
                }
            }
        }
    }
    Ok(false)
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

fn interactive(socket: &Socket) -> std::io::Result<()> {
    let color = render::supports_color(true);
    let _guard = TermGuard::enter()?;

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
    if render::motion_enabled(true) && intro(color)? {
        return Ok(());
    }
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

    let mut status: Option<Status> = None;
    let mut down = String::new();
    let mut notice = String::new();
    let mut pending_switch = None;
    let mut picking: Vec<sock::LocalModel> = Vec::new();

    loop {
        let (w, h) = terminal::size().unwrap_or((80, 24));
        let frame = match &status {
            Some(s) => render::render(s, w, h, now_unix(), &notice, color),
            None if down.is_empty() => render::render_logo(if h < 6 { w.min(39) } else { w }, color),
            None => render::render_down(w, &down, color),
        };
        draw(&frame, h)?;

        match rx.recv() {
            Ok(Msg::Status(s)) => {
                clear_finished_switch(&s, &mut pending_switch, &mut notice);
                status = Some(s);
            }
            Ok(Msg::Down(why)) => {
                status = None;
                down = why;
            }
            Ok(Msg::Resize) => write!(std::io::stdout(), "\x1b[2J")?,
            Ok(Msg::Key(k)) => {
                if k.modifiers.contains(KeyModifiers::CONTROL) && quit_key(k) {
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
                                    Err(e) => format!("! {e}"),
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
                    _ if quit_key(k) => return Ok(()),
                    KeyCode::Char('u') | KeyCode::Char('U') if status.is_some() => match sock::get_models(socket) {
                        Ok(m) if m.is_empty() => notice = "no models installed".into(),
                        Ok(m) => {
                            let list: Vec<String> =
                                m.iter().enumerate().map(|(i, x)| format!("[{}] {} {}", i + 1, x.id, x.quant)).collect();
                            notice = format!("Use which model?  {}  (any other key cancels)", list.join("  "));
                            picking = m;
                        }
                        Err(e) => notice = format!("! {e}"),
                    },
                    KeyCode::Char('f') | KeyCode::Char('F') if status.is_some() => {
                        notice = free_card(socket).unwrap_or_else(|e| format!("! {e}"));
                    }
                    KeyCode::Char('p') | KeyCode::Char('P') => {
                        if let Some(s) = &status {
                            if let Err(e) = sock::set_pause(socket, !s.paused) {
                                notice = format!("! {e}");
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
