use std::{
    borrow::Cow,
    fs::File,
    io::{self, IsTerminal, Read, Write},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Instant,
};

use crossterm::{execute, terminal::{self, EnterAlternateScreen, LeaveAlternateScreen}};
use rustix::{
    event::{poll, PollFd, PollFlags, Timespec},
    fs::{fcntl_getfl, fcntl_setfl, open, Mode, OFlags},
    io::Errno,
    termios::{tcflush, tcgetattr, tcsetattr, LocalModes, OptionalActions, QueueSelector, Termios},
};
use signal_hook::{
    consts::{SIGHUP, SIGINT, SIGQUIT, SIGTERM},
    flag, low_level, SigId,
};

use crate::render;

const POLL_TIMEOUT: Timespec = Timespec {
    tv_sec: 0,
    tv_nsec: 50_000_000,
};
const LOGIN_SIGNALS: [i32; 4] = [SIGINT, SIGTERM, SIGHUP, SIGQUIT];

pub(super) struct Interrupt(Arc<AtomicBool>, Vec<SigId>);

impl Interrupt {
    pub(super) fn install() -> io::Result<Self> {
        Self::for_signals(&[SIGINT])
    }

    fn for_signals(signals: &[i32]) -> io::Result<Self> {
        let mut handler = Self(Arc::new(AtomicBool::new(false)), Vec::new());
        for &signal in signals {
            handler.1.push(flag::register(signal, handler.0.clone())?);
        }
        Ok(handler)
    }

    pub(super) fn take(&self) -> bool {
        self.0.swap(false, Ordering::Relaxed)
    }
}

impl Drop for Interrupt {
    fn drop(&mut self) {
        for id in self.1.drain(..) {
            low_level::unregister(id);
        }
    }
}

pub(super) fn open_tty() -> io::Result<File> {
    Ok(File::from(open(
        "/dev/tty",
        OFlags::RDONLY | OFlags::NONBLOCK | OFlags::CLOEXEC,
        Mode::empty(),
    )?))
}

pub(super) enum Input {
    Line(String),
    Eof,
    Interrupted,
}

pub(super) fn read_tty(tty: &mut File, interrupt: &Interrupt, mut idle: impl FnMut() -> io::Result<()>) -> io::Result<Input> {
    let mut bytes = [0; 8192];
    loop {
        if interrupt.take() {
            return Ok(Input::Interrupted);
        }
        let mut fds = [PollFd::new(&*tty, PollFlags::IN)];
        match poll(&mut fds, Some(&POLL_TIMEOUT)) {
            Ok(0) => {
                idle()?;
                continue;
            }
            Err(Errno::INTR) => continue,
            Err(e) => return Err(e.into()),
            Ok(_) => {}
        }
        // A nonblocking read also covers Ctrl-C flushing input between poll and read.
        match tty.read(&mut bytes) {
            Err(e) if matches!(e.kind(), io::ErrorKind::Interrupted | io::ErrorKind::WouldBlock) => continue,
            _ if interrupt.take() => return Ok(Input::Interrupted),
            Ok(0) => return Ok(Input::Eof),
            Ok(n) => {
                return String::from_utf8(bytes[..n].to_vec())
                    .map(Input::Line)
                    .map_err(io::Error::other)
            }
            Err(e) => return Err(e),
        }
    }
}

pub(super) fn empty_prompt(
    tty: &mut File,
    out: &mut io::StderrLock<'_>,
    interrupt: &Interrupt,
    model: &str,
    color: bool,
) -> io::Result<Input> {
    let (width, height) = terminal::size().unwrap_or((80, 24));
    let start = Instant::now();
    let mut eyes = render::chat_eyes(width, color, Some(start.elapsed()));
    let mut rows = render::render_empty_chat(model, width, color, Some(start.elapsed()));
    if !render::motion_enabled(io::stdout().is_terminal()) || eyes.is_empty()
        || usize::from(height) <= rows.len()
    {
        write!(out, "{}", render::render_empty_chat(model, width, color, None).join("\n"))?;
        out.flush()?;
        return read_tty(tty, interrupt, || Ok(()));
    }

    let prompt_row = rows.len();
    let prompt = rows.pop().unwrap();
    let eye_row = rows.len() - eyes.len();
    let input = {
        let screen = ChatScreen(out);
        execute!(screen.0, EnterAlternateScreen)?;
        // Keep the kernel's line editing. Only the prompt may scroll if a question wraps.
        let frame = rows.join("\x1b[E");
        write!(
            screen.0,
            "\x1b[2J\x1b[H{frame}\x1b[{prompt_row};{height}r\x1b[{prompt_row};1H{prompt}"
        )?;
        screen.0.flush()?;
        let mut animate = true;
        let mut redraw = |out: &mut io::StderrLock<'_>| -> io::Result<()> {
            // A resize can move the prompt. Leave its cursor alone until submission.
            if !animate || !terminal::size().is_ok_and(|size| size == (width, height)) {
                animate = false;
                return Ok(());
            }
            let next = render::chat_eyes(width, color, Some(start.elapsed()));
            if eyes != next {
                // Save the cursor where canonical input last echoed it.
                let frame = next.join("\x1b[E\x1b[2K");
                let update = format!("\x1b7\x1b[{eye_row};1H\x1b[2K{frame}\x1b8");
                out.write_all(update.as_bytes())?;
                out.flush()?;
                eyes = next;
            }
            Ok(())
        };
        loop {
            let input = read_tty(tty, interrupt, || redraw(screen.0))?;
            match input {
                Input::Interrupted => {}
                Input::Line(ref line) if line.trim().is_empty() => {}
                _ => break input,
            }
            write!(screen.0, "\r\x1b[2K{prompt}")?;
            screen.0.flush()?;
        }
    };
    match &input {
        Input::Line(line) if line.trim() != "/exit" => {
            // The alternate screen is temporary. Keep the question with its answer in scrollback.
            let (width, _) = terminal::size().unwrap_or((width, height));
            let header = render::chat_header(model, width, color).join("\n");
            let question = escape_controls(line.trim_end_matches(['\n', '\r']), false);
            writeln!(out, "{header}\n{prompt}{question}")?;
        }
        _ => {}
    }
    Ok(input)
}

struct ChatScreen<'a, 'b>(&'a mut io::StderrLock<'b>);

impl Drop for ChatScreen<'_, '_> {
    fn drop(&mut self) {
        let _ = write!(self.0, "\x1b[r");
        let _ = execute!(self.0, LeaveAlternateScreen);
    }
}

pub(super) fn credentials() -> Result<(String, String), String> {
    let mut tty = File::from(
        open(
            "/dev/tty",
            OFlags::RDWR | OFlags::NONBLOCK | OFlags::CLOEXEC,
            Mode::empty(),
        )
        .map_err(|_| "login requires a terminal")?,
    );
    // Restore the usual signal behavior after echo is safe, including during the login request.
    let echo_restored = Arc::new(AtomicBool::new(false));
    for signal in LOGIN_SIGNALS {
        flag::register_conditional_default(signal, echo_restored.clone())
            .map_err(|e| format!("cannot handle login signals: {e}"))?;
    }
    let interrupt = Interrupt::for_signals(&LOGIN_SIGNALS).map_err(|e| format!("cannot handle login signals: {e}"))?;
    let read = |tty: &mut File| match read_tty(tty, &interrupt, || Ok(())) {
        Ok(Input::Line(line)) if !line.trim().is_empty() => Ok(line.trim().to_owned()),
        Ok(Input::Interrupted) => Err("login cancelled"),
        Ok(_) => Err("email and access key are required"),
        Err(_) => Err("cannot read credentials from the terminal"),
    };
    let result = (|| {
        // A single paste can queue both lines before the email read returns.
        let _no_echo = NoEcho::enter(&tty).map_err(|_| "cannot disable terminal echo")?;
        write!(tty, "Email: ").map_err(|_| "cannot write to the terminal")?;
        let email = read(&mut tty)?;
        write!(tty, "\nAccess key: ").map_err(|_| "cannot write to the terminal")?;
        Ok((email, read(&mut tty)?))
    })();
    echo_restored.store(true, Ordering::SeqCst);
    let _ = writeln!(tty);
    if interrupt.take() {
        Err("login cancelled".into())
    } else {
        result
    }
}

struct NoEcho(File, Termios);

impl NoEcho {
    fn enter(tty: &File) -> io::Result<Self> {
        let guard = Self(tty.try_clone()?, tcgetattr(tty)?);
        let mut hidden = guard.1.clone();
        hidden.local_modes.remove(LocalModes::ECHO | LocalModes::ECHONL);
        tcsetattr(tty, OptionalActions::Flush, &hidden)?;
        Ok(guard)
    }
}

pub(super) fn escape_controls(text: &str, multiline: bool) -> Cow<'_, str> {
    let escape = |c: char| c.is_control() && !(multiline && matches!(c, '\n' | '\t'));
    if !text.chars().any(escape) {
        return Cow::Borrowed(text);
    }
    let mut safe = String::with_capacity(text.len());
    for c in text.chars() {
        if escape(c) {
            safe.extend(c.escape_default());
        } else {
            safe.push(c);
        }
    }
    Cow::Owned(safe)
}

pub(super) struct AnswerOutput<'a> {
    stdout: io::StdoutLock<'static>,
    cancel: &'a AtomicBool,
    original_flags: OFlags,
    pub(super) terminal: bool,
}

impl<'a> AnswerOutput<'a> {
    pub(super) fn open(cancel: &'a AtomicBool) -> io::Result<Self> {
        let stdout = io::stdout().lock();
        let terminal = stdout.is_terminal();
        // A full pipe must still let the worker stop on Ctrl-C or at the deadline.
        let original_flags = fcntl_getfl(&stdout)?;
        fcntl_setfl(&stdout, original_flags | OFlags::NONBLOCK)?;
        Ok(Self {
            stdout,
            cancel,
            original_flags,
            terminal,
        })
    }

    pub(super) fn write(&mut self, text: &str) -> io::Result<()> {
        let text = if self.terminal {
            escape_controls(text, true)
        } else {
            Cow::Borrowed(text)
        };
        let mut bytes = text.as_bytes();
        while !bytes.is_empty() {
            if self.cancel.load(Ordering::Relaxed) {
                return Err(io::ErrorKind::Interrupted.into());
            }
            match rustix::io::write(&self.stdout, bytes) {
                Ok(0) => return Err(io::ErrorKind::WriteZero.into()),
                Ok(n) => bytes = &bytes[n..],
                Err(Errno::INTR) => continue,
                Err(Errno::AGAIN) => {
                    let mut fds = [PollFd::new(&self.stdout, PollFlags::OUT)];
                    match poll(&mut fds, Some(&POLL_TIMEOUT)) {
                        Ok(_) | Err(Errno::INTR) => {}
                        Err(e) => return Err(e.into()),
                    }
                }
                Err(e) => return Err(e.into()),
            }
        }
        Ok(())
    }
}

impl Drop for AnswerOutput<'_> {
    fn drop(&mut self) {
        while fcntl_setfl(&self.stdout, self.original_flags) == Err(Errno::INTR) {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escapes_terminal_controls_and_preserves_answer_layout() {
        let text = "é\u{1b}]52;c;data\u{7}\u{9b}\u{7f}\r\n\tend";
        let escaped = "é\\u{1b}]52;c;data\\u{7}\\u{9b}\\u{7f}\\r";
        assert_eq!(escape_controls(text, true), format!("{escaped}\n\tend"));
        assert_eq!(escape_controls(text, false), format!("{escaped}\\n\\tend"));
    }
}

impl Drop for NoEcho {
    fn drop(&mut self) {
        // A cancelled partial credential must not become input to the caller's shell.
        while tcflush(&self.0, QueueSelector::IFlush) == Err(Errno::INTR) {}
        while tcsetattr(&self.0, OptionalActions::Now, &self.1) == Err(Errno::INTR) {}
    }
}
