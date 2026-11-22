use std::{
    borrow::Cow,
    fs::File,
    io::{self, IsTerminal, Read, Write},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

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

pub(super) fn read_tty(tty: &mut File, interrupt: &Interrupt) -> io::Result<Input> {
    let mut bytes = [0; 8192];
    loop {
        if interrupt.take() {
            return Ok(Input::Interrupted);
        }
        let mut fds = [PollFd::new(&*tty, PollFlags::IN)];
        match poll(&mut fds, Some(&POLL_TIMEOUT)) {
            Ok(0) | Err(Errno::INTR) => continue,
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
    let read = |tty: &mut File| match read_tty(tty, &interrupt) {
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
    cancel: Option<&'a AtomicBool>,
    original_flags: Option<OFlags>,
    pub(super) terminal: bool,
}

impl<'a> AnswerOutput<'a> {
    pub(super) fn open(cancel: Option<&'a AtomicBool>) -> io::Result<Self> {
        let stdout = io::stdout().lock();
        let terminal = stdout.is_terminal();
        let original_flags = if cancel.is_some() {
            let flags = fcntl_getfl(&stdout)?;
            // A full redirected pipe must not block the worker after an interrupt.
            fcntl_setfl(&stdout, flags | OFlags::NONBLOCK)?;
            Some(flags)
        } else {
            None
        };
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
        let Some(cancel) = self.cancel else {
            self.stdout.write_all(bytes)?;
            return self.stdout.flush();
        };
        while !bytes.is_empty() {
            if cancel.load(Ordering::Relaxed) {
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
        if let Some(flags) = self.original_flags {
            while fcntl_setfl(&self.stdout, flags) == Err(Errno::INTR) {}
        }
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
