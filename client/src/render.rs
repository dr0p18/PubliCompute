//! Build terminal rows without doing terminal I/O.

use std::time::Duration;

use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use crate::status::{status_line, Status, Tone};

const NAME: &str = "PubliCompute";
const KEYS: [&str; 5] = ["[U] use a model", "[F] free card", "[P] pause/resume", "[C] chat link", "[Q] quit"];

// The 12×11 MARK from tools/sito/pixel-art.py (also in site/favicon.svg).
// Each terminal cell holds two vertical pixels.
const MARK: [&str; 6] = [
    "█▀▀█    █▀▀█",
    "█▄▄█▀▀▀▀█▄▄█",
    " ▀▄      ▄▀",
    "   ▀▄▄▄▄▀",
    "    █  █",
    "    ▀▀▀▀",
];

const LOGO: [&str; 4] = [
    "█▀▀▀▄ █   █ █▀▀▀▄ █     ▀█▀ ▄▀▀▀▀ ▄▀▀▀▄ █▄ ▄█ █▀▀▀▄ █   █ ▀▀█▀▀ █▀▀▀▀",
    "█▄▄▄▀ █   █ █▄▄▄▀ █      █  █     █   █ █ █ █ █▄▄▄▀ █   █   █   █▄▄▄",
    "█     █   █ █   █ █      █  █     █   █ █   █ █     █   █   █   █",
    "▀      ▀▀▀  ▀▀▀▀  ▀▀▀▀▀ ▀▀▀  ▀▀▀▀  ▀▀▀  ▀   ▀ ▀      ▀▀▀    ▀   ▀▀▀▀▀",
];

// Almonds, irises and lids from eyes() in tools/sito/pixel-art.py.
// Crop the empty space around and between the eyes to fit a narrow terminal.
const EYES_WIDTH: usize = 29;
const EYES_BLANK: [&str; 5] = [""; 5];
const EYES_CLOSED: [&str; 5] = [
    "",
    "",
    "            ▄   ▀▄▄▄▄    ▄▄▄▄",
    "▀█▀▀▄▄▄▄▀▀▀█     ▀   ▀█▀▀  ▀",
    "      ▀",
];
const EYES_OPEN: [&str; 5] = [
    "    ▄▄ ▄▄▄       ▄▄█▀▀ ▀▀▄▄",
    " ▄██ ▄   ███▄   ████ ▀   ███▄",
    "████     ████   ▀███▄   ▄████",
    "▀▀███▄▄▄██▀▀       ▀▀▀▀▀▀▀▀",
    "",
];
const EYES_RIGHT: [&str; 5] = [
    "    ▄▄▄▄▄        ▄▄█████▀",
    " ▄█████▀▄       ███████ ▀",
    "███████         ▀██████▄    ▄",
    "▀▀██████▄▄         ▀▀▀▀▀▀▀▀",
    "",
];
const EYES_LEFT: [&str; 5] = [
    "    ▄▄▄▄▄▄          ▀████▄▄",
    "  ▄  ▀██████▄     ▀   ██████▄",
    "      ███████        ▄███████",
    "▀ ▄▄▄█████▀▀       ▀▀▀▀▀▀▀▀",
    "",
];
const EYES_UP: [&str; 5] = [
    "     ▄   ▄       ▄▄█ ▀   ▄▄",
    " ▄██     ███▄   ████▄   ▄███▄",
    "█████▄▄▄█████   ▀████████████",
    "▀▀████████▀▀       ▀▀▀▀▀▀▀▀",
    "",
];
// BULB from the same source, sampled down to seven pixels across.
const BULB: [&str; 5] = [" ▄███▄", "▄█████▄", "▀█████▀", "  ███", "  ▀█▀"];

#[derive(Clone, Copy)]
enum Style {
    Plain,
    Dim,
    Signal,
    Warn,
}

struct Row(String, Style);

pub fn supports_color(tty: bool) -> bool {
    tty && std::env::var_os("NO_COLOR").is_none()
        && std::env::var("TERM").is_ok_and(|term| {
            !matches!(term.as_str(), "" | "dumb" | "mono")
                && !term.starts_with("vt")
                && !term.ends_with("-mono")
                && !term.ends_with("-m")
        })
}

pub fn motion_enabled(tty: bool) -> bool {
    tty && std::env::var_os("PUBLICOMPUTE_REDUCED_MOTION").is_none()
        && std::env::var("TERM").is_ok_and(|term| !term.is_empty() && term != "dumb")
}

fn cells(text: &str) -> usize {
    UnicodeWidthStr::width(text)
}

fn fit(text: &str, width: usize) -> String {
    let text = text.replace(char::is_control, " ");
    if cells(text.as_str()) <= width {
        return text;
    }
    let Some(mut left) = width.checked_sub(1) else {
        return String::new();
    };
    let mut result = String::new();
    for ch in text.chars() {
        let size = ch.width().unwrap_or(0);
        if size > left {
            break;
        }
        result.push(ch);
        left -= size;
    }
    result.push('…');
    result
}

fn paint(text: &str, style: Style, color: bool) -> String {
    if !color || text.is_empty() {
        return text.into();
    }
    let code = match style {
        Style::Plain => "38;2;244;245;247",
        Style::Dim => "38;2;154;160;171",
        Style::Signal => "38;2;198;255;61",
        Style::Warn => "1;38;2;255;255;255",
    };
    format!("\x1b[{code}m{text}\x1b[0m")
}

fn finish(rows: Vec<Row>, width: u16, color: bool) -> Vec<String> {
    rows.into_iter()
        .map(|Row(text, style)| paint(&fit(&text, width.into()), style, color))
        .collect()
}

fn rule(width: u16, color: bool) -> String {
    paint(&"─".repeat(width.into()), Style::Dim, color)
}

fn section(title: &str, width: u16, color: bool) -> String {
    paint(&fit(title, width.into()), Style::Dim, color)
}

fn gib(bytes: u64) -> String {
    format!("{:.1}", bytes as f64 / (1u64 << 30) as f64)
}

fn hours(seconds: u64) -> u64 {
    seconds.saturating_add(1800) / 3600
}

fn hhmm_utc(unix: i64) -> String {
    let m = unix.rem_euclid(86400) / 60;
    format!("{:02}:{:02}", m / 60, m % 60)
}

fn vram_bar(used: u64, total: u64) -> String {
    let filled = if total == 0 { 0 } else { ((used as f64 / total as f64) * 10.0).round() as usize };
    let filled = filled.min(10);
    format!("[{}{}]", "█".repeat(filled), "·".repeat(10 - filled))
}

fn field(label: &str, value: impl std::fmt::Display) -> Row {
    Row(format!("  {label:<11}{value}"), Style::Plain)
}

fn wrap_groups<'a>(groups: impl IntoIterator<Item = &'a str>, separator: &str, width: usize) -> Vec<String> {
    let mut rows = Vec::new();
    let mut line = String::new();
    for group in groups {
        if !line.is_empty() && cells(&line) + cells(separator) + cells(group) > width {
            rows.push(std::mem::take(&mut line));
        }
        if !line.is_empty() {
            line.push_str(separator);
        }
        line.push_str(group);
    }
    if !line.is_empty() {
        rows.push(line);
    }
    rows
}

fn message(text: &str, width: u16, style: Style, color: bool) -> Vec<String> {
    let rows = wrap_groups(text.split_whitespace(), " ", usize::from(width).saturating_sub(2));
    finish(rows.into_iter().map(|line| Row(format!("  {line}"), style)).collect(), width, color)
}

fn status_rows(s: &Status, width: u16, now: i64, color: bool) -> Vec<String> {
    let (line, tone) = status_line(s);
    let mut rows = vec![paint(&fit(NAME, width.into()), Style::Plain, color), rule(width, color)];
    match tone {
        Tone::Good => {
            let mut status = message(&format!("ONLINE {line}"), width, Style::Plain, false).into_iter();
            if let Some(first) = status.next() {
                let detail = first.strip_prefix("  ONLINE");
                let first = match detail {
                    Some(detail) => format!(
                        "  {}{}", paint("ONLINE", Style::Signal, color), paint(detail, Style::Plain, color)
                    ),
                    None => paint(&first, Style::Plain, color),
                };
                rows.push(first);
            }
            rows.extend(status.map(|row| paint(&row, Style::Plain, color)));
        }
        Tone::Plain => rows.extend(message(&line, width, Style::Plain, color)),
        Tone::Warn => rows.extend(message(&format!("! {line}"), width, Style::Warn, color)),
    }
    if let Some(detail) = s.download_line() {
        let style = if detail.starts_with('!') { Style::Warn } else { Style::Plain };
        rows.extend(message(&detail, width, style, color));
    }
    if let Some(error) = &s.model_error {
        rows.extend(message(&format!("! {error}"), width, Style::Warn, color));
    }
    if let Some(until) = s.contribution.as_ref().and_then(|c| c.stable_until).filter(|u| *u > now) {
        rows.extend(message(&format!("change pending until {} UTC", hhmm_utc(until)), width, Style::Dim, color));
    }

    rows.push(String::new());
    rows.push(section("CARD", width, color));
    let mut card = Vec::new();
    match &s.gpu {
        Some(g) => {
            card.push(field("GPU", &g.name));
            let used = format!("{} / {} GiB", gib(g.vram_used_bytes), gib(g.vram_total_bytes));
            let bar = vram_bar(g.vram_used_bytes, g.vram_total_bytes);
            let vram = if cells(&used) + cells(&bar) + 15 <= usize::from(width) {
                format!("{used}  {bar}")
            } else {
                used
            };
            card.push(field("VRAM", vram));
            card.push(field("Temp", format_args!("{} °C", g.temperature_celsius)));
        }
        None => card.push(field("GPU", "not reported")),
    }
    card.push(field("Model", s.loaded_model.as_deref().unwrap_or("none loaded")));
    let chats = match s.active_leases {
        0 => "none".into(),
        n => format!("{n} in progress"),
    };
    card.push(field("Chats", chats));
    let chain = match &s.chain_id {
        Some(id) => format!("yes ({id})"),
        None => "no".into(),
    };
    card.push(field("Chain", chain));
    rows.extend(finish(card, width, color));

    rows.push(String::new());
    match &s.summary {
        Some(m) => {
            rows.push(section(&format!("LAST {} DAYS", m.days), width, color));
            rows.extend(finish(vec![
                field("Free chats", m.free_chats),
                field("In service", format_args!("{} h", hours(m.service_seconds))),
                field("Working", format_args!("{} h", hours(m.work_seconds))),
            ], width, color));
        }
        None => rows.push(section("LAST 30 DAYS: NO DATA YET", width, color)),
    }
    rows
}

/// Reserve space for command replies and the keys when the status is tall.
pub fn render(s: &Status, width: u16, height: u16, now: i64, notice: &str, color: bool) -> Vec<String> {
    let mut rows = status_rows(s, width, now, color);
    let style = if notice.starts_with('!') { Style::Warn } else { Style::Plain };
    let mut notices = message(notice, width, style, color);
    let mut keys = finish(
        wrap_groups(KEYS, "  ", width.into()).into_iter().map(|s| Row(s, Style::Dim)).collect(),
        width, color,
    );
    if height > 0 {
        let height = usize::from(height);
        if keys.len() > height {
            keys.drain(..keys.len() - height);
        }
        let available = height - keys.len();
        notices.truncate(available);
        rows.truncate(available - notices.len());
    }
    if height == 0 || rows.len() + notices.len() + keys.len() < usize::from(height) {
        rows.push(String::new());
    }
    rows.extend(notices);
    rows.extend(keys);
    rows
}

pub fn render_down(width: u16, notice: &str, color: bool) -> Vec<String> {
    let detail = if notice.is_empty() { crate::sock::START } else { notice };
    let mut rows = vec![paint(&fit(NAME, width.into()), Style::Plain, color), rule(width, color)];
    rows.extend(message(&format!("! {detail}"), width, Style::Warn, color));
    rows.extend(message("Retrying every 2 s.", width, Style::Dim, color));
    rows.push(String::new());
    rows.push(section("[Q] quit", width, color));
    rows
}

pub fn render_logo(width: u16, color: bool) -> Vec<String> {
    logo(width, color, None)
}

// The site intro: the green block blinks three times in the middle of the
// screen (three 600 ms periods), then the name is written in steps.
pub const INTRO_BLINK: Duration = Duration::from_millis(1800);
pub const INTRO_REVEAL: Duration = Duration::from_millis(1200);
pub const INTRO: Duration = Duration::from_millis(1800 + 1200 + 500);

pub fn render_intro(width: u16, height: u16, color: bool, elapsed: Duration) -> Vec<String> {
    let (w, h) = (usize::from(width), usize::from(height));
    let Some(writing) = elapsed.checked_sub(INTRO_BLINK) else {
        // Two rows of one cell: about the tall, narrow shape of the site cursor.
        let on = elapsed.as_millis() % 600 < 300;
        let block = if on { paint("█", Style::Signal, color) } else { String::new() };
        return center(vec![block.clone(), block], 1, w, h);
    };
    let reveal = (writing.as_millis() * 600 / INTRO_REVEAL.as_millis()).min(600) as usize;
    center(logo(width, color, Some((reveal, true))), logo_cells(width), w, h)
}

// Visible width of logo(width): the space the name will take once written,
// so the logo does not move while it is revealed.
fn logo_cells(width: u16) -> usize {
    let width = usize::from(width);
    let mark = width >= 40;
    let block = width >= 16 + cells(LOGO[0]) + 1;
    let word = if block { cells(LOGO[0]) + 1 } else { cells(NAME) + 1 };
    if mark { 16 + word } else { word }.min(width)
}

fn center(lines: Vec<String>, cells: usize, w: usize, h: usize) -> Vec<String> {
    let top = h.saturating_sub(lines.len()) / 2;
    let pad = " ".repeat(w.saturating_sub(cells) / 2);
    let mut out = vec![String::new(); top];
    out.extend(lines.into_iter().map(|l| if l.is_empty() { l } else { format!("{pad}{l}") }));
    out
}

fn logo(width: u16, color: bool, intro: Option<(usize, bool)>) -> Vec<String> {
    let width = usize::from(width);
    let mark = width >= 40;
    let block = width >= 16 + cells(LOGO[0]) + 1;
    let word = if block { &LOGO[..] } else { &[NAME] };
    let word_width = cells(word[0]);
    let (shown, cursor) = intro.map_or((word_width, false), |(reveal, cursor)| {
        (reveal * word_width / 600, cursor && reveal < 600)
    });
    let word_top = if block { 1 } else if mark { 2 } else { 0 };
    let height = if mark { MARK.len() } else { 1 };
    (0..height).map(|y| {
        let mut line = String::new();
        if mark {
            line.push_str(&paint(&format!("{:<12}    ", MARK[y]), Style::Signal, color));
        }
        if let Some(row) = y.checked_sub(word_top).and_then(|i| word.get(i)) {
            let ink: String = row.chars().take(shown).collect();
            let show_cursor = cursor && y == word_top + usize::from(block);
            let prefix = if show_cursor { format!("{ink:<shown$}") } else { ink };
            line.push_str(&paint(&fit(&prefix, width.saturating_sub(if mark { 16 } else { 0 })), Style::Plain, color));
            if show_cursor && shown < width.saturating_sub(if mark { 16 } else { 0 }) {
                line.push_str(&paint("▍", Style::Signal, color));
            }
        }
        if !color {
            line.truncate(line.trim_end().len());
        }
        line
    }).collect()
}

pub fn chat_header(model: &str, width: u16, color: bool) -> Vec<String> {
    let mut rows = vec![paint(&fit("PubliCompute / CHAT", width.into()), Style::Plain, color), rule(width, color)];
    rows.extend(finish(vec![field("Model", model), Row("  /exit to quit".into(), Style::Dim)], width, color));
    rows
}

pub fn chat_prompt(color: bool) -> String {
    paint("> ", Style::Signal, color)
}

pub fn chat_eyes(width: u16, color: bool, elapsed: Option<Duration>) -> Vec<String> {
    if usize::from(width) < EYES_WIDTH {
        return Vec::new();
    }
    let (eyes, bulb) = match elapsed.map(|t| t.as_millis() % 13_000) {
        Some(1820..=2859 | 8580..=9619) => (EYES_CLOSED, false),
        Some(2860..=3899 | 7800..=8579) | None => (EYES_OPEN, false),
        Some(3900..=4939) => (EYES_RIGHT, false),
        Some(4940..=5979) => (EYES_LEFT, false),
        Some(5980..=7799) => (EYES_UP, true),
        _ => (EYES_BLANK, false),
    };
    let place = |line: &str, art_width: usize, style| {
        if line.is_empty() {
            String::new()
        } else {
            let margin = " ".repeat((usize::from(width) - art_width) / 2);
            format!("{margin}{}", paint(line, style, color))
        }
    };
    let mut rows = Vec::new();
    if elapsed.is_some() {
        rows.extend(BULB.map(|line| place(if bulb { line } else { "" }, 7, Style::Signal)));
    }
    rows.extend(eyes.map(|line| place(line, EYES_WIDTH, Style::Plain)));
    if elapsed.is_none() {
        while rows.last().is_some_and(String::is_empty) {
            rows.pop();
        }
    }
    rows
}

pub fn render_empty_chat(model: &str, width: u16, color: bool, elapsed: Option<Duration>) -> Vec<String> {
    let mut rows = chat_header(model, width, color);
    rows.push(String::new());
    rows.extend(chat_eyes(width, color, elapsed));
    rows.push(String::new());
    rows.push(if width >= 2 { chat_prompt(color) } else { fit(">", width.into()) });
    rows
}
