//! The renderer builds styled rows from a status snapshot and terminal size.
//! Keeping terminal I/O at the caller makes the layout testable. Rows are
//! truncated to fit, which prevents wrapping from changing the layout.

use crate::status::{status_line, Status, Tone};

pub const LOGO: [&str; 3] = [
    "  +--+   +--+   +--+",
    "  |##|---|##|---|##|    P U B L I C O M P U T E",
    "  +--+   +--+   +--+    p2p distributed inference",
];

pub const NO_GUARANTEE: &str = "No earnings guarantee: credits depend on demand.";
pub const KEYS: &str = "[U] use a model  [F] FARM mode  [P] pause/resume  [C] chat link  [Q] quit";

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Style {
    Plain,
    Dim,
    Accent,
    Warn,
}

pub struct Row(pub String, pub Style);

/// Limit the line by character count, including an ellipsis when truncating.
pub fn fit(text: &str, width: usize) -> String {
    if text.chars().count() <= width {
        return text.to_string();
    }
    if width == 0 {
        return String::new();
    }
    let mut s: String = text.chars().take(width - 1).collect();
    s.push('…');
    s
}

fn paint(text: &str, style: Style, color: bool) -> String {
    let code = match (color, style) {
        (false, _) | (_, Style::Plain) => return text.to_string(),
        (_, Style::Dim) => "38;2;141;138;132",
        (_, Style::Accent) => "38;2;198;255;61",
        (_, Style::Warn) => "38;2;255;106;61",
    };
    format!("\x1b[{code}m{text}\x1b[0m")
}

fn finish(rows: Vec<Row>, width: u16, color: bool) -> Vec<String> {
    rows.into_iter().map(|Row(t, s)| paint(&fit(&t, width as usize), s, color)).collect()
}

fn gib(bytes: u64) -> String {
    format!("{:.1}", bytes as f64 / (1u64 << 30) as f64)
}

pub fn thousands(n: i64) -> String {
    let digits = n.unsigned_abs().to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    if n < 0 {
        out.insert(0, '-');
    }
    out
}

fn hours(seconds: u64) -> u64 {
    (seconds + 1800) / 3600
}

fn hhmm_utc(unix: i64) -> String {
    let m = unix.rem_euclid(86400) / 60;
    format!("{:02}:{:02}", m / 60, m % 60)
}

fn vram_bar(used: u64, total: u64, cells: usize) -> String {
    let filled = if total == 0 { 0 } else { ((used as f64 / total as f64) * cells as f64).round() as usize };
    let filled = filled.min(cells);
    format!("[{}{}]", "#".repeat(filled), ".".repeat(cells - filled))
}

/// Build styled rows before fitting them to the viewport. The pending-change
/// message compares the stored deadline with Unix seconds.
pub fn rows(s: &Status, now: i64, notice: &str) -> Vec<Row> {
    use Style::*;
    let mut r = Vec::new();
    r.push(Row("PubliCompute".into(), Accent));
    let (line, tone) = status_line(s);
    let style = match tone {
        Tone::Good => Accent,
        Tone::Plain => Plain,
        Tone::Warn => Warn,
    };
    r.push(Row(format!("  {line}"), style));
    if let Some(until) = s.contribution.as_ref().and_then(|c| c.stable_until).filter(|u| *u > now) {
        r.push(Row(format!("  change pending until {} UTC", hhmm_utc(until)), Dim));
    }
    r.push(Row(String::new(), Plain));

    r.push(Row("Card".into(), Dim));
    match &s.gpu {
        Some(g) => {
            r.push(Row(format!("  GPU      {}", g.name), Plain));
            let bar = vram_bar(g.vram_used_bytes, g.vram_total_bytes, 10);
            r.push(Row(format!("  VRAM     {} / {} GiB  {bar}", gib(g.vram_used_bytes), gib(g.vram_total_bytes)), Plain));
            r.push(Row(format!("  Temp     {} °C", g.temperature_celsius), Plain));
        }
        None => r.push(Row("  GPU      not reported".into(), Dim)),
    }
    r.push(Row(format!("  Model    {}", s.loaded_model.as_deref().unwrap_or("none loaded")), Plain));
    let chats = match s.active_leases {
        0 => "none".to_string(),
        1 => "1 in progress".to_string(),
        n => format!("{n} in progress"),
    };
    r.push(Row(format!("  Chats    {chats}"), Plain));
    let chain = match &s.chain_id {
        Some(id) => format!("yes ({id})"),
        None => "no".into(),
    };
    r.push(Row(format!("  Chain    {chain}"), Plain));
    r.push(Row(String::new(), Plain));

    match &s.summary {
        Some(m) => {
            r.push(Row(format!("Last {} days", m.days), Dim));
            r.push(Row(format!("  free service received  {:>9} mc  ({} chats)", thousands(m.free_value_mc), m.free_chats), Plain));
            r.push(Row(format!("  credits earned         {:>9} mc", thousands(m.earned_mc)), Plain));
            r.push(Row(
                format!("  in service             {:>9} h   (of which working {} h)", hours(m.service_seconds), hours(m.work_seconds)),
                Plain,
            ));
        }
        None => r.push(Row("Last 30 days: no data yet".into(), Dim)),
    }
    r.push(Row(format!("  {NO_GUARANTEE}"), Dim));
    r.push(Row(String::new(), Plain));
    if !notice.is_empty() {
        r.push(Row(notice.to_string(), Accent));
    }
    r.push(Row(KEYS.into(), Dim));
    r
}

fn wrap_message(message: &str, width: u16, style: Style) -> Vec<Row> {
    let mut rows = Vec::new();
    let mut line = String::from("  ");
    for word in message.split_whitespace() {
        if line.len() > 2 && line.chars().count() + 1 + word.chars().count() > usize::from(width) {
            rows.push(Row(line, style));
            line = String::from("  ");
        }
        if line.len() > 2 {
            line.push(' ');
        }
        line.push_str(word);
    }
    rows.push(Row(line, style));
    rows
}

/// Preserve the key legend on the last row when the body is too tall to fit.
pub fn render(s: &Status, width: u16, height: u16, now: i64, notice: &str, color: bool) -> Vec<String> {
    let mut rows = rows(s, now, notice);
    if s.gpu_lost {
        rows.splice(1..2, wrap_message(crate::status::GPU_LOST, width, Style::Warn));
    }
    let h = height as usize;
    if h > 0 && rows.len() > h {
        let last = rows.pop().unwrap();
        rows.truncate(h - 1);
        rows.push(last);
    }
    finish(rows, width, color)
}

/// Build a view for a lost or unavailable connection.
pub fn render_down(width: u16, notice: &str, color: bool) -> Vec<String> {
    let message = if notice.is_empty() { crate::sock::START } else { notice };
    let mut rows = vec![Row("PubliCompute".into(), Style::Accent)];
    rows.extend(wrap_message(message, width, Style::Warn));
    rows.push(Row("  Retrying every 2 s.".into(), Style::Dim));
    rows.push(Row(String::new(), Style::Plain));
    rows.push(Row("[Q] quit".into(), Style::Dim));
    finish(rows, width, color)
}

pub fn render_logo(width: u16, color: bool) -> Vec<String> {
    finish(LOGO.iter().map(|l| Row(l.to_string(), Style::Accent)).collect(), width, color)
}
