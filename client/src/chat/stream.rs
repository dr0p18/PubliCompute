use std::{
    io::{self, BufRead, BufReader, IsTerminal, Read},
    time::Instant,
};

use serde_json::Value;

use super::session::Session;

const MAX_EVENT: usize = 1024 * 1024;
pub(super) const UNFINISHED_REASONING: &str = "the stream ended during reasoning";
pub(super) const REASONING_TOKEN_LIMIT: &str = "the token limit was reached during reasoning";

struct Sse<R> {
    reader: R,
    skip_lf: bool,
    first_line: bool,
}

impl<R: BufRead> Sse<R> {
    fn new(reader: R) -> Self {
        Self {
            reader,
            skip_lf: false,
            first_line: true,
        }
    }

    fn read_line(&mut self, line: &mut Vec<u8>) -> io::Result<bool> {
        line.clear();
        loop {
            let bytes = self.reader.fill_buf()?;
            if bytes.is_empty() {
                return Ok(false);
            }
            if std::mem::take(&mut self.skip_lf) && bytes[0] == b'\n' {
                self.reader.consume(1);
                continue;
            }
            let end = bytes.iter().position(|&c| c == b'\r' || c == b'\n');
            let n = end.unwrap_or(bytes.len());
            if line.len() + n > MAX_EVENT {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "SSE line exceeds 1 MiB"));
            }
            line.extend_from_slice(&bytes[..n]);
            self.skip_lf = end.is_some_and(|i| bytes[i] == b'\r');
            self.reader.consume(n + usize::from(end.is_some()));
            if end.is_some() {
                return Ok(true);
            }
        }
    }

    fn next(&mut self) -> io::Result<Option<(String, String)>> {
        let (mut name, mut data, mut line) = (String::new(), String::new(), Vec::new());
        while self.read_line(&mut line)? {
            let mut text = std::str::from_utf8(&line).map_err(io::Error::other)?;
            if std::mem::take(&mut self.first_line) {
                text = text.strip_prefix('\u{feff}').unwrap_or(text);
            }
            if text.is_empty() {
                if data.pop().is_some() {
                    return Ok(Some((name, data)));
                }
                name.clear();
                continue;
            }
            let (field, value) = text.split_once(':').unwrap_or((text, ""));
            let value = value.strip_prefix(' ').unwrap_or(value);
            match field {
                "event" => {
                    name.clear();
                    name.push_str(value);
                }
                "data" => {
                    if data.len() + value.len() >= MAX_EVENT {
                        return Err(io::Error::new(io::ErrorKind::InvalidData, "SSE event exceeds 1 MiB"));
                    }
                    data.push_str(value);
                    data.push('\n');
                }
                _ => {}
            }
        }
        Ok(None)
    }
}

pub(super) struct Completion {
    pub(super) free_seat: Option<bool>,
    pub(super) tokens_per_second: Option<f64>,
}

pub(super) enum Progress {
    Delta(Instant),
    Error(&'static str),
}

struct Thinking {
    terminal: bool,
    shown: bool,
}

impl Thinking {
    fn update(&mut self, hidden: bool) {
        if self.terminal && hidden != self.shown {
            if hidden {
                eprint!("\rThinking…\x1b[K");
            } else {
                eprint!("\r\x1b[K");
            }
            self.shown = hidden;
        }
    }
}

impl Drop for Thinking {
    fn drop(&mut self) {
        self.update(false);
    }
}

pub(super) fn consume(
    input: impl Read,
    model: &str,
    session: &Session,
    mut emit: impl FnMut(&str) -> io::Result<()>,
    mut progress: impl FnMut(Progress),
) -> Result<Completion, String> {
    let mut reader = Sse::new(BufReader::new(input));
    let mut reasoning = Reasoning::new(model);
    let mut thinking = Thinking { terminal: io::stderr().is_terminal(), shown: false };
    let (mut finish_reason, mut free_seat, mut tokens) = (None, None, None);
    let (mut first, mut last): (Option<Instant>, Option<Instant>) = (None, None);
    let mut token_prefix = String::new();
    let mut emit_visible = |text: &str, done: bool| {
        token_prefix.push_str(text);
        let token = session.token();
        let pending = if done {
            0
        } else {
            // A credential echoed by the API may straddle SSE deltas.
            partial_suffix(token_prefix.split(token).last().unwrap_or_default(), token)
        };
        let tail = token_prefix.split_off(token_prefix.len() - pending);
        let text = session.redact(&std::mem::replace(&mut token_prefix, tail));
        if !text.is_empty() {
            emit(&text).map_err(output_error)?;
        }
        Ok::<(), String>(())
    };
    let result = loop {
        let (event, data) = match reader.next() {
            Ok(Some(event)) => event,
            Ok(None) => {
                if (first.is_some() || finish_reason.is_some()) && reasoning.hidden() {
                    break Err(reasoning_error(finish_reason));
                }
                break Err("incomplete chat stream: missing final chunk or [DONE]".into());
            }
            Err(e) => break Err(format!("incomplete chat stream: {e}")),
        };
        if event == "publicompute.route" {
            continue;
        }
        if data == "[DONE]" && event != "error" {
            if reasoning.hidden() && (first.is_some() || finish_reason.is_some()) {
                break Err(reasoning_error(finish_reason));
            }
            if finish_reason.is_none() {
                break Err("incomplete chat stream: missing final chunk".into());
            }
            let tokens_per_second = match (first, last, tokens) {
                (Some(start), Some(end), Some(n)) if end > start && n > 1 => {
                    Some((n - 1) as f64 / end.duration_since(start).as_secs_f64())
                }
                _ => None,
            };
            break Ok(Completion {
                free_seat,
                tokens_per_second,
            });
        }
        let parsed = serde_json::from_str::<Value>(&data);
        if event == "error" || parsed.as_ref().is_ok_and(|v| !v["error"].is_null()) {
            thinking.update(false);
            let kind = parsed.as_ref().ok().and_then(|v| v["error"]["code"].as_str().or(v["error"]["type"].as_str()));
            let (code, message) = match kind {
                Some("no_capacity") => ("no_capacity", "no free seat on this model, try again in a moment"),
                Some("timeout") => ("timeout", "the answer took longer than the time limit"),
                Some("context_length_exceeded") => ("context_length_exceeded", "the conversation is too long for this model"),
                Some("invalid_request") => ("invalid_request", "the chain could not prepare the conversation"),
                Some("slow_client") => ("slow_client", "the connection was too slow, the answer stopped"),
                _ => ("server_error", "the answer was not completed"),
            };
            progress(Progress::Error(code));
            break Err(format!("chat stream error: {message}"));
        }
        let Ok(chunk) = parsed else {
            break Err("invalid JSON in chat stream".into());
        };
        let choice = &chunk["choices"][0];
        if let Some(text) = choice["delta"]["content"].as_str().filter(|s| !s.is_empty()) {
            let now = Instant::now();
            progress(Progress::Delta(now));
            first.get_or_insert(now);
            last = Some(now);
            let visible = reasoning.push(text)?;
            if !visible.is_empty() {
                thinking.update(false);
            }
            emit_visible(&visible, false)?;
            thinking.update(reasoning.hidden());
        }
        if let Some(reason) = choice["finish_reason"].as_str().filter(|s| !s.is_empty()) {
            finish_reason = Some(reason == "length");
            free_seat = chunk["publicompute"]["free_seat"].as_bool();
        }
        tokens = chunk["usage"]["completion_tokens"].as_u64().or(tokens);
    };
    thinking.update(false);
    emit_visible(&reasoning.finish(), true)?;
    result
}

fn reasoning_error(hit_limit: Option<bool>) -> String {
    if hit_limit == Some(true) {
        REASONING_TOKEN_LIMIT.into()
    } else {
        UNFINISHED_REASONING.into()
    }
}

fn output_error(e: io::Error) -> String {
    format!("cannot write chat response: {e}")
}

const OPEN: &str = "<think>";
const CLOSE: &str = "</think>";

fn partial_suffix(text: &str, pattern: &str) -> usize {
    (1..=(pattern.len() - 1).min(text.len()))
        .rev()
        .find(|&n| text.ends_with(&pattern[..n]))
        .unwrap_or(0)
}

enum Phase {
    Hidden,
    Visible,
}

struct Reasoning {
    phase: Phase,
    trim_start: bool,
    pending: String,
}

impl Reasoning {
    fn new(model: &str) -> Self {
        // The Qwen templates open the reasoning block in the prompt, so the reply carries only the
        // closing tag. Other models stream at once and hide only an explicit <think> block, as the web
        // chat does: waiting for a tag that may never come would defeat streaming.
        Self {
            phase: if model.starts_with("qwen3") { Phase::Hidden } else { Phase::Visible },
            trim_start: false,
            pending: String::new(),
        }
    }

    fn hidden(&self) -> bool {
        matches!(self.phase, Phase::Hidden)
    }

    fn push(&mut self, text: &str) -> Result<String, String> {
        let mut input = std::mem::take(&mut self.pending);
        input.push_str(text);
        let mut text = input.as_str();
        let mut visible = String::new();
        loop {
            if self.trim_start {
                text = text.trim_start();
                if text.is_empty() {
                    break;
                }
                self.trim_start = false;
            }
            let in_answer = matches!(self.phase, Phase::Visible);
            let tag = if in_answer { OPEN } else { CLOSE };
            if let Some((before, after)) = text.split_once(tag) {
                if in_answer {
                    visible.push_str(before);
                }
                self.trim_start = !in_answer;
                self.phase = if in_answer { Phase::Hidden } else { Phase::Visible };
                text = after;
                continue;
            }
            // Keep only the suffix that could be the start of a tag split across chunks.
            let partial = partial_suffix(text, tag);
            let (before, suffix) = text.split_at(text.len() - partial);
            self.pending.push_str(suffix);
            match self.phase {
                Phase::Visible => visible.push_str(before),
                Phase::Hidden => {}
            }
            break;
        }
        Ok(visible)
    }

    fn finish(self) -> String {
        match self.phase {
            Phase::Hidden => String::new(),
            Phase::Visible => self.pending,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::session::Endpoint;
    use ureq::http::HeaderMap;

    fn session() -> Session {
        let mut headers = HeaderMap::new();
        headers.append("Set-Cookie", "pc_session=fixture_token; Max-Age=60".parse().unwrap());
        Session::from_headers(&headers, &Endpoint::parse("http://127.0.0.1").unwrap()).unwrap()
    }

    struct Split<'a>(&'a [u8]);

    impl Read for Split<'_> {
        fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
            let max = bytes.len().min(3);
            self.0.read(&mut bytes[..max])
        }
    }

    const TEXT: &str = "data: {\"choices\":[{\"delta\":{\"content\":\"caffè\"},\"finish_reason\":null}]}\n\n";
    const FINAL: &str =
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"publicompute\":{\"free_seat\":false}}\n\n";
    const DONE: &str = "data: [DONE]\n\n";

    #[test]
    fn reports_stream_ending_during_reasoning_without_printing_it() {
        for (final_chunk, end, expected) in [
            (FINAL.replace("stop", "length"), DONE, REASONING_TOKEN_LIMIT),
            (FINAL.into(), DONE, UNFINISHED_REASONING),
            (String::new(), "", UNFINISHED_REASONING),
        ] {
            let input = format!("{TEXT}{final_chunk}{end}");
            let result = consume(input.as_bytes(), "qwen3", &session(), |_| panic!("reasoning leaked"), |_| {});
            assert!(matches!(result, Err(message) if message == expected));
        }
        let input = format!("{}{TEXT}{}{DONE}", TEXT.replace("caffè", "</think>"), FINAL.replace("stop", "length"));
        assert!(consume(input.as_bytes(), "qwen3", &session(), |_| Ok(()), |_| {}).is_ok());
    }

    #[test]
    fn reads_split_sse_lines_utf8_comments_and_named_route() {
        let input = format!("\u{feff}: ok\r\n\r\n: keepalive\r\revent: publicompute.route\r\ndata: route\r\ndata: details\r\n\r\n{TEXT}{FINAL}{DONE}");
        let mut reader = Sse::new(BufReader::new(Split(input.as_bytes())));
        assert_eq!(
            reader.next().unwrap().unwrap(),
            ("publicompute.route".into(), "route\ndetails".into())
        );
        let mut answer = String::new();
        let done = consume(Split(input.as_bytes()), "gemma", &session(), |text| {
            answer.push_str(text);
            Ok(())
        }, |_| {})
        .unwrap();
        assert_eq!(answer, "caffè");
        assert_eq!(done.free_seat, Some(false));
    }

    #[test]
    fn reports_named_and_unnamed_stream_errors_after_partial_output() {
        for event in ["", "event: error\n"] {
            let input = format!("{TEXT}{event}data: {{\"error\":{{\"message\":\"rete piena fixture_token\",\"type\":\"no_capacity\"}}}}\n\n{DONE}");
            let mut answer = String::new();
            let result = consume(input.as_bytes(), "gemma", &session(), |text| {
                answer.push_str(text);
                Ok(())
            }, |_| {});
            assert!(matches!(result, Err(message) if message == "chat stream error: no free seat on this model, try again in a moment"));
            assert_eq!(answer, "caffè");
        }
    }

    #[test]
    fn requires_both_final_chunk_and_complete_done_event() {
        for suffix in [
            String::new(),
            DONE.into(),
            FINAL.into(),
            format!("{FINAL}data: [DONE]\n"),
        ] {
            let input = format!("{TEXT}{suffix}");
            assert!(matches!(consume(input.as_bytes(), "gemma", &session(), |_| Ok(()), |_| {}),
                Err(error) if error.starts_with("incomplete chat stream")));
        }
    }

    #[test]
    fn redacts_a_session_token_split_across_deltas() {
        let mut events = String::new();
        for text in ["before fixture_", "token after"] {
            let chunk = serde_json::json!({"choices": [{"delta": {"content": text}}]});
            events.push_str(&format!("data: {chunk}\n\n"));
        }
        events.push_str(FINAL);
        events.push_str(DONE);
        let mut answer = String::new();
        consume(events.as_bytes(), "gemma", &session(), |text| {
            answer.push_str(text);
            Ok(())
        }, |_| {})
        .unwrap();
        assert_eq!(answer, "before [redacted] after");
    }

    fn strip(model: &str, parts: &[&str]) -> String {
        let mut reasoning = Reasoning::new(model);
        let mut answer = String::new();
        for part in parts {
            answer.push_str(&reasoning.push(part).unwrap());
        }
        answer.push_str(&reasoning.finish());
        answer
    }

    #[test]
    fn strips_explicit_and_implicit_reasoning_across_chunks() {
        assert_eq!(
            strip("gemma", &["<th", "ink>private</th", "ink>", " \n", "answer"]),
            "answer"
        );
        assert_eq!(strip("qwen3.5-9b", &["思考", "</", "think>\n\n", "answer"]), "answer");
        assert_eq!(strip("qwen3.5-9b", &["<think>explicit", "</think> answer"]), "answer");
        assert_eq!(strip("gemma", &["start <think>secret</think> end"]), "start end");
        assert_eq!(strip("gemma", &["<think>unfinished reasoning"]), "");
        assert_eq!(
            strip("gemma", &["ordinary <th", "reat and <", "end", "<th"]),
            "ordinary <threat and <end<th"
        );
    }

    #[test]
    fn streams_other_models_at_once_and_hides_qwen_until_the_closing_tag() {
        let mut other = Reasoning::new("gemma");
        assert_eq!(other.push("plain answer").unwrap(), "plain answer");
        let mut qwen = Reasoning::new("qwen3.5-9b");
        assert!(qwen.push("private notes </thi").unwrap().is_empty());
        assert_eq!(qwen.push("nk>\n public").unwrap(), "public");
        assert_eq!(qwen.push(" text").unwrap(), " text");
        assert!(qwen.finish().is_empty());
    }
}
