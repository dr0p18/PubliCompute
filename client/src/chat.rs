//! Account commands and streaming chat use the coordinator's HTTP API directly.

mod session;
mod stream;
mod terminal;

use std::{
    env,
    fs::File,
    io::{self, BufRead, IsTerminal, Write},
    net::{Ipv4Addr, Ipv6Addr, SocketAddr},
    sync::atomic::{AtomicBool, Ordering},
    thread,
    time::Duration,
};

use serde::Serialize;
use serde_json::{json, Value};
use ureq::{
    config::Config,
    http::{Response, Uri},
    unversioned::{
        resolver::{DefaultResolver, ResolvedSocketAddrs, Resolver},
        transport::{DefaultConnector, NextTimeout},
    },
    Agent, Body,
};

use crate::{render, sock::{self, Socket}};
use session::{directory, Endpoint, Session, SessionDir, EXPIRED};
use terminal::{AnswerOutput, Input, Interrupt};

const DEFAULT_API: &str = "https://api.publicompute.com:8444";
const MAX_MESSAGES: usize = 200;
const MAX_REQUEST: usize = 256 * 1024;

fn api(override_url: Option<&str>) -> Result<Endpoint, String> {
    if let Some(url) = override_url {
        return Endpoint::parse(url);
    }
    match env::var("PUBLICOMPUTE_API") {
        Ok(url) => Endpoint::parse(&url),
        Err(env::VarError::NotPresent) => Endpoint::parse(DEFAULT_API),
        Err(_) => Err("invalid PUBLICOMPUTE_API".into()),
    }
}

#[derive(Debug)]
struct DirectResolver;

impl Resolver for DirectResolver {
    fn resolve(&self, uri: &Uri, config: &Config, timeout: NextTimeout) -> Result<ResolvedSocketAddrs, ureq::Error> {
        if uri.scheme_str() == Some("http") && uri.host() == Some("localhost") {
            // Do not allow a hosts file or DNS override to send a cleartext credential off loopback.
            let port = uri.port_u16().unwrap_or(80);
            let mut addrs = self.empty();
            addrs.push(SocketAddr::from((Ipv4Addr::LOCALHOST, port)));
            addrs.push(SocketAddr::from((Ipv6Addr::LOCALHOST, port)));
            return Ok(addrs);
        }
        DefaultResolver::default().resolve(uri, config, timeout)
    }
}

fn http() -> Agent {
    let config = Config::builder()
        .proxy(None)
        .max_redirects(0)
        .http_status_as_error(false)
        .timeout_resolve(Some(Duration::from_secs(3)))
        .timeout_connect(Some(Duration::from_secs(5)))
        .timeout_recv_response(Some(Duration::from_secs(30)))
        .timeout_per_read(Some(Duration::from_secs(45)))
        .timeout_per_write(Some(Duration::from_secs(15)))
        .build();
    Agent::with_parts(config, DefaultConnector::default(), DirectResolver)
}

pub fn login() -> Result<(), String> {
    let api = api(None)?;
    let (email, key) = terminal::credentials()?;
    let reply = http()
        .post(api.url("/v1/auth/key"))
        .send_json(json!({"email": email, "access_key": key}))
        .map_err(|_| "cannot reach the API for login")?;
    match reply.status().as_u16() {
        200 => {}
        403 => return Err("invalid_credentials: email or access key is incorrect".into()),
        429 => {
            let retry = reply
                .headers()
                .get("Retry-After")
                .and_then(|h| h.to_str().ok())
                .and_then(|h| h.parse::<u64>().ok());
            return Err(match retry {
                Some(seconds) => format!("rate_limited: too many login attempts, retry in {seconds} seconds"),
                None => "rate_limited: too many login attempts, try again later".into(),
            });
        }
        status => return Err(format!("login failed (HTTP {status})")),
    }
    let session = Session::from_headers(reply.headers(), &api)?;
    SessionDir::open(&directory()?, true)
        .and_then(|dir| dir.save(&session))
        .map_err(|e| format!("cannot save session: {e}"))?;
    eprintln!("Logged in.");
    Ok(())
}

pub fn logout() -> Result<(), String> {
    let session = match SessionDir::open(&directory()?, false) {
        Ok(dir) => dir.take().map_err(|e| format!("cannot remove session: {e}"))?,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(format!("cannot open session directory: {e}")),
    };
    let revoked = match session {
        Ok(session) => session.endpoint().is_ok_and(|api| {
            http()
                .post(api.url("/v1/auth/logout"))
                .header("Cookie", session.cookie())
                .send_empty()
                .is_ok_and(|reply| reply.status().is_success())
        }),
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(_) => false,
    };
    if revoked {
        eprintln!("Logged out.");
    } else {
        eprintln!("Local session removed, but server revocation could not be confirmed.");
    }
    Ok(())
}

pub fn run(socket: &Socket, args: &[String]) -> Result<(), String> {
    let (mut model, mut url, mut question) = (None, None, None);
    let mut options = true;
    let mut args = args.iter().map(String::as_str);
    while let Some(arg) = args.next() {
        match (options, arg) {
            (true, "--") => options = false,
            (true, "--model") => model = Some(args.next().filter(|s| !s.is_empty()).ok_or("--model needs an ID")?),
            (true, "--api") => url = Some(args.next().ok_or("--api needs a URL")?),
            (true, flag) if flag.starts_with("--") => return Err("unknown chat option".into()),
            _ if question.is_none() => question = Some(arg),
            _ => return Err("chat accepts one question, quote it as a single argument".into()),
        }
    }
    let api = api(url)?;
    let session = SessionDir::open(&directory()?, false)
        .and_then(|dir| dir.load())
        .map_err(|e| {
            if e.kind() == io::ErrorKind::NotFound {
                "no session, run publicompute login".into()
            } else {
                format!("cannot read session: {e}")
            }
        })?;
    session.check(&api)?;
    let model = match model {
        Some(model) => model.to_owned(),
        None => sock::get_status(socket)
            .ok()
            .and_then(|s| s.contribution?.serving)
            .filter(|model| !model.is_empty())
            .ok_or_else(|| {
                format!(
                    "the agent has no serving model, {}",
                    model_help(&http(), &api, &session)
                )
            })?,
    };
    let chat = Chat { api, session, model };
    if let Some(question) = question {
        chat.answer(&http(), &[Message::user(question.into())], None)?;
        return Ok(());
    }
    chat.conversation()
}

fn model_help(agent: &Agent, api: &Endpoint, session: &Session) -> String {
    let models = (|| {
        session.check_expiry().ok()?;
        let mut reply = agent
            .get(api.url("/v1/models"))
            .header("Cookie", session.cookie())
            .call()
            .ok()?;
        if !reply.status().is_success() {
            return None;
        }
        let body: Value = reply.body_mut().read_json().ok()?;
        let ids: Vec<_> = body["data"]
            .as_array()?
            .iter()
            .filter_map(|model| model["id"].as_str())
            .collect();
        (!ids.is_empty()).then(|| session.redact(&ids.join(", ")))
    })();
    match models {
        Some(models) => format!(
            "use --model ID (available: {})",
            terminal::escape_controls(&models, false)
        ),
        None => "use --model ID (model list unavailable from /v1/models)".into(),
    }
}

#[derive(Clone, Serialize)]
struct Message {
    role: &'static str,
    content: String,
}

impl Message {
    fn user(content: String) -> Self {
        Self { role: "user", content }
    }
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    stream: bool,
    messages: &'a [Message],
}

fn request_body(model: &str, messages: &[Message]) -> Vec<u8> {
    serde_json::to_vec(&ChatRequest {
        model,
        stream: true,
        messages,
    })
    .expect("a chat request is always serializable")
}

fn trim_history(model: &str, messages: &mut Vec<Message>) -> bool {
    // The empty request gives the envelope length. Each message adds its JSON and a comma except the first.
    let envelope = request_body(model, &[]).len();
    let sizes: Vec<_> = messages
        .iter()
        .map(|message| {
            serde_json::to_vec(message)
                .expect("a chat message is always serializable")
                .len()
                + 1
        })
        .collect();
    let mut bytes = sizes.iter().sum::<usize>();
    let mut skip = 0;
    while messages.len() - skip >= 2
        && (messages.len() - skip > MAX_MESSAGES || envelope + bytes.saturating_sub(1) > MAX_REQUEST)
    {
        bytes -= sizes[skip] + sizes[skip + 1];
        skip += 2;
    }
    messages.drain(..skip);
    skip != 0
}

struct Chat {
    api: Endpoint,
    session: Session,
    model: String,
}

impl Chat {
    fn chat_error(&self, agent: &Agent, mut reply: Response<Body>) -> String {
        match reply.status().as_u16() {
            401 => EXPIRED.into(),
            404 => format!("model_not_found: {}", model_help(agent, &self.api, &self.session)),
            429 => "too_many_chats: wait for an active chat to finish and try again".into(),
            503 => {
                let body: Value = reply.body_mut().read_json().unwrap_or_default();
                match body["error"]["code"].as_str().or(body["error"]["type"].as_str()) {
                    Some("no_chain_ready") => "no_chain_ready: no chain is ready for this model yet, try again in a few minutes".into(),
                    Some("network_full") | Some("no_capacity") => "network_full: no free seat on this model, try again later".into(),
                    _ => "the network is unavailable (HTTP 503), try again later".into(),
                }
            }
            status => format!("chat failed (HTTP {status})"),
        }
    }

    fn answer(
        &self,
        agent: &Agent,
        messages: &[Message],
        cancel: Option<&AtomicBool>,
    ) -> Result<Option<String>, String> {
        let body = request_body(&self.model, messages);
        if body.len() > MAX_REQUEST {
            return Err("the question exceeds the 256 KiB chat request limit".into());
        }
        self.session.check_expiry()?;
        let reply = agent
            .post(self.api.url("/v1/chat/completions"))
            .header("Cookie", self.session.cookie())
            .header("Accept", "text/event-stream")
            .header("Content-Type", "application/json")
            .send(body.as_slice())
            .map_err(|_| "cannot reach the API for chat")?;
        if !reply.status().is_success() {
            return Err(self.chat_error(agent, reply));
        }
        let mut stdout = AnswerOutput::open(cancel).map_err(|e| format!("cannot open chat output: {e}"))?;
        let mut answer = cancel.is_some().then(String::new);
        let mut printed = false;
        let completion = stream::consume(reply.into_body().into_reader(), &self.model, &self.session, |text| {
            stdout.write(text)?;
            printed = true;
            if let Some(saved) = &mut answer {
                if saved.len() + text.len() <= MAX_REQUEST {
                    saved.push_str(text);
                } else {
                    answer = None;
                }
            }
            Ok(())
        });
        let printed_to_terminal = printed && stdout.terminal;
        drop(stdout);
        if printed_to_terminal {
            eprintln!();
        }
        let completion = completion?;
        if completion.free_seat == Some(false) {
            eprintln!("contribution seat not used");
        }
        if let Some(rate) = completion.tokens_per_second {
            eprintln!("{rate:.1} tok/s");
        }
        Ok(answer)
    }

    fn conversation(&self) -> Result<(), String> {
        let interrupt = Interrupt::install().map_err(|e| format!("cannot handle Ctrl-C: {e}"))?;
        let stdin = io::stdin();
        let mut tty = if stdin.is_terminal() {
            Some(terminal::open_tty().map_err(|e| format!("cannot open terminal: {e}"))?)
        } else {
            None
        };
        let mut stdin = stdin.lock();
        let mut history = Vec::new();
        let mut first_prompt = true;
        loop {
            let model = first_prompt.then_some(self.model.as_str());
            let input = prompt(&mut tty, &mut stdin, &interrupt, model).map_err(|e| format!("cannot read question: {e}"))?;
            first_prompt = false;
            let question = match input {
                Input::Line(line) if line.trim() == "/exit" => return Ok(()),
                Input::Line(line) if line.trim().is_empty() => continue,
                Input::Line(line) => line.trim_end_matches(['\n', '\r']).to_owned(),
                Input::Eof => return Ok(()),
                Input::Interrupted => continue,
            };
            let user = Message::user(question);
            if request_body(&self.model, std::slice::from_ref(&user)).len() > MAX_REQUEST {
                eprintln!("! the question exceeds the 256 KiB chat request limit");
                continue;
            }
            let mut candidate = history.clone();
            candidate.push(user);
            let trimmed = trim_history(&self.model, &mut candidate);
            let notice = match self.interruptible_answer(&candidate, &interrupt) {
                Some(Ok(Some(content))) => {
                    candidate.push(Message {
                        role: "assistant",
                        content,
                    });
                    let trimmed_after_answer = trim_history(&self.model, &mut candidate);
                    if !candidate.is_empty() {
                        history = candidate;
                        if trimmed || trimmed_after_answer {
                            eprintln!("chat history limit reached, dropped oldest turns");
                        }
                        continue;
                    }
                    "chat history limit reached, the last turn was too large to keep".into()
                }
                Some(Err(error)) if error == EXPIRED => return Err(error),
                Some(Err(error)) => format!("! {error}"),
                Some(Ok(None)) => "chat history limit reached, the last turn was too large to keep".into(),
                None => "answer stopped".into(),
            };
            eprintln!("{notice}");
        }
    }

    fn interruptible_answer(
        &self,
        history: &[Message],
        interrupt: &Interrupt,
    ) -> Option<Result<Option<String>, String>> {
        let agent = http();
        let cancel = AtomicBool::new(false);
        thread::scope(|scope| {
            let worker = scope.spawn(|| self.answer(&agent, history, Some(&cancel)));
            while !worker.is_finished() {
                if interrupt.take() {
                    cancel.store(true, Ordering::Relaxed);
                    // Shut down the socket even when the worker is waiting for the next SSE event.
                    agent.clone().shutdown();
                    let _ = worker.join();
                    return None;
                }
                thread::sleep(Duration::from_millis(20));
            }
            Some(worker.join().expect("chat worker panicked"))
        })
    }
}

fn prompt(tty: &mut Option<File>, stdin: &mut impl BufRead, interrupt: &Interrupt, model: Option<&str>) -> io::Result<Input> {
    if let Some(tty) = tty {
        let mut out = io::stderr().lock();
        let color = render::supports_color(out.is_terminal());
        if let Some(model) = model.filter(|_| out.is_terminal()) {
            return terminal::empty_prompt(tty, &mut out, interrupt, model, color);
        }
        write!(out, "{}", render::chat_prompt(color))?;
        out.flush()?;
        return terminal::read_tty(tty, interrupt, || Ok(()));
    }
    let mut line = String::new();
    let size = stdin.read_line(&mut line)?;
    if interrupt.take() {
        Ok(Input::Interrupted)
    } else if size == 0 {
        Ok(Input::Eof)
    } else {
        Ok(Input::Line(line))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MODEL: &str = "gemma-4-12b";

    fn assistant(content: &str) -> Message {
        Message {
            role: "assistant",
            content: content.into(),
        }
    }

    #[test]
    fn trims_whole_turns_at_the_message_limit() {
        let mut history = Vec::new();
        for n in 0..100 {
            history.push(Message::user(n.to_string()));
            history.push(assistant("reply"));
        }
        assert!(!trim_history(MODEL, &mut history));
        history.push(Message::user("latest".into()));
        assert!(trim_history(MODEL, &mut history));
        assert_eq!(history.len(), 199);
        assert_eq!(history[0].content, "1");
        assert_eq!(history.last().unwrap().content, "latest");
    }

    #[test]
    fn trims_exact_request_size_with_model_and_escaped_content() {
        let mut history = vec![
            Message::user("old".into()),
            assistant(&"\"".repeat(MAX_REQUEST / 2)),
            Message::user("latest".into()),
        ];
        assert!(trim_history(MODEL, &mut history));
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].content, "latest");

        let overhead = request_body(MODEL, &[Message::user(String::new()), assistant("")]).len();
        let mut boundary = vec![Message::user("a".repeat(MAX_REQUEST - overhead)), assistant("")];
        assert_eq!(request_body(MODEL, &boundary).len(), MAX_REQUEST);
        assert!(!trim_history(MODEL, &mut boundary));
        assert!(trim_history("longer-model-id", &mut boundary.clone()));
        boundary[0].content.push('b');
        assert!(trim_history(MODEL, &mut boundary));
        assert!(boundary.is_empty());
    }
}
