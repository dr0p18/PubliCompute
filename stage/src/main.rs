//! Stage command line. Subcommands run a layer range in a chain, run the
//! reference in one process, replay tokens, or compare recorded outputs.

use publicompute_stage::{experts, output};
use std::collections::HashMap;
use std::process::ExitCode;

const USAGE: &str = r#"publicompute-stage <command> [options]

Commands:
  info MODEL.gguf    Read the layer count and embedding width from the GGUF.
  reference          Run the full model in a single process as a reference.
  head               Run the head. Tokenize, prefill and record the transcript.
  middle             Run a middle stage. Forward residual state to the next stage.
  tail               Run the tail. Sample and send tokens back to the head.
  replay             Replay the tokens in --tokens and record their log probabilities.
  compare A.jsonl B.jsonl [--identical-tokens | --bit-identical | --all-positions | --tolerance NAT]
                    Compare transcripts. --tolerance bounds the maximum log-probability
                    difference before tokens first diverge. Use --all-positions when B
                    is a replay of A to compare every forced position.
  calibrate A.summary.json [B...] [--conversations N] [--g-max M] [--stages S | --shares 1,1]
           [--latency-ms MS] [--min-per-conv N] [--exclude-g G] [--json FILE]
                    Estimate throughput from stage timing summaries and recommend the
                    smallest group count that reaches 95% of peak throughput.
  compare-experts A B
                    Compare router traces written by --log-experts. Each
                    argument may be a comma-separated list of trace files.

Common options:
  --model FILE                  Full GGUF file.
  --layer-start N               First layer in this stage (default 0).
  --layer-end N                 Exclusive end layer (default -1, through the final layer).
  --ngl N                       GPU layers (default 0, use 999 for all layers).
  --threads N                   CPU threads (default 6).
  --slots N                     Sequence slots. Use the same count on every stage.
  --ctx-per-slot N              Context per slot (default 1024).
  --groups N                    In-flight groups per stage (default 2).
  --n-batch N                   Maximum batch size in tokens (default 512).
  --cache-type-k T              KV cache type for K: f16 (default), q8_0 or q4_0. Use the same on every stage.
  --cache-type-v T              KV cache type for V: f16 (default), q8_0 or q4_0. Quantized V requires
                                --flash-attn on.
  --flash-attn M                auto (default), on or off. Must be on when --cache-type-v is quantized.
  --log-experts F               Write selected MoE experts for each generation step as JSONL.
  --log-top F                   Tail or reference: write the top 20 log probabilities per sampled token.
  --sampling-threads N          Sampling threads on the tail or reference (default 0, choose automatically).
  --log-llama                   Print engine logs to stderr.
  --timeout-s N                 Connection and reply timeout (default 300 s).
  --park-ttl-s N                Keep a completed conversation's checkpoint for N seconds (default 0).

Replay and comparison:
  --tokens FILE                 Replay a JSONL transcript. Match prompts by ID.
  --verifier-mode               Replay each conversation on its own and score all generated positions.
                                Split long sequences into chunks of at most --n-batch tokens.
  --seed-tolerance X            Flag sampling mismatches above this probability margin (default 0.001).

Head and reference:
  --prompt FILE                 Text lines or JSONL objects with id and prompt fields.
  --n-prompt N                  Number of prompts (default 4).
  --skip N                      Skip this many input prompts (default 0).
  --max-tokens N                Maximum new tokens per conversation (default 64).
  --block N                     Tokens per prefill block (default up to 512, limited by batch and frame).
  --temperature X               Sampling temperature (default 0, greedy).
  --seed N                      Root sampling seed (default 1).
  --classes LIST                Comma-separated prompt classes: i or interactive, b or batch.
                                Interactive requests can suspend batch work and replay its prefix later.
  --interactive-reserve N       Slots reserved for interactive requests (default 0).
  --arrivals LIST               Admit each prompt after the listed number of generation steps.
                                Requests wait in arrival order when the slots are full.
  --first-token-delay N         Admission rounds before recording a new conversation's first token.
                                Zero uses twice the number of groups and one waits for the first token.
  --prefill-order MODE          Prefill in arrival order (fifo, default) or round-robin.
  --output PREFIX               Write PREFIX.jsonl and PREFIX.summary.json.

Head of a chain:
  --work-timeout-s N            Deadline for unconfirmed work (default --timeout-s).
  --recovery-timeout-s N        Deadline for restoring links and replaying the prefix (default 120 s).
  --max-recoveries N            Maximum recoveries per conversation (default 3).
  --service ADDR                Accept requests from the agent on a loopback address,
                                instead of reading --prompt. Open only once the chain is ready.
  --service-pipe                Accept requests on stdin and send frames on stdout.
  --next ADDR                   Next stage's listen address.
  --listen-token ADDR           Listen for the tail's token stream.
  --park-grace-s N              Protect parked member slots from public eviction for N seconds (default 0).
  --test-state-int8             Test only: quantize each outgoing row to int8 and restore FP32.

Head and middle stage:
  --state-format F              Outgoing state: fp32 (default), fp16, int8 (scale per row),
                                or int8-stream (scale per residual stream).

Speculative decoding:
  --draft-max-tokens N          Draft tokens per conversation per step (default 0, disabled).
                                Use the same maximum at every stage. Recurrent state uses
                                additional snapshots so the tail can reject a draft without
                                changing the sampled sequence.
  --draft-source F              Head: ngram (default, prompt lookup) or mtp (drafts from the tail).
  --test-draft-oracle FILE      Test only: take draft tokens from a reference transcript.
  --test-draft-corrupt-every N   Test only: change every Nth reference draft token.
  --draft-max-rows N            Head: stop adding draft rows once a batch reaches N rows (default 64).
  --test-mtp-depth N            Test only: request this draft depth for each step.
  --log-mtp FILE                Tail: write a binary trace of draft logits.
  --mtp                         Tail: load an MTP block from the GGUF. The tail can propose up
                                to --draft-max-tokens tokens using the recurrent draft context.

Stage links:
  --hello-timeout-s N           Handshake timeout (default 10 s).
  --write-timeout-s N           Write and send-queue timeout (default 120 s).
  --read-timeout-s N            Read timeout (default 0, allow idle links between requests).
  --queue-max-mib N             Maximum queued bytes per outgoing link in MiB (default 256).
  --ingress-max-mib N           Maximum queued incoming work in MiB (default 512).
  --tls-cert FILE               Node certificate chain (PEM) with URI SAN publicompute:node:ID.
  --tls-key FILE                Node private key (PEM PKCS#8).
  --tls-ca FILE                 Coordinator CA (PEM).
  --capability FILE             Signed chain capability. The stage reloads it on change
                                and closes the links at expiry.
  --capability-pub FILE         Coordinator's Ed25519 key (32 bytes or 64 hex digits).
  --clock-offset-s N            Coordinator time minus local time in seconds (default 0).
                                Applies to capability expiry. TLS uses local time.
  For a TLS link, supply the certificate, key, CA, capability, and public key.
  Without TLS credentials the stage uses plaintext links for local tests.

Middle and tail stages:
  --input-format F              Require a specific incoming state encoding.
  --index N                     Stage position in the chain, counting from 1 at the head.
  --listen ADDR                 Listen for the preceding stage.
  --next ADDR                   Middle stage: connect to the next stage.
  --head-token ADDR             Tail: connect to the head's --listen-token address.
  --stop-on-stdin-eof           Release retained state and exit when the agent closes stdin.
  --reconnect-grace-s N         Allow this long for the preceding stage to reconnect (default 3 s).
  --reserve-timeout-s N         Reserved-slot timeout (default 30 s).
  --idle-timeout-s N            Idle-slot timeout (default 120 s).
  --test-conv-error ID:N        Test only: fail the Nth generation row of conversation ID.
  --test-pilot F                Test only: sample the highest-scoring alternative token in a
                                fraction F of positions.
"#;

struct Args {
    m: HashMap<String, String>,
    pos: Vec<String>,
}

fn parse(v: &[String]) -> Result<Args, String> {
    let mut m = HashMap::new();
    let mut pos = Vec::new();
    let mut i = 0;
    while i < v.len() {
        if let Some(k) = v[i].strip_prefix("--") {
            if matches!(k, "identical-tokens" | "bit-identical" | "log-llama" | "all-positions" | "verifier-mode" | "test-state-int8" | "mtp" | "service-pipe" | "stop-on-stdin-eof") {
                m.insert(k.to_string(), String::new());
                i += 1;
                continue;
            }
            if !matches!(
                k,
                "model" | "layer-start" | "layer-end" | "ngl" | "threads" | "slots" | "ctx-per-slot" | "groups" | "n-batch"
                    | "cache-type-k" | "cache-type-v" | "flash-attn"
                    | "log-experts" | "log-top" | "sampling-threads" | "timeout-s" | "tokens" | "seed-tolerance"
                    | "prompt" | "n-prompt" | "skip" | "max-tokens" | "block" | "temperature" | "seed"
                    | "classes" | "interactive-reserve" | "arrivals" | "first-token-delay" | "prefill-order" | "output"
                    | "work-timeout-s" | "recovery-timeout-s" | "max-recoveries" | "service" | "next" | "listen-token"
                    | "state-format" | "draft-max-tokens" | "draft-source" | "test-draft-oracle"
                    | "test-draft-corrupt-every" | "draft-max-rows" | "test-mtp-depth" | "log-mtp"
                    | "hello-timeout-s" | "write-timeout-s" | "read-timeout-s" | "queue-max-mib" | "ingress-max-mib"
                    | "tls-cert" | "tls-key" | "tls-ca" | "capability" | "capability-pub" | "clock-offset-s"
                    | "input-format" | "index" | "listen" | "head-token" | "reconnect-grace-s"
                    | "reserve-timeout-s" | "idle-timeout-s" | "park-ttl-s" | "park-grace-s" | "test-conv-error" | "test-pilot"
                    | "tolerance" | "conversations" | "g-max" | "stages" | "shares" | "latency-ms" | "min-per-conv"
                    | "exclude-g" | "json"
            ) {
                return Err(format!("unknown option --{k}"));
            }
            // A following option means the value was forgotten, not that it is the value.
            let val = v.get(i + 1).filter(|x| !x.starts_with("--")).ok_or_else(|| format!("missing value for --{k}"))?;
            m.insert(k.to_string(), val.clone());
            i += 2;
        } else {
            pos.push(v[i].clone());
            i += 1;
        }
    }
    let args = Args { m, pos };
    if args.num("slots", 1_u32)? == 0 {
        return Err("--slots must be positive".into());
    }
    // Check the cache options up front, before any model is read.
    args.cache()?;
    Ok(args)
}

#[cfg_attr(not(feature = "llama"), allow(dead_code))]
impl Args {
    fn text(&self, k: &str) -> Option<&str> {
        self.m.get(k).map(|s| s.as_str())
    }
    fn cache(&self) -> Result<publicompute_stage::kv_cache::CacheConfig, String> {
        publicompute_stage::kv_cache::CacheConfig::parse(self.text("cache-type-k"), self.text("cache-type-v"), self.text("flash-attn"))
    }
    fn richiesto(&self, k: &str) -> Result<String, String> {
        self.m.get(k).cloned().ok_or_else(|| format!("missing --{k}"))
    }
    fn num<T: std::str::FromStr>(&self, k: &str, def: T) -> Result<T, String> {
        match self.m.get(k) {
            None => Ok(def),
            Some(s) => s.parse().map_err(|_| format!("invalid value for --{k}: {s}")),
        }
    }
}

fn compare(a: &Args) -> Result<ExitCode, String> {
    if a.pos.len() != 2 {
        return Err("Usage: compare A.jsonl B.jsonl".into());
    }
    let (x, y) = (output::leggi_jsonl(&a.pos[0])?, output::leggi_jsonl(&a.pos[1])?);
    if a.m.contains_key("all-positions") {
        let s = output::scarti_tutte_le_posizioni(&x, &y)?;
        println!("conversations compared: {}", s.per_conv.len());
        println!("positions: {}", s.n_posizioni);
        println!("mean difference per position: {:e}", s.medio);
        println!("p99.9 difference: {:e}", s.p999);
        println!("max difference: {:e}", s.max);
        println!("bit-identical log probabilities: {}", if s.bit_identici { "yes" } else { "no" });
        for (id, n, medio, max) in &s.per_conv {
            println!("  {id}: {n} positions, mean {medio:.4e}, max {max:.4e}");
        }
        return Ok(ExitCode::SUCCESS);
    }
    let c = output::compare(&x, &y)?;
    println!("conversations compared: {}", c.per_conv.len());
    println!("identical tokens: {}", if c.token_identici() { "yes" } else { "no" });
    for p in c.per_conv.iter().filter(|p| !p.token_identici) {
        println!("  mismatch: {} at step {}", p.prompt_id, p.primo_passo_diverso.unwrap());
    }
    match c.primo_passo_diverso() {
        None => println!("first differing step: none"),
        Some((p, s)) => println!("first differing step: {p}, step {s}"),
    }
    println!("max logprob difference: {:e}", c.scarto_max());
    println!("mean logprob difference: {:e}", c.scarto_medio());
    println!("bit-identical log probabilities: {}", if c.bit_identici() { "yes" } else { "no" });
    if let Some(t) = a.text("tolerance") {
        let tol: f64 = t.parse().map_err(|_| format!("--tolerance {t}: expected a number"))?;
        let ok = c.entro_tolleranza(tol);
        println!("max difference {:e} {} tolerance {tol}; mismatched conversations: {}/{}", c.scarto_max(), if ok { "within" } else { "above" }, c.n_token_diversi(), c.per_conv.len());
        return Ok(if ok { ExitCode::SUCCESS } else { ExitCode::from(1) });
    }
    let ok = if a.m.contains_key("bit-identical") {
        c.bit_identici()
    } else if a.m.contains_key("identical-tokens") {
        c.token_identici()
    } else {
        true
    };
    Ok(if ok { ExitCode::SUCCESS } else { ExitCode::from(1) })
}

fn compare_experts(a: &Args) -> Result<ExitCode, String> {
    if a.pos.len() != 2 {
        return Err("Usage: compare-experts A B (files or comma-separated file lists)".into());
    }
    let (x, y) = (experts::read(&a.pos[0])?, experts::read(&a.pos[1])?);
    let c = experts::compare(&x, &y);
    println!("token-layer pairs compared: {}", c.confrontati);
    println!("only in A: {}  only in B: {}", c.solo_in_a, c.solo_in_b);
    let perc = |n: usize, d: usize| if d == 0 { 0.0 } else { 100.0 * n as f64 / d as f64 };
    println!("pairs with different experts: {} ({:.3}%)", c.diversi, perc(c.diversi, c.confrontati));
    println!("different expert selections: {} of {} ({:.3}%)", c.esperti_scambiati, c.esperti_totali, perc(c.esperti_scambiati, c.esperti_totali));
    let elenco: Vec<String> = c.per_layer.iter().map(|(l, (n, d))| format!("{l}:{d}/{n}")).collect();
    println!("by layer (mismatched/total): {}", elenco.join(" "));
    Ok(ExitCode::SUCCESS)
}

#[cfg(feature = "llama")]
mod execution {
    use super::*;
    use publicompute_stage::engine::{Engine, EngineOptions};
    use publicompute_stage::slot::Deadlines;
    use publicompute_stage::stage::{self, StageOptions, Role};
    use publicompute_stage::head::{self, HeadOptions, HeadLinks};
    use std::time::Duration;

    pub fn info(a: &Args) -> Result<(), String> {
        let p = a.pos.first().ok_or("Usage: info MODEL.gguf")?;
        let (n_layer, n_embd) = publicompute_stage::engine::info_gguf(p)?;
        println!("n_layer={n_layer} n_embd={n_embd}");
        Ok(())
    }

    /// Network timeouts shared by the head and stages (defaults in `network::LinkTimeouts`).
    fn scadenze_rete(a: &Args) -> Result<(), String> {
        let d = publicompute_stage::network::LinkTimeouts::default();
        publicompute_stage::network::set_link_timeouts(publicompute_stage::network::LinkTimeouts {
            hello_s: a.num("hello-timeout-s", d.hello_s)?,
            lettura_s: a.num("read-timeout-s", d.lettura_s)?,
            scrittura_s: a.num("write-timeout-s", d.scrittura_s)?,
            coda_max_mib: a.num("queue-max-mib", d.coda_max_mib)?,
        });
        Ok(())
    }

    fn engine(a: &Args, slots: u32, mtp: bool) -> Result<Engine, String> {
        publicompute_stage::engine::set_llama_logging(a.m.contains_key("log-llama"));
        Engine::load(&EngineOptions {
            model: a.richiesto("model")?,
            layer_start: a.num("layer-start", 0)?,
            layer_end: a.num("layer-end", -1)?,
            ngl: a.num("ngl", 0)?,
            threads: a.num("threads", 6)?,
            slots,
            groups: a.num("groups", 2)?,
            ctx_per_slot: a.num("ctx-per-slot", 1024)?,
            n_batch: a.num("n-batch", 512)?,
            experts: a.m.contains_key("log-experts"),
            max_draft_tokens: a.num("draft-max-tokens", 0)?,
            mtp,
            cache: a.cache()?,
        })
    }

    /// With a TLS certificate configured, authenticate peers and enforce the
    /// signed chain assignment. Without it, open plaintext links.
    fn sicurezza_rete(a: &Args, index: u32) -> Result<(), String> {
        use publicompute_stage::tls::{TlsOptions, StageSecurity};
        let altre = ["tls-key", "tls-ca", "capability", "capability-pub", "clock-offset-s"];
        if a.text("tls-cert").is_none() {
            if let Some(k) = altre.iter().find(|k| a.text(k).is_some()) {
                return Err(format!("--{k} requires --tls-cert"));
            }
            eprintln!("[network] warning: stage links use plaintext because --tls-cert is missing");
            return Ok(());
        }
        if index == 0 {
            return Err("TLS requires --index (stage position starting at 1)".into());
        }
        let p = |k: &str| a.richiesto(k).map(std::path::PathBuf::from);
        let s = StageSecurity::new(&TlsOptions { cert: p("tls-cert")?, key: p("tls-key")?, ca: p("tls-ca")?, capability: p("capability")?, capability_pub: p("capability-pub")?, index, clock_offset_s: a.num("clock-offset-s", 0)? })?;
        let c = s.capability();
        eprintln!(
            "[network] mutual TLS 1.3 enabled: node {} at position {index} of {} in chain {} (epoch {}); capability expires in {} s",
            s.mio_id(),
            c.stages.len(),
            c.chain_id,
            c.epoch,
            c.expires_unix - s.now()
        );
        publicompute_stage::network::set_security(s);
        Ok(())
    }

    pub fn head(a: &Args, reference: bool, forzato: bool) -> Result<(), String> {
        scadenze_rete(a)?;
        if !reference {
            sicurezza_rete(a, 1)?;
        }
        let n_prompt: usize = a.num("n-prompt", 4)?;
        let slots: u32 = a.num("slots", n_prompt as u32)?;
        let attesa_s: u64 = a.num("timeout-s", 300)?;
        let service = a.text("service").map(String::from);
        let service_pipe = a.text("service-pipe").is_some();
        if service_pipe && service.is_some() {
            return Err("--service and --service-pipe are mutually exclusive".into());
        }
        if (service.is_some() || service_pipe) && reference {
            return Err("service mode requires a chain head".into());
        }
        let block = a.num("block", 0)?;
        if block == 0 && a.text("block").is_some() {
            return Err("--block must be positive".into());
        }
        let o = HeadOptions {
            // Service requests supply their own prompts after startup.
            prompt: if service.is_some() || service_pipe { a.text("prompt").unwrap_or_default().to_string() } else { a.richiesto("prompt")? },
            service,
            service_pipe,
            park_ttl_s: a.num("park-ttl-s", 0)?,
            park_grace_s: a.num("park-grace-s", 0)?,
            n_prompt,
            salta: a.num("skip", 0)?,
            max_tokens: a.num("max-tokens", 64)?,
            block,
            groups: a.num("groups", 2)?,
            temperature: a.num("temperature", 0.0)?,
            seed: a.num("seed", 1)?,
            ctx_per_slot: a.num("ctx-per-slot", 1024)?,
            output: a.text("output").map(String::from),
            attesa_s,
            t_lavoro_s: a.num("work-timeout-s", attesa_s)?,
            t_ripresa_s: a.num("recovery-timeout-s", 120)?,
            k_riprese: a.num("max-recoveries", 3)?,
            forzati: if forzato { Some(a.richiesto("tokens")?) } else { None },
            experts: a.text("log-experts").map(String::from),
            top: a.text("log-top").map(String::from),
            thread_campiona: a.num("sampling-threads", 0)?,
            prova_stato_int8: a.m.contains_key("test-state-int8"),
            state_format: publicompute_stage::network::parse_state_format(a.text("state-format").unwrap_or("fp32"))?,
            forma_verificatore: a.m.contains_key("verifier-mode"),
            tol_seme: a.num("seed-tolerance", 1e-3)?,
            classi: a.text("classes").map(String::from),
            riserva_interattive: a.num("interactive-reserve", 0)?,
            arrivi: a.text("arrivals").map(String::from),
            ritardo_primo: a.num("first-token-delay", 0)?,
            prefill_order: publicompute_stage::head::PrefillOrder::da_testo(a.text("prefill-order").unwrap_or("fifo"))?,
            max_draft_tokens: if reference { 0 } else { a.num("draft-max-tokens", 0)? },
            max_draft_rows: a.num("draft-max-rows", 64)?,
            draft_oracle: a.text("test-draft-oracle").map(String::from),
            draft_oracle_corrupt_every: a.num("test-draft-corrupt-every", 0)?,
            draft_source: publicompute_stage::draft::DraftSource::da_testo(a.text("draft-source").unwrap_or("ngram"))?,
            mtp_k_prova: a.num("test-mtp-depth", 0)?,
        };
        if a.m.contains_key("mtp") {
            return Err("--mtp requires the tail; use --draft-source mtp on the head".into());
        }
        let network = if reference {
            None
        } else {
            Some(HeadLinks { next: a.richiesto("next")?, ascolta_token: a.richiesto("listen-token")? })
        };
        let m = engine(a, slots, false)?;
        head::run(m, o, network.as_ref(), slots)
    }

    pub fn stage(a: &Args, role: Role) -> Result<(), String> {
        scadenze_rete(a)?;
        sicurezza_rete(a, a.num("index", 0)?)?;
        let slots: u32 = a.num("slots", 4)?;
        let o = StageOptions {
            role,
            listen: a.richiesto("listen")?,
            next: a.text("next").map(String::from),
            testa_token: a.text("head-token").map(String::from),
            park_ttl_s: a.num("park-ttl-s", 0)?,
            stop_on_stdin_eof: a.m.contains_key("stop-on-stdin-eof"),
            scadenze: Deadlines {
                riserva: Duration::from_secs(a.num("reserve-timeout-s", 30)?),
                inattivo: Duration::from_secs(a.num("idle-timeout-s", 120)?),
            },
            slots,
            ctx_per_slot: a.num("ctx-per-slot", 1024)?,
            attesa_s: a.num("timeout-s", 300)?,
            grazia_s: a.num("reconnect-grace-s", 3)?,
            ingress_max_mib: a.num("ingress-max-mib", 512)?,
            experts: a.text("log-experts").map(String::from),
            top: a.text("log-top").map(String::from),
            thread_campiona: a.num("sampling-threads", 0)?,
            state_format: publicompute_stage::network::parse_state_format(a.text("state-format").unwrap_or("fp32"))?,
            index: a.num("index", 0)?,
            formato_ingresso: match a.text("input-format") {
                None => None,
                Some(t) => Some(publicompute_stage::network::parse_state_format(t)?),
            },
            prova_pilota: match a.text("test-pilot") {
                None => None,
                Some(v) => Some(v.parse::<f64>().map_err(|_| "--test-pilot: invalid fraction")?).filter(|f| *f > 0.0),
            },
            max_draft_tokens: a.num("draft-max-tokens", 0)?,
            registro_mtp: a.text("log-mtp").map(String::from),
            prova_errore_conv: match a.text("test-conv-error") {
                None => None,
                Some(v) => {
                    let (id, n) = v.split_once(':').ok_or("--test-conv-error requires ID:N")?;
                    Some((id.parse().map_err(|_| "--test-conv-error: invalid ID")?, n.parse().map_err(|_| "--test-conv-error: invalid N")?))
                }
            },
        };
        let mtp = a.m.contains_key("mtp");
        if mtp && role != Role::Tail {
            return Err("--mtp requires the tail".into());
        }
        if a.text("draft-source").is_some() {
            return Err("--draft-source requires the head; use --mtp on the tail".into());
        }
        let m = engine(a, slots, mtp)?;
        stage::run(m, &o)
    }
}

fn main() -> ExitCode {
    let v: Vec<String> = std::env::args().skip(1).collect();
    let Some(cmd) = v.first() else {
        eprint!("{USAGE}");
        return ExitCode::from(2);
    };
    if matches!(cmd.as_str(), "-h" | "--help" | "help") {
        print!("{USAGE}");
        return ExitCode::SUCCESS;
    }
    if v.as_slice() == ["--version"] {
        println!(concat!("publicompute-stage ", env!("CARGO_PKG_VERSION")));
        return ExitCode::SUCCESS;
    }
    let args = match parse(&v[1..]) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };
    let r: Result<ExitCode, String> = match cmd.as_str() {
        "compare" => compare(&args),
        "compare-experts" => compare_experts(&args),
        "calibrate" => publicompute_stage::calibration::comando(&args.pos, &args.m).map(|_| ExitCode::SUCCESS),
        #[cfg(feature = "llama")]
        "info" => execution::info(&args).map(|_| ExitCode::SUCCESS),
        #[cfg(feature = "llama")]
        "reference" => execution::head(&args, true, false).map(|_| ExitCode::SUCCESS),
        #[cfg(feature = "llama")]
        "replay" => execution::head(&args, true, true).map(|_| ExitCode::SUCCESS),
        #[cfg(feature = "llama")]
        "head" => execution::head(&args, false, false).map(|_| ExitCode::SUCCESS),
        #[cfg(feature = "llama")]
        "middle" => execution::stage(&args, publicompute_stage::stage::Role::Middle).map(|_| ExitCode::SUCCESS),
        #[cfg(feature = "llama")]
        "tail" => execution::stage(&args, publicompute_stage::stage::Role::Tail).map(|_| ExitCode::SUCCESS),
        altro => Err(format!("unknown command or missing llama feature: {altro}")),
    };
    match r {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(v: &[&str]) -> Result<Args, String> {
        parse(&v.iter().map(|s| s.to_string()).collect::<Vec<_>>())
    }

    #[test]
    fn parse_accepts_cache_options() {
        let a = args(&["--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "--flash-attn", "on"]).unwrap();
        let c = a.cache().unwrap();
        assert_eq!((c.k.to_string(), c.v.to_string(), c.flash_attn.to_string()), ("q8_0".into(), "q8_0".into(), "on".into()));
    }

    #[test]
    fn parse_rejects_quantized_v_without_flash_attention() {
        let err = args(&["--cache-type-v", "q4_0", "--flash-attn", "off"]).err().unwrap();
        assert!(err.contains("requires --flash-attn on"), "{err}");
        let err_auto = args(&["--cache-type-v", "q4_0"]).err().unwrap();
        assert!(err_auto.contains("requires --flash-attn on"), "{err_auto}");
        assert!(args(&["--cache-type-k", "q9"]).is_err());
    }
}
