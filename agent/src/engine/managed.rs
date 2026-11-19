//! A managed engine listens on loopback and runs in its own process group.
//! On Unix, terminate the whole group, escalate after the grace period and
//! wait for exit before allowing a replacement to use the GPU. Windows creates
//! a process group and stops the child with forced termination.

use std::{io, path::PathBuf, process::Stdio, time::Duration};

use anyhow::{bail, Context, Result};
use tokio::process::{Child, Command};
use tracing::{info, warn};

#[derive(Debug, Clone)]
pub struct ManagedCfg {
    pub bin: PathBuf,
    pub model_path: PathBuf,
    pub port: u16,
    pub gpu_index: Option<String>,
    /// Additional arguments forwarded unchanged to the child.
    pub extra_args: Vec<String>,
    /// File receiving both output streams from the child.
    pub log_path: PathBuf,
    /// Grace period between SIGTERM and SIGKILL.
    pub stop_grace: Duration,
    /// Wait this long after forced termination before reporting a stuck process
    /// to the supervisor for quarantine.
    pub kill_wait: Duration,
}

/// Extract the port used by the server listener.
pub fn port_from_url(url: &str) -> Result<u16> {
    let rest = url.trim_end_matches('/').rsplit(':').next().unwrap_or("");
    rest.parse::<u16>().with_context(|| format!("invalid port in --llama-url {url:?}"))
}

/// Reject a port already in use so the child cannot collide with an existing listener.
pub async fn ensure_port_free(port: u16) -> Result<()> {
    match tokio::time::timeout(Duration::from_secs(2), tokio::net::TcpStream::connect(("127.0.0.1", port))).await {
        Ok(Ok(_)) => bail!("port {port} is already in use; managed mode needs a free port"),
        _ => Ok(()),
    }
}

/// Parse the launch options that describe device offload. The agent reports
/// these settings as observations for verification.
pub fn settings_from_args(args: &[String]) -> publicompute_proto::node::EngineSettings {
    let mut s = publicompute_proto::node::EngineSettings { known: true, gpu_layers: "auto".into(), ..Default::default() };
    let mut i = 0;
    while i < args.len() {
        let (name, inline) = match args[i].split_once('=') {
            Some((n, v)) if n.starts_with("--") => (n, Some(v.to_string())),
            _ => (args[i].as_str(), None),
        };
        let mut value = || match &inline {
            Some(v) => Some(v.clone()),
            None => {
                i += 1;
                args.get(i).cloned()
            }
        };
        match name {
            "-ngl" | "--gpu-layers" | "--n-gpu-layers" => {
                if let Some(v) = value() {
                    s.gpu_layers = v;
                }
            }
            "-ncmoe" | "--n-cpu-moe" => {
                if let Some(v) = value().and_then(|v| v.parse().ok()) {
                    s.n_cpu_moe = v;
                }
            }
            "-dev" | "--device" => {
                if let Some(v) = value() {
                    s.device = v;
                }
            }
            "-cmoe" | "--cpu-moe" => s.cpu_moe = true,
            "-nkvo" | "--no-kv-offload" => s.kv_offload_disabled = true,
            "-kvo" | "--kv-offload" => s.kv_offload_disabled = false,
            _ => {}
        }
        i += 1;
    }
    s
}

/// Default context size in tokens for each concurrent slot.
pub const CTX_PER_POSTO: u64 = 4096;

/// Expand the total context to hold one context per slot. For one slot,
/// preserve the supplied arguments. Reject an explicit parallel option because
/// the agent owns the slot count.
pub fn args_con_posti(extra: &[String], slots: u32) -> Result<Vec<String>> {
    if slots <= 1 {
        return Ok(extra.to_vec());
    }
    let mut out: Vec<String> = Vec::with_capacity(extra.len() + 4);
    let mut ctx: Option<u64> = None;
    let mut i = 0;
    while i < extra.len() {
        let (name, inline) = match extra[i].split_once('=') {
            Some((n, v)) if n.starts_with("--") => (n, Some(v.to_string())),
            _ => (extra[i].as_str(), None),
        };
        match name {
            "-np" | "--parallel" => bail!("--llama-arg {name}: use --slots to set the slot count"),
            "-c" | "--ctx-size" => {
                let v = match inline {
                    Some(v) => v,
                    None => {
                        i += 1;
                        extra.get(i).cloned().with_context(|| format!("missing value for {name}"))?
                    }
                };
                let n: u64 = v.parse().with_context(|| format!("{name}: expected a number, got {v:?}"))?;
                if n == 0 {
                    bail!("{name} 0 uses the model context; --slots requires an explicit context per slot");
                }
                ctx = Some(n);
            }
            _ => out.push(extra[i].clone()),
        }
        i += 1;
    }
    out.extend(["--ctx-size".to_string(), (ctx.unwrap_or(CTX_PER_POSTO) * u64::from(slots)).to_string(), "--parallel".to_string(), slots.to_string()]);
    Ok(out)
}

impl ManagedCfg {
    pub fn spawn(&self, gpu_uuid: Option<&str>) -> Result<Process> {
        let open = || std::fs::OpenOptions::new().create(true).append(true).open(&self.log_path);
        let mut log = open().with_context(|| format!("open log {}", self.log_path.display()))?;
        {
            use std::io::Write;
            let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
            let _ = writeln!(log, "--- llama-server startup (Unix time {now}) ---");
        }
        let mut cmd = Command::new(&self.bin);
        if let Some(selector) = gpu_uuid.or(self.gpu_index.as_deref()) {
            // Use the device UUID because the runtime and driver utility may enumerate cards differently.
            cmd.env("CUDA_VISIBLE_DEVICES", selector);
        }
        cmd.arg("-m")
            .arg(&self.model_path)
            .args(["--host", "127.0.0.1", "--port"])
            .arg(self.port.to_string())
            .args(&self.extra_args)
            .stdin(Stdio::null())
            .stdout(Stdio::from(log.try_clone()?))
            .stderr(Stdio::from(log));
        let process = Process::spawn(cmd).with_context(|| format!("start {}", self.bin.display()))?;
        info!(pid = process.pid, log = %self.log_path.display(), "llama-server started");
        Ok(process)
    }
}

pub struct Process {
    pub child: Child,
    // Child::id disappears when the launcher is reaped, even if its group survives.
    pid: u32,
}

impl Process {
    pub fn spawn(mut cmd: Command) -> io::Result<Self> {
        cmd.kill_on_drop(true);
        #[cfg(unix)]
        cmd.process_group(0);
        #[cfg(windows)]
        cmd.creation_flags(0x0000_0200); // CREATE_NEW_PROCESS_GROUP
        let child = cmd.spawn()?;
        let pid = child.id().expect("new child has a PID");
        Ok(Self { child, pid })
    }

    #[cfg(unix)]
    fn signal_group(&self, sig: i32) -> io::Result<bool> {
        // A negative PID addresses the entire process group.
        if unsafe { libc::kill(-(self.pid as i32), sig) } == 0 {
            return Ok(true);
        }
        let err = io::Error::last_os_error();
        if err.raw_os_error() == Some(libc::ESRCH) {
            return Ok(false);
        }
        Err(err)
    }

    pub async fn wait_for_exit(&mut self, process: &str) -> Result<()> {
        let status = self.child.wait().await.with_context(|| format!("could not reap {process}"))?;
        #[cfg(unix)]
        while self.signal_group(0).with_context(|| format!("could not check the {process} process group"))? {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        info!(process, %status, "process exited");
        Ok(())
    }
}

/// Stop the process group before another engine can claim the same GPU.
pub async fn stop(child: &mut Process, grace: Duration, kill_wait: Duration, process: &str) -> Result<()> {
    let pid = child.pid;
    #[cfg(unix)]
    {
        let _ = child.signal_group(libc::SIGTERM);
        // A stopped child cannot handle SIGTERM until execution resumes.
        let _ = child.signal_group(libc::SIGCONT);
        if let Ok(Ok(())) = tokio::time::timeout(grace, child.wait_for_exit(process)).await {
            return Ok(());
        }
        warn!(process, pid, "process group has not exited; sending SIGKILL");
        let _ = child.signal_group(libc::SIGKILL);
    }
    #[cfg(not(unix))]
    {
        let _ = grace;
        let _ = child.child.start_kill();
    }
    match tokio::time::timeout(kill_wait, child.wait_for_exit(process)).await {
        Ok(result) => result,
        Err(_) => bail!("{process} (pid {pid}) still running {} s after SIGKILL", kill_wait.as_secs()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offload_settings_are_read_from_the_launch_args() {
        let a = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let d = settings_from_args(&a(&[]));
        assert!(d.known && d.gpu_layers == "auto" && d.n_cpu_moe == 0 && !d.cpu_moe && !d.kv_offload_disabled && d.device.is_empty());
        let s = settings_from_args(&a(&["-c", "4096", "-ngl", "20", "--n-cpu-moe=7", "-nkvo", "-dev", "CUDA0"]));
        assert_eq!((s.gpu_layers.as_str(), s.n_cpu_moe, s.kv_offload_disabled, s.device.as_str()), ("20", 7, true, "CUDA0"));
        let s = settings_from_args(&a(&["--n-gpu-layers", "all", "--cpu-moe"]));
        assert_eq!((s.gpu_layers.as_str(), s.cpu_moe), ("all", true));
    }

    #[test]
    fn slots_scale_context_and_set_parallelism() {
        let a = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(args_con_posti(&a(&["-c", "8192", "-ngl", "99"]), 1).unwrap(), a(&["-c", "8192", "-ngl", "99"]));
        assert_eq!(args_con_posti(&a(&["-c", "8192", "-ngl", "99"]), 4).unwrap(), a(&["-ngl", "99", "--ctx-size", "32768", "--parallel", "4"]));
        assert_eq!(args_con_posti(&a(&["--ctx-size=2048"]), 2).unwrap(), a(&["--ctx-size", "4096", "--parallel", "2"]));
        assert_eq!(args_con_posti(&a(&["-ngl", "99"]), 3).unwrap(), a(&["-ngl", "99", "--ctx-size", "12288", "--parallel", "3"]));
        assert!(args_con_posti(&a(&["--parallel", "2"]), 2).is_err());
        assert!(args_con_posti(&a(&["-c", "0"]), 2).is_err());
    }

    #[test]
    fn port_is_taken_from_url() {
        assert_eq!(port_from_url("http://127.0.0.1:8080").unwrap(), 8080);
        assert_eq!(port_from_url("http://localhost:18090/").unwrap(), 18090);
        assert!(port_from_url("http://localhost").is_err());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn wait_error_does_not_prove_exit() {
        let mut cmd = Command::new("sh");
        cmd.args(["-c", "exit 0"]);
        let mut child = Process::spawn(cmd).unwrap();
        let pid = child.pid as libc::pid_t;
        // Reap externally to exercise the error returned by Child::wait.
        assert_eq!(unsafe { libc::waitpid(pid, std::ptr::null_mut(), 0) }, pid);
        let error = stop(&mut child, Duration::ZERO, Duration::from_secs(1), "llama-server").await.unwrap_err();
        assert!(format!("{error:#}").contains("could not reap llama-server"), "{error:#}");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn stop_waits_for_the_group_after_the_launcher_exits() {
        let mut cmd = Command::new("sh");
        cmd.args(["-c", "trap '' TERM; sleep 30 &"]);
        let mut child = Process::spawn(cmd).unwrap();
        assert!(child.child.wait().await.unwrap().success());
        assert!(child.signal_group(0).unwrap(), "the launcher left a descendant");

        stop(&mut child, Duration::from_millis(50), Duration::from_secs(3), "launcher").await.unwrap();
        assert!(!child.signal_group(0).unwrap(), "the group must be empty before switching");
    }

    #[tokio::test]
    async fn busy_port_is_refused() {
        let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = l.local_addr().unwrap().port();
        assert!(ensure_port_free(port).await.is_err());
        drop(l);
    }
}

#[cfg(test)]
mod stop_tests {
    use super::*;

    /// A child that ignores SIGTERM must still be killed and reaped.
    #[tokio::test]
    async fn stop_kills_a_process_that_ignores_sigterm() {
        let bin = std::env::current_exe().unwrap().parent().unwrap().parent().unwrap().join("fake-llama-server");
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = l.local_addr().unwrap().port();
        drop(l);
        let cfg = ManagedCfg {
            bin,
            model_path: "/m.gguf".into(),
            port,
            gpu_index: None,
            extra_args: vec!["--fake-ignore-term".into()],
            log_path: std::env::temp_dir().join(format!("publicompute-ignore-term-{}.log", std::process::id())),
            stop_grace: Duration::from_millis(500),
            kill_wait: Duration::from_secs(3),
        };
        let mut child = cfg.spawn(None).unwrap();
        let pid = child.pid;
        tokio::time::sleep(Duration::from_millis(400)).await;
        let t0 = std::time::Instant::now();
        stop(&mut child, cfg.stop_grace, cfg.kill_wait, "llama-server").await.unwrap();
        assert!(t0.elapsed() >= Duration::from_millis(500), "should wait for the grace period");
        assert!(unsafe { libc::kill(pid as i32, 0) } != 0, "the process should have exited");
    }
}
