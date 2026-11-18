//! Standalone HTTP fixture for managed-engine supervision. It accepts the
//! supervisor's launch arguments and can delay startup, hang on completion
//! while health stays available, or ignore termination. The first-run hang
//! records its activation in a file so a replacement process can respond.

#[path = "../fake_llama.rs"]
mod fake_llama;

use std::sync::atomic::Ordering;

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let val = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    let flag = |name: &str| args.iter().any(|a| a == name);
    let port: u16 = val("--port").and_then(|p| p.parse().ok()).unwrap_or(8080);
    let model = val("-m").unwrap_or_default();
    if let Some(ms) = val("--fake-start-delay-ms").and_then(|v| v.parse().ok()) {
        tokio::time::sleep(std::time::Duration::from_millis(ms)).await;
    }
    let st = fake_llama::FakeState::new(&model);
    if flag("--fake-always-hang") {
        st.hang_completion.store(true, Ordering::SeqCst);
    }
    if let Some(f) = val("--fake-hang-first-run") {
        if !std::path::Path::new(&f).exists() {
            let _ = std::fs::write(&f, "1");
            st.hang_completion.store(true, Ordering::SeqCst);
        }
    }
    #[cfg(unix)]
    if flag("--fake-ignore-term") {
        let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).unwrap();
        tokio::spawn(async move {
            loop {
                term.recv().await;
            }
        });
    }
    let listener = match tokio::net::TcpListener::bind(("127.0.0.1", port)).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("fake-llama-server: porta {port}: {e}");
            std::process::exit(1);
        }
    };
    println!("fake-llama-server listening on {port}, model {model}");
    fake_llama::serve(listener, st).await;
}
