//! Enrollment and session handshakes report the available CPU, memory, storage,
//! operating system and accelerators. Configuration can override the inventory
//! reported by the driver utility.

use publicompute_proto::{
    common::{EngineKind, Money},
    node::{CpuInfo, EngineInfo, EngineSettings, GpuInfo, GpuVendor, HardwareProfile, HostPolicy, OsFamily, OsInfo},
};

/// One accelerator from the driver inventory or an explicit configuration override.
pub struct ManualGpu {
    pub name: String,
    pub vram_bytes: u64,
    /// Driver version from configuration or the driver query. An empty value means it is unknown.
    pub driver_version: String,
}

/// Add launch settings and the build reported by the running engine's properties
/// endpoint. These self-reported values are inputs to observational verification.
pub fn describe_engine(hw: &mut HardwareProfile, build: Option<String>, settings: EngineSettings) {
    if let Some(e) = hw.engines.first_mut() {
        e.build = build.unwrap_or_default();
        e.settings = Some(settings);
    }
}

pub fn set_detected_gpu(hw: &mut HardwareProfile, card: &crate::gpu::Lettura, build: Option<String>) {
    hw.gpus = vec![GpuInfo {
        vendor: GpuVendor::Nvidia as i32,
        name: card.name.clone(),
        vram_bytes: card.vram_totale,
        driver_version: card.driver.clone(),
        ..Default::default()
    }];
    if let Some(engine) = hw.engines.first_mut() {
        engine.backends = vec!["cuda".into()];
        engine.build = build.unwrap_or_default();
    }
}

/// Report the smaller of system memory and the cgroup limit so the advertised
/// capacity reflects what the process can use in a container.
pub fn ram_usabile(sys: &sysinfo::System) -> u64 {
    let totale = sys.total_memory();
    match sys.cgroup_limits() {
        Some(c) if c.total_memory > 0 => totale.min(c.total_memory),
        _ => totale,
    }
}

/// Return free bytes on the filesystem containing the given path, or zero if unavailable.
pub fn disco_libero(temp_dir: &std::path::Path) -> u64 {
    let percorso = std::fs::canonicalize(temp_dir).unwrap_or_else(|_| temp_dir.to_path_buf());
    let dischi = sysinfo::Disks::new_with_refreshed_list();
    dischi
        .list()
        .iter()
        .filter(|d| percorso.starts_with(d.mount_point()))
        .max_by_key(|d| d.mount_point().as_os_str().len())
        .map(|d| d.available_space())
        .unwrap_or(0)
}

/// Build the advertised inventory and measure storage at the weights directory.
/// An empty accelerator list selects the CPU backend.
pub fn profile(
    gpus: &[ManualGpu],
    engine: EngineKind,
    electricity: Option<&Money>,
    power_limit_percent: u32,
    cartella_modelli: &std::path::Path,
) -> HardwareProfile {
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    sys.refresh_cpu_all();
    let family = match std::env::consts::OS {
        "windows" => OsFamily::Windows,
        "macos" => OsFamily::Macos,
        "linux" => OsFamily::Linux,
        _ => OsFamily::Unspecified,
    };
    let ha_gpu = !gpus.is_empty();
    let gpus: Vec<GpuInfo> = gpus
        .iter()
        .map(|g| GpuInfo {
            vendor: if family == OsFamily::Macos { GpuVendor::Apple } else { GpuVendor::Nvidia } as i32,
            name: g.name.clone(),
            vram_bytes: g.vram_bytes,
            driver_version: g.driver_version.clone(),
            unified_memory: family == OsFamily::Macos,
            ..Default::default()
        })
        .collect();
    HardwareProfile {
        os: Some(OsInfo {
            family: family as i32,
            version: sysinfo::System::long_os_version().unwrap_or_default(),
            arch: std::env::consts::ARCH.to_string(),
        }),
        cpu: Some(CpuInfo {
            model: sys.cpus().first().map(|c| c.brand().trim().to_string()).unwrap_or_default(),
            physical_cores: sysinfo::System::physical_core_count().unwrap_or(0) as u32,
            logical_cores: sys.cpus().len() as u32,
        }),
        ram_bytes: sys.total_memory(),
        usable_ram_bytes: ram_usabile(&sys),
        models_free_disk_bytes: disco_libero(cartella_modelli),
        gpus,
        network: None,
        // The backend label comes from the declared inventory, independent of job execution.
        engines: vec![EngineInfo {
            kind: engine as i32,
            version: String::new(),
            backends: vec![if !ha_gpu {
                "cpu"
            } else if family == OsFamily::Macos {
                "metal"
            } else {
                "cuda"
            }
            .to_string()],
            build: String::new(),
            settings: None,
        }],
        policy: Some(HostPolicy {
            power_limit_percent,
            temperature_limit_celsius: 83,
            electricity_price_per_kwh: electricity.cloned(),
            max_vram_bytes: 0,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_driver_and_engine_description_reach_the_profile() {
        let gpu = ManualGpu { name: "RTX Test".into(), vram_bytes: 8 << 30, driver_version: "555.42".into() };
        let mut hw = profile(&[gpu], EngineKind::LlamaCpp, None, 70, std::path::Path::new("."));
        assert_eq!(hw.gpus[0].driver_version, "555.42");
        assert_eq!(hw.engines[0].backends.len(), 1);
        describe_engine(&mut hw, Some("b1-abc".into()), EngineSettings { known: true, gpu_layers: "20".into(), ..Default::default() });
        assert_eq!(hw.engines[0].build, "b1-abc");
        assert_eq!(hw.engines[0].settings.as_ref().unwrap().gpu_layers, "20");
    }

    #[tokio::test]
    async fn hardware_probe_works_without_gpu() {
        // An empty GPU inventory must still produce a complete CPU profile.
        let hw = profile(&[], EngineKind::Unspecified, None, 70, std::path::Path::new("."));
        assert!(hw.gpus.is_empty());
        assert_eq!(hw.engines[0].backends, vec!["cpu".to_string()]);
        assert!(hw.ram_bytes > 0 && hw.usable_ram_bytes > 0 && hw.usable_ram_bytes <= hw.ram_bytes);
        assert!(hw.cpu.as_ref().unwrap().logical_cores > 0);
        assert!(hw.models_free_disk_bytes > 0, "the current directory should report free disk space");
        let _ = crate::gpu::read(None).await;
    }

    #[test]
    fn free_disk_space_works_for_missing_directory() {
        let _ = disco_libero(std::path::Path::new("/path/that/does/not/exist"));
    }
}
