use std::{
    collections::HashMap,
    fs::{self, File, Metadata, OpenOptions},
    io::{self, Read, Write},
    path::{Path, PathBuf},
    sync::{LazyLock, Mutex},
    time::Duration,
};

use anyhow::{ensure, Context, Result};
use publicompute_proto::common::ModelRef;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const GIB: u64 = 1 << 30;
const RESERVE: u64 = 512 << 20;
const MAX_STATE_BYTES: u64 = 16 << 10;

#[derive(Debug, Deserialize)]
pub struct Model {
    pub id: String,
    pub quant: String,
    pub url: String,
    pub bytes: u64,
    pub sha256: String,
    pub file: String,
}

pub static MANIFEST: LazyLock<Vec<Model>> =
    LazyLock::new(|| serde_json::from_str(include_str!("../models.json")).expect("invalid embedded model manifest"));

pub fn find(id: &str, quant: Option<&str>) -> Option<&'static Model> {
    MANIFEST
        .iter()
        .find(|m| m.id == id && quant.is_none_or(|q| q == m.quant))
}

impl Model {
    pub fn reference(&self) -> ModelRef {
        ModelRef {
            model_id: self.id.clone(),
            quantization: self.quant.clone(),
            weights_sha256: hex::decode(&self.sha256).expect("invalid embedded weights hash"),
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ActiveModel {
    pub id: String,
    pub quant: String,
    // Apply the selection to FARM mode on startup only if no later preference replaced it.
    #[serde(default)]
    pub use_revision: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
struct FileStamp {
    bytes: u64,
    mtime_ns: i128,
    inode: u64,
    device: u64,
    ctime_ns: i128,
}

impl FileStamp {
    #[cfg(unix)]
    fn read(meta: &Metadata) -> Result<Self> {
        use std::os::unix::fs::MetadataExt;
        Ok(Self {
            bytes: meta.len(),
            mtime_ns: i128::from(meta.mtime()) * 1_000_000_000 + i128::from(meta.mtime_nsec()),
            inode: meta.ino(),
            device: meta.dev(),
            ctime_ns: i128::from(meta.ctime()) * 1_000_000_000 + i128::from(meta.ctime_nsec()),
        })
    }

    #[cfg(not(unix))]
    fn read(meta: &Metadata) -> Result<Self> {
        Ok(Self {
            bytes: meta.len(),
            mtime_ns: meta.modified()?.duration_since(std::time::UNIX_EPOCH)?.as_nanos() as i128,
            inode: 0,
            device: 0,
            ctime_ns: 0,
        })
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct Verified {
    id: String,
    quant: String,
    stamp: FileStamp,
    sha256: String,
}

pub struct Store {
    data_dir: PathBuf,
    pub worker: tokio::sync::Mutex<()>,
    cache: Mutex<()>,
    rejected: Mutex<HashMap<String, FileStamp>>,
}

impl Store {
    pub fn new(data_dir: &Path) -> Self {
        Self {
            data_dir: data_dir.into(),
            worker: Default::default(),
            cache: Default::default(),
            rejected: Default::default(),
        }
    }

    fn dir(&self) -> PathBuf {
        self.data_dir.join("models")
    }

    pub fn path(&self, model: &Model) -> PathBuf {
        self.dir().join(&model.file)
    }

    pub fn model_at(&self, path: &Path) -> Option<&'static Model> {
        let model = MANIFEST.iter().find(|m| path.file_name().is_some_and(|name| name == m.file.as_str()))?;
        if self.path(model) == path {
            return Some(model);
        }
        let parent = path.parent()?.canonicalize().ok()?;
        (parent == self.dir().canonicalize().ok()?).then_some(model)
    }

    pub fn partial(&self, model: &Model) -> PathBuf {
        self.dir().join(format!("{}.agent.part", model.file))
    }

    fn check_dir(&self) -> Result<()> {
        let dir = self.dir();
        let metadata = fs::symlink_metadata(&dir).with_context(|| format!("read {}", dir.display()))?;
        ensure!(metadata.is_dir(), "expected a model directory at {}", dir.display());
        Ok(())
    }

    pub fn prepare(&self) -> Result<()> {
        match fs::create_dir(self.dir()) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => self.check_dir(),
            Err(e) => Err(e).context("create model directory"),
        }
    }

    pub fn installed(&self, model: &Model) -> Result<bool> {
        self.check_dir()?;
        Ok(file_size(&self.path(model))? == Some(model.bytes))
    }

    pub fn download_size(&self, model: &Model) -> Result<Option<u64>> {
        self.check_dir()?;
        // Read the partial first. If promotion removes it, the final file is
        // already visible by the time we inspect the destination.
        let partial = file_size(&self.partial(model));
        self.download_size_after_partial(model, partial)
    }

    fn download_size_after_partial(&self, model: &Model, partial: Result<Option<u64>>) -> Result<Option<u64>> {
        if self.download_candidate(model)? {
            return Ok(None);
        }
        Ok(Some(model.bytes - partial?.filter(|n| *n <= model.bytes).unwrap_or(0)))
    }

    pub fn download_candidate(&self, model: &Model) -> Result<bool> {
        self.check_dir()?;
        let Some(meta) = download_destination(&self.path(model))? else {
            return Ok(false);
        };
        if meta.len() != model.bytes {
            return Ok(false);
        }
        let stamp = FileStamp::read(&meta)?;
        Ok(self.rejected.lock().unwrap().get(&model.file) != Some(&stamp))
    }

    pub fn space(&self, missing: u64) -> Result<()> {
        self.check_dir()?;
        #[cfg(unix)]
        {
            let fs = rustix::fs::statvfs(self.dir()).context("check free model disk space")?;
            check_space(missing, fs.f_bavail.saturating_mul(fs.f_frsize))
        }
        #[cfg(not(unix))]
        {
            let _ = missing;
            Ok(())
        }
    }

    pub fn active(&self) -> Result<Option<(&'static Model, ActiveModel)>> {
        let Some(active): Option<ActiveModel> = read_json(&self.data_dir.join("active-model.json"), false)? else {
            return Ok(None);
        };
        let model = find(&active.id, Some(&active.quant)).context("model is absent from the manifest")?;
        self.verify(model, || true)?;
        Ok(Some((model, active)))
    }

    pub fn save_active(&self, model: &Model, use_revision: Option<u64>) -> Result<()> {
        ensure!(self.installed(model)?, "the verified model file is no longer present");
        write_json(
            &self.data_dir.join("active-model.json"),
            &ActiveModel {
                id: model.id.clone(),
                quant: model.quant.clone(),
                use_revision,
            },
        )
    }

    fn verified(&self) -> Vec<Verified> {
        read_json(&self.dir().join("verified.json"), true)
            .ok()
            .flatten()
            .unwrap_or_default()
    }

    fn remember(&self, model: &Model, stamp: FileStamp) {
        let _cache = self.cache.lock().unwrap();
        let mut entries = self.verified();
        entries.retain(|v| v.id != model.id || v.quant != model.quant);
        entries.push(Verified {
            id: model.id.clone(),
            quant: model.quant.clone(),
            stamp,
            sha256: model.sha256.clone(),
        });
        if let Err(e) = write_json(&self.dir().join("verified.json"), &entries) {
            tracing::warn!(error = %e, "could not save the verified model cache");
        }
    }

    pub fn verify(&self, model: &Model, current: impl Fn() -> bool) -> Result<()> {
        self.check_dir()?;
        let mut file = open_file(&self.path(model), false)?;
        let stamp = FileStamp::read(&file.metadata()?)?;
        ensure!(stamp.bytes == model.bytes, "incorrect model size for {}", model.id);
        ensure!(
            self.rejected.lock().unwrap().get(&model.file) != Some(&stamp),
            "SHA-256 mismatch for {}",
            model.id
        );
        #[cfg(unix)]
        if self
            .verified()
            .iter()
            .any(|v| v.id == model.id && v.quant == model.quant && v.sha256 == model.sha256 && v.stamp == stamp)
        {
            return Ok(());
        }
        if !verify_hash(&mut file, model, &stamp, current)? {
            self.rejected.lock().unwrap().insert(model.file.clone(), stamp);
            anyhow::bail!("SHA-256 mismatch for {}", model.id);
        }
        self.remember(model, stamp);
        Ok(())
    }

    pub fn promote(&self, model: &Model, current: impl Fn() -> bool) -> Result<()> {
        self.check_dir()?;
        let partial = self.partial(model);
        let mut file = open_file(&partial, false)?;
        let stamp = FileStamp::read(&file.metadata()?)?;
        ensure!(stamp.bytes == model.bytes, "incomplete model download");
        let valid = verify_hash(&mut file, model, &stamp, &current)?;
        ensure!(
            stamp == FileStamp::read(&fs::symlink_metadata(&partial)?)?,
            "partial model file was replaced during verification"
        );
        if !valid {
            fs::remove_file(&partial)?;
            anyhow::bail!("SHA-256 mismatch for {}", model.id);
        }
        ensure!(current(), "model request replaced");
        let dest = self.path(model);
        download_destination(&dest)?;
        fs::rename(&partial, &dest).context("install verified weights")?;
        File::open(self.dir())?.sync_all()?;
        self.remember(model, FileStamp::read(&file.metadata()?)?);
        Ok(())
    }
}

// A verified partial can replace a symlink by rename without opening its target.
fn download_destination(path: &Path) -> Result<Option<Metadata>> {
    match fs::symlink_metadata(path) {
        Ok(m) if m.is_file() => Ok(Some(m)),
        Ok(m) if m.is_symlink() => Ok(None),
        Ok(_) => anyhow::bail!("expected a regular file at {}", path.display()),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e).with_context(|| format!("read {}", path.display())),
    }
}

fn verify_hash(file: &mut File, model: &Model, stamp: &FileStamp, current: impl Fn() -> bool) -> Result<bool> {
    let mut hash = Sha256::new();
    let mut buf = vec![0; 1024 * 1024];
    loop {
        ensure!(current(), "model request replaced");
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hash.update(&buf[..n]);
    }
    ensure!(
        *stamp == FileStamp::read(&file.metadata()?)?,
        "model file changed during verification"
    );
    Ok(hex::encode(hash.finalize()) == model.sha256)
}

#[cfg(unix)]
fn check_space(missing: u64, available: u64) -> Result<()> {
    let required = missing.saturating_add(RESERVE);
    ensure!(
        available >= required,
        "insufficient free disk space (need {:.1} GiB, have {:.1} GiB)",
        gib(required),
        gib(available)
    );
    Ok(())
}

pub fn gib(bytes: u64) -> f64 {
    bytes as f64 / GIB as f64
}

fn file_size(path: &Path) -> Result<Option<u64>> {
    match fs::symlink_metadata(path) {
        Ok(m) => {
            ensure!(m.is_file(), "expected a regular file at {}", path.display());
            Ok(Some(m.len()))
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e).with_context(|| format!("read {}", path.display())),
    }
}

pub fn open_file(path: &Path, write: bool) -> Result<File> {
    let mut options = OpenOptions::new();
    options.read(true).write(write).create(write);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK).mode(0o600);
    }
    let file = options.open(path).with_context(|| format!("open {}", path.display()))?;
    let meta = file.metadata()?;
    ensure!(meta.is_file(), "expected a regular file at {}", path.display());
    #[cfg(unix)]
    if write {
        use std::os::unix::fs::MetadataExt;
        ensure!(meta.nlink() == 1, "model partial has multiple links");
    }
    Ok(file)
}

fn read_json<T: DeserializeOwned>(path: &Path, private: bool) -> Result<Option<T>> {
    if file_size(path)?.is_none() {
        return Ok(None);
    }
    let file = open_file(path, false)?;
    #[cfg(unix)]
    if private {
        use std::os::unix::fs::MetadataExt;
        let meta = file.metadata()?;
        ensure!(
            meta.mode() & 0o7777 == 0o600 && meta.nlink() == 1 && meta.uid() == rustix::process::geteuid().as_raw(),
            "untrusted model cache"
        );
    }
    #[cfg(not(unix))]
    let _ = private;
    let mut bytes = Vec::new();
    file.take(MAX_STATE_BYTES + 1).read_to_end(&mut bytes)?;
    ensure!(bytes.len() as u64 <= MAX_STATE_BYTES, "model state file is too large");
    serde_json::from_slice(&bytes)
        .map(Some)
        .with_context(|| format!("read {}", path.display()))
}

fn write_json(path: &Path, data: &impl Serialize) -> Result<()> {
    let parent = path.parent().context("model state has no parent directory")?;
    let tmp = parent.join(format!(".model-state-{:016x}.tmp", rand::random::<u64>()));
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(&tmp)?;
    let result = (|| -> Result<()> {
        serde_json::to_writer(&mut file, data)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
        fs::rename(&tmp, path)?;
        // A sync failure after rename cannot roll the selection back.
        if let Err(e) = File::open(parent).and_then(|dir| dir.sync_all()) {
            tracing::warn!(path = %path.display(), error = %e, "could not sync the model state directory");
        }
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(tmp);
    }
    result.with_context(|| format!("save {}", path.display()))
}

pub fn download_client() -> Result<reqwest::Client> {
    let roots = rustls::RootCertStore::from_iter(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    Ok(download_client_builder(roots)?.build()?)
}

pub fn download_client_builder(roots: rustls::RootCertStore) -> Result<reqwest::ClientBuilder> {
    let tls =
        rustls::ClientConfig::builder_with_provider(std::sync::Arc::new(rustls::crypto::ring::default_provider()))
            .with_safe_default_protocol_versions()?
            .with_root_certificates(roots)
            .with_no_client_auth();
    Ok(reqwest::Client::builder()
        .tls_backend_preconfigured(tls)
        .https_only(true)
        .redirect(reqwest::redirect::Policy::custom(|attempt| {
            if !allowed_url(attempt.url()) || attempt.previous().len() > 10 {
                return attempt.error("model redirect not allowed");
            }
            attempt.follow()
        }))
        .no_gzip()
        .no_brotli()
        .no_deflate()
        .no_zstd()
        .connect_timeout(Duration::from_secs(15))
        .read_timeout(Duration::from_secs(60)))
}

pub fn allowed_url(url: &reqwest::Url) -> bool {
    url.scheme() == "https"
        && url.host_str().is_some_and(|host| {
            host == "huggingface.co" || host.ends_with(".huggingface.co") || host.ends_with(".hf.co")
        })
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use serde_json::Value;

    pub fn fixture_model(id: &str, bytes: &[u8]) -> Model {
        Model {
            id: id.into(),
            quant: "Q4".into(),
            url: String::new(),
            bytes: bytes.len() as u64,
            sha256: hex::encode(Sha256::digest(bytes)),
            file: format!("{id}-Q4.gguf"),
        }
    }

    pub fn fixture_store(name: &str) -> std::sync::Arc<Store> {
        let dir = crate::stages::tests::tempdir(name);
        let store = std::sync::Arc::new(Store::new(&dir));
        store.prepare().unwrap();
        store
    }

    pub fn cached_weights(store: &Store, model: &Model) {
        File::create(store.path(model)).unwrap().set_len(model.bytes).unwrap();
        store.remember(
            model,
            FileStamp::read(&fs::metadata(store.path(model)).unwrap()).unwrap(),
        );
    }

    #[test]
    fn manifest_matches_catalog_and_installer() {
        let catalog: Value = serde_json::from_str(include_str!("../../infra/hetzner/models.json")).unwrap();
        let catalog = catalog["models"].as_array().unwrap();
        assert_eq!(MANIFEST.len(), 3);
        assert_eq!(MANIFEST.len(), catalog.len());
        for model in MANIFEST.iter() {
            let row = catalog.iter().find(|row| row["id"] == model.id).unwrap();
            assert_eq!(row["quantization"], model.quant);
            assert_eq!(row["weights_sha256"], model.sha256);
            assert_eq!(model.reference().weights_sha256.len(), 32);
            assert_eq!(model.file, format!("{}-{}.gguf", model.id, model.quant));
            let url = reqwest::Url::parse(&model.url).unwrap();
            assert!(allowed_url(&url));
            let (_, path) = url.path().split_once("/resolve/").unwrap();
            let (commit, _) = path.split_once('/').unwrap();
            assert_eq!(hex::decode(commit).unwrap().len(), 20);
        }
        let inst = include_str!("../../site/install.sh");
        assert_eq!(inst, include_str!("../../install.sh"));
        let row = inst
            .lines()
            .find_map(|line| line.strip_prefix("model_manifest='"))
            .unwrap();
        let small = find("qwen3.5-9b", None).unwrap();
        assert_eq!(
            row.trim_end_matches('\''),
            format!(
                "{} {} {} {} {}",
                small.id, small.quant, small.bytes, small.sha256, small.url
            )
        );
        assert_eq!(find("qwen3.6-35b-a3b", None).unwrap().bytes, 22_360_456_160);
        assert_eq!(find("deepseek-v4-flash", None).unwrap().bytes, 86_720_111_488);
    }

    #[cfg(unix)]
    #[test]
    fn cache_is_invalidated_by_mtime_inode_or_permissions() {
        use std::os::unix::fs::PermissionsExt;
        let store = fixture_store("verified-cache");
        let model = fixture_model("weights", b"abcdef");
        let path = store.path(&model);
        fs::write(&path, b"abcdef").unwrap();
        store.verify(&model, || true).unwrap();
        assert!(store.verify(&model, || false).is_ok());

        let changed = fs::metadata(&path).unwrap().modified().unwrap() - Duration::from_secs(2);
        File::open(&path).unwrap().set_modified(changed).unwrap();
        assert!(store.verify(&model, || false).is_err());
        store.verify(&model, || true).unwrap();

        let replacement = path.with_extension("replacement");
        fs::write(&replacement, b"xyzxyz").unwrap();
        File::open(&replacement).unwrap().set_modified(changed).unwrap();
        fs::rename(&replacement, &path).unwrap();
        assert!(store
            .verify(&model, || true)
            .unwrap_err()
            .to_string()
            .contains("SHA-256 mismatch"));

        fs::write(&path, b"abcdef").unwrap();
        store.verify(&model, || true).unwrap();
        let cache = store.dir().join("verified.json");
        fs::set_permissions(&cache, fs::Permissions::from_mode(0o640)).unwrap();
        assert!(store.verify(&model, || false).is_err());
        store.verify(&model, || true).unwrap();
        assert_eq!(fs::metadata(&cache).unwrap().permissions().mode() & 0o777, 0o600);
        fs::remove_dir_all(&store.data_dir).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn active_model_requires_manifest_and_verified_weights() {
        let store = fixture_store("active-model");
        let model = find("qwen3.6-35b-a3b", None).unwrap();
        let active = store.data_dir.join("active-model.json");
        assert!(store.active().unwrap().is_none());

        fs::write(&active, b"{ broken").unwrap();
        assert!(store.active().is_err());
        fs::write(&active, br#"{"id":"unknown","quant":"q"}"#).unwrap();
        assert!(store.active().is_err());

        cached_weights(&store, model);
        store.save_active(model, Some(42)).unwrap();
        let (restored, selection) = store.active().unwrap().unwrap();
        assert_eq!(restored.id, model.id);
        assert_eq!(selection.use_revision, Some(42));
        File::options()
            .write(true)
            .open(store.path(model))
            .unwrap()
            .set_len(40)
            .unwrap();
        assert!(store.active().unwrap_err().to_string().contains("incorrect model size"));
        fs::remove_dir_all(&store.data_dir).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn weight_paths_refuse_symlinks_and_partial_hard_links() {
        use std::os::unix::fs::symlink;
        let store = fixture_store("model-links");
        let model = fixture_model("weights", b"weights");
        let target = store.data_dir.join("target");
        fs::write(&target, b"weights").unwrap();
        symlink(&target, store.path(&model)).unwrap();
        assert!(store.installed(&model).is_err());
        assert!(store.verify(&model, || true).is_err());
        fs::hard_link(&target, store.partial(&model)).unwrap();
        assert!(open_file(&store.partial(&model), true).is_err());
        fs::remove_dir_all(&store.data_dir).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn second_use_observes_a_final_file_when_promotion_removed_the_partial() {
        let store = fixture_store("model-promotion-race");
        let model = fixture_model("weights", b"weights");
        assert!(!store.download_candidate(&model).unwrap());
        fs::write(store.partial(&model), b"weights").unwrap();
        store.promote(&model, || true).unwrap();
        let partial = file_size(&store.partial(&model));
        assert_eq!(partial.as_ref().unwrap(), &None);
        assert_eq!(store.download_size_after_partial(&model, partial).unwrap(), None);
        assert_eq!(store.download_size(&model).unwrap(), None);
    }

    #[cfg(unix)]
    #[test]
    fn disk_reserve_is_added_to_the_missing_bytes() {
        assert!(check_space(GIB, GIB + RESERVE - 1)
            .unwrap_err()
            .to_string()
            .contains("insufficient free disk space"));
        assert!(check_space(GIB, GIB + RESERVE).is_ok());
    }
}
