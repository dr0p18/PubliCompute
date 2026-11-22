use std::{
    env,
    fs::File,
    io::{self, Read},
    net::IpAddr,
    os::fd::OwnedFd,
    path::{Component, Path, PathBuf},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use rustix::{
    fs::{self, AtFlags, FileType, FlockOperation, Mode, OFlags, CWD},
    io::Errno,
    process::geteuid,
};
use serde::{Deserialize, Serialize};
use ureq::http::{HeaderMap, Uri};

pub(super) const COOKIE: &str = "pc_session";
pub(super) const EXPIRED: &str = "session expired, run publicompute login";
const FILE: &str = "session";
const MAX_FILE: u64 = 8192;

#[derive(PartialEq, Eq)]
pub(super) struct Endpoint(String);

impl Endpoint {
    pub(super) fn parse(value: &str) -> Result<Self, String> {
        let bad = || "API URL must be an http or https origin (scheme, host and optional port)".to_owned();
        let uri: Uri = value.parse().map_err(|_| bad())?;
        let scheme = uri.scheme_str().ok_or_else(bad)?;
        let default_port = match scheme {
            "https" => 443,
            "http" => 80,
            _ => return Err(bad()),
        };
        let authority = uri.authority().ok_or_else(bad)?;
        if uri.path() != "/" || uri.query().is_some() || value.contains('#') || authority.as_str().contains('@') {
            return Err(bad());
        }
        let raw_host = authority.host();
        let mut host = raw_host.to_ascii_lowercase();
        let port = match authority.as_str().strip_prefix(raw_host) {
            Some("") => default_port,
            Some(_) => authority.port_u16().filter(|&port| port != 0).ok_or_else(bad)?,
            None => return Err(bad()),
        };
        let ip = host.trim_start_matches('[').trim_end_matches(']').parse::<IpAddr>();
        if let Ok(ip) = ip {
            host = match ip {
                IpAddr::V4(_) => ip.to_string(),
                IpAddr::V6(_) => format!("[{ip}]"),
            };
        } else if host.is_empty()
            || !host
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'.' | b'-'))
        {
            return Err(bad());
        }
        if scheme == "http" && host != "localhost" && !ip.is_ok_and(|ip| ip.is_loopback()) {
            return Err("HTTP is only allowed on loopback".into());
        }
        let suffix = if port == default_port {
            String::new()
        } else {
            format!(":{port}")
        };
        Ok(Self(format!("{scheme}://{host}{suffix}")))
    }

    pub(super) fn url(&self, path: &str) -> String {
        format!("{}{path}", self.0)
    }
}

#[derive(Serialize, Deserialize)]
pub(super) struct Session {
    token: String,
    api: String,
    expires_at: u64,
}

impl Session {
    pub(super) fn from_headers(headers: &HeaderMap, api: &Endpoint) -> Result<Self, String> {
        let now = SystemTime::now();
        for header in headers.get_all(ureq::http::header::SET_COOKIE) {
            if let Some((token, expiry)) = header.to_str().ok().and_then(|s| parse_cookie(s, now)) {
                return Ok(Self {
                    token: token.into(),
                    api: api.0.clone(),
                    expires_at: unix_time(expiry),
                });
            }
        }
        Err("login succeeded but the API did not return a usable session cookie".into())
    }

    pub(super) fn endpoint(&self) -> Result<Endpoint, String> {
        Endpoint::parse(&self.api)
    }

    pub(super) fn check(&self, api: &Endpoint) -> Result<(), String> {
        self.check_expiry()?;
        if self.endpoint()? != *api {
            return Err("session belongs to a different API, run publicompute login for this API".into());
        }
        Ok(())
    }

    pub(super) fn check_expiry(&self) -> Result<(), String> {
        if self.expires_at <= unix_time(SystemTime::now()) {
            return Err(EXPIRED.into());
        }
        Ok(())
    }

    pub(super) fn cookie(&self) -> String {
        format!("{COOKIE}={}", self.token)
    }

    pub(super) fn token(&self) -> &str {
        &self.token
    }

    pub(super) fn redact(&self, text: &str) -> String {
        text.replace(&self.token, "[redacted]")
    }
}

fn unix_time(time: SystemTime) -> u64 {
    time.duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
}

fn valid_token(token: &str) -> bool {
    !token.is_empty()
        && token
            .bytes()
            .all(|c| matches!(c, 0x21 | 0x23..=0x2b | 0x2d..=0x3a | 0x3c..=0x5b | 0x5d..=0x7e))
}

fn parse_cookie(header: &str, now: SystemTime) -> Option<(&str, SystemTime)> {
    let mut parts = header.split(';');
    let (name, value) = parts.next()?.trim().split_once('=')?;
    let token = value
        .strip_prefix('"')
        .and_then(|s| s.strip_suffix('"'))
        .unwrap_or(value);
    if name != COOKIE || !valid_token(token) {
        return None;
    }
    let (mut max_age, mut expires) = (None, None);
    for part in parts {
        if let Some((name, value)) = part.trim().split_once('=') {
            if name.eq_ignore_ascii_case("Max-Age") {
                max_age = value.trim().parse::<i64>().ok();
            } else if name.eq_ignore_ascii_case("Expires") {
                expires = httpdate::parse_http_date(value.trim()).ok();
            }
        }
    }
    let expiry = match max_age {
        Some(age) if age <= 0 => return None,
        Some(age) => now.checked_add(Duration::from_secs(age as u64))?,
        None => expires?,
    };
    (expiry > now).then_some((token, expiry))
}

pub(super) fn directory() -> Result<PathBuf, String> {
    env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .or_else(|| {
            env::var_os("HOME")
                .map(|home| PathBuf::from(home).join(".config"))
                .filter(|p| p.is_absolute())
        })
        .map(|base| base.join("publicompute"))
        .ok_or_else(|| "set XDG_CONFIG_HOME or HOME to locate the session file".into())
}

pub(super) struct SessionDir(OwnedFd);

impl SessionDir {
    pub(super) fn open(path: &Path, create: bool) -> io::Result<Self> {
        let flags = OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC;
        let mut dir = fs::openat(CWD, "/", flags, Mode::empty())?;
        // Anchor each lookup to the directory we opened, including the config path.
        for component in path.components() {
            let name = match component {
                Component::RootDir => continue,
                Component::Normal(name) => name,
                _ => return Err(invalid("invalid session directory path")),
            };
            dir = match fs::openat(&dir, name, flags, Mode::empty()) {
                Err(Errno::NOENT) if create => {
                    match fs::mkdirat(&dir, name, Mode::RWXU) {
                        Ok(()) | Err(Errno::EXIST) => {}
                        Err(e) => return Err(e.into()),
                    }
                    fs::openat(&dir, name, flags, Mode::empty())?
                }
                result => result?,
            };
        }
        let st = fs::fstat(&dir)?;
        if st.st_uid != geteuid().as_raw() {
            return Err(invalid("session directory is owned by another user"));
        }
        if create && st.st_mode & 0o7777 != 0o700 {
            fs::fchmod(&dir, Mode::RWXU)?;
        }
        Ok(Self(dir))
    }

    pub(super) fn load(&self) -> io::Result<Session> {
        if fs::fstat(&self.0)?.st_mode & 0o7777 != 0o700 {
            return Err(invalid("session directory must have permissions 0700"));
        }
        let fd = fs::openat(
            &self.0,
            FILE,
            OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::CLOEXEC | OFlags::NONBLOCK,
            Mode::empty(),
        )?;
        let st = fs::fstat(&fd)?;
        if !FileType::from_raw_mode(st.st_mode).is_file()
            || st.st_mode & 0o7777 != 0o600
            || st.st_uid != geteuid().as_raw()
            || st.st_nlink > 1
        {
            return Err(invalid("session must be a private regular file with permissions 0600"));
        }
        let mut bytes = Vec::new();
        File::from(fd).take(MAX_FILE + 1).read_to_end(&mut bytes)?;
        if bytes.len() as u64 <= MAX_FILE {
            if let Ok(session) = serde_json::from_slice::<Session>(&bytes) {
                if valid_token(&session.token) {
                    return Ok(session);
                }
            }
        }
        Err(invalid("invalid session file"))
    }

    pub(super) fn save(self, session: &Session) -> io::Result<()> {
        fs::flock(&self.0, FlockOperation::LockExclusive)?;
        match fs::statat(&self.0, FILE, AtFlags::SYMLINK_NOFOLLOW) {
            Ok(st) if !FileType::from_raw_mode(st.st_mode).is_file() => {
                return Err(invalid("refusing to replace a non-regular session file"));
            }
            Ok(_) | Err(Errno::NOENT) => {}
            Err(e) => return Err(e.into()),
        }
        let random = getrandom::u64().map_err(|_| io::Error::other("cannot create a temporary session file"))?;
        let temp = format!(".session-{random:016x}");
        let fd = fs::openat(
            &self.0,
            temp.as_str(),
            OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL | OFlags::CLOEXEC,
            Mode::RUSR | Mode::WUSR,
        )?;
        let mut file = File::from(fd);
        let result = (|| {
            fs::fchmod(&file, Mode::RUSR | Mode::WUSR)?;
            serde_json::to_writer(&mut file, session)?;
            file.sync_all()?;
            fs::renameat(&self.0, temp.as_str(), &self.0, FILE)?;
            Ok(())
        })();
        if result.is_err() {
            let _ = fs::unlinkat(&self.0, temp.as_str(), AtFlags::empty());
        }
        result
    }

    pub(super) fn take(self) -> io::Result<io::Result<Session>> {
        // Remove the local credential before the network call so a concurrent login can save a new one.
        fs::flock(&self.0, FlockOperation::LockExclusive)?;
        let session = self.load();
        match fs::unlinkat(&self.0, FILE, AtFlags::empty()) {
            Ok(()) | Err(Errno::NOENT) => Ok(session),
            Err(e) => Err(e.into()),
        }
    }
}

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::{symlink, MetadataExt, PermissionsExt};

    const API: &str = "https://api.example.test:8444";

    fn session(token: &str) -> Session {
        Session {
            token: token.into(),
            api: API.into(),
            expires_at: unix_time(SystemTime::now()) + 60,
        }
    }

    #[test]
    fn saves_private_session_and_replaces_it_atomically() {
        let temp = tempfile::tempdir().unwrap();
        let dir = temp.path().join("publicompute");
        let path = dir.join(FILE);
        SessionDir::open(&dir, true).unwrap().save(&session("old")).unwrap();
        assert_eq!(dir.metadata().unwrap().permissions().mode() & 0o7777, 0o700);
        assert_eq!(path.metadata().unwrap().permissions().mode() & 0o7777, 0o600);

        let old_file = File::open(&path).unwrap();
        SessionDir::open(&dir, true).unwrap().save(&session("new")).unwrap();
        assert_ne!(old_file.metadata().unwrap().ino(), path.metadata().unwrap().ino());
        assert!(serde_json::from_reader::<_, Session>(old_file).unwrap().token == "old");
        assert!(SessionDir::open(&dir, false).unwrap().load().unwrap().token == "new");
        assert_eq!(dir.read_dir().unwrap().count(), 1);
    }

    #[test]
    fn refuses_session_and_directory_symlinks() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("publicompute");
        let target = temp.path().join("target");
        std::fs::write(&target, "keep me").unwrap();
        let dir = SessionDir::open(&path, true).unwrap();
        symlink(&target, path.join(FILE)).unwrap();
        assert!(dir.load().is_err());
        assert!(dir.save(&session("new")).is_err());
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "keep me");
        let link = temp.path().join("link");
        symlink(&path, &link).unwrap();
        assert!(SessionDir::open(&link, true).is_err());
        assert!(SessionDir::open(&link.join("nested"), true).is_err());
    }

    #[test]
    fn checks_expiry_and_binds_to_scheme_host_and_port() {
        let mut session = session("session");
        for url in [
            "http://localhost:8444",
            "https://other.example.test:8444",
            "https://api.example.test",
        ] {
            assert!(session.check(&Endpoint::parse(url).unwrap()).is_err(), "{url}");
        }
        assert!(session
            .check(&Endpoint::parse("https://API.EXAMPLE.TEST:8444/").unwrap())
            .is_ok());
        session.api = "http://localhost:8444".into();
        assert!(session
            .check(&Endpoint::parse("https://localhost:8444").unwrap())
            .is_err());
        assert!(session
            .check(&Endpoint::parse("http://127.0.0.1:8444").unwrap())
            .is_err());
        session.expires_at = 1;
        assert_eq!(
            session
                .check(&Endpoint::parse("http://localhost:8444").unwrap())
                .unwrap_err(),
            EXPIRED
        );
    }

    #[test]
    fn validates_api_origins() {
        for url in [
            "http://localhost",
            "http://127.0.0.2:3000",
            "http://[::1]",
            "https://api.example.test:443/",
        ] {
            assert!(Endpoint::parse(url).is_ok(), "{url}");
        }
        for url in [
            "http://localhost.example.test",
            "http://127.1",
            "http://[::ffff:127.0.0.1]",
            "http://127.0.0.1:99999",
            "https://user@api.example.test",
            "https://api.example.test/path",
            "https://api.example.test/?query",
            "https://api.example.test/#fragment",
            "file:///tmp/api",
        ] {
            assert!(Endpoint::parse(url).is_err(), "{url}");
        }
        assert!(
            Endpoint::parse("https://api.example.test:443/").unwrap()
                == Endpoint::parse("https://api.example.test").unwrap()
        );
    }

    #[test]
    fn selects_session_cookie_and_prefers_max_age() {
        let now = UNIX_EPOCH + Duration::from_secs(1_000_000);
        let expiry = now + Duration::from_secs(10);
        for header in [
            "pc_session=abc; Max-Age=10; Path=/; Secure; HttpOnly",
            "pc_session=\"abc\"; expires=Thu, 01 Jan 1970 00:00:00 GMT; MAX-AGE=10",
        ] {
            assert_eq!(parse_cookie(header, now), Some(("abc", expiry)));
        }
        for header in [
            "other=abc; Max-Age=10",
            "pc_session=abc; Max-Age=0",
            "pc_session=abc; Max-Age=-1",
            "pc_session=; Max-Age=10",
            "pc_session=two words; Max-Age=10",
            "pc_session=abc; Path=/",
        ] {
            assert!(parse_cookie(header, now).is_none(), "{header}");
        }
        let header = format!("pc_session=abc; Expires={}", httpdate::fmt_http_date(expiry));
        assert_eq!(parse_cookie(&header, now), Some(("abc", expiry)));

        let mut headers = HeaderMap::new();
        headers.append("Set-Cookie", "other=first; Max-Age=60".parse().unwrap());
        headers.append("Set-Cookie", "pc_session=second; Max-Age=60".parse().unwrap());
        let session = Session::from_headers(&headers, &Endpoint::parse(API).unwrap()).unwrap();
        assert!(session.cookie() == "pc_session=second");
    }
}
