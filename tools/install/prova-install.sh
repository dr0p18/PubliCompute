#!/bin/sh
# Exercise install and uninstall with Podman. The container runs systemd as PID 1.
set -eu

if [ "${1:-}" != --inside ]; then
    repo=$(cd "$(dirname "$0")/../.." && pwd)
    build=$(mktemp -d)
    container=
    cleanup() {
        if [ -n "$container" ]; then podman rm -f "$container" >/dev/null 2>&1 || :; fi
        rm -rf -- "$build"
    }
    trap cleanup 0
    trap 'exit 130' HUP INT TERM
    cat > "$build/Containerfile" <<'IMAGE'
FROM ubuntu:24.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates curl openssh-client procps python3 sudo systemd util-linux zstd && rm -rf /var/lib/apt/lists/*
CMD ["/lib/systemd/systemd"]
IMAGE
    podman build --tag publicompute-install-test "$build"
    container=$(podman run --detach --systemd=always --cap-add SYS_ADMIN --security-opt label=disable \
        --volume "$repo:/src:ro" publicompute-install-test)
    podman exec "$container" timeout 60 systemctl is-system-running --wait || :
    if ! podman exec "$container" sh /src/tools/install/prova-install.sh --inside; then
        # Capture the service journal before the container is removed so failures remain diagnosable.
        podman exec "$container" journalctl -u publicompute-agent.service --no-pager -n 40 || :
        exit 1
    fi
    exit
fi

[ "$(cat /proc/1/comm)" = systemd ]
work=$(mktemp -d)
http_pid=
bound=
cleanup() {
    if [ -n "$bound" ]; then umount "$bound" || :; fi
    if [ -n "$http_pid" ]; then kill "$http_pid" 2>/dev/null || :; fi
    rm -rf -- "$work"
}
trap cleanup 0
trap 'exit 130' HUP INT TERM

# Probe credential delivery in the container. If the unit cannot read its
# credential, install a fixture-only override that uses the runtime directory.
if ! systemd-run --wait --collect -q -p LoadCredential=probe:/etc/hostname \
        /bin/sh -c 'test -s "$CREDENTIALS_DIRECTORY/probe"' 2>/dev/null; then
    echo "LoadCredential unavailable in this container: using the fixture drop-in"
    mkdir -p /etc/systemd/system/publicompute-agent.service.d
    cat > /etc/systemd/system/publicompute-agent.service.d/fixture-credential.conf <<'UNIT'
[Service]
LoadCredential=
Environment=CREDENTIALS_DIRECTORY=/run/publicompute-agent/credential
ExecStartPre=+/bin/sh -c 'install -d -m 0700 -o publicompute-agent -g publicompute-agent /run/publicompute-agent/credential && install -m 0400 -o publicompute-agent -g publicompute-agent /etc/publicompute-agent/owner-token /run/publicompute-agent/credential/owner-token'
UNIT
fi
version=0.1.4
older_version=0.1.3
name=publicompute-linux-x86_64-$version
older_name=publicompute-linux-x86_64-$older_version
package=$work/$name
mkdir -p "$package/bin" "$package/share" "$work/http"
signing_key=$work/signer/.config/publicompute/firma/publicompute-release
mkdir -p "$(dirname "$signing_key")"
ssh-keygen -q -t ed25519 -N '' -f "$signing_key"
public_key=$(awk '{ print $1, $2 }' "$signing_key.pub")
installer=$work/http/install.sh
sed "s|^release_pubkey=.*|release_pubkey='$public_key'|" /src/install.sh > "$installer"
grep -Fxq "release_pubkey='$public_key'" "$installer"
cp /src/tools/install/publicompute-agent.service /src/tools/install/uninstall.sh "$package/share/"

# Stand-ins let the fixture exercise package installation without external services or GPU resources.
cat > "$package/bin/publicompute-agent" <<'PY'
#!/usr/bin/python3
import argparse
import ctypes
import json
import os
from pathlib import Path
import signal
import socket
import sys

if sys.argv[1] == "nat-check":
    print("blocked")
    sys.exit(0)
if sys.argv[1] == "coordinator-check":
    assert Path(sys.argv[sys.argv.index("--ca") + 1]).is_file()
    print("Coordinator fixture reachable")
    sys.exit(0)
if sys.argv[1] == "swap-install":
    staging, installed = map(Path, sys.argv[2:])
    if installed.exists():
        libc = ctypes.CDLL(None, use_errno=True)
        rc = libc.renameat2(-100, os.fsencode(staging), -100, os.fsencode(installed), 2)
        if rc:
            raise OSError(ctypes.get_errno(), "renameat2")
    else:
        staging.rename(installed)
    sys.exit(0)

if sys.argv[1] == "--data-dir":
    _, _, data_dir, command, choice = sys.argv
    if command == "backup-preference":
        assert choice in ("contributo.json", "active-model.json")
        try:
            fd = os.open(Path(data_dir, choice), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except FileNotFoundError:
            sys.exit(3)
        with os.fdopen(fd, "rb") as source:
            sys.stdout.buffer.write(source.read())
        sys.exit(0)
    assert command == "install-preference"
    path = Path(data_dir, "contributo.json")
    revision = json.loads(path.read_text())["revisione"] + 1 if path.exists() else 1
    id_quant = {
        "medio": ("qwen3.6-35b-a3b", "UD-Q4_K_XL"),
        "deepseek": ("deepseek-v4-flash", "IQ2XXS-w2Q2K-0731"),
        "free": None,
    }[choice]
    model = dict(zip(("id", "quant"), id_quant)) if id_quant else None
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"modo": "uso" if model else "libera", "modello": model, "revisione": revision}))
    tmp.replace(path)
    sys.exit(0)

parser = argparse.ArgumentParser()
for arg in ("engine", "stage-bin", "coordinator", "server-name", "ca", "data-dir", "control-socket", "owner-token-file"):
    parser.add_argument("--" + arg, required=True)
args = parser.parse_args()
assert args.engine == "stages"
assert args.stage_bin == "/opt/publicompute-agent/bin/publicompute-stage"
assert os.environ["PUBLICOMPUTE_RELAY"] == "https://relay.publicompute.com:8443"

token = Path(args.owner_token_file).read_text().strip()
assert token and token not in sys.argv and token not in os.environ.values()
identity = Path(args.data_dir, "identity.bin")
if not identity.exists():
    identity.write_text("fixture identity " + str(os.getpid()))
    Path(args.data_dir, "key.pem").write_text("fixture key " + str(os.getpid()))
if token == "fixture-rejected-code":
    sys.exit(1)

def stop(_signum, _frame):
    print("fixture received SIGTERM", flush=True)
    sys.exit(0)

signal.signal(signal.SIGTERM, stop)
with socket.socket(socket.AF_UNIX) as listener:
    mask = os.umask(0o117)
    listener.bind(args.control_socket)
    os.umask(mask)
    listener.listen()
    while True:
        stream, _ = listener.accept()
        with stream, stream.makefile("rwb") as io:
            assert json.loads(io.readline()) == {"op": "status"}
            state = {"engine": "parked", "connected": token != "fixture-wait-code", "gpu_lost": False}
            io.write(json.dumps({"ok": True, "status": state}, separators=(",", ":")).encode() + b"\n")
PY
cat > "$package/bin/publicompute" <<'PY'
#!/usr/bin/python3
import json
import os
import socket
import sys

args = sys.argv[1:]
if args[0] == "uninstall":
    script = "/opt/publicompute-agent/uninstall.sh"
    os.execvp("sudo", ["sudo", "--", script, *args[1:]])
path = "/run/publicompute-agent/agent.sock"
if args[0] == "--socket":
    path, args = args[1], args[2:]
assert args[0] == "status"
with socket.socket(socket.AF_UNIX) as stream:
    stream.connect(path)
    stream.sendall(b'{"op":"status"}\n')
    status = json.load(stream.makefile("rb"))["status"]
print(json.dumps(status, separators=(",", ":")) if "--json" in args else "fixture agent ready")
PY
printf '#!/bin/sh\nexit 1\n' > "$package/bin/llama-server"
printf '#!/bin/sh\nexit 1\n' > "$package/bin/publicompute-stage"
chmod 0755 "$package/bin/"*
printf 'original\n' > "$package/RELEASE"
printf '%s\n' "$version" > "$package/VERSION"
tar --zstd -cf "$work/http/$name.tar.zst" -C "$work" "$name"
(cd "$work/http" && sha256sum "$name.tar.zst" > SHA256SUMS)
HOME=$work/signer /src/tools/pacchetto/firma.sh "$work/http/SHA256SUMS"
sha=$(cut -d ' ' -f 1 "$work/http/SHA256SUMS")
mkdir "$work/http/older"
cp -a "$package" "$work/$older_name"
printf '%s\n' "$older_version" > "$work/$older_name/VERSION"
tar --zstd -cf "$work/http/older/$older_name.tar.zst" -C "$work" "$older_name"
(cd "$work/http/older" && sha256sum "$older_name.tar.zst" > SHA256SUMS)
HOME=$work/signer /src/tools/pacchetto/firma.sh "$work/http/older/SHA256SUMS"
printf 'candidate\n' > "$package/RELEASE"
printf '\n# Candidate fixture unit\n' >> "$package/share/publicompute-agent.service"
tar --zstd -cf "$work/rejected.tar.zst" -C "$work" "$name"
rejected_sha=$(sha256sum "$work/rejected.tar.zst" | cut -d ' ' -f 1)
mkdir "$work/http/replacement"
cp "$work/rejected.tar.zst" "$work/http/replacement/$name.tar.zst"
(cd "$work/http/replacement" && sha256sum "$name.tar.zst" > SHA256SUMS)
HOME=$work/signer /src/tools/pacchetto/firma.sh "$work/http/replacement/SHA256SUMS"
url=http://127.0.0.1:18773
printf 'fixture CA\n' > "$work/ca.pem"
python3 -m http.server 18773 --bind 127.0.0.1 --directory "$work/http" > "$work/http.log" 2>&1 &
http_pid=$!
timeout 10 sh -c 'until curl -fsS "$1/SHA256SUMS" >/dev/null 2>&1; do sleep 1; done' sh "$url"

useradd --create-home fixture
printf 'fixture ALL=(root) NOPASSWD: ALL\n' > /etc/sudoers.d/publicompute-fixture
chmod 0440 /etc/sudoers.d/publicompute-fixture
PUBLICOMPUTE_SKIP_GPU_CHECK=1
PUBLICOMPUTE_TEST_CPUINFO=$work/cpuinfo
printf 'processor\t: 0\nflags\t\t: ssse3 sse4_1 sse4_2 popcnt cx16 avx avx2 fma f16c bmi1 bmi2 abm movbe\n' > "$PUBLICOMPUTE_TEST_CPUINFO"
PUBLICOMPUTE_CA_FILE=$work/ca.pem
PUBLICOMPUTE_BASE_URL=$url
PUBLICOMPUTE_VERSION=$version
PUBLICOMPUTE_LINK_CODE=fixture-first-code
unset PUBLICOMPUTE_MODEL
export PUBLICOMPUTE_SKIP_GPU_CHECK PUBLICOMPUTE_TEST_CPUINFO PUBLICOMPUTE_CA_FILE \
    PUBLICOMPUTE_BASE_URL PUBLICOMPUTE_VERSION PUBLICOMPUTE_LINK_CODE
expect_failure() {
    message=$1
    shift
    if "$@" > "$work/failure.log" 2>&1; then
        printf 'Expected failure: %s\n' "$message" >&2
        exit 1
    fi
    grep -F "$message" "$work/failure.log" || { cat "$work/failure.log" >&2; exit 1; }
}
expect_failure 'HTTP packages are only allowed on loopback for tests.' \
    env PUBLICOMPUTE_TEST_INSECURE=0 sh /src/install.sh
expect_failure 'HTTP packages are only allowed on loopback for tests.' \
    env PUBLICOMPUTE_TEST_INSECURE=1 PUBLICOMPUTE_BASE_URL=http://example.invalid sh /src/install.sh
PUBLICOMPUTE_TEST_INSECURE=1
export PUBLICOMPUTE_TEST_INSECURE
prefix=/opt/publicompute-agent
staging=$prefix.staging
state=/var/lib/publicompute-agent
conf=/etc/publicompute-agent
unit_file=/etc/systemd/system/publicompute-agent.service
lock=/run/publicompute-agent-lock/lock
legacy_lock=/run/lock/publicompute-agent.lock
chmod 1777 /run/lock
runuser -u fixture -- touch "$legacy_lock"

printf 'processor\t: 0\nflags\t\t: ssse3 sse4_1 sse4_2 popcnt cx16 avx fma f16c bmi1 bmi2 abm movbe\n' > "$work/cpuinfo-no-avx2"
requests=$(wc -l < "$work/http.log")
expect_failure 'This CPU lacks avx2. PubliCompute needs a CPU with AVX2' \
    env PUBLICOMPUTE_TEST_CPUINFO="$work/cpuinfo-no-avx2" sh /src/install.sh
if [ "$(wc -l < "$work/http.log")" -ne "$requests" ]; then
    printf '%s\n' 'Unexpected HTTP request from a CPU without AVX2.' >&2
    exit 1
fi
# A virtual CPU can mask a single feature of the baseline.
printf 'processor\t: 0\nflags\t\t: ssse3 sse4_1 sse4_2 popcnt cx16 avx avx2 fma f16c bmi1 bmi2 abm\n' > "$work/cpuinfo-no-movbe"
expect_failure 'This CPU lacks movbe.' env PUBLICOMPUTE_TEST_CPUINFO="$work/cpuinfo-no-movbe" sh /src/install.sh
expect_failure 'PUBLICOMPUTE_PACKAGE requires PUBLICOMPUTE_PACKAGE_SHA256.' \
    env PUBLICOMPUTE_PACKAGE="$work/http/$name.tar.zst" PUBLICOMPUTE_PACKAGE_SHA256= sh /src/install.sh

mkdir "$work/bin"
cat > "$work/bin/nvidia-smi" <<'SMI'
#!/bin/sh
[ "$LC_ALL" = C ] || exit 1
printf '%s\n' "$1" >> "$FIXTURE_SMI_LOG"
# Print only the requested columns, so a missing fallback query cannot pass.
case "$1" in
    --query-gpu=memory.total,uuid,driver_version,compute_cap)
        [ "${FIXTURE_NVIDIA_OLD_CLI:-0}" = 0 ] || exit 1
        printf '%s\n' "$FIXTURE_GPU_CSV" ;;
    --query-gpu=memory.total,uuid,driver_version)
        printf '%s\n' "$FIXTURE_GPU_CSV" | cut -d, -f1-3 ;;
    *) exit 1 ;;
esac
SMI
chmod 0755 "$work/bin/nvidia-smi"
(
    PATH=$work/bin:$PATH
    PUBLICOMPUTE_SKIP_GPU_CHECK=0
    FIXTURE_SMI_LOG=$work/nvidia-smi.log
    export PATH PUBLICOMPUTE_SKIP_GPU_CHECK FIXTURE_SMI_LOG
    unset PUBLICOMPUTE_PACKAGE PUBLICOMPUTE_PACKAGE_SHA256
    expect_gpu_install_failure() {
        gpu_message=$1 gpu_csv=$2
        shift 2
        requests=$(wc -l < "$work/http.log")
        expect_failure "$gpu_message" env FIXTURE_GPU_CSV="$gpu_csv" "$@" sh /src/install.sh
        if [ "$(wc -l < "$work/http.log")" -ne "$requests" ]; then
            printf 'Unexpected HTTP request with GPU data %s\n' "$gpu_csv" >&2
            exit 1
        fi
    }
    expect_gpu_install_failure 'NVIDIA driver 525.147.05 is too old for this GPU. PubliCompute needs 535 or newer.' \
        '8192, GPU-aaaa, 580.12, 6.1
16384, GPU-bbbb, 525.147.05, 8.9'
    expect_gpu_install_failure 'NVIDIA driver 565.57.01 is too old for this GPU. PubliCompute needs 570 or newer.' \
        '16384, GPU-aaaa, 565.57.01, 12.0'
    grep -F 'sudo ubuntu-drivers install' "$work/failure.log"
    grep -F 'update the Windows NVIDIA driver' "$work/failure.log"
    expect_gpu_install_failure 'Unsupported NVIDIA GPU (compute capability 6.1).' '8192, GPU-aaaa, 580.12, 6.1'
    expect_gpu_install_failure 'NVIDIA GPU data unavailable. Update the NVIDIA driver.' '8192, GPU-aaaa, 535.183.01, [N/A]'
    : > "$FIXTURE_SMI_LOG"
    expect_gpu_install_failure 'NVIDIA GPU data unavailable. Update the NVIDIA driver.' \
        '8192, GPU-aaaa, 535.183.01, 8.9' FIXTURE_NVIDIA_OLD_CLI=1
    printf '%s\n' --query-gpu=memory.total,uuid,driver_version,compute_cap \
        --query-gpu=memory.total,uuid,driver_version | cmp -s - "$FIXTURE_SMI_LOG" || {
        printf '%s\n' 'nvidia-smi fallback query missing or out of order' >&2
        exit 1
    }
    expect_gpu_install_failure 'NVIDIA GPU data unavailable. Update the NVIDIA driver.' '[N/A], GPU-aaaa, 580.12, 8.9'

    expect_gpu_install_failure 'PUBLICOMPUTE_PACKAGE requires PUBLICOMPUTE_PACKAGE_SHA256.' \
        '8192, GPU-aaaa, 535.183.01, 7.5' PUBLICOMPUTE_PACKAGE="$work/http/$name.tar.zst"
    grep -Fx 'Using NVIDIA GPU GPU-aaaa.' "$work/failure.log"
    expect_gpu_install_failure 'PUBLICOMPUTE_PACKAGE requires PUBLICOMPUTE_PACKAGE_SHA256.' \
        '16384, GPU-aaaa, 525.147.05, 8.9
8192, GPU-bbbb, 570, 12.0' PUBLICOMPUTE_PACKAGE="$work/http/$name.tar.zst"
    grep -Fx 'Using NVIDIA GPU GPU-bbbb.' "$work/failure.log"
)

expect_rejected_release() {
    expect_failure "$1" sh "$installer"
    if grep -F 'Coordinator fixture reachable' "$work/failure.log"; then
        printf '%s\n' 'The agent ran from a rejected release.' >&2
        exit 1
    fi
    for path in "$prefix" "$staging" "$state" "$conf" "$unit_file" /usr/local/bin/publicompute; do
        test ! -e "$path" && test ! -L "$path" || { printf 'Leftover path: %s\n' "$path" >&2; exit 1; }
    done
    if getent passwd publicompute-agent || getent group publicompute-agent; then
        printf '%s\n' 'The rejected release left an account or group.' >&2
        exit 1
    fi
}
mv "$work/http/SHA256SUMS.sig" "$work/SHA256SUMS.sig"
expect_rejected_release 'Cannot download SHA256SUMS.sig.'
mv "$work/SHA256SUMS.sig" "$work/http/SHA256SUMS.sig"
cp "$work/http/SHA256SUMS" "$work/SHA256SUMS"
printf '\n' >> "$work/http/SHA256SUMS"
expect_rejected_release 'Invalid SHA256SUMS signature.'
mv "$work/SHA256SUMS" "$work/http/SHA256SUMS"
if grep -F "GET /$name.tar.zst" "$work/http.log"; then
    printf '%s\n' 'The archive was fetched without a verified signature.' >&2
    exit 1
fi
mv "$work/http/$name.tar.zst" "$work/original.tar.zst"
cp "$work/rejected.tar.zst" "$work/http/$name.tar.zst"
expect_rejected_release 'Package SHA-256 mismatch.'
mv "$work/original.tar.zst" "$work/http/$name.tar.zst"
expect_failure 'Invalid region.' sh "$installer" --region 'EU/IT'
expect_failure 'PUBLICOMPUTE_MODEL must be medio, deepseek or free.' env PUBLICOMPUTE_MODEL=invalid sh "$installer"
runuser -u fixture -- sh -c 'curl -fsSL "$PUBLICOMPUTE_BASE_URL/install.sh" | PUBLICOMPUTE_REGION=EU-DE sh -s -- --region EU-IT'
grep -Fx 'PUBLICOMPUTE_REGION=EU-IT' "$conf/agent.conf"
grep -Fx 'PUBLICOMPUTE_RELAY=https://relay.publicompute.com:8443' "$conf/agent.conf"
test "$(stat -c %u "$legacy_lock")" = "$(id -u fixture)"
runuser -u fixture -- rm -- "$legacy_lock"
model=$state/models/qwen3.5-9b-Q4_K_M.gguf
preference=$state/contributo.json
expect_preference() {
    python3 - "$preference" "$@" <<'PYFIX'
import json
from pathlib import Path
import sys

pref = json.loads(Path(sys.argv[1]).read_text())
assert pref["modo"] == sys.argv[2], pref
assert pref["modello"] == ({"id": sys.argv[3], "quant": sys.argv[4]} if len(sys.argv) == 5 else None), pref
assert pref["revisione"] > 0
PYFIX
}
test -z "$(ls -A "$state/models")"
expect_preference libera
# Retained weights must survive subsequent updates and uninstall --keep-models.
printf 'legacy weights\n' > "$model"
chown publicompute-agent:publicompute-agent "$model"
socket=/run/publicompute-agent/agent.sock
test "$(stat -c %a "$socket")" = 660
test "$(stat -c %a "$(dirname "$socket")")" = 750
test "$(stat -c %a "$state")" = 700
test "$(stat -c %u:%g:%a "$conf/owner-token")" = 0:0:600
test "$(stat -c %u:%g:%a "$(dirname "$lock")")" = 0:0:700
test "$(stat -c %u:%g:%a "$lock")" = 0:0:600
lock_inode=$(stat -c %i "$lock")
runuser -u fixture -- publicompute status

expect_failure "Public release v$version has a different SHA-256" env \
    PUBLICOMPUTE_BASE_URL="$url/replacement" sh "$installer"
test "$(cat "$prefix/RELEASE")" = original

identity=$(cat "$state/identity.bin")
model_inode=$(stat -c %i "$model")
expect_failure "Downgrade from $version to $older_version requires --allow-downgrade." env \
    PUBLICOMPUTE_VERSION="$older_version" PUBLICOMPUTE_BASE_URL="$url/older" sh "$installer"
test "$(cat "$prefix/VERSION")" = "$version"
PUBLICOMPUTE_VERSION=$older_version PUBLICOMPUTE_BASE_URL=$url/older sh "$installer" --allow-downgrade
test "$(cat "$prefix/VERSION")" = "$older_version"
sh "$installer"
test "$(cat "$prefix/VERSION")" = "$version"
test "$(cat "$state/identity.bin")" = "$identity"
test "$(stat -c %i "$model")" = "$model_inode"

PUBLICOMPUTE_LINK_CODE=ignored-second-code
export PUBLICOMPUTE_LINK_CODE
mkdir "$staging"
touch "$staging/stale"
test "$(stat -c %u:%g:%a "$legacy_lock")" = 0:0:600
exec 8<"$legacy_lock"
flock -x 8
PUBLICOMPUTE_PACKAGE=$work/http/$name.tar.zst PUBLICOMPUTE_PACKAGE_SHA256=$sha \
    sh /src/install.sh > "$work/local-install.log" 2>&1 8<&- &
local_pid=$!
timeout 10 sh -c 'while flock -n "$1" true; do sleep 0.1; done' sh "$lock"
test -f "$staging/stale"
flock -u 8
exec 8<&-
wait "$local_pid" || { cat "$work/local-install.log" >&2; exit 1; }
test ! -e "$staging"
test "$(cat "$conf/owner-token")" = fixture-first-code
test "$(cat "$state/identity.bin")" = "$identity"
test "$(stat -c %i "$model")" = "$model_inode"

key=$(cat "$state/key.pem")
# Simulate an earlier package layout with a different agent basename. The
# candidate must still be able to restore the package and its original unit.
systemctl stop publicompute-agent.service
mv "$prefix/bin/publicompute-agent" "$prefix/bin/previous-agent"
sed -i 's|/bin/publicompute-agent |/bin/previous-agent |' "$unit_file"
systemctl daemon-reload
systemctl start publicompute-agent.service
timeout 10 sh -c 'until publicompute status >/dev/null 2>&1; do sleep 1; done'
sha256sum "$conf/agent.conf" "$conf/ca.pem" "$unit_file" "$preference" > "$work/before-upgrade"
printf 'candidate CA\n' > "$work/candidate-ca.pem"
expect_failure 'Restoring the previous PubliCompute installation.' env \
    PUBLICOMPUTE_PACKAGE="$work/rejected.tar.zst" PUBLICOMPUTE_PACKAGE_SHA256="$rejected_sha" \
    PUBLICOMPUTE_COORDINATOR=localhost:7777 PUBLICOMPUTE_CA_FILE="$work/candidate-ca.pem" \
    PUBLICOMPUTE_LINK_CODE=fixture-rejected-code PUBLICOMPUTE_MODEL=medio sh /src/install.sh --relink
test "$(cat "$prefix/RELEASE")" = original
test -x "$prefix/bin/previous-agent"
test ! -e "$prefix/bin/publicompute-agent"
sha256sum -c "$work/before-upgrade"
test "$(cat "$conf/owner-token")" = fixture-first-code
test "$(cat "$state/identity.bin")" = "$identity"
test "$(cat "$state/key.pem")" = "$key"
test ! -e "$staging"
systemctl is-active --quiet publicompute-agent.service
timeout 10 sh -c 'until publicompute status >/dev/null 2>&1; do sleep 1; done'

PUBLICOMPUTE_LINK_CODE=fixture-wait-code sh "$installer" --relink > "$work/interrupted.log" 2>&1 &
installer_pid=$!
if ! timeout 20 sh -c 'until grep -q "Waiting for the coordinator session" "$1"; do sleep 1; done' sh "$work/interrupted.log"; then
    kill -TERM "$installer_pid" 2>/dev/null || :
    wait "$installer_pid" || :
    cat "$work/interrupted.log" >&2
    exit 1
fi
kill -TERM "$installer_pid"
if wait "$installer_pid"; then printf '%s\n' 'Expected the interrupted install to fail.' >&2; exit 1; fi
grep -F 'Restoring the previous PubliCompute installation.' "$work/interrupted.log"
test "$(cat "$conf/owner-token")" = fixture-first-code
test "$(cat "$state/identity.bin")" = "$identity"
test "$(cat "$state/key.pem")" = "$key"
test ! -e "$staging"
systemctl is-active --quiet publicompute-agent.service

mkdir "$work/source" "$state/models/mounted"
touch "$work/source/keep"
if mount --bind "$work/source" "$state/models/mounted"; then
    bound=$state/models/mounted
    expect_failure 'Mounted path in' sh "$installer"
    expect_failure 'Mounted path in' publicompute uninstall --remove-models
    test -f "$work/source/keep"
    systemctl is-active --quiet publicompute-agent.service
    umount "$bound"
    bound=
else
    printf '%s\n' 'SKIP bind mount test: the container runtime denied the mount.' >&2
fi
rmdir "$state/models/mounted"

PUBLICOMPUTE_LINK_CODE=fixture-relinked-code sh "$installer" --relink
test "$(cat "$conf/owner-token")" = fixture-relinked-code
test "$(cat "$state/identity.bin")" != "$identity"
test "$(stat -c %i "$model")" = "$model_inode"
identity=$(cat "$state/identity.bin")
PUBLICOMPUTE_LINK_CODE=fixture-relinked-code sh "$installer" --relink
test "$(cat "$state/identity.bin")" != "$identity"

# Recreate the model settings and preference left by a 0.1.3 installation.
printf '%s\n' "$older_version" > "$prefix/VERSION"
sed -i -e '/^PUBLICOMPUTE_REGION=/d' -e '/^PUBLICOMPUTE_RELAY=/d' "$conf/agent.conf"
printf 'MODEL_PATH=%s\nMODEL_ID=qwen3.5-9b\nQUANTIZATION=Q4_K_M\n' "$model" >> "$conf/agent.conf"
printf '%s\n' '{"modo":"uso","modello":{"id":"qwen3.5-9b","quant":"Q4_K_M"},"revisione":900}' > "$preference"
partial=$state/models/qwen3.6-35b-a3b-UD-Q4_K_XL.gguf.agent.part
printf 'resumable service download\n' > "$partial"
chown publicompute-agent:publicompute-agent "$partial"
sha256sum "$model" "$partial" > "$work/model-state-before"
for step in migration double-quoted single-quoted; do
    expected_region=EU
    set --
    case "$step" in
        double-quoted)
            sed -i 's/^PUBLICOMPUTE_REGION=.*/PUBLICOMPUTE_REGION="EU-IT"/' "$conf/agent.conf"
            expected_region=EU-IT
            set -- --region EU-FR
            ;;
        single-quoted)
            sed -i "s/^PUBLICOMPUTE_REGION=.*/PUBLICOMPUTE_REGION='EU-DE'/" "$conf/agent.conf"
            expected_region=EU-DE
            ;;
    esac
    sh "$installer" "$@"
    expect_preference libera
    test -z "$(sed -n '/^MODEL_/p; /^QUANTIZATION=/p' "$conf/agent.conf")"
    grep -Fx "PUBLICOMPUTE_REGION=$expected_region" "$conf/agent.conf"
    grep -Fx 'PUBLICOMPUTE_RELAY=https://relay.publicompute.com:8443' "$conf/agent.conf"
done
test "$(cat "$prefix/VERSION")" = "$version"
test "$(stat -c %i "$model")" = "$model_inode"
PUBLICOMPUTE_MODEL=medio sh "$installer"
expect_preference uso qwen3.6-35b-a3b UD-Q4_K_XL
PUBLICOMPUTE_MODEL=deepseek sh "$installer"
expect_preference uso deepseek-v4-flash IQ2XXS-w2Q2K-0731
PUBLICOMPUTE_MODEL=free sh "$installer"
expect_preference libera
sha256sum -c "$work/model-state-before"
touch "$state/active-model.json"
touch "$model.part" "$state/models/verified.json"
runuser -u fixture -- publicompute uninstall --keep-models
test ! -e "$state/active-model.json"
test ! -e "$partial"
test ! -e "$preference"
test ! -e "$model.part"
test ! -e "$state/models/verified.json"
test -f "$model"
test "$(stat -c %u "$model")" = 0
test ! -e "$state/identity.bin"
PUBLICOMPUTE_LINK_CODE=fixture-reinstall-code sh "$installer"
test "$(stat -c %i "$model")" = "$model_inode"
# Verify that the CLI can remove an installation when no agent or coordinator is available.
uid=$(id -u publicompute-agent)
systemctl stop publicompute-agent.service
test ! -S "$socket"
mkdir "$staging"
touch "$staging/stale"
runuser -u fixture -- publicompute uninstall --remove-models
test "$(stat -c %i "$lock")" = "$lock_inode"
journalctl -u publicompute-agent.service --no-pager | awk '/fixture received SIGTERM/ { found=1 } END { exit !found }'
for path in "$state" "$conf" "$prefix" "$staging" /run/publicompute-agent \
    /etc/systemd/system/publicompute-agent.service /usr/local/bin/publicompute; do
    test ! -e "$path" && test ! -L "$path" || { printf 'Leftover path: %s\n' "$path" >&2; exit 1; }
done
if getent passwd publicompute-agent || getent group publicompute-agent || pgrep -u "$uid"; then
    printf '%s\n' 'Leftover account or process.' >&2
    exit 1
fi
printf '%s\n' 'Install, upgrade, relink and uninstall fixture passed.'
