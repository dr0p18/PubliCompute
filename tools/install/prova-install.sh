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
    ca-certificates curl procps python3 sudo systemd util-linux zstd && rm -rf /var/lib/apt/lists/*
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
name=publicompute-linux-x86_64-fixture
package=$work/$name
mkdir -p "$package/bin" "$package/share" "$work/http"
cp /src/install.sh "$work/http/"
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

parser = argparse.ArgumentParser()
for arg in ("engine", "llama-mode", "llama-bin", "model-path", "model-id", "quantization",
            "coordinator", "server-name", "ca", "data-dir", "control-socket", "owner-token-file"):
    parser.add_argument("--" + arg, required=True)
args = parser.parse_args()
assert (args.engine, args.llama_mode, args.model_id, args.quantization) == ("llama", "managed", "qwen3.5-9b", "Q4_K_M")
assert Path(args.model_path).is_file()
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
            state = {"engine": "ready", "connected": token != "fixture-wait-code", "gpu_lost": False}
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
tar --zstd -cf "$work/http/$name.tar.zst" -C "$work" "$name"
(cd "$work/http" && sha256sum "$name.tar.zst" > SHA256SUMS)
sha=$(cut -d ' ' -f 1 "$work/http/SHA256SUMS")
printf 'candidate\n' > "$package/RELEASE"
printf '\n# Candidate fixture unit\n' >> "$package/share/publicompute-agent.service"
tar --zstd -cf "$work/rejected.tar.zst" -C "$work" "$name"
rejected_sha=$(sha256sum "$work/rejected.tar.zst" | cut -d ' ' -f 1)
dd if=/dev/zero of="$work/http/model.gguf" bs=1024 count=4 status=none
model_sha=$(sha256sum "$work/http/model.gguf" | cut -d ' ' -f 1)
url=http://127.0.0.1:18773
printf 'qwen3.5-9b Q4_K_M 4096 %s %s/model.gguf\n' "$model_sha" "$url" > "$work/model.manifest"
printf 'fixture CA\n' > "$work/ca.pem"
python3 -m http.server 18773 --bind 127.0.0.1 --directory "$work/http" > "$work/http.log" 2>&1 &
http_pid=$!
timeout 10 sh -c 'until curl -fsS "$1/SHA256SUMS" >/dev/null 2>&1; do sleep 1; done' sh "$url"

useradd --create-home fixture
printf 'fixture ALL=(root) NOPASSWD: ALL\n' > /etc/sudoers.d/publicompute-fixture
chmod 0440 /etc/sudoers.d/publicompute-fixture
PUBLICOMPUTE_SKIP_GPU_CHECK=1
PUBLICOMPUTE_TEST_MODEL_MANIFEST=$work/model.manifest
PUBLICOMPUTE_CA_FILE=$work/ca.pem
PUBLICOMPUTE_BASE_URL=$url
PUBLICOMPUTE_VERSION=fixture
PUBLICOMPUTE_LINK_CODE=fixture-first-code
export PUBLICOMPUTE_SKIP_GPU_CHECK PUBLICOMPUTE_TEST_MODEL_MANIFEST PUBLICOMPUTE_CA_FILE PUBLICOMPUTE_BASE_URL \
    PUBLICOMPUTE_VERSION PUBLICOMPUTE_LINK_CODE
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
runuser -u fixture -- sh -c 'curl -fsSL "$PUBLICOMPUTE_BASE_URL/install.sh" | sh'

prefix=/opt/publicompute-agent
staging=$prefix.staging
state=/var/lib/publicompute-agent
conf=/etc/publicompute-agent
unit_file=/etc/systemd/system/publicompute-agent.service
model=$state/models/qwen3.5-9b-Q4_K_M.gguf
socket=/run/publicompute-agent/agent.sock
test "$(stat -c %a "$socket")" = 660
test "$(stat -c %a "$(dirname "$socket")")" = 750
test "$(stat -c %a "$state")" = 700
test "$(stat -c %u:%g:%a "$conf/owner-token")" = 0:0:600
test "$(stat -c %u:%g:%a /run/lock/publicompute-agent.lock)" = 0:0:600
runuser -u fixture -- publicompute status

identity=$(cat "$state/identity.bin")
model_inode=$(stat -c %i "$model")
PUBLICOMPUTE_LINK_CODE=ignored-second-code
export PUBLICOMPUTE_LINK_CODE
mkdir "$staging"
touch "$staging/stale"
PUBLICOMPUTE_PACKAGE=$work/http/$name.tar.zst PUBLICOMPUTE_PACKAGE_SHA256=$sha sh /src/install.sh
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
sha256sum "$conf/agent.conf" "$conf/ca.pem" "$unit_file" > "$work/before-upgrade"
printf 'candidate CA\n' > "$work/candidate-ca.pem"
expect_failure 'Restoring the previous PubliCompute installation.' env \
    PUBLICOMPUTE_PACKAGE="$work/rejected.tar.zst" PUBLICOMPUTE_PACKAGE_SHA256="$rejected_sha" \
    PUBLICOMPUTE_COORDINATOR=localhost:7777 PUBLICOMPUTE_CA_FILE="$work/candidate-ca.pem" \
    PUBLICOMPUTE_LINK_CODE=fixture-rejected-code sh /src/install.sh --relink
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

PUBLICOMPUTE_LINK_CODE=fixture-wait-code sh /src/install.sh --relink > "$work/interrupted.log" 2>&1 &
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
    expect_failure 'Mounted path in' sh /src/install.sh
    expect_failure 'Mounted path in' publicompute uninstall --remove-models
    test -f "$work/source/keep"
    systemctl is-active --quiet publicompute-agent.service
    umount "$bound"
    bound=
else
    printf '%s\n' 'SKIP bind mount test: the container runtime denied the mount.' >&2
fi
rmdir "$state/models/mounted"

PUBLICOMPUTE_LINK_CODE=fixture-relinked-code sh /src/install.sh --relink
test "$(cat "$conf/owner-token")" = fixture-relinked-code
test "$(cat "$state/identity.bin")" != "$identity"
test "$(stat -c %i "$model")" = "$model_inode"
identity=$(cat "$state/identity.bin")
PUBLICOMPUTE_LINK_CODE=fixture-relinked-code sh /src/install.sh --relink
test "$(cat "$state/identity.bin")" != "$identity"

runuser -u fixture -- publicompute uninstall --keep-models
test -f "$model"
test "$(stat -c %u "$model")" = 0
test ! -e "$state/identity.bin"
PUBLICOMPUTE_LINK_CODE=fixture-reinstall-code sh /src/install.sh
test "$(stat -c %i "$model")" = "$model_inode"
# Verify that the CLI can remove an installation when no agent or coordinator is available.
uid=$(id -u publicompute-agent)
systemctl stop publicompute-agent.service
test ! -S "$socket"
mkdir "$staging"
touch "$staging/stale"
runuser -u fixture -- publicompute uninstall --remove-models
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
