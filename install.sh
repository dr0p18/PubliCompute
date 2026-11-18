#!/bin/sh
# Install the Linux package, configure the agent service and enroll a node.
# The privileged phase stages and verifies a candidate before replacing the
# active installation. Use PUBLICOMPUTE_CA_FILE for a private coordinator.
# The loopback HTTP, manifest and GPU overrides are for installer tests.
set +x
set -eu

# Keep the link code out of the environment inherited by child processes.
unset link_code
link_code=${PUBLICOMPUTE_LINK_CODE:-}
unset PUBLICOMPUTE_LINK_CODE

fail() { printf '%s\n' "$*" >&2; exit 1; }

relink=0
case "$#:${1:-}" in
    0:) ;;
    1:--relink) relink=1 ;;
    *) fail 'Usage: install.sh [--relink]' ;;
esac

started=$(date +%s)
[ "$(uname -s)" = Linux ] && [ "$(uname -m)" = x86_64 ] || fail 'PubliCompute requires Linux x86_64.'
command -v systemctl >/dev/null && [ -d /run/systemd/system ] || fail 'A running systemd is required.'
for tool in zstd sha256sum flock runuser timeout pgrep findmnt; do
    command -v "$tool" >/dev/null || fail "$tool is required."
done
if command -v curl >/dev/null; then
    downloader=curl
elif command -v wget >/dev/null; then
    downloader=wget
else
    fail 'Install curl or wget first.'
fi

as_root() {
    if [ "$(id -u)" -eq 0 ]; then
        exec "$@"
    else
        exec sudo -- "$@"
    fi
}
if [ "$(id -u)" -ne 0 ]; then
    command -v sudo >/dev/null || fail 'Run as root or install sudo.'
    sudo -v || fail 'Administrator privileges are required.'
fi

# Read interactive input from the terminal because stdin may carry the installer.
link_file=
root_pid=
trap '[ -z "$link_file" ] || rm -f -- "$link_file"' 0
interrupt() {
    trap '' HUP INT TERM
    if [ -n "$root_pid" ]; then
        kill -TERM "$root_pid" 2>/dev/null || :
        wait "$root_pid" || :
    fi
    exit 130
}
trap interrupt HUP INT TERM
if [ -n "$link_code" ]; then
    link_file=$(umask 077; mktemp)
    printf '%s\n' "$link_code" > "$link_file"
    unset link_code
fi

version=${PUBLICOMPUTE_VERSION:-0.1.1}
case "$version" in
    ''|*[!A-Za-z0-9._-]*) fail 'Invalid package version.' ;;
esac
package=${PUBLICOMPUTE_PACKAGE:-}
case "$package" in
    ''|/*) ;;
    *) package=$PWD/$package ;;
esac
ca=${PUBLICOMPUTE_CA_FILE:-}
case "$ca" in
    ''|/*) ;;
    *) ca=$PWD/$ca ;;
esac
# Allow the service fixture to supply its own unit and credential root.
skip_gpu=${PUBLICOMPUTE_SKIP_GPU_CHECK:-0}
test_manifest=${PUBLICOMPUTE_TEST_MODEL_MANIFEST:-}
if [ -n "$test_manifest" ]; then
    [ "$skip_gpu" = 1 ] || fail 'The test model requires PUBLICOMPUTE_SKIP_GPU_CHECK=1.'
    case "$test_manifest" in /*) ;; *) test_manifest=$PWD/$test_manifest ;; esac
fi

base_url=${PUBLICOMPUTE_BASE_URL:-https://github.com/dr0p18/PubliCompute/releases/download/v$version}
insecure=${PUBLICOMPUTE_TEST_INSECURE:-0}
http_fixture=0
case "$base_url" in
    https://*) ;;
    http://*)
        authority=${base_url#http://}
        authority=${authority%%/*}
        host=${authority%%:*}
        case "$host" in localhost|127.0.0.1) ;; *) fail 'HTTP packages are only allowed on loopback for tests.' ;; esac
        if [ "$authority" != "$host" ]; then
            case "${authority#*:}" in ''|*[!0-9]*) fail 'Invalid loopback package port.' ;; esac
        fi
        [ "$insecure" = 1 ] || fail 'HTTP packages are only allowed on loopback for tests.'
        http_fixture=1 ;;
    *) fail 'The package base URL must use HTTPS.' ;;
esac

# Keep the privileged phase inline so piped installation works without a script file.
as_root sh -s -- "$started" "$downloader" "$version" "$base_url" \
    "$package" "${PUBLICOMPUTE_PACKAGE_SHA256:-}" "${PUBLICOMPUTE_COORDINATOR:-api.publicompute.com:7443}" \
    "${PUBLICOMPUTE_SERVER_NAME:-}" "$ca" "${PUBLICOMPUTE_SPEED_URL:-}" "$skip_gpu" "$test_manifest" \
    "$link_file" "$relink" "${SUDO_USER:-$(id -un)}" "$http_fixture" <<'ROOT' &
set -eu
umask 022

fail() { printf '%s\n' "$*" >&2; exit 1; }

started=$1 downloader=$2 version=$3 base_url=$4 package=$5 package_sha=$6 coordinator=$7 server_name=$8 ca=$9
shift 9
speed_url=$1 skip_gpu=$2 test_manifest=$3 link_file=$4 relink=$5 installer=$6 http_fixture=$7

account=publicompute-agent
prefix=/opt/$account
scratch=$prefix.staging
stage=$scratch/release
previous=$scratch/previous
state=/var/lib/$account
conf=/etc/$account
socket=/run/$account/agent.sock
unit=$account.service
unit_file=/etc/systemd/system/$unit
link=/usr/local/bin/publicompute
name=publicompute-linux-x86_64-$version

require_unmounted() {
    mounts=$(findmnt --kernel --raw -n -o TARGET) || fail 'Cannot read the mount table.'
    for path do
        while IFS= read -r mount; do
            case "$mount" in "$path"|"$path"/*) fail "Mounted path in $path: $mount." ;; esac
        done <<MOUNTS
$mounts
MOUNTS
    done
}

root_directory() {
    if [ -e "$1" ] || [ -L "$1" ]; then
        [ ! -L "$1" ] && [ -d "$1" ] && [ "$(stat -c %u "$1")" = 0 ] ||
            fail "Expected a root-owned directory at $1."
        mode=$(stat -c %a "$1")
        [ "$((0$mode & 022))" -eq 0 ] || fail "Directory writable by other users: $1."
    fi
}

# Keep a stable lock inode across install and uninstall so all invocations contend on the same lock.
lock=/run/lock/publicompute-agent.lock
if [ ! -e "$lock" ]; then (umask 077; set -C; : > "$lock") 2>/dev/null || :; fi
[ ! -L "$lock" ] && [ -f "$lock" ] && [ "$(stat -c %u "$lock")" = 0 ] || fail 'Invalid installer lock file.'
exec 9>>"$lock"
fd=/proc/$$/fd/9
[ -f "$fd" ] && [ "$(stat -Lc %u:%h "$fd")" = 0:1 ] && [ ! -L "$lock" ] &&
    [ "$(stat -Lc %d:%i "$fd")" = "$(stat -c %d:%i "$lock")" ] || fail 'Invalid opened installer lock file.'
chmod 0600 "$fd"
flock -x -w 120 9 || fail 'Timed out after 120 s waiting for the PubliCompute installer lock.'

root_directory "$scratch"
root_directory "$prefix"
root_directory "$conf"
[ ! -L "$state" ] && { [ ! -e "$state" ] || [ -d "$state" ]; } || fail "Invalid state directory: $state."
require_unmounted "$scratch" "$prefix" "$state" "$conf"

restore_file() {
    if [ -f "$previous/$1" ]; then
        cp -p --remove-destination -- "$previous/$1" "$2"
    else
        rm -f -- "$2"
    fi
}

rollback() {
    printf '%s\n' 'Restoring the previous PubliCompute installation.' >&2
    systemctl stop "$unit" || { [ ! -f "$unit_file" ] && [ ! -f "$previous/unit" ]; } || return 1
    if [ -f "$previous/enabling" ] && [ ! -f "$previous/enabled" ]; then
        systemctl disable "$unit" || return 1
    fi
    old_inode=$(cat "$previous/package") || return 1
    [ -z "$old_inode" ] || [ -d "$prefix" ] || return 1
    if [ -e "$prefix" ] && [ "$(stat -c %d:%i "$prefix")" != "$old_inode" ]; then
        if [ -n "$old_inode" ]; then
            [ "$(stat -c %d:%i "$stage")" = "$old_inode" ] || return 1
            # The verified candidate still occupies the active directory.
            # Use its swap helper without depending on the staged package layout.
            "$prefix/bin/publicompute-agent" swap-install "$stage" "$prefix" || return 1
        else
            mv -T -- "$prefix" "$stage" || return 1
        fi
    fi
    restore_file agent.conf "$conf/agent.conf" && restore_file ca.pem "$conf/ca.pem" &&
        restore_file owner-token "$conf/owner-token" && restore_file unit "$unit_file" || return 1
    if [ -f "$previous/identity-saved" ]; then
        restore_file identity.bin "$state/identity.bin" && restore_file key.pem "$state/key.pem" || return 1
    fi
    if [ -f "$previous/new-link" ] && [ "$(readlink "$link" 2>/dev/null || :)" = "$prefix/bin/publicompute" ]; then
        rm -f -- "$link" || return 1
    fi
    systemctl daemon-reload || return 1
    # Reset the start limit so rollback can restart the restored service.
    systemctl reset-failed "$unit" 2>/dev/null || :
    if [ -f "$previous/running" ]; then systemctl start "$unit" || return 1; fi
    # Leave staging for the next installer to clear once rollback has restored service.
    mv -T -- "$previous" "$scratch/restored"
}

if [ -d "$previous" ]; then rollback || fail "Rollback failed. Recovery files remain in $scratch."; fi
rm -rf -- "$scratch"
install -d -m 0700 "$scratch"
cleanup() {
    result=$?
    trap - 0
    trap '' HUP INT TERM
    if [ -n "${stty_restore:-}" ]; then
        stty "$stty_restore" < /dev/tty 2>/dev/null || :
        printf '\n' > /dev/tty 2>/dev/null || :
    fi
    require_unmounted "$scratch" "$prefix" "$state" "$conf"
    if [ -d "$previous" ]; then rollback || fail "Rollback failed. Recovery files remain in $scratch."; fi
    rm -rf -- "$scratch"
    exit "$result"
}
trap cleanup 0
trap 'exit 130' HUP INT TERM

case "$coordinator" in
    ''|*[!][A-Za-z0-9.:_-]*) fail 'Invalid coordinator address.' ;;
esac
# Validate the address before writing it to the unit environment file. Allow IPv6 colons and brackets.
case "$coordinator" in *:*) ;; *) fail 'The coordinator address must include a UDP port.' ;; esac
if [ -z "$server_name" ]; then
    server_name=${coordinator%:*}
    server_name=${server_name#\[}
    server_name=${server_name%\]}
fi
case "$server_name" in
    ''|*[!A-Za-z0-9.:_-]*) fail 'Invalid coordinator TLS name.' ;;
esac

gpu_uuid=
if [ "$skip_gpu" = 1 ]; then
    printf '%s\n' 'Skipping the NVIDIA check for the install fixture.'
else
    PATH=$PATH:/usr/lib/wsl/lib
    export PATH
    command -v nvidia-smi >/dev/null || fail 'nvidia-smi is missing. Install the NVIDIA driver.'
    cards=$(LC_ALL=C timeout 10 nvidia-smi --query-gpu=memory.total,uuid --format=csv,noheader,nounits) ||
        fail 'nvidia-smi failed or timed out. Check the NVIDIA driver.'
    gpu_uuid=$(printf '%s\n' "$cards" | awk -F, '$1 ~ /^[[:space:]]*[0-9]+[[:space:]]*$/ && $1 >= 8192 {
        gsub(/[[:space:]]/, "", $2); if ($2 ~ /^GPU-[a-fA-F0-9-]+$/) {print $2; exit}
    }')
    [ -n "$gpu_uuid" ] || fail 'At least one NVIDIA GPU with 8 GiB of VRAM is required.'
fi

model_manifest='qwen3.5-9b Q4_K_M 5680522464 03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8 https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/3885219b6810b007914f3a7950a8d1b469d598a5/Qwen3.5-9B-Q4_K_M.gguf'
if [ -n "$test_manifest" ]; then
    model_manifest=$(cat -- "$test_manifest")
fi
IFS=' ' read -r model_id quant model_size model_sha model_url extra <<MANIFEST
$model_manifest
MANIFEST
[ -z "$extra" ] || fail 'Invalid model manifest.'
case "$model_id:$quant" in *[!A-Za-z0-9._:-]*|:*|*:) fail 'Invalid model identifier or quantization.' ;; esac
case "$model_size" in ''|*[!0-9]*) fail 'Invalid model size.' ;; esac
case "$model_url" in http://*|https://*) ;; *) fail 'Invalid model URL.' ;; esac
valid_hash() {
    [ "${#1}" -eq 64 ] || return 1
    case "$1" in *[!0123456789abcdefABCDEF]*) return 1 ;; esac
}
valid_hash "$model_sha" || fail 'Invalid model SHA-256.'
model_sha=$(printf '%s' "$model_sha" | tr 'A-F' 'a-f')
if [ -n "$package" ]; then
    valid_hash "$package_sha" || fail 'PUBLICOMPUTE_PACKAGE requires PUBLICOMPUTE_PACKAGE_SHA256.'
fi

if [ -e "$link" ] || [ -L "$link" ]; then
    [ "$(readlink "$link")" = "$prefix/bin/publicompute" ] ||
        fail "$link belongs to another installation."
fi

install -d -m 0755 -o root -g root "$conf"
if entry=$(getent passwd "$account"); then
    sys_uid_max=$(awk '$1 == "UID_MIN" { min=$2 } $1 == "SYS_UID_MAX" { max=$2 }
        END { print max ? max : (min ? min - 1 : 999) }' /etc/login.defs)
    printf '%s\n' "$entry" | awk -F: -v home="$state" -v max="$sys_uid_max" '
        $3 ~ /^[0-9]+$/ && $3 > 0 && $3 <= max && $6 == home &&
        $7 ~ /^(\/usr)?\/s?bin\/nologin$/ { ok=1 }
        END { exit !ok }' || fail "Existing $account must be a system account with home $state and a nologin shell."
fi
if ! getent group "$account" >/dev/null; then
    groupadd --system "$account"
    getent group "$account" | cut -d: -f3 > "$conf/group.gid"
fi
if [ -z "$entry" ]; then
    useradd --system --gid "$account" --home-dir "$state" --no-create-home --shell /usr/sbin/nologin "$account"
    id -u "$account" > "$conf/account.uid"
fi
install -d -m 0700 -o "$account" -g "$account" "$state"
if [ -d "$state/models" ] && [ ! -L "$state/models" ] && [ "$(stat -c %u "$state/models")" = 0 ]; then
    # Retained weights belong to root after uninstall so a recycled service UID cannot read them.
    require_unmounted "$state/models"
    chown -hR "$account:$account" "$state/models"
fi
as_agent() { runuser -u "$account" -- "$@"; }
as_agent mkdir -p "$state/models"
model=$state/models/$model_id-$quant.gguf
part=$model.part

hash_matches() {
    expected=$(printf '%s' "$1" | tr 'A-F' 'a-f')
    printf '%s  %s\n' "$expected" "$2" | sha256sum --check --status -
}

model_ok() {
    as_agent test ! -L "$1" && as_agent test -f "$1" && as_agent test "$(as_agent stat -c %s "$1")" = "$model_size" &&
        printf '%s  %s\n' "$model_sha" "$1" | as_agent sha256sum --check --status -
}

# Reserve space for the compressed package, expanded files and temporary downloads.
package_kb=$((2 * 1024 * 1024))
model_kb=$((512 * 1024))
need_model=1
if model_ok "$model"; then
    need_model=0
elif ! model_ok "$part"; then
    partial=$(as_agent stat -c %s "$part" 2>/dev/null || printf 0)
    [ "$partial" -lt "$model_size" ] || partial=0
    model_kb=$((model_kb + (model_size - partial + 1023) / 1024))
fi
check_space() {
    available=$(df -Pk "$1" | awk 'NR == 2 { print $4 }')
    [ "${available:-0}" -ge "$2" ] || fail "Insufficient free space on $1 (need $2 KiB)."
}
if [ "$(stat -c %d /opt)" = "$(stat -c %d "$state")" ]; then
    check_space /opt "$((package_kb + model_kb))"
else
    check_space /opt "$package_kb"
    check_space "$state" "$model_kb"
fi

download_seconds=0
fetch() {
    url=$1
    output=$2
    if [ "$http_fixture" = 1 ] && [ "${url#http://}" != "$url" ]; then
        # Restrict the HTTP fixture exception to the validated loopback URL.
        case "$downloader" in
            curl) curl -q -fsS --proto =http --noproxy '*' --connect-timeout 15 --retry 3 --output "$output" -- "$url" ;;
            wget) wget --no-config --no-proxy --max-redirect=0 -q --timeout=30 --tries=3 -O "$output" -- "$url" ;;
        esac
        return
    fi
    case "$downloader" in
        curl) curl -q -fLsS --proto =https --proto-redir =https --connect-timeout 15 --retry 3 --output "$output" -- "$url" ;;
        wget)
            # Validate each redirect before following it. Wget's HTTPS-only option
            # applies to recursive traversal.
            redirects=0
            while [ "$redirects" -le 10 ]; do
                case "$url" in https://*) ;; *) return 1 ;; esac
                if LC_ALL=C wget --no-config --max-redirect=0 --server-response --timeout=30 --tries=3 \
                    -O "$output" -- "$url" > /dev/null 2> "$scratch/wget.response"; then
                    return 0
                fi
                next=$(awk '
                    $1 ~ /^HTTP\// { status=$2; location="" }
                    tolower($1) == "location:" { location=$2 }
                    END { if (status ~ /^(301|302|303|307|308)$/) print location }
                ' "$scratch/wget.response")
                case "$next" in
                    https://*) url=$next ;;
                    //*) url=https:$next ;;
                    /*) origin=${url#https://}; url=https://${origin%%/*}$next ;;
                    *) return 1 ;;
                esac
                redirects=$((redirects + 1))
            done
            return 1 ;;
    esac
}

printf '%s\n' 'Preparing the Linux package.'
if [ -n "$package" ]; then
    cp -- "$package" "$scratch/package.tar.zst"
else
    begin=$(date +%s)
    fetch "${base_url%/}/SHA256SUMS" "$scratch/SHA256SUMS" || fail 'Cannot download SHA256SUMS.'
    package_sha=$(awk -v name="$name.tar.zst" '$2 == name || $2 == "*" name { print $1 }' "$scratch/SHA256SUMS")
    valid_hash "$package_sha" || fail "No unique SHA-256 for $name.tar.zst in SHA256SUMS."
    fetch "${base_url%/}/$name.tar.zst" "$scratch/package.tar.zst" || fail 'Cannot download the Linux package.'
    download_seconds=$((download_seconds + $(date +%s) - begin))
fi
hash_matches "$package_sha" "$scratch/package.tar.zst" || fail 'Package SHA-256 mismatch.'
tar --no-same-owner --no-same-permissions --zstd -xf "$scratch/package.tar.zst" -C "$scratch" ||
    fail 'Cannot extract the Linux package.'
mv -T -- "$scratch/$name" "$stage"
root_directory "$stage"
require_unmounted "$stage"
chmod -R go-w "$stage"
for binary in publicompute-agent publicompute-stage publicompute llama-server; do
    [ -x "$stage/bin/$binary" ] || fail "Package is missing bin/$binary."
done
[ -f "$stage/share/$unit" ] || fail "Package is missing share/$unit."
install -m 0755 "$stage/share/uninstall.sh" "$stage/uninstall.sh"

if [ -n "$ca" ]; then
    install -m 0644 -- "$ca" "$scratch/ca.pem"
else
    begin=$(date +%s)
    fetch https://publicompute.com/coordinator-ca.pem "$scratch/ca.pem" ||
        fail 'Cannot download the PubliCompute QUIC CA. Set PUBLICOMPUTE_CA_FILE for a private coordinator.'
    download_seconds=$((download_seconds + $(date +%s) - begin))
fi
"$stage/bin/publicompute-agent" coordinator-check --coordinator "$coordinator" --server-name "$server_name" --ca "$scratch/ca.pem" ||
    fail "Cannot reach the coordinator at $coordinator over QUIC."
nat=$("$stage/bin/publicompute-agent" nat-check)
printf 'NAT type: %s\n' "$nat"
if [ -n "$speed_url" ]; then
    if [ "$downloader" = curl ]; then
        if speed=$(curl -fLsS --max-time 20 -o /dev/null -w '%{speed_download}' -- "$speed_url"); then
            printf 'Download speed: %s B/s\n' "$speed"
        else
            printf '%s\n' 'Warning: optional bandwidth check failed.' >&2
        fi
    else
        speed_start=$(date +%s)
        if timeout 20 wget -q -O "$scratch/speed" -- "$speed_url"; then
            speed_elapsed=$(($(date +%s) - speed_start))
            [ "$speed_elapsed" -gt 0 ] || speed_elapsed=1
            printf 'Download speed: about %s B/s\n' "$(($(stat -c %s "$scratch/speed") / speed_elapsed))"
        else
            printf '%s\n' 'Warning: optional bandwidth check failed.' >&2
        fi
    fi
fi

new_token=0
if [ "$relink" = 1 ] || [ ! -s "$conf/owner-token" ]; then
    unset token
    if [ "$relink" = 0 ] && [ -s "$conf/.owner-token.new" ]; then
        # Resume a pending relink using the staged credential when no new one was requested.
        IFS= read -r token < "$conf/.owner-token.new"
    elif [ -n "$link_file" ]; then
        IFS= read -r token < "$link_file"
    else
        [ -t 3 ] 3< /dev/tty 2>/dev/null ||
            fail 'A terminal is needed for the link code. Use PUBLICOMPUTE_LINK_CODE in automated tests.'
        stty_restore=$(stty -g < /dev/tty)
        stty -echo < /dev/tty
        printf 'Link code: ' > /dev/tty
        read_ok=0
        IFS= read -r token < /dev/tty || read_ok=$?
        stty "$stty_restore" < /dev/tty
        stty_restore=
        printf '\n' > /dev/tty
        [ "$read_ok" = 0 ] || fail 'Cannot read the link code.'
    fi
    case "$token" in ''|*[[:space:]]*) fail 'Invalid link code.' ;; esac
    (umask 077; printf '%s\n' "$token" > "$scratch/owner-token")
    unset token
    new_token=1
fi

if [ "$need_model" = 1 ]; then
    printf 'Preparing %s %s.\n' "$model_id" "$quant"
    if ! model_ok "$part"; then
        as_agent test ! -L "$part" || fail 'Model partial file must not be a symbolic link.'
        partial=$(as_agent stat -c %s "$part" 2>/dev/null || printf 0)
        if [ "$partial" -ge "$model_size" ]; then as_agent rm -f -- "$part"; fi
        begin=$(date +%s)
        case "$downloader" in
            curl) as_agent curl -fLsS --connect-timeout 15 --retry 3 -C - -o "$part" -- "$model_url" ||
                fail 'Model download failed. Rerun to resume.' ;;
            wget) as_agent wget --timeout=30 --tries=3 -c -O "$part" -- "$model_url" ||
                fail 'Model download failed. Rerun to resume.' ;;
        esac
        download_seconds=$((download_seconds + $(date +%s) - begin))
        if ! model_ok "$part"; then
            as_agent rm -f -- "$part"
            fail 'Model size or SHA-256 mismatch.'
        fi
    fi
    as_agent mv -fT -- "$part" "$model"
fi

printf 'COORDINATOR=%s\nSERVER_NAME=%s\nMODEL_PATH=%s\nMODEL_ID=%s\nQUANTIZATION=%s\n' \
    "$coordinator" "$server_name" "$model" "$model_id" "$quant" > "$scratch/agent.conf"
if [ -n "$gpu_uuid" ]; then printf 'PUBLICOMPUTE_GPU_UUID=%s\n' "$gpu_uuid" >> "$scratch/agent.conf"; fi
backup_file() {
    if [ -e "$1" ] || [ -L "$1" ]; then
        [ -f "$1" ] && [ ! -L "$1" ] || fail "Expected a regular file at $1."
        cp -p -- "$1" "$2"
    fi
}

# Once the package directory moves, the exit trap owns rollback.
snapshot=$scratch/snapshot
mkdir "$snapshot"
if [ -d "$prefix" ]; then stat -c %d:%i "$prefix" > "$snapshot/package"; else : > "$snapshot/package"; fi
backup_file "$conf/agent.conf" "$snapshot/agent.conf"
backup_file "$conf/ca.pem" "$snapshot/ca.pem"
backup_file "$conf/owner-token" "$snapshot/owner-token"
backup_file "$unit_file" "$snapshot/unit"
if systemctl is-active --quiet "$unit"; then : > "$snapshot/running"; fi
if systemctl is-enabled --quiet "$unit"; then : > "$snapshot/enabled"; fi
if [ ! -L "$link" ]; then : > "$snapshot/new-link"; fi
mv -T -- "$snapshot" "$previous"

if [ -f "$previous/unit" ]; then systemctl stop "$unit"; fi
# Stop the running agent before copying enrollment so the snapshot is consistent.
backup_file "$state/identity.bin" "$previous/identity.bin"
backup_file "$state/key.pem" "$previous/key.pem"
: > "$previous/identity-saved"
"$stage/bin/publicompute-agent" swap-install "$stage" "$prefix" || fail 'Cannot replace the installation directory.'
install -m 0644 "$scratch/ca.pem" "$conf/ca.pem"
install -m 0644 "$scratch/agent.conf" "$conf/agent.conf"
if [ "$new_token" = 1 ]; then
    rm -f -- "$state/identity.bin" "$state/key.pem"
    install -m 0600 -o root -g root "$scratch/owner-token" "$conf/owner-token"
fi
install -m 0644 "$prefix/share/$unit" "$unit_file"
systemctl daemon-reload
# Clear the start limit before launching the candidate so failed attempts cannot suppress this start.
systemctl reset-failed "$unit" 2>/dev/null || :
systemctl start "$unit"

printf '%s\n' 'Waiting for the agent.'
deadline=$(($(date +%s) + 300))
last=
while [ "$(date +%s)" -lt "$deadline" ]; do
    case "$(systemctl show --property=SubState --value "$unit")" in
        failed|dead|auto-restart) fail "The new service failed to start. See journalctl -u $unit." ;;
    esac
    remaining=$((deadline - $(date +%s)))
    [ "$remaining" -gt 0 ] || break
    if reply=$(timeout "$remaining" "$prefix/bin/publicompute" --socket "$socket" status --json 2>/dev/null); then
        case "$reply" in
            *'"gpu_lost":true'*) phase='GPU unavailable; waiting for the NVIDIA driver' ;;
            *'"engine":"starting"'*) phase='Checking the model and starting the engine' ;;
            *'"engine":"quarantined"'*) fail "The engine is quarantined. See journalctl -u $unit." ;;
            *'"engine":"recovering"'*) phase='Recovering the engine' ;;
            *'"connected":false'*) phase='Waiting for the coordinator session' ;;
            *'"connected":true'*)
                phase='Waiting for the service to be active'
                if systemctl is-active --quiet "$unit"; then
                    "$prefix/bin/publicompute" --socket "$socket" status
                    install -d -m 0755 /usr/local/bin
                    ln -sfn -- "$prefix/bin/publicompute" "$link"
                    if [ "$installer" != root ]; then
                        usermod -aG "$account" -- "$installer"
                        printf 'Log out and back in to use the %s group.\n' "$account"
                    fi
                    : > "$previous/enabling"
                    systemctl enable "$unit"
                    mv -T -- "$previous" "$scratch/committed"
                    rm -f -- "$conf/.owner-token.new"
                    total=$(($(date +%s) - started))
                    printf 'Installation took %s s (%s s excluding downloads).\n' "$total" "$((total - download_seconds))"
                    exit 0
                fi ;;
            *) phase='Waiting for agent status' ;;
        esac
    else
        phase='Waiting for the control socket'
    fi
    if [ "$phase" != "$last" ]; then printf '%s\n' "$phase"; last=$phase; fi
    sleep 1
done
fail "Agent startup did not complete within 300 s. See journalctl -u $unit."
ROOT
root_pid=$!
wait "$root_pid"
