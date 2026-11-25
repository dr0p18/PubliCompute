#!/bin/sh
# Install the Linux package, configure the agent service and enroll a node.
# The privileged phase stages and verifies a candidate before replacing the
# active installation. Use PUBLICOMPUTE_CA_FILE for a private coordinator.
# The loopback HTTP and hardware overrides are for installer tests.
set +x
set -eu

# The CUDA package targets the 535 driver branch and needs 570 for Blackwell.
MIN_DRIVER=535
MIN_DRIVER_BLACKWELL=570

# Keep the link code out of the environment inherited by child processes.
unset link_code
link_code=${PUBLICOMPUTE_LINK_CODE:-}
unset PUBLICOMPUTE_LINK_CODE

fail() { printf '%s\n' "$*" >&2; exit 1; }

relink=0
allow_downgrade=0
region=${PUBLICOMPUTE_REGION:-EU}
while [ "$#" -gt 0 ]; do
    case "$1" in
        --relink) relink=1 ;;
        --allow-downgrade) allow_downgrade=1 ;;
        --region)
            [ "$#" -ge 2 ] || fail '--region requires a value.'
            shift
            region=$1 ;;
        *) fail 'Usage: install.sh [--relink] [--allow-downgrade] [--region REGION]' ;;
    esac
    shift
done

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

default_version=0.1.4
version=${PUBLICOMPUTE_VERSION:-$default_version}
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
cpuinfo=${PUBLICOMPUTE_TEST_CPUINFO:-/proc/cpuinfo}
case "$cpuinfo" in /*) ;; *) cpuinfo=$PWD/$cpuinfo ;; esac
model_choice=${PUBLICOMPUTE_MODEL:-}
case "$model_choice" in
    ''|medio|deepseek|free) ;;
    *) fail 'PUBLICOMPUTE_MODEL must be medio, deepseek or free.' ;;
esac

release_url=https://github.com/dr0p18/PubliCompute/releases/download
base_url=${PUBLICOMPUTE_BASE_URL:-$release_url/v$version}
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
    "${PUBLICOMPUTE_SERVER_NAME:-}" "$ca" "${PUBLICOMPUTE_SPEED_URL:-}" "$skip_gpu" "$model_choice" \
    "$link_file" "$relink" "${SUDO_USER:-$(id -un)}" "$http_fixture" "$allow_downgrade" "$region" \
    "$MIN_DRIVER" "$MIN_DRIVER_BLACKWELL" "$cpuinfo" <<'ROOT' &
set -eu
umask 022

fail() { printf '%s\n' "$*" >&2; exit 1; }

started=$1 downloader=$2 version=$3 base_url=$4 package=$5 package_sha=$6 coordinator=$7 server_name=$8 ca=$9
shift 9
speed_url=$1 skip_gpu=$2 model_choice=$3 link_file=$4 relink=$5 installer=$6 http_fixture=$7 allow_downgrade=$8 region=$9
MIN_DRIVER=${10} MIN_DRIVER_BLACKWELL=${11} cpuinfo=${12}

valid_region() {
    printf '%s\n' "$1" | LC_ALL=C awk '/^[A-Za-z][A-Za-z0-9-]*$/ && length <= 32 { valid++ }
        END { exit !(NR == 1 && valid == 1) }'
}
valid_region "$region" || fail 'Invalid region. Use a letter followed by up to 31 letters, digits or hyphens.'

# The packaged binaries are built for x86-64-v3 and use these instructions on
# any core. A fake cpuinfo is accepted only with the loopback test fixture.
[ "$http_fixture" = 1 ] || cpuinfo=/proc/cpuinfo
[ -f "$cpuinfo" ] && [ -r "$cpuinfo" ] || fail "Cannot read $cpuinfo to check the CPU."
if missing_cpu_flags=$(LC_ALL=C awk '
    BEGIN { count = split("ssse3 sse4_1 sse4_2 popcnt cx16 avx avx2 fma f16c bmi1 bmi2 abm movbe", required) }
    /^flags[[:space:]]*:/ {
        seen++
        sub(/^flags[[:space:]]*:/, "")
        gsub(/[[:space:]]+/, " ")
        flags = " " $0 " "
        for (i = 1; i <= count; i++)
            if (!index(flags, " " required[i] " ")) missing[i] = 1
    }
    END {
        for (i = 1; i <= count; i++) {
            if (!seen || missing[i]) {
                printf "%s%s", separator, required[i]
                separator = " "
            }
        }
    }
' 2>/dev/null < "$cpuinfo"); then
    [ -z "$missing_cpu_flags" ] || fail "This CPU lacks $missing_cpu_flags. PubliCompute needs a CPU with AVX2 (Intel Core from 2013 or newer, or AMD Ryzen or newer). Most Pentium and Celeron models lack it."
else
    fail "Cannot read $cpuinfo to check the CPU."
fi

release_pubkey='ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIDv7F+43HnKvzUnfc8dADFSHP98ZKG5uKNvvvVD8oymj'
if [ -z "$package" ]; then
    case "$release_pubkey" in
        'ssh-ed25519 '*) ;;
        *) fail 'The PubliCompute release signing key is not configured in this installer.' ;;
    esac
    command -v ssh-keygen >/dev/null || fail 'Install OpenSSH with ssh-keygen -Y support to verify the release.'
fi

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

root_directory /run
lock_dir=/run/publicompute-agent-lock
mkdir -m 0700 -- "$lock_dir" 2>/dev/null || :
[ ! -L "$lock_dir" ] && [ -d "$lock_dir" ] && [ "$(stat -c %u:%a "$lock_dir")" = 0:700 ] ||
    fail 'Invalid installer lock directory.'
lock=$lock_dir/lock
if [ ! -e "$lock" ]; then (umask 077; set -C; : > "$lock") 2>/dev/null || :; fi
[ ! -L "$lock" ] && [ -f "$lock" ] && [ "$(stat -c %u:%h "$lock")" = 0:1 ] || fail 'Invalid installer lock file.'
exec 9>>"$lock"
chmod 0600 "$lock"
flock -x -w 120 9 || fail 'Timed out after 120 s waiting for the PubliCompute installer lock.'

# Older installers and uninstallers still lock the root-owned file in /run/lock.
legacy_dir=/run/lock
legacy_lock=$legacy_dir/$account.lock
if [ ! -L "$legacy_dir" ] && [ -d "$legacy_dir" ] && [ "$(stat -c %u "$legacy_dir")" = 0 ]; then
    mode=$(stat -c %a "$legacy_dir")
    if [ "$((0$mode & 022))" -eq 0 ] || [ "$((0$mode & 01000))" -ne 0 ]; then
        if [ ! -e "$legacy_lock" ] && [ ! -L "$legacy_lock" ]; then
            if legacy_temp=$(mktemp -- "$legacy_dir/.$account.XXXXXXXX"); then
                ln -T -- "$legacy_temp" "$legacy_lock" 2>/dev/null || :
                rm -f -- "$legacy_temp"
            fi
        fi
        if [ ! -L "$legacy_lock" ] && [ -f "$legacy_lock" ] && [ "$(stat -c %u:%h "$legacy_lock")" = 0:1 ]; then
            exec 8<"$legacy_lock"
            flock -x -w 120 8 || fail 'Timed out after 120 s waiting for the legacy PubliCompute installer lock.'
        fi
    fi
fi

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
    if [ -f "$previous/preference-saved" ]; then
        restore_file contributo.json "$state/contributo.json" &&
            restore_file active-model.json "$state/active-model.json" || return 1
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

release_number() {
    case "$1" in ''|*[!0-9.]*) return 1 ;; esac
    printf '%s\n' "$1" | LC_ALL=C grep -Eq '^(0|[1-9][0-9]*)(\.(0|[1-9][0-9]*)){2}$'
}
installed_version=
if [ -d "$prefix" ] && [ "$allow_downgrade" = 0 ]; then
    if [ -f "$prefix/VERSION" ] && [ ! -L "$prefix/VERSION" ]; then
        installed_version=$(cat -- "$prefix/VERSION")
    fi
    if [ "$installed_version" != "$version" ]; then
        if ! release_number "$installed_version" || ! release_number "$version"; then
            fail 'Cannot order the installed and requested versions. Use --allow-downgrade to proceed.'
        fi
        first=$(printf '%s\n' "$installed_version" "$version" | LC_ALL=C sort -V | head -n 1)
        [ "$first" = "$installed_version" ] ||
            fail "Downgrade from $installed_version to $version requires --allow-downgrade."
    fi
fi

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
    # Older drivers can reject the compute capability field.
    cards=$(LC_ALL=C timeout 10 nvidia-smi --query-gpu=memory.total,uuid,driver_version,compute_cap \
        --format=csv,noheader,nounits 2>/dev/null) ||
    cards=$(LC_ALL=C timeout 10 nvidia-smi --query-gpu=memory.total,uuid,driver_version --format=csv,noheader,nounits) ||
        fail 'nvidia-smi failed or timed out. Check the NVIDIA driver.'
    gpu_check=$(printf '%s\n' "$cards" | LC_ALL=C awk -F, -v min_driver="$MIN_DRIVER" -v blackwell_driver="$MIN_DRIVER_BLACKWELL" '
        function version_at_least(version, minimum, actual, required, n, m, i) {
            n = split(version, actual, /\./)
            m = split(minimum, required, /\./)
            for (i = 1; i <= n || i <= m; i++) {
                if (actual[i] + 0 != required[i] + 0) return actual[i] + 0 > required[i] + 0
            }
            return 1
        }
        BEGIN { failure = "vram" }
        {
            for (i = 1; i <= NF; i++) gsub(/^[[:space:]]+|[[:space:]]+$/, "", $i)
            if ($1 !~ /^[0-9]+$/) {
                if (failure == "vram") failure = "unavailable"
                next
            }
            if ($1 + 0 < 8192) next
            if ($4 !~ /^[0-9]+\.[0-9]+$/) {
                reason = "unavailable"
            } else if ($4 !~ /^(7\.5|8\.[6-9]|12\.[0-9]+)$/) {
                reason = "unsupported " $4
            } else if ($2 !~ /^GPU-[a-fA-F0-9-]+$/ || $3 !~ /^[0-9]+(\.[0-9]+)*$/) {
                reason = "unavailable"
            } else {
                minimum = ($4 ~ /^12\./) ? blackwell_driver : min_driver
                if (version_at_least($3, minimum)) {
                    selected = $2
                    exit
                }
                reason = "old " $3 " " minimum
            }
            # When no card works, report the one with the most VRAM.
            if ($1 + 0 > largest) {
                largest = $1 + 0
                failure = reason
            }
        }
        END { print selected != "" ? "ok " selected : failure }
    ')
    IFS=' ' read -r gpu_status gpu_value gpu_minimum <<GPU
$gpu_check
GPU
    case "$gpu_status" in
        ok)
            gpu_uuid=$gpu_value
            printf 'Using NVIDIA GPU %s.\n' "$gpu_uuid" ;;
        vram) fail 'At least one NVIDIA GPU with 8 GiB of VRAM is required.' ;;
        unsupported) fail "Unsupported NVIDIA GPU (compute capability $gpu_value)." ;;
        unavailable) fail 'NVIDIA GPU data unavailable. Update the NVIDIA driver.' ;;
        old) fail "NVIDIA driver $gpu_value is too old for this GPU. PubliCompute needs $gpu_minimum or newer.
Ubuntu: run sudo ubuntu-drivers install or install the nvidia-driver-570 package.
WSL: update the Windows NVIDIA driver." ;;
    esac
fi

valid_hash() {
    [ "${#1}" -eq 64 ] || return 1
    case "$1" in *[!0123456789abcdefABCDEF]*) return 1 ;; esac
}
if [ -n "$package" ]; then
    valid_hash "$package_sha" || fail 'PUBLICOMPUTE_PACKAGE requires PUBLICOMPUTE_PACKAGE_SHA256.'
fi

if [ -e "$link" ] || [ -L "$link" ]; then
    [ "$(readlink "$link")" = "$prefix/bin/publicompute" ] ||
        fail "$link belongs to another installation."
fi

hash_matches() {
    printf '%s  %s\n' "$1" "$2" | sha256sum --check --status -
}

check_space() {
    available=$(df -Pk "$1" | awk 'NR == 2 { print $4 }')
    [ "${available:-0}" -ge "$2" ] || fail "Insufficient free space on $1 (need $2 KiB)."
}
# Reserve space for both the archive and the extracted package.
check_space /opt "$((2 * 1024 * 1024))"

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
    package_sha=$(printf '%s' "$package_sha" | tr 'A-F' 'a-f')
    cp -- "$package" "$scratch/package.tar.zst"
else
    begin=$(date +%s)
    fetch "${base_url%/}/SHA256SUMS" "$scratch/SHA256SUMS" || fail 'Cannot download SHA256SUMS.'
    fetch "${base_url%/}/SHA256SUMS.sig" "$scratch/SHA256SUMS.sig" || fail 'Cannot download SHA256SUMS.sig.'
    printf 'publicompute-release %s\n' "$release_pubkey" > "$scratch/allowed-signers"
    ssh-keygen -Y verify -f "$scratch/allowed-signers" -I publicompute-release -n publicompute-release \
        -s "$scratch/SHA256SUMS.sig" < "$scratch/SHA256SUMS" || fail 'Invalid SHA256SUMS signature.'
    package_sha=$(awk -v name="$name.tar.zst" '$2 == name || $2 == "*" name { print $1 }' "$scratch/SHA256SUMS")
    valid_hash "$package_sha" || fail "No unique SHA-256 for $name.tar.zst in SHA256SUMS."
    package_sha=$(printf '%s' "$package_sha" | tr 'A-F' 'a-f')
    if [ "$installed_version" = "$version" ] &&
        [ -f "$prefix/.release-sha256" ] && [ ! -L "$prefix/.release-sha256" ]; then
        [ "$(cat -- "$prefix/.release-sha256")" = "$package_sha" ] ||
            fail "Public release v$version has a different SHA-256 from the installed package. Publish it under a new version."
    fi
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
[ -f "$stage/VERSION" ] && [ ! -L "$stage/VERSION" ] && [ "$(cat -- "$stage/VERSION")" = "$version" ] ||
    fail "Package VERSION does not match $version."
for binary in publicompute-agent publicompute-stage publicompute llama-server; do
    [ -x "$stage/bin/$binary" ] || fail "Package is missing bin/$binary."
done
[ -f "$stage/share/$unit" ] || fail "Package is missing share/$unit."
install -m 0755 "$stage/share/uninstall.sh" "$stage/uninstall.sh"
printf '%s\n' "$package_sha" > "$scratch/release-sha256"
install -m 0644 "$scratch/release-sha256" "$stage/.release-sha256"

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
    find "$state/models" -name '*.agent.part' -prune -o -exec chown -h "$account:$account" -- {} +
fi
as_agent() { runuser -u "$account" -- "$@"; }
as_agent mkdir -p "$state/models"

config_value() {
    sed -n "s/^$1=//p" "$conf/agent.conf" | head -n 1
}

trusted_config() {
    [ -f "$conf/agent.conf" ] && [ ! -L "$conf/agent.conf" ] &&
        [ "$(stat -c %u "$conf/agent.conf")" = 0 ] || return 1
    mode=$(stat -c %a "$conf/agent.conf")
    [ "$((0$mode & 022))" -eq 0 ]
}

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

# stdin carries the privileged script, even when the user has a terminal.
if [ -z "$model_choice" ] && [ -t 3 ] 2>/dev/null 3< /dev/tty; then
    while :; do
        printf '%s\n' 'Choose how to use this GPU:' \
            '  1. Qwen3.6-35B-A3B UD-Q4_K_XL (about 21 GiB)' \
            '  2. DeepSeek V4 Flash 0731 (about 81 GiB)' \
            '  3. Keep the card free' > /dev/tty
        printf 'Choice [3]: ' > /dev/tty
        IFS= read -r answer < /dev/tty || answer=
        case "$answer" in
            1|medio) model_choice=medio; break ;;
            2|deepseek) model_choice=deepseek; break ;;
            3|free|'') model_choice=free; break ;;
        esac
    done
fi
model_choice=${model_choice:-free}

printf 'COORDINATOR=%s\nSERVER_NAME=%s\n' "$coordinator" "$server_name" > "$scratch/agent.conf"
if trusted_config; then
    old_region=$(config_value PUBLICOMPUTE_REGION | sed 's/^[[:blank:]]*//; s/[[:blank:]]*$//')
    # EnvironmentFile accepts either style of quotes around the region.
    case "$old_region" in
        \"*\") old_region=${old_region#\"}; old_region=${old_region%\"} ;;
        \'*\') old_region=${old_region#\'}; old_region=${old_region%\'} ;;
    esac
    if [ -n "$old_region" ]; then
        valid_region "$old_region" || fail 'Invalid existing PUBLICOMPUTE_REGION in agent.conf.'
        region=$old_region
    fi
fi
printf 'PUBLICOMPUTE_REGION=%s\nPUBLICOMPUTE_RELAY=https://relay.publicompute.com:8443\n' "$region" >> "$scratch/agent.conf"
if [ -n "$gpu_uuid" ]; then printf 'PUBLICOMPUTE_GPU_UUID=%s\n' "$gpu_uuid" >> "$scratch/agent.conf"; fi
backup_file() {
    if [ -e "$1" ] || [ -L "$1" ]; then
        [ -f "$1" ] && [ ! -L "$1" ] || fail "Expected a regular file at $1."
        cp -p -- "$1" "$2"
    fi
}

backup_preference() {
    # Root creates the snapshot. The service user opens and reads the source.
    if (umask 077; as_agent "$prefix/bin/publicompute-agent" --data-dir "$state" backup-preference "$1" > "$2"); then
        [ -f "$2" ] && [ ! -L "$2" ] && [ -s "$2" ] || fail "Invalid snapshot of $1."
        chown -- "$account:$account" "$2"
    elif [ "$?" = 3 ]; then
        # The state file is absent on this installation.
        rm -f -- "$2"
    else
        fail "Cannot back up $1."
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
backup_preference contributo.json "$previous/contributo.json"
backup_preference active-model.json "$previous/active-model.json"
: > "$previous/preference-saved"
as_agent "$prefix/bin/publicompute-agent" --data-dir "$state" install-preference "$model_choice" ||
    fail 'Cannot save the contribution preference.'
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
    service_state=$(systemctl show --property=SubState --property=ExecMainStatus "$unit")
    case "$service_state" in
        *SubState=auto-restart*ExecMainStatus=75*|*ExecMainStatus=75*SubState=auto-restart*)
            # A cached model can finish switching before the installer sees the first session.
            sleep 1
            continue ;;
        *SubState=failed*|*SubState=dead*|*SubState=auto-restart*)
            fail "The new service failed to start. See journalctl -u $unit." ;;
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
                    "$prefix/bin/publicompute" --socket "$socket" status || :
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
