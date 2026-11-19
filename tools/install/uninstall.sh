#!/bin/sh
# Stop and remove the service under the same lock used by installation.
# Keep downloaded weights only when requested, and restrict their ownership
# before removing the service account.
set -eu

fail() { printf '%s\n' "$*" >&2; exit 1; }

[ "$(id -u)" -eq 0 ] || fail 'Run publicompute uninstall to remove the agent with sudo.'
case "$#:${1:-}" in
    0:)
        [ -t 3 ] 3< /dev/tty 2>/dev/null || fail 'Use --keep-models or --remove-models without a terminal.'
        printf 'Keep downloaded models? [Y/n] ' > /dev/tty
        IFS= read -r answer < /dev/tty || fail 'Cannot read your choice.'
        case "$answer" in
            ''|[yY]|[yY][eE][sS]) keep=1 ;;
            [nN]|[nN][oO]) keep=0 ;;
            *) fail 'Answer yes or no, or specify --keep-models or --remove-models.' ;;
        esac ;;
    1:--keep-models) keep=1 ;;
    1:--remove-models) keep=0 ;;
    *) fail 'Usage: publicompute uninstall [--keep-models | --remove-models]' ;;
esac

account=publicompute-agent
prefix=/opt/$account
scratch=$prefix.staging
state=/var/lib/$account
conf=/etc/$account
unit=$account.service
unit_file=/etc/systemd/system/$unit
link=/usr/local/bin/publicompute

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

root_directory "$scratch"
root_directory "$conf"
root_directory "$prefix"
[ ! -L "$state" ] && { [ ! -e "$state" ] || [ -d "$state" ]; } || fail "Invalid state directory: $state."
require_unmounted "$scratch" "$state" "$conf" "$prefix" "/run/$account"

command -v pgrep >/dev/null || fail 'pgrep is required to verify the agent has stopped.'
uid=$(id -u "$account" 2>/dev/null || :)
gid=$(getent group "$account" | cut -d: -f3)
marked_id() {
    [ -n "$2" ] && [ -f "$1" ] && [ ! -L "$1" ] && [ "$(stat -c %u "$1")" = 0 ] &&
        [ "$(cat -- "$1")" = "$2" ]
}
has_processes() {
    [ -n "$uid" ] || return 1
    for selector in -u -U; do
        result=0
        pgrep "$selector" "$uid" >/dev/null || result=$?
        case "$result" in
            0) return 0 ;;
            1) ;;
            *) fail 'Cannot check for remaining agent processes.' ;;
        esac
    done
    return 1
}
if [ -f "$unit_file" ]; then
    # Send SIGTERM so the agent can announce its departure before the service stops its child.
    systemctl stop "$unit" || fail "Cannot stop $unit."
    case "$(systemctl show --property=ActiveState --value "$unit")" in
        inactive|failed) ;;
        *) fail "$unit has not stopped." ;;
    esac
fi
if has_processes; then fail "Processes still run as $account; removal stopped."; fi
require_unmounted "$scratch" "$state" "$conf" "$prefix" "/run/$account"

if [ -f "$unit_file" ]; then
    systemctl disable "$unit"
    rm -f -- "$unit_file"
    systemctl daemon-reload
fi
if [ "$keep" = 1 ] && [ -d "$state/models" ] && [ ! -L "$state/models" ]; then
    find "$state" -mindepth 1 -maxdepth 1 ! -name models -exec rm -rf -- {} +
    require_unmounted "$state"
    chown -hR root:root "$state"
    chmod 0700 "$state"
    printf 'Models kept in %s/models.\n' "$state"
else
    rm -rf -- "$state"
fi
if marked_id "$conf/account.uid" "$uid"; then userdel "$account"; fi
if marked_id "$conf/group.gid" "$gid" && ! id -u "$account" >/dev/null 2>&1 &&
    getent group "$account" >/dev/null; then groupdel "$account"; fi
if [ -L "$link" ] && [ "$(readlink "$link")" = "$prefix/bin/publicompute" ]; then
    rm -f -- "$link"
fi
require_unmounted "$scratch" "$conf" "$prefix" "/run/$account"
rm -rf -- "$conf" "/run/$account" "$scratch" "$prefix"
if has_processes; then fail "Processes still run as UID $uid."; fi
printf 'PubliCompute agent uninstalled. No processes remain for %s.\n' "$account"
