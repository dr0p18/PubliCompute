#!/bin/sh
set -eu

fail() { printf '%s\n' "$*" >&2; exit 1; }
[ "$#" -le 1 ] || fail 'Usage: firma.sh [SHA256SUMS]'
here=$(cd "$(dirname "$0")" && pwd)
manifest=${1:-$here/out/SHA256SUMS}
key=$HOME/.config/publicompute/firma/publicompute-release
[ -s "$manifest" ] || fail "Missing or empty manifest: $manifest"
[ -f "$key" ] || fail "Missing release signing key: $key"
command -v ssh-keygen >/dev/null || fail 'Install OpenSSH with ssh-keygen -Y support.'

signature=$(mktemp -- "$manifest.sig.XXXXXX")
trap 'rm -f -- "$signature"' 0
trap 'exit 130' HUP INT TERM
SSH_ASKPASS_REQUIRE=never ssh-keygen -Y sign -f "$key" -n publicompute-release < "$manifest" > "$signature"
chmod 0644 "$signature"
mv -fT -- "$signature" "$manifest.sig"
printf 'Signed %s\n' "$manifest"
