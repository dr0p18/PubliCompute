#!/usr/bin/env bash
# Check the packaged binaries on the oldest supported Ubuntu runtime.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ARCHIVE=${1:-}
if [ -z "$ARCHIVE" ]; then
    shopt -s nullglob
    for candidate in "$HERE"/out/publicompute-linux-x86_64-*.tar.zst; do
        if [[ -z $ARCHIVE || $candidate -nt $ARCHIVE ]]; then ARCHIVE=$candidate; fi
    done
fi
[ -n "$ARCHIVE" ] || { echo "No package found. Supply a PACKAGE.tar.zst." >&2; exit 1; }
ARCHIVE=$(realpath -e "$ARCHIVE")
[ -f "$ARCHIVE" ]
if podman info >/dev/null 2>&1; then
    ENGINE=podman
elif docker info >/dev/null 2>&1; then
    ENGINE=docker
else
    echo "A working Podman or Docker installation is required." >&2
    exit 1
fi
mkdir -p "$HERE/out"
LOG="$HERE/out/package-test.log"
SCRATCH=$(mktemp -d "$HERE/out/runtime-test.XXXXXX")
trap 'rm -rf -- "$SCRATCH"' EXIT
cp -- "$ARCHIVE" "$SCRATCH/package.tar.zst"

cat > "$SCRATCH/check.sh" <<'INNER'
set -euo pipefail
apt-get update -qq
apt-get install -y -qq --no-install-recommends zstd >/dev/null
mkdir /pkg
tar --zstd -xf /in/package.tar.zst -C /pkg --strip-components=1
for binary in publicompute publicompute-agent publicompute-stage llama-server; do
    test -x "/pkg/bin/$binary"
done
for lib in libllama.so libggml-cuda.so; do
    test -f "/pkg/lib/$lib"
done

for file in /pkg/bin/* /pkg/lib/*.so*; do
    dependencies=$(ldd "$file")
    # The host driver supplies libcuda.so.1 when the program runs on a GPU.
    if awk '/not found/ && $1 != "libcuda.so.1" { print; missing=1 } END { exit !missing }' <<< "$dependencies"; then
        echo "Missing dependencies for $file" >&2
        exit 1
    fi
    echo "PASS ldd ${file##*/}"
done
/pkg/bin/publicompute --help >/dev/null
/pkg/bin/publicompute-agent --help >/dev/null
check_startup() {
    if output=$("$@" 2>&1); then
        echo "PASS $*"
    elif [[ $output == *'libcuda.so.1: cannot open shared object file'* ]]; then
        echo "SKIP $* (no NVIDIA driver)"
    else
        echo "FAIL $*: $output" >&2
        exit 1
    fi
}
check_startup /pkg/bin/publicompute-stage --help
touch /empty.jsonl
check_startup /pkg/bin/publicompute-stage compare /empty.jsonl /empty.jsonl --bit-identical
check_startup /pkg/bin/llama-server --version
INNER

if "$ENGINE" run --rm -v "$SCRATCH:/in:ro,Z" docker.io/library/ubuntu:22.04 bash /in/check.sh > "$LOG" 2>&1; then
    grep -E '^(PASS|SKIP)' "$LOG"
else
    rc=$?
    tail -n 25 "$LOG" >&2
    exit "$rc"
fi
