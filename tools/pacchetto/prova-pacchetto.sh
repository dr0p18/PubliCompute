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
apt-get install -y -qq --no-install-recommends binutils zstd >/dev/null
mkdir /pkg
tar --zstd -xf /in/package.tar.zst -C /pkg --strip-components=1
for binary in publicompute publicompute-agent publicompute-stage llama-server; do
    test -x "/pkg/bin/$binary"
done
for lib in libllama.so libggml-cpu.so libggml-cuda.so; do
    test -f "/pkg/lib/$lib"
done

# EVEX also encodes AVX-512 instructions that only use xmm or ymm registers.
if ! command -v objdump >/dev/null; then
    echo 'SKIP libggml-cpu AVX-512 check (objdump is missing)'
elif LC_ALL=C objdump -dw -M att /pkg/lib/libggml-cpu.so | awk -F '\t' '
    $1 ~ /^[[:space:]]*[[:xdigit:]]+:$/ &&
        ($2 ~ /^[[:space:]]*((26|2e|36|3e|64|65|67)[[:space:]]+)*62[[:space:]]/ ||
         $3 ~ /%(zmm|k)[0-9]+([^[:alnum:]_]|$)/) {
        if (!found) print
        found = 1
    }
    END { exit found }
'; then
    echo 'PASS libggml-cpu has no AVX-512 instructions'
else
    echo 'FAIL libggml-cpu AVX-512 check' >&2
    exit 1
fi

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

cat > "$SCRATCH/check-sass.sh" <<'INNER'
set -euo pipefail
if ! command -v cuobjdump >/dev/null; then
    echo 'SKIP libggml-cuda SASS check (cuobjdump is missing)'
    exit 0
fi
pkg=$(mktemp -d)
trap 'rm -rf -- "$pkg"' EXIT
tar --zstd -xf "$1" -C "$pkg" --strip-components=1 --wildcards '*/lib/libggml-cuda.so*'
if ! sass=$(cuobjdump --list-elf "$pkg/lib/libggml-cuda.so"); then
    echo 'FAIL libggml-cuda SASS check (cuobjdump failed)' >&2
    exit 1
fi
for arch in 75 86 89 120; do
    pattern=$arch
    if [ "$arch" = 120 ]; then pattern='120a?'; fi
    # Match the architecture suffix of each ELF record, not any part of its name.
    if grep -Eq "^ELF file.*\.sm_${pattern}\.cubin$" <<< "$sass"; then
        echo "PASS libggml-cuda SASS for sm_$arch"
    else
        echo "FAIL libggml-cuda missing SASS for sm_$arch" >&2
        exit 1
    fi
done
INNER

check_sass() {
    # The clean runtime image has no CUDA toolkit. The build image includes it.
    if command -v cuobjdump >/dev/null && command -v zstd >/dev/null; then
        bash "$SCRATCH/check-sass.sh" "$SCRATCH/package.tar.zst"
    elif "$ENGINE" image inspect publicompute-build >/dev/null 2>&1; then
        "$ENGINE" run --rm -v "$SCRATCH:/in:ro,Z" publicompute-build bash /in/check-sass.sh /in/package.tar.zst
    else
        echo 'SKIP libggml-cuda SASS check (cuobjdump unavailable without a local CUDA build image or host CUDA tools)'
    fi
}

if "$ENGINE" run --rm -v "$SCRATCH:/in:ro,Z" docker.io/library/ubuntu:22.04 bash /in/check.sh > "$LOG" 2>&1 &&
    check_sass >> "$LOG" 2>&1; then
    grep -E '^(PASS|SKIP)' "$LOG"
else
    rc=$?
    tail -n 25 "$LOG" >&2
    exit "$rc"
fi
