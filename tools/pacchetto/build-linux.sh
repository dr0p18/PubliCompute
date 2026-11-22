#!/usr/bin/env bash
# Build the Linux archive and its SHA-256 manifest in a container.
# The outer invocation selects the version and captures the build log in
# the output directory. The inner invocation reconstructs the engine,
# compiles the binaries, bundles their libraries and checks their linkage.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
OUT="$HERE/out"
mkdir -p "$OUT"

if [ "${1:-}" != "--inside" ]; then
    VERSION=${1:-$(git -C "$REPO" describe --tags --always --dirty 2>/dev/null || echo dev)}
    if podman info >/dev/null 2>&1; then
        ENGINE=podman
    elif docker info >/dev/null 2>&1; then
        ENGINE=docker
    else
        echo "A working Podman or Docker installation is required." >&2
        exit 1
    fi
    LOG="$OUT/build.log"
    : > "$LOG"
    echo "Building $VERSION (log: $LOG)"
    "$ENGINE" build -t publicompute-build -f "$HERE/Containerfile" "$HERE" >> "$LOG" 2>&1 \
        || { tail -n 20 "$LOG"; exit 1; }
    # Relabel the bind mount so SELinux permits access from the container.
    "$ENGINE" run --rm -v "$REPO:/work:Z" -e VERSION="$VERSION" -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
        publicompute-build bash /work/tools/pacchetto/build-linux.sh --inside >> "$LOG" 2>&1 \
        || { echo "Build failed. Last log lines:"; tail -n 25 "$LOG"; exit 1; }
    grep -A40 '^== Summary' "$LOG" || tail -n 10 "$LOG"
    exit 0
fi

# Build and assemble the package inside the container.
VERSION=${VERSION:-dev}
cd /work
B=/work/tools/pacchetto/build        # Keep the reconstructed source, build directory and Cargo target under this scratch root.
PKG=/work/tools/pacchetto/out/publicompute
NAME="publicompute-linux-x86_64-$VERSION"
rm -rf "$PKG"
mkdir -p "$B" "$PKG/bin" "$PKG/lib"

echo "== 1. fork sources (engine/source.sh checks TREE)"
# Verify the pinned engine tree before compiling or packaging its libraries.
engine/source.sh "$B/llama-src"

echo "== 2. fork build (CUDA, shared libs)"
# Driver 535 can load speed-compressed CUDA code. Size compression needs a newer driver.
# Use the same AVX2 baseline on every build machine.
cmake -S "$B/llama-src" -B "$B/llama-build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON -DGGML_CUDA_NCCL=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_UI=OFF -DCMAKE_CUDA_ARCHITECTURES="75;86;89;120" -DBUILD_SHARED_LIBS=ON \
    -DGGML_CUDA_COMPRESSION_MODE=speed \
    -DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_BMI2=ON \
    -DGGML_AVX_VNNI=OFF -DGGML_AVX512=OFF -DGGML_AVX512_VBMI=OFF -DGGML_AVX512_VNNI=OFF -DGGML_AVX512_BF16=OFF \
    -DGGML_AMX_TILE=OFF -DGGML_AMX_INT8=OFF -DGGML_AMX_BF16=OFF \
    -DCMAKE_C_FLAGS=-march=x86-64-v3 -DCMAKE_CXX_FLAGS=-march=x86-64-v3 \
    -DCMAKE_CUDA_FLAGS=-Xcompiler=-march=x86-64-v3 \
    -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
    -DCMAKE_EXE_LINKER_FLAGS="-Wl,--allow-shlib-undefined"
# Check what the fork passes to the compilers before starting the build.
LC_ALL=C awk '
    function check_cuda() {
        if (cuda_rule && !speed) {
            print "Missing -compress-mode=speed: " cuda_rule
            bad = 1
        }
    }
    /(^|[[:space:]="\047,])(-march=native|-mavx512[^[:space:]"\047,]*|-mamx[^[:space:]"\047,]*|-mavxvnni)([[:space:]"\047,]|$)/ {
        print "Unsupported CPU flag in " FILENAME ":" FNR ": " $0
        bad = 1
    }
    /^build / {
        check_cuda()
        cuda_rule = ""
        speed = 0
        if ($0 ~ /: CUDA_COMPILER[^[:space:]]* /) {
            cuda_rule = $0
            cuda_count++
        }
    }
    cuda_rule && /^[[:space:]]+FLAGS = / && /(^|[[:space:]"\047])-compress-mode=speed([[:space:]"\047]|$)/ {
        speed = 1
    }
    END {
        check_cuda()
        if (!cuda_count) print "No CUDA compile rules found in build.ninja"
        exit (bad || !cuda_count)
    }
' "$B/llama-build/build.ninja" "$B/llama-build/CMakeFiles/rules.ninja" >&2
# The build image lacks the NVIDIA driver. Permit unresolved references to
# its shared library at link time because the installed driver supplies it.
cmake --build "$B/llama-build" --target llama-server llama ggml ggml-base ggml-cpu ggml-cuda -j"$(nproc)"
LLAMA_BIN="$B/llama-build/bin"

echo "== 3. rust binaries"
export CARGO_TARGET_DIR="$B/target"
# Point the stage bindings at the same headers and libraries built for the package.
export PUBLICOMPUTE_LLAMA_DIR="$B/llama-src"
export PUBLICOMPUTE_LLAMA_LIB="$LLAMA_BIN"
# The stage links against the CUDA backend, whose driver dependency resolves at runtime.
# Keep Rust on its default CPU target even if the caller has set compiler flags.
unset CARGO_ENCODED_RUSTFLAGS
export RUSTFLAGS="-C link-arg=-Wl,--allow-shlib-undefined"
cargo build --locked --release -p publicompute-agent -p publicompute-stage -p publicompute \
    --features publicompute-stage/llama --bin publicompute-agent --bin publicompute-stage --bin publicompute

echo "== 4. collect files"
mkdir -p "$PKG/share"
cp tools/install/publicompute-agent.service tools/install/uninstall.sh "$PKG/share/"
cp "$B/target/release/publicompute-agent" "$B/target/release/publicompute-stage" "$B/target/release/publicompute" "$PKG/bin/"
cp "$LLAMA_BIN/llama-server" "$PKG/bin/"
# Preserve library symlink chains so SONAMEs resolve to the bundled binaries.
cp -a "$LLAMA_BIN"/lib*.so* "$PKG/lib/"
# Copy runtime libraries under their SONAMEs so the dynamic linker can find them.
CUDA_LIB=/usr/local/cuda/lib64
for l in libcudart libcublas libcublasLt; do
    f=$(ls "$CUDA_LIB/$l.so".[0-9]* 2>/dev/null | sort -V | tail -n1 || true)
    [ -n "$f" ] || { echo "$l is missing from $CUDA_LIB" >&2; exit 1; }
    cp -L "$f" "$PKG/lib/$(patchelf --print-soname "$f")"
done
# Bundle the OpenMP runtime required by the CPU backend for hosts without it.
cp -L "$(ldconfig -p | awk '/libgomp\.so\.1 .*x86-64/{print $NF; exit}')" "$PKG/lib/libgomp.so.1"
# Search the package library directory at runtime instead of relying on toolkit paths.
for f in "$PKG"/bin/*; do patchelf --set-rpath '$ORIGIN/../lib' "$f"; done
for f in "$PKG"/lib/*.so*; do if [ ! -L "$f" ]; then patchelf --set-rpath '$ORIGIN' "$f"; fi; done
# Check the bundled dependencies in the build container. Also check from a
# clean runtime container to catch dependencies present only in the build image.
MISSING=$(for f in "$PKG"/bin/* "$PKG"/lib/*.so*; do ldd "$f" 2>/dev/null; done | awk '/not found/ && $1 != "libcuda.so.1" {print $1}' | sort -u)
[ -z "$MISSING" ] || { echo "Missing shared libraries: $MISSING" >&2; exit 1; }
strip --strip-unneeded "$PKG"/bin/* 2>/dev/null || true
echo "$VERSION" > "$PKG/VERSION"

echo "== 5. archive"
cd /work/tools/pacchetto/out
rm -f "$NAME.tar.zst" SHA256SUMS SHA256SUMS.sig
tar --transform "s,^publicompute,$NAME," -cf - publicompute | zstd -19 -T0 -q -o "$NAME.tar.zst"
sha256sum "$NAME.tar.zst" > SHA256SUMS
# Return ownership of the package outputs to the host user.
if [ -n "${HOST_UID:-}" ]; then
    chown -R "$HOST_UID:$HOST_GID" /work/tools/pacchetto/out /work/tools/pacchetto/build 2>/dev/null || true
fi

echo "== Summary"
echo "archive   : tools/pacchetto/out/$NAME.tar.zst ($(du -h "$NAME.tar.zst" | cut -f1), unpacked $(du -sh publicompute | cut -f1))"
echo "sha256    : $(cut -d' ' -f1 SHA256SUMS)"
echo "binaries  : $(ls "$PKG/bin" | tr '\n' ' ')"
echo "libraries : $(ls "$PKG/lib" | wc -l)"
