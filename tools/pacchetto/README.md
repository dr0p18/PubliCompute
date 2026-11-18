# Linux binary package

Builds `publicompute-linux-x86_64-<version>.tar.zst` and `SHA256SUMS`, the files that `install.sh` downloads.

## Contents

- `bin/`: `publicompute-agent`, `publicompute-stage` (with the `llama` feature), `llama-server` from the engine fork (the
  patches in `engine/patches`, source tree checked by `engine/source.sh`), and the `publicompute` client;
- `lib/`: `libllama` and `libggml*` from the fork, plus the CUDA 12.8 runtime libraries (`libcudart`, `libcublas`,
  `libcublasLt`);
- the binaries use `RPATH=$ORIGIN/../lib`, so the target machine only needs the NVIDIA driver (`libcuda.so.1`), not the
  CUDA toolkit;
- CUDA architectures 75, 86, 89 and 120; built in an Ubuntu 22.04 container (glibc 2.35).

## Usage

On a Linux machine with podman or docker (a GPU is not needed to build):

    tools/pacchetto/build-linux.sh 0.1.1
    tools/pacchetto/prova-pacchetto.sh tools/pacchetto/out/publicompute-linux-x86_64-0.1.1.tar.zst

The package check uses a clean Ubuntu 22.04 container.

The build is heavy (CUDA for four architectures plus a cargo release build). The full log goes to
`tools/pacchetto/out/build.log`; the terminal only shows a summary.

## Output

`tools/pacchetto/out/`: `publicompute-linux-x86_64-<version>.tar.zst`, `SHA256SUMS` and `publicompute/` (the
uncompressed tree). `tools/pacchetto/build/` holds the intermediate work (fork source, cmake and cargo targets). The
compressed package is about 600-700 MB, mostly `libcublasLt` and `libggml-cuda`. It is published as a GitHub release.
