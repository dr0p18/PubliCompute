# Linux binary package

Builds `publicompute-linux-x86_64-<version>.tar.zst` and `SHA256SUMS`. Remote installation also requires `SHA256SUMS.sig`.

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

## Signing a release

On the release signer's laptop, outside the build container, create the Ed25519 key once with a passphrase:

    mkdir -p -m 700 ~/.config/publicompute/firma
    ssh-keygen -t ed25519 -f ~/.config/publicompute/firma/publicompute-release

Keep the private key on the laptop, separate from the release hosting account. Pin only the first two fields of the
`.pub` file (`ssh-ed25519` and the key data) in the installer. After the final build and package checks, run
`tools/pacchetto/firma.sh` from a terminal and enter the passphrase when prompted. It writes `SHA256SUMS.sig` next to
the manifest. Publish the archive, `SHA256SUMS`, and `SHA256SUMS.sig` together, after the release preflight.

For a local installation, an administrator can use `PUBLICOMPUTE_PACKAGE` with a separately trusted
`PUBLICOMPUTE_PACKAGE_SHA256`; no release signature is required for that path.

The installer compares numbered releases (major.minor.patch) with the active package's `VERSION` file. Pass
`--allow-downgrade` to install an older release or switch between versions that cannot be ordered. Package
verification still applies. When the active package has a recorded digest, a remote archive with the same version
and a different digest also requires explicit approval via this flag.

To keep an alternate model on upgrade, set `MODEL_PATH`, `MODEL_ID`, `QUANTIZATION`, `MODEL_SIZE` (bytes), and
`MODEL_SHA256` together in the root-owned `/etc/publicompute-agent/agent.conf`. Get the expected size and digest from
trusted model metadata. If the configured weights cannot be verified, the installer prepares the default model.

The installer and uninstaller share a lock under `/run/publicompute-agent-lock`. During migration they also honor a
valid root-owned lock under `/run/lock` when the directory protects that entry from replacement by other users.
Use the updated scripts for both operations if the legacy lock path is occupied by an unrelated file.

## Output

`tools/pacchetto/out/`: `publicompute-linux-x86_64-<version>.tar.zst`, `SHA256SUMS` and `publicompute/` (the
uncompressed tree). `tools/pacchetto/build/` holds the intermediate work (fork source, cmake and cargo targets). The
compressed package is about 600-700 MB, mostly `libcublasLt` and `libggml-cuda`. It is published as a GitHub release.
