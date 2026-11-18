# Stage engine source

The stage engine uses a pinned upstream `llama.cpp` revision and the patch
series in this directory. The patches let a stage load and execute a
half-open range of layers from a full GGUF file. The head turns token IDs
into activations, intermediate stages forward activations, and the tail
produces logits. Each stage keeps the weights and runtime state needed by
its assigned layers.

The patches use the [MIT license](LICENSE), as does the pinned upstream source.

## Reconstruct the source

`BASE` records the upstream URL, the required commit and the expected
Git tree after applying the series. Run `engine/source.sh DEST` to fetch
the commit, apply `patches/*.patch` in filename order and verify the
result. Set `PUBLICOMPUTE_UPSTREAM` to use a mirror with the same commit.

The script builds the candidate in a temporary directory and replaces
the destination only after the tree matches. A clean checkout with the
expected tree can be reused for incremental builds.

## Build the CPU test tool

Create the destination's parent directory outside this repository first.

    engine/source.sh ~/publicompute-llama/fork
    cd ~/publicompute-llama/fork
    cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_VULKAN=OFF \
          -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_EXAMPLES=OFF
    nice -n 19 cmake --build build --target llama-publicompute-stage-test -j 4

The tool checks full-range and split execution, batched sequences,
state removal and reuse, cache rollback, memory use, and rejected
assignments. For architectures that route experts by token ID, keep
all token-routed layers in the head. Some architectures pass multiple
residual streams between stages, so callers must use the boundary
width returned by `llama_model_n_embd_stage`.

## Maintain the patch series

1. Apply the existing patches in order to the pinned upstream commit, with one
   change per commit.
2. After editing the series, use `git diff --full-index COMMIT^ COMMIT` to
   regenerate each patch in order. When updating upstream, rebase the series
   onto the new revision first.
3. Record the resulting `HEAD^{tree}` in `BASE`. The tree includes source
   comments, so rewriting a comment in the resulting source also changes the hash.
4. Reconstruct into a fresh destination and run the single-stage and
   split-stage parity checks.
