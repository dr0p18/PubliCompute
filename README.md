<p align="center">
  <img src="logo.svg" alt="PubliCompute logo" width="160">
</p>

<h1 align="center">PubliCompute</h1>

<p align="center">
  A network of gaming GPUs that run large language models together.<br>
  Connect your card and use models that don't fit in its memory, <b>for free</b>.
</p>

<p align="center">
  <a href="https://publicompute.com">Website</a> ·
  <a href="#start-here">Install</a> ·
  <a href="#privacy-of-your-prompts">Privacy</a> ·
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

---

A model that does not fit on one card is split into stages and distributed over
several GPUs, forming a **Chain**. Every card contributing to a Chain gets free
use of the model it hosts.

While part of a Chain, your card may also compute tokens for other users. This
extra work is paid in **credits**, which you can spend on PubliCompute Chat when
your PC is offline.

This project would not exist without **llama.cpp and GGML**: read the
[acknowledgements](#acknowledgements).

## Status

**Beta, by invitation.** Get your invite code by registering on
[publicompute.com](https://publicompute.com). Things change fast and the network
is small: expect rough edges, and tell us about them.

## Supported hardware

- **NVIDIA GPUs with 8 GB or more**, Turing (RTX 20) and newer.
- **Linux x86_64** today. Windows is a work in progress.
- **A decent internet connection**: about 100 Mbit/s up and down, and under
  25 ms to the other nodes. Chains are built inside one region to keep latency
  down.

## Start here

```sh
curl -fsSLO https://github.com/dr0p18/PubliCompute/releases/download/v0.1.1/install.sh
sh install.sh
```

The installer asks for the **link code** that comes with your invitation,
verifies the package against `SHA256SUMS`, installs the agent as a system
service and connects your card. Run it again with `--relink` to move the card
to another account.

## Everyday use

```sh
publicompute status            # what your card is doing, credits, Chain
publicompute use <model>       # pick the model you want to use
publicompute farm              # FARM mode: the scheduler decides
publicompute pause             # leave the network
publicompute start             # resume
publicompute uninstall         # remove everything, keep or delete the models
```

## What can I do with this?

- **Use models bigger than your PC can handle.** Pick a model: if the scheduler
  assigns your GPU to a Chain running that model, you use it without spending
  credits while your GPU contributes.
- **Farm credits.** If you don't pick a model, your card goes into FARM mode:
  the scheduler puts it wherever it is needed and every verified token earns you
  credits.
- **Spend credits when you are offline**, on any model of the network, through
  the web chat.
- **Try it without a GPU**: everyone gets a small daily quota on the base model.

## Launch models

| Model | Quantization | Weights | Runs on |
| --- | --- | --- | --- |
| Qwen3.5-9B | Q4_K_M | 5.3 GiB | one 8 GB card |
| Qwen3.6-35B-A3B | Q4 | 21.3 GiB | one 24 GB card, or a Chain of smaller cards |
| DeepSeek V4 Flash | Q2 | 81 GiB | a Chain of 24 GB cards |

The context cache needs memory on top of the weights; the scheduler accounts for
it when it places a model.

The scheduler assigns work dynamically, aiming to reduce the total latency of
each Chain. This usually means short Chains (2-5 stages), so cards with less
VRAM are less likely to join a Chain running a larger model. Smaller cards are
still used to run the network and to verify work, which also gives them access
to the models and earns credits.

Want a different model? [Let us know](https://publicompute.com/#roster).

## How it works

1. **The app** runs on your PC as a small background service. It talks only to
   the coordinator and to the other nodes of your Chain. Connect to and
   disconnect from the network whenever you want.
2. **The coordinator** assigns each card a whole model or a range of layers,
   builds Chains from nearby nodes, and routes conversations.
3. **Verification.** A share of every node's work is recomputed by independent
   verifiers on different cards. A node that cheats or runs a degraded model is
   caught and stops earning.
4. **Credits.** One price list per model: what users spend is what the GPUs
   earn, split across a Chain by the share of the model each card carries.

## Speed

Measuring. Numbers for single cards and Chains will be published here, with the
hardware, the line and the commands to reproduce them.

## Privacy of your prompts

Read this before sending anything personal.

- **The nodes that serve your request can read it.** The first stage of a Chain
  receives your prompt as tokens, the last one produces the answer, and a model
  that runs on a single card sees both. These are home PCs run by other people.
- **The stages in the middle see only hidden states**, not text. This is not a
  protection: the model weights are public, and published techniques turn
  hidden states back into the original tokens. A malicious node with enough
  effort can reconstruct your prompts.
- So **do not send passwords, personal data, or anything confidential**. Use
  PubliCompute for what you would not mind a stranger reading.
- Verification catches nodes that compute wrong results, not nodes that read.

As the network grows, we plan to look at schemes such as **Cascade**, which
split the computation of every token across many nodes so that no single node
sees all of it. They need many more nodes and much more bandwidth than a young
network has, so for now they remain a goal.

## Motivations

- Many perfectly capable gaming GPUs are held back by their VRAM when running
  local LLMs. We want to bring their owners into the realm of local(-ish)
  inference.
- Before someone else decides we shouldn't have any computing power of our own,
  we should get organized.
- **Running open models should not require a datacenter or a subscription. AI
  is getting too powerful to be a closed product. Demand democratization.**

## AI full disclosure

This software is developed with **strong assistance from AI coding agents**,
with humans leading the ideas, the design, the testing and the debugging. We
say this openly because it shaped how the project was built. If you are not
happy with AI-developed code, this software is not for you.

The acknowledgements below are just as important: none of this would exist
without llama.cpp and GGML, largely written by hand.

## Acknowledgements

### llama.cpp and GGML

The engine is a **small fork of [llama.cpp](https://github.com/ggml-org/llama.cpp)**:
a set of patches in `engine/patches` that let a process load and run only a
range of layers of a GGUF model, and pass the hidden state to the next stage.
Everything else (the kernels, the quantization formats, the GGUF ecosystem) is
the work of Georgi Gerganov and the llama.cpp contributors. We are grateful,
and the patches keep their MIT license.

### antirez and DwarfStar

The idea for PubliCompute came from studying
[DwarfStar](https://github.com/antirez/ds4), the inference engine by antirez.
Thanks to him for his indirect contribution to the project.

## Detailed guides

Coming soon: Install, The Chain, Credits, Verification, Privacy, Building from
source.

## What is open source

Everything you install is open source: the app, the agent, the Chain stage, the
installer and the engine patches. The coordinator and the scheduler run on our
servers and are not part of this repository; we are thinking about
open-sourcing them in the future.

## Contributing

Contributions are welcome, especially bug reports from real hardware and real
connections.

- Please open an issue before a pull request that adds a feature, so we can
  discuss it first.
- Send pull requests to the `dev` branch, not `main`.
- Follow the existing style: simple code, comments that explain why, no
  abstractions without a reason.
- By contributing you agree that your work is released under the license of
  the files you change.

Read [CONTRIBUTING.md](CONTRIBUTING.md) before sending a pull request.

## License

The app, the agent, the Chain stage and the installer are released under the
[Apache License 2.0](LICENSE). The engine patches in `engine/patches` are
released under the MIT license, like llama.cpp. "PubliCompute" and the logo are
not covered by these licenses.
