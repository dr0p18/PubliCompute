<p align="center">
  <img src="logo.svg" alt="PubliCompute logo" width="160">
</p>

**PubliCompute** is a network of (smaller) gaming GPUs that run large language
models together. Connect to the network to use models bigger than your memory,
**for free**. When part of a Chain, you may also compute tokens for other users:
this extra work is paid in **credits**, which you spend on PubliCompute when your
PC is offline, through the chat.

A model that does not fit on one card is split into stages, each stage on a
different home GPU, forming a **Chain**. A 16 GB card cannot hold
DeepSeek V4 Flash, but four or five cards (in the same region) can. Every card
contributing to the computation of a model gets free use of it. As we see it,
this is an alternative to paying for an AI subscription, or a way to use it less.

## Supported hardware

* **NVIDIA GPUs with 8 GB or more**, Turing (RTX 20) and newer. 8-12 GB cards run
  the base model and verify other nodes' work; 16 GB cards pair up for the
  medium model; 24 GB and up run it alone and carry stages of the top model.
  These are general indications: the scheduler assigns work dynamically,
  following the network's needs.
* **Linux x86_64** today. Windows is currently a work in progress.
* **A decent home line** for the Chain: about 100 Mbit/s up and down, and under
  25 ms to the other nodes. Chains are (generally) built inside one region to
  keep latency down.

This project would not exist without **llama.cpp and GGML**: read the
acknowledgements below.

# So, what can I do with this?

* **Use models bigger than your PC.** Pick a model; when the scheduler puts your
  card in a Chain for it, you use it for free while you are in.
* **Farm credits.** If you don't pick a model, your card goes into FARM mode:
  the scheduler puts it wherever it's needed and every verified token earns
  credits.
* **Spend credits when you are offline**, on any model of the network, through
  the web chat.
* **Try it without a GPU**: everyone gets a small daily quota on the base model.

## Launch models

| Model | Quantization | Where it runs (generally) |
| --- | --- | --- |
| Qwen3.5-9B | Q4_K_M | one 8 GB card (also on the free quota) |
| Qwen3.6-35B-A3B | Q4 | one 24 GB card, or a Chain of smaller ones |
| DeepSeek V4 Flash | Q2 | a Chain of 4-5 cards with 24 GB or more |

The roster follows the best open-weight models for gaming hardware, and changes
when a better one arrives. Vote for the next one on
[publicompute.com](https://publicompute.com/#roster).

## How it works

1. **The app** runs on your PC as a small background service. It talks only to
   the coordinator and to the other nodes of your Chain. Connect and disconnect
   from the network whenever you want.
2. **The coordinator** assigns each card a whole model or a range of layers,
   builds Chains from nearby nodes, and routes conversations.
3. **Verification.** A share of every node's work is recomputed by independent
   verifiers, on different cards. A node that cheats or runs a degraded model
   is caught and stops earning.
4. **Credits.** One price list per model: what users spend is what the GPUs earn,
   split across a Chain by the share of the model each card carries. No coin,
   no speculation, no cash-out for now.

## Motivations

* There are many perfectly capable gaming GPUs held back by their VRAM when
  running local LLMs.
* Mixture-of-experts models generate fast even when split across cards: only
  the activations travel between stages, never the weights.
* **Running open models should not require a datacenter, or a subscription.**

# AI full disclosure

* This software is developed with **strong assistance from AI coding agents**,
  with humans leading the ideas, the design, the testing and the debugging. We
  say this openly because it shaped how the project was built. If you are not
  happy with AI-developed code, this software is not for you. The
  acknowledgement below is equally important: this would not exist without
  `llama.cpp` and GGML, largely written by hand.

## Acknowledgements to llama.cpp and GGML

The engine is a **small fork of [llama.cpp](https://github.com/ggml-org/llama.cpp)**:
a set of patches in `engine/patches` that let a process load and run only a
range of layers of a GGUF model, and pass the hidden state to the next stage.
Everything else, the kernels, the quantization formats, the GGUF ecosystem, is
the work of Georgi Gerganov and the llama.cpp contributors. We are grateful,
and the patches keep their MIT license.

## Acknowledgements to antirez and DwarfStar

PubliCompute started from [DwarfStar](https://github.com/antirez/ds4) by
Salvatore Sanfilippo (antirez): the idea that a few excellent open models can
run well on hardware people actually own, and the way of working, with code
kept small and readable, measurements before claims, and coding agents used
openly. DwarfStar is still our reference for what efficient inference on
consumer hardware looks like. Thank you.

## Status

Beta, by invitation. Things change fast, and the network is small: expect
rough edges, and tell us about them.

# Start here

```sh
curl -fsSLO https://github.com/dr0p18/PubliCompute/releases/download/v0.1.0/install.sh
sh install.sh
```

The installer asks for the **link code** that comes with your invitation,
verifies the package against `SHA256SUMS`, installs the agent as a system
service and connects your card. Run it again with `--relink` to move the card
to another account.

## Everyday use

```sh
publicompute status            # what your card is doing, credits, Chain
publicompute use <model>       # pick the model you want to use for free
publicompute free              # FARM mode: the scheduler decides
publicompute pause on          # leave the network (gaming, night...)
publicompute pause off         # come back
publicompute uninstall         # remove everything, keep or delete the models
```

## Speed

Measuring. Numbers for single cards and Chains will be published here, with
the hardware, the line and the commands to reproduce them.

## Privacy of your prompts

Read this before sending anything personal.

* **The nodes that serve your request can read it.** The first stage of a Chain
  receives your prompt as tokens, the last one produces the answer, and a model
  that runs on a single card sees both. These are home PCs run by other people.
* **The stages in the middle see only hidden states**, not text. This is not a
  protection: the model weights are public, and published techniques turn hidden
  states back into the original tokens. A malicious node with enough effort and
  dedication can reconstruct your prompts.
* So **do not send passwords, personal data, or anything confidential**. Use
  PubliCompute for what you would not mind a stranger reading.
* Verification catches nodes that compute wrong results, not nodes that read.

As the network grows, we plan to look at schemes such as **Cascade**, which
split the computation of every token across many nodes so that no single node
sees the whole of it. They need many more nodes and much more bandwidth than a
young network has, so they are a goal, not a feature.

## Detailed guides

*(to be written before the code is published)*: Install, The Chain, Credits,
Verification, Privacy, Building from source.

## What is open source

Everything you install is open source: the app, the agent, the Chain stage, the
installer, and the engine patches. The coordinator and the scheduler run on our
servers and are not part of this repository. We are thinking about
open-sourcing them in the future.

## Contributing

Contributions are welcome, especially bug reports from real hardware and lines.

* Please open an issue before a pull request that adds a feature, so we can
  discuss it first.
* Send pull requests to the `dev` branch, not `main`.
* Follow the existing style: simple code, comments that explain why, no
  abstractions without a reason.
* By contributing you agree that your work is released under the license of
  the files you change.

Read [CONTRIBUTING.md](CONTRIBUTING.md) before sending a pull request.

## License

The app, the agent, the Chain stage and the installer are released under the
[Apache License 2.0](LICENSE). The engine patches in `engine/patches` are released
under the MIT license, like llama.cpp. "PubliCompute" and the logo are not covered
by these licenses.
