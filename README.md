# tinyclass

JEV-like classification from the terminal.

```bash
tinyclass decide "Payroll asks for your password on a non-company sign-in page." \
           Legitimate Spam Phishing
#   Legitimate    1.7%  █
#   Spam          0.9%
#   Phishing     97.4%  █████████████████████████████
# → Phishing
```

## Installing

With [mise](https://mise.jdx.dev):

```bash
mise use -g github:monorkin/tinyclass                      # install it and put tinyclass on your PATH
mise exec github:monorkin/tinyclass -- tinyclass --help    # or run it once without installing
```

On [Omarchy](https://omarchy.org):

```bash
omarchy-mise-install github:monorkin/tinyclass tinyclass
```

From source:

```bash
cargo install --path .
```

Then, either way:

```bash
tinyclass shell-completion install zsh
```

The release binaries carry llama.cpp inside them, so there's nothing else to
install. They need glibc 2.39 or newer and the Vulkan loader
(`libvulkan.so.1`), which any Linux with GPU drivers already has.

## Usage

```bash
tinyclass decide "Payroll asks for your password on a non-company sign-in page." \
           Legitimate Spam Phishing
#   Legitimate    1.7%  █
#   Spam          0.9%
#   Phishing     97.4%  █████████████████████████████
# → Phishing


tinyclass decide "Best purchase ever" Positive Negative \
           --instruction "Classify the sentiment of the review." --json
# {
#   "chosen": "Positive",
#   "scores": [
#     {
#       "choice": "Positive",
#       "logit": 36.610538482666016,
#       "logprob": -0.0000152587890625,
#       "probability": 0.9999847412109375
#     },
#     {
#       "choice": "Negative",
#       "logit": 25.53368377685547,
#       "logprob": -11.07686996459961,
#       "probability": 0.000015465946489712223
#     }
#   ]
# }


tinyclass play Positive Negative # keeps the model loaded, decides per line typed
# Loading qwen3-1.7b…
# Deciding between Positive, Negative for every line; Ctrl-D ends.
# > This is great!
#   Positive  100.0%  ██████████████████████████████
#   Negative    0.0%
# → Positive
# > This sucks!
#   Positive    0.0%
#   Negative  100.0%  ██████████████████████████████
# → Negative
# > The moon is made of cheese
#   Positive   54.9%  ████████████████
#   Negative   45.1%  ██████████████
# → Positive

tinyclass noul "The moon is made of cheese."   # Simple YES/NO questions: how likely it's true
# 0.125

tinyclass model list           # the Qwen3 sizes tinyclass knows
tinyclass model set qwen3-1.7b # pick one
tinyclass model pull           # fetch it from Hugging Face

tinyclass device list          # the CPU and GPUs llama.cpp sees
tinyclass device set gpu       # auto (the default), cpu, gpu, or gpu:N
tinyclass decide "…" A B --device cpu   # override for one run
```

Models live under `~/.local/share/tinyclass/models`, or `$XDG_DATA_HOME/tinyclass`.

`qwen3-0.6b` is enough for `decide` with clear-cut choices, but it leans hard
toward whichever option is listed first and says yes to nearly everything, so
`noul` needs at least `qwen3-1.7b`.

## Acknowledgements

The idea from [Jev in 25 lines of Python](https://www.nobodywho.ai/posts/jev-in-25-lines/): ask a chat model a multiple-choice question, stop after the prompt, and read the logits of the
answer letters instead of sampling. The softmax over just those
letters is the decision, with a probability for each choice.

Inference is done via [llama.cpp](https://github.com/ggml-org/llama.cpp).

## Building

```bash
mise trust
mise install
cargo build --release # → target/release/tinyclass
```

Always build and run with `--release`. Without it, the inference library takes about 200x longer
per given answer.

When developing use `cargo run --release -- decide …` instead of plain `cargo run`.

llama.cpp is compiled in, so the build needs cmake, a C++ compiler, and for
the default Vulkan backend the Vulkan and SPIR-V headers and `glslc`. On
Arch that's `cmake clang vulkan-headers spirv-headers shaderc`; on Debian
and Ubuntu `cmake g++ libvulkan-dev spirv-headers glslc`. `--no-default-features` builds a CPU-only binary with
none of the Vulkan requirements, and `--features cuda`, `rocm`, or `metal`
swap the GPU backend.

## Benchmark

`script/benchmark` times every pulled model on the phishing example above and
prints a table like the ones below. Load is what a one-off `decide` pays
before it answers; `play` pays it once. `DEVICE=cpu script/benchmark` picks
the device the same way `--device` does.

```
CPU: AMD Ryzen 9 9950X3D 16-Core Processor
Device: auto
Running on AMD Radeon RX 7900 XTX (RADV NAVI31) (Vulkan).
20 answers of "Payroll asks for your password on a non-company sign-in page." between Legitimate Spam Phishing

| Model      | Load   | Per answer | Answers/s |
|------------|--------|------------|-----------|
| qwen3-0.6b | 0.47 s | 14 ms      | 70.3      |
| qwen3-1.7b | 0.64 s | 17 ms      | 58.0      |
```

```
Device: cpu
Running on CPU.

| Model      | Load   | Per answer | Answers/s |
|------------|--------|------------|-----------|
| qwen3-0.6b | 0.41 s | 35 ms      | 28.6      |
| qwen3-1.7b | 0.47 s | 85 ms      | 11.8      |
```

Most of the load time is the Vulkan driver coming up, about 0.35 s here,
which a CPU-only build (`--no-default-features`) skips.

## License

tinyclass is released under the MIT License, see [LICENSE](LICENSE) for details.
