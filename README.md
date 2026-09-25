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

```bash
cargo install --path .
tinyclass shell-completion install zsh
```

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
```

Models live under `~/.local/share/tinyclass/models`, or `$XDG_DATA_HOME/tinyclass`.

`qwen3-0.6b` is enough for `decide` with clear-cut choices, but it leans hard
toward whichever option is listed first and says yes to nearly everything, so
`noul` needs at least `qwen3-1.7b`.

## Acknowledgements

The idea from [Jev in 25 lines of Python](https://www.nobodywho.ai/posts/jev-in-25-lines/): ask a chat model a multiple-choice question, stop after the prompt, and read the logits of the
answer letters instead of sampling. The softmax over just those
letters is the decision, with a probability for each choice.

## Building

```bash
mise trust
mise install
cargo build --release # → target/release/tinyclass
```

Always build and run with `--release`. Without it, the inference library takes about 200x longer
per given answer.

When developing use `cargo run --release -- decide …` instead of plain `cargo run`.

## Benchmark

`script/benchmark` times every pulled model on the phishing example above and
prints a table like this one. Load is what a one-off `decide` pays before it
answers; `play` pays it once.

```
CPU: AMD Ryzen 9 9950X3D 16-Core Processor
20 answers of "Payroll asks for your password on a non-company sign-in page." between Legitimate Spam Phishing

| Model      | Load   | Per answer | Answers/s |
|------------|--------|------------|-----------|
| qwen3-0.6b | 0.61 s | 136 ms     | 7.4       |
| qwen3-1.7b | 1.12 s | 300 ms     | 3.3       |
```

## Embedding

tinyclass is a library with a thin binary on top, the same shape as
[ax](https://github.com/monorkin/ax) and [katami](https://github.com/monorkin/katami).
A program that links it calls `tinyclass::decision::decide` or
`tinyclass::decision::noul` with a `Model::current().load()`, or reuses whole
commands through `tinyclass::cli::run`. Before it touches either, it says
where its models live and what it's called, once:

```rust
tinyclass::settings::configure(tinyclass::settings::Settings {
    data_dir: Some(my_data_dir.join("tinyclass")),
    invoked_as: Some("anna classify".to_string()),
});
```

## License

tinyclass is released under the MIT License, see [LICENSE](LICENSE) for details.
