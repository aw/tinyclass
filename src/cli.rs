//! The command line: what `tinyclass` understands and what each command runs.

use anyhow::Result;
use llama_cpp_2::LlamaBackendDeviceType;
use std::io::{self, BufRead, Write};
use usage::{Cli, Subcommands};

use crate::completions;
use crate::config::Config;
use crate::decision::{self, Decision};
use crate::device::{self, Choice};
use crate::model::{self, Model};
use crate::paths;

/// A local decision model out of Qwen3: pick one of N choices with a probability for each
#[derive(Cli)]
#[usage(bin = "tinyclass", version, unknown_flags = "error", completion)]
pub struct Cli {
    #[usage(subcommand)]
    pub command: Command,
}

#[derive(Subcommands)]
pub enum Command {
    /// Choose, pull, and list models
    Model {
        #[usage(subcommand)]
        command: ModelCommand,
    },
    /// Choose and list the CPU and GPUs the model can run on
    Device {
        #[usage(subcommand)]
        command: DeviceCommand,
    },
    /// Decide once: classify an input into one of the choices
    Decide {
        /// The text to decide about
        input: String,
        /// The choices, at least two
        choices: Vec<String>,
        /// What the model is told to do with the input
        #[usage(long, default = "Choose one option.")]
        instruction: String,
        /// Where to run: auto, cpu, gpu, or gpu:N; defaults to the set device
        #[usage(long)]
        device: Option<String>,
        /// Print the decision as JSON
        #[usage(long)]
        json: bool,
    },
    /// Ask a yes/no question: how likely the statement is true, from 0 to 1
    Noul {
        /// The statement to judge
        statement: String,
        /// What the model is told to do with the statement
        #[usage(long, default = "Is the statement true?")]
        instruction: String,
        /// Where to run: auto, cpu, gpu, or gpu:N; defaults to the set device
        #[usage(long)]
        device: Option<String>,
        /// Print the probability as JSON
        #[usage(long)]
        json: bool,
    },
    /// Keep the model loaded and decide about every line typed on stdin
    Play {
        /// The choices, at least two
        choices: Vec<String>,
        /// What the model is told to do with each line
        #[usage(long, default = "Choose one option.")]
        instruction: String,
        /// Where to run: auto, cpu, gpu, or gpu:N; defaults to the set device
        #[usage(long)]
        device: Option<String>,
    },
    /// Print or install the shell completion script
    ShellCompletion {
        #[usage(subcommand)]
        command: ShellCompletionCommand,
    },
}

#[derive(Subcommands)]
pub enum ModelCommand {
    /// List the models tinyclass knows, marking the set one and the pulled ones
    List,
    /// Make a model the one that answers
    Set {
        /// A name from `tinyclass model list`
        name: String,
    },
    /// Fetch the set model, or a named one, from Hugging Face
    Pull {
        /// A name from `tinyclass model list`; defaults to the set model
        name: Option<String>,
    },
}

#[derive(Subcommands)]
pub enum DeviceCommand {
    /// List the devices llama.cpp sees, with the index `gpu:N` refers to
    List,
    /// Make a device the one the model runs on: auto, cpu, gpu, or gpu:N
    Set {
        /// auto, cpu, gpu, or gpu:N from `tinyclass device list`
        device: String,
    },
}

#[derive(Subcommands)]
pub enum ShellCompletionCommand {
    /// Write the completion script to stdout
    Print {
        /// The shell to generate for
        #[usage(
            choices("bash", "elvish", "zsh", "fish", "nu", "powershell"),
            choices_strict = false
        )]
        shell: String,
    },
    /// Write the completion script where the shell looks for it
    Install {
        /// The shell to install for
        #[usage(
            choices("bash", "elvish", "zsh", "fish", "nu", "powershell"),
            choices_strict = false
        )]
        shell: String,
    },
}

pub fn main() {
    if let Err(error) = run(Cli::parse()) {
        eprintln!("error: {error:#}");
        std::process::exit(1);
    }
}

pub fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Command::Model { command } => match command {
            ModelCommand::List => list_models(),
            ModelCommand::Set { name } => set_model(&name),
            ModelCommand::Pull { name } => pull_model(name.as_deref()),
        },
        Command::Device { command } => match command {
            DeviceCommand::List => list_devices(),
            DeviceCommand::Set { device } => set_device(&device),
        },
        Command::Decide { input, choices, instruction, device, json } => {
            decide(&input, &choices, &instruction, device.as_deref(), json)
        }
        Command::Noul { statement, instruction, device, json } => {
            noul(&statement, &instruction, device.as_deref(), json)
        }
        Command::Play { choices, instruction, device } => play(&choices, &instruction, device.as_deref()),
        Command::ShellCompletion { command } => match command {
            ShellCompletionCommand::Print { shell } => completions::print(&shell),
            ShellCompletionCommand::Install { shell } => completions::install(&shell),
        },
    }
}

fn list_models() -> Result<()> {
    let current = Config::load()?.model;
    for model in model::CATALOG {
        let mut marker = " ";
        if current.as_deref() == Some(model.name) {
            marker = "*";
        }
        let mut state = "not pulled";
        if model.available() {
            state = "pulled";
        }
        println!("{marker} {:<12} {:>7}  {state}", model.name, model.size);
    }
    Ok(())
}

fn set_model(name: &str) -> Result<()> {
    let model = Model::find(name)?;
    let mut config = Config::load()?;
    config.model = Some(model.name.to_string());
    config.save()?;

    if model.available() {
        println!("{} is set.", model.name);
    } else {
        println!("{} is set — fetch it with `{} model pull`.", model.name, paths::invoked_as());
    }
    Ok(())
}

fn pull_model(name: Option<&str>) -> Result<()> {
    let model = match name {
        Some(name) => Model::find(name)?,
        None => Model::current()?,
    };

    if model.available() {
        println!("{} is already pulled.", model.name);
    } else {
        model.pull()?;
        println!("{} is ready.", model.name);
    }
    Ok(())
}

fn list_devices() -> Result<()> {
    model::backend()?;
    let choice = Choice::current(None)?;
    let chosen = choice.place()?.device.map(|it| it.index);
    println!("Set: {choice}");
    for device in device::all() {
        let mut marker = " ";
        if chosen == Some(device.index) || (chosen.is_none() && device.device_type == LlamaBackendDeviceType::Cpu) {
            marker = "*";
        }
        let mut memory = String::new();
        if device.memory_total > 0 {
            memory = format!("{:.1} GB", device.memory_total as f64 / 1e9);
        }
        println!("{marker} {:<3} {:<8} {:<40} {memory}", device.index, device.backend, device.description);
    }
    Ok(())
}

fn set_device(text: &str) -> Result<()> {
    let choice = Choice::parse(text)?;
    model::backend()?;
    let placement = choice.place()?;

    let mut config = Config::load()?;
    config.device = Some(choice.to_string());
    config.save()?;
    println!("{choice} is set — the model runs on {placement}.");
    Ok(())
}

fn decide(input: &str, choices: &[String], instruction: &str, device: Option<&str>, json: bool) -> Result<()> {
    let model = Model::current()?.load(Choice::current(device)?)?;
    let decision = decision::decide(&model, instruction, input, choices)?;
    if json {
        print_json(&decision)?;
    } else {
        print_decision(&decision);
    }
    Ok(())
}

fn noul(statement: &str, instruction: &str, device: Option<&str>, json: bool) -> Result<()> {
    let model = Model::current()?.load(Choice::current(device)?)?;
    let probability = decision::noul(&model, instruction, statement)?;
    if json {
        println!("{}", serde_json::json!({ "probability": probability }));
    } else {
        println!("{probability:.3}");
    }
    Ok(())
}

fn play(choices: &[String], instruction: &str, device: Option<&str>) -> Result<()> {
    let current = Model::current()?;
    eprintln!("Loading {}…", current.name);
    let model = current.load(Choice::current(device)?)?;
    eprintln!("Running on {}.", model.placement);
    eprintln!("Deciding between {} for every line; Ctrl-D ends.", choices.join(", "));

    let stdin = io::stdin();
    loop {
        eprint!("> ");
        io::stderr().flush()?;

        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            eprintln!();
            break;
        }
        let input = line.trim();
        if !input.is_empty() {
            print_decision(&decision::decide(&model, instruction, input, choices)?);
        }
    }
    Ok(())
}

fn print_decision(decision: &Decision) {
    let width = decision.scores.iter().map(|it| it.choice.len()).max().unwrap_or(0);
    for score in &decision.scores {
        let bar = "█".repeat((score.probability * 30.0).round() as usize);
        println!(
            "  {:<width$}  {:>5.1}%  {bar}",
            score.choice,
            score.probability * 100.0,
            width = width
        );
    }
    println!("→ {}", decision.chosen().choice);
}

fn print_json(decision: &Decision) -> Result<()> {
    let scores: Vec<serde_json::Value> = decision
        .scores
        .iter()
        .map(|it| {
            serde_json::json!({
                "choice": it.choice,
                "logit": it.logit,
                "logprob": it.logprob,
                "probability": it.probability,
            })
        })
        .collect();
    let value = serde_json::json!({
        "chosen": decision.chosen().choice,
        "scores": scores,
    });
    println!("{}", serde_json::to_string_pretty(&value)?);
    Ok(())
}
