//! Where the model runs: `auto` takes a GPU when ggml sees one and the CPU
//! otherwise, `cpu` and `gpu` force one, and `gpu:N` picks one of several by
//! the index `device list` prints. The choice is saved next to the model
//! choice, and a `--device` flag overrides it for one run.

use anyhow::{Result, bail};
use llama_cpp_2::{LlamaBackendDevice, LlamaBackendDeviceType, list_llama_ggml_backend_devices};
use std::fmt;

use crate::config::Config;
use crate::paths;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Choice {
    Auto,
    Cpu,
    Gpu(Option<usize>),
}

/// What a choice came to, once the machine's devices were consulted.
pub struct Placement {
    pub device: Option<LlamaBackendDevice>,
}

impl Choice {
    pub fn parse(text: &str) -> Result<Choice> {
        match text {
            "auto" => Ok(Choice::Auto),
            "cpu" => Ok(Choice::Cpu),
            "gpu" => Ok(Choice::Gpu(None)),
            _ => match text.strip_prefix("gpu:").and_then(|it| it.parse().ok()) {
                Some(index) => Ok(Choice::Gpu(Some(index))),
                None => bail!("unknown device '{text}' — use auto, cpu, gpu, or gpu:N"),
            },
        }
    }

    /// The flag if given, else the saved choice, else auto.
    pub fn current(flag: Option<&str>) -> Result<Choice> {
        match flag {
            Some(text) => Choice::parse(text),
            None => match Config::load()?.device {
                Some(text) => Choice::parse(&text),
                None => Ok(Choice::Auto),
            },
        }
    }

    pub fn place(self) -> Result<Placement> {
        let gpus: Vec<LlamaBackendDevice> = gpus();
        let device = match self {
            Choice::Auto => gpus.into_iter().next(),
            Choice::Cpu => None,
            Choice::Gpu(None) => match gpus.into_iter().next() {
                Some(gpu) => Some(gpu),
                None => bail!("no GPU found — see `{} device list`", paths::invoked_as()),
            },
            Choice::Gpu(Some(index)) => match gpus.into_iter().find(|it| it.index == index) {
                Some(gpu) => Some(gpu),
                None => bail!("no GPU has index {index} — see `{} device list`", paths::invoked_as()),
            },
        };
        Ok(Placement { device })
    }
}

impl fmt::Display for Choice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Choice::Auto => write!(f, "auto"),
            Choice::Cpu => write!(f, "cpu"),
            Choice::Gpu(None) => write!(f, "gpu"),
            Choice::Gpu(Some(index)) => write!(f, "gpu:{index}"),
        }
    }
}

impl fmt::Display for Placement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.device {
            Some(device) => write!(f, "{} ({})", device.description, device.backend),
            None => write!(f, "CPU"),
        }
    }
}

pub fn all() -> Vec<LlamaBackendDevice> {
    list_llama_ggml_backend_devices()
}

fn gpus() -> Vec<LlamaBackendDevice> {
    all()
        .into_iter()
        .filter(|it| {
            matches!(
                it.device_type,
                LlamaBackendDeviceType::Gpu | LlamaBackendDeviceType::IntegratedGpu
            )
        })
        .collect()
}
