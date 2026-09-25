//! The Qwen3 models tinyclass knows how to run: a GGUF from Qwen's quantized
//! repo, fetched once into the data folder. The tokenizer and chat template
//! travel inside the file. Nothing downloads on the answering path.

use anyhow::{Context, Result, anyhow, bail};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::{LogOptions, send_logs_to_tracing};
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::sync::OnceLock;

use crate::config::Config;
use crate::device::{Choice, Placement};
use crate::paths;

pub struct Model {
    pub name: &'static str,
    pub repo: &'static str,
    pub file: &'static str,
    pub size: &'static str,
}

pub const CATALOG: &[Model] = &[
    Model {
        name: "qwen3-0.6b",
        repo: "Qwen/Qwen3-0.6B-GGUF",
        file: "Qwen3-0.6B-Q8_0.gguf",
        size: "640 MB",
    },
    Model {
        name: "qwen3-1.7b",
        repo: "Qwen/Qwen3-1.7B-GGUF",
        file: "Qwen3-1.7B-Q8_0.gguf",
        size: "1.8 GB",
    },
    Model {
        name: "qwen3-4b",
        repo: "Qwen/Qwen3-4B-GGUF",
        file: "Qwen3-4B-Q8_0.gguf",
        size: "4.3 GB",
    },
    Model {
        name: "qwen3-8b",
        repo: "Qwen/Qwen3-8B-GGUF",
        file: "Qwen3-8B-Q8_0.gguf",
        size: "8.7 GB",
    },
];

pub struct Loaded {
    pub backend: &'static LlamaBackend,
    pub model: LlamaModel,
    pub placement: Placement,
}

impl Model {
    pub fn find(name: &str) -> Result<&'static Model> {
        CATALOG.iter().find(|it| it.name == name).ok_or_else(|| {
            anyhow!("no model named '{name}' — see `{} model list` for the known ones", paths::invoked_as())
        })
    }

    pub fn current() -> Result<&'static Model> {
        match Config::load()?.model {
            Some(name) => Model::find(&name),
            None => bail!("no model set — pick one with `{} model set <name>`", paths::invoked_as()),
        }
    }

    pub fn directory(&self) -> PathBuf {
        paths::models_dir().join(self.name)
    }

    pub fn available(&self) -> bool {
        self.path().exists()
    }

    pub fn pull(&self) -> Result<()> {
        fs::create_dir_all(self.directory())
            .with_context(|| format!("could not create {}", self.directory().display()))?;
        self.fetch()
    }

    pub fn load(&self, choice: Choice) -> Result<Loaded> {
        if !self.available() {
            bail!("{} isn't on disk yet — fetch it with `{} model pull`", self.name, paths::invoked_as());
        }

        let backend = backend()?;
        let placement = choice.place()?;
        let mut params = LlamaModelParams::default();
        if let Some(device) = &placement.device {
            params = params.with_n_gpu_layers(u32::MAX).with_devices(&[device.index])?;
        } else {
            params = params.with_n_gpu_layers(0);
        }

        let model = LlamaModel::load_from_file(backend, self.path(), &params)
            .with_context(|| format!("could not load {}", self.path().display()))?;
        Ok(Loaded { backend, model, placement })
    }

    fn fetch(&self) -> Result<()> {
        let destination = self.path();
        let url = format!("https://huggingface.co/{}/resolve/main/{}", self.repo, self.file);
        let mut response = ureq::get(&url)
            .call()
            .with_context(|| format!("could not download {url}"))?;
        let total = response
            .headers()
            .get("content-length")
            .and_then(|it| it.to_str().ok())
            .and_then(|it| it.parse::<u64>().ok());

        let temporary = destination.with_extension("tmp");
        let mut output = File::create(&temporary)?;
        let mut reader = response.body_mut().as_reader();
        let mut buffer = vec![0u8; 1 << 20];
        let mut received = 0u64;
        let mut reported = 0u64;
        loop {
            let read = reader.read(&mut buffer).with_context(|| format!("could not download {url}"))?;
            if read == 0 {
                break;
            }
            output.write_all(&buffer[..read])?;
            received += read as u64;
            if received - reported >= 10_000_000 {
                report_progress(self.file, received, total);
                reported = received;
            }
        }
        report_progress(self.file, received, total);
        eprintln!();

        fs::rename(&temporary, destination)?;
        Ok(())
    }

    fn path(&self) -> PathBuf {
        self.directory().join(self.file)
    }
}

/// llama.cpp is initialized once per process and narrates every load to
/// stderr unless told not to; the device list needs it up as well.
pub fn backend() -> Result<&'static LlamaBackend> {
    static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();
    if BACKEND.get().is_none() {
        send_logs_to_tracing(LogOptions::default().with_logs_enabled(false));
        let backend = LlamaBackend::init().context("could not initialize llama.cpp")?;
        BACKEND.set(backend).ok();
    }
    Ok(BACKEND.get().expect("the backend was just initialized"))
}

/// llama.cpp defaults to four threads. The matmuls are memory-bound, so a
/// second thread per core only adds contention: one per physical core is
/// the fast setting. Only Linux says how many threads share a core; Apple
/// silicon has no SMT, so counting every hardware thread is right there.
pub fn physical_cores() -> usize {
    let hardware_threads = std::thread::available_parallelism().map(|it| it.get()).unwrap_or(1);
    let threads_per_core = fs::read_to_string("/sys/devices/system/cpu/cpu0/topology/thread_siblings_list")
        .map(|it| it.trim().split(',').count())
        .unwrap_or(1);
    (hardware_threads / threads_per_core).max(1)
}

fn report_progress(file: &str, received: u64, total: Option<u64>) {
    let received_mb = received as f64 / 1e6;
    match total {
        Some(total) => {
            let percent = received * 100 / total;
            eprint!("\r{file}: {received_mb:.0} MB of {:.0} MB ({percent}%)", total as f64 / 1e6)
        }
        None => eprint!("\r{file}: {received_mb:.0} MB"),
    }
    io::stderr().flush().ok();
}
