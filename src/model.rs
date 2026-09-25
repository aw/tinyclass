//! The Qwen3 models tinyclass knows how to run: a GGUF from Qwen's quantized repo
//! and the tokenizer from the matching base repo, fetched once into the data
//! folder. Nothing downloads on the answering path.

use anyhow::{Context, Result, anyhow, bail};
use candle_core::Device;
use candle_core::quantized::gguf_file;
use candle_transformers::models::quantized_qwen3::ModelWeights;
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::PathBuf;
use tokenizers::Tokenizer;

use crate::config::Config;
use crate::paths;

pub struct Model {
    pub name: &'static str,
    pub gguf_repo: &'static str,
    pub gguf_file: &'static str,
    pub tokenizer_repo: &'static str,
    pub size: &'static str,
}

pub const CATALOG: &[Model] = &[
    Model {
        name: "qwen3-0.6b",
        gguf_repo: "Qwen/Qwen3-0.6B-GGUF",
        gguf_file: "Qwen3-0.6B-Q8_0.gguf",
        tokenizer_repo: "Qwen/Qwen3-0.6B",
        size: "640 MB",
    },
    Model {
        name: "qwen3-1.7b",
        gguf_repo: "Qwen/Qwen3-1.7B-GGUF",
        gguf_file: "Qwen3-1.7B-Q8_0.gguf",
        tokenizer_repo: "Qwen/Qwen3-1.7B",
        size: "1.8 GB",
    },
    Model {
        name: "qwen3-4b",
        gguf_repo: "Qwen/Qwen3-4B-GGUF",
        gguf_file: "Qwen3-4B-Q8_0.gguf",
        tokenizer_repo: "Qwen/Qwen3-4B",
        size: "4.3 GB",
    },
    Model {
        name: "qwen3-8b",
        gguf_repo: "Qwen/Qwen3-8B-GGUF",
        gguf_file: "Qwen3-8B-Q8_0.gguf",
        tokenizer_repo: "Qwen/Qwen3-8B",
        size: "8.7 GB",
    },
];

pub struct Loaded {
    pub tokenizer: Tokenizer,
    pub weights: ModelWeights,
    pub device: Device,
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
        self.gguf_path().exists() && self.tokenizer_path().exists()
    }

    pub fn pull(&self) -> Result<()> {
        fs::create_dir_all(self.directory())
            .with_context(|| format!("could not create {}", self.directory().display()))?;

        self.fetch(self.tokenizer_repo, "tokenizer.json", &self.tokenizer_path())?;
        self.fetch(self.gguf_repo, self.gguf_file, &self.gguf_path())?;
        Ok(())
    }

    pub fn load(&self) -> Result<Loaded> {
        if !self.available() {
            bail!("{} isn't on disk yet — fetch it with `{} model pull`", self.name, paths::invoked_as());
        }

        let tokenizer = Tokenizer::from_file(self.tokenizer_path()).map_err(|error| {
            anyhow!("could not load {}: {error}", self.tokenizer_path().display())
        })?;

        let device = Device::Cpu;
        let mut file = File::open(self.gguf_path())
            .with_context(|| format!("could not open {}", self.gguf_path().display()))?;
        let content = gguf_file::Content::read(&mut file)
            .with_context(|| format!("could not read {}", self.gguf_path().display()))?;
        let weights = ModelWeights::from_gguf(content, &mut file, &device)
            .with_context(|| format!("could not load {}", self.gguf_path().display()))?;

        Ok(Loaded { tokenizer, weights, device })
    }

    fn fetch(&self, repo: &str, file: &str, destination: &PathBuf) -> Result<()> {
        if destination.exists() {
            return Ok(());
        }

        let url = format!("https://huggingface.co/{repo}/resolve/main/{file}");
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
                report_progress(file, received, total);
                reported = received;
            }
        }
        report_progress(file, received, total);
        eprintln!();

        fs::rename(&temporary, destination)?;
        Ok(())
    }

    fn gguf_path(&self) -> PathBuf {
        self.directory().join(self.gguf_file)
    }

    fn tokenizer_path(&self) -> PathBuf {
        self.directory().join("tokenizer.json")
    }
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
