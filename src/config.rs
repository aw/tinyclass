//! The settings: which model answers, and where it runs.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;

use crate::fsutil;
use crate::paths;

#[derive(Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct Config {
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device: Option<String>,
}

impl Config {
    pub fn load() -> Result<Config> {
        let path = paths::config_path();
        if path.exists() {
            let contents = fs::read_to_string(&path)?;
            serde_json::from_str(&contents)
                .with_context(|| format!("could not parse {}", path.display()))
        } else {
            Ok(Config::default())
        }
    }

    pub fn save(&self) -> Result<()> {
        fsutil::write_json_atomically(&paths::config_path(), self)
    }
}
