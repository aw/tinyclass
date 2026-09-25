//! The one setting: which model answers.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;

use crate::fsutil;
use crate::paths;

#[derive(Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct Config {
    pub model: Option<String>,
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
