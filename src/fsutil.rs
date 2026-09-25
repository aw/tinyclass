use anyhow::{Context, Result};
use serde::Serialize;
use std::fs;
use std::path::Path;

pub fn write_json_atomically<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let parent = path.parent().context("path has no parent directory")?;
    fs::create_dir_all(parent)?;

    let temporary = path.with_extension("tmp");
    fs::write(&temporary, serde_json::to_string_pretty(value)?)?;
    fs::rename(&temporary, path).with_context(|| format!("could not write {}", path.display()))?;
    Ok(())
}
