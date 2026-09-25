//! What a program built on tinyclass decides for it.
//!
//! On its own, tinyclass keeps its models in the person's data folder and
//! tells them to run `tinyclass …`. A program that embeds it — one with a
//! data folder of its own, reached under a name of its own — says so here,
//! once, before it does anything else. Nothing is read from the environment
//! for this: variables set in a process to reach a crate it links leak into
//! everything that process starts.

use std::path::PathBuf;
use std::sync::OnceLock;

#[derive(Debug, Default, Clone)]
pub struct Settings {
    /// Where the models and the config live, instead of the person's
    /// tinyclass folder.
    pub data_dir: Option<PathBuf>,
    /// What someone types to reach these commands, for messages that tell
    /// them what to run next: `anna classify`, say.
    pub invoked_as: Option<String>,
}

static SETTINGS: OnceLock<Settings> = OnceLock::new();

/// Once, before anything reads a path. A second call is a programming error
/// and says so, rather than quietly leaving the first in place.
pub fn configure(settings: Settings) {
    if SETTINGS.set(settings).is_err() {
        panic!("tinyclass was configured twice");
    }
}

pub fn settings() -> &'static Settings {
    static ON_ITS_OWN: Settings = Settings { data_dir: None, invoked_as: None };
    SETTINGS.get().unwrap_or(&ON_ITS_OWN)
}
