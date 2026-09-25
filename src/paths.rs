use std::env;
use std::path::PathBuf;

use crate::settings;

/// A program built on tinyclass that keeps models of its own says where in
/// `settings`.
pub fn data_dir() -> PathBuf {
    if let Some(dir) = &settings::settings().data_dir {
        dir.clone()
    } else if let Some(dir) = env::var_os("XDG_DATA_HOME").filter(|it| !it.is_empty()) {
        PathBuf::from(dir).join("tinyclass")
    } else {
        home().join(".local/share/tinyclass")
    }
}

pub fn config_path() -> PathBuf {
    data_dir().join("config.json")
}

pub fn models_dir() -> PathBuf {
    data_dir().join("models")
}

/// What someone types to reach these commands, for messages that tell them
/// what to run next.
pub fn invoked_as() -> String {
    settings::settings().invoked_as.clone().unwrap_or_else(|| "tinyclass".to_string())
}

fn home() -> PathBuf {
    dirs::home_dir().expect("could not determine the home directory")
}
