//! Unified cache management for melops
//!
//! Cache directory structure:
//! ```text
//! <system_cache_dir>/melops/
//! ├── index.json          # Multi-level cache (pages → audio → SRT)
//! ├── models/             # ONNX models from melops-export
//! └── ort/                # OpenVINO execution provider cache
//! ```

use crate::cli::CacheConfig;
use clap::{Args, Subcommand, ValueEnum};
use eyre::{OptionExt, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

/// Index filename
pub const INDEX_FILENAME: &str = "index.json";

/// Models directory name
pub const MODELS_DIR: &str = "models";

/// ONNX Runtime cache directory name
pub const ORT_DIR: &str = "ort";

/// Get default cache directory
///
/// Returns system cache directory with "melops" subdirectory.
/// This is the unified cache root for all melops components.
pub fn default_dir() -> Result<PathBuf> {
    dirs::cache_dir()
        .map(|d| d.join("melops"))
        .ok_or_eyre("failed to determine cache directory")
}

/// CLI arguments for cache management
#[derive(Debug, Args)]
pub struct CacheCommand {
    /// Cache directory (default: system cache directory)
    #[arg(long)]
    pub cache_dir: Option<PathBuf>,

    #[command(subcommand)]
    pub command: CacheSubcommand,
}

#[derive(Debug, Subcommand)]
pub enum CacheSubcommand {
    /// Show cache directory path
    Dir,

    /// Clean cache directory
    Clean {
        /// Cache type to clean
        #[arg(value_enum, default_value_t = CacheType::default())]
        cache_type: CacheType,
    },
}

#[derive(Debug, Clone, Copy, Default, ValueEnum)]
pub enum CacheType {
    /// All cache (entire melops directory)
    All,
    /// Index file (index.json) - tracks file mappings
    #[default]
    Index,
    /// Exported models
    Models,
    /// ONNX Runtime cache
    Ort,
}

/// Three-level cache structure
///
/// Level 1: page_url → Vec<resolved_youtube_url>
/// Level 2: resolved_youtube_url → audio_path
/// Level 3: audio_path_key → srt_path
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct CacheData {
    pub pages: HashMap<String, Vec<String>>,
    pub audio: HashMap<String, PathBuf>,
    pub srt: HashMap<String, PathBuf>,
}

/// Web cache wrapper with persistent storage and refresh control
pub struct Cache {
    pub data: CacheData,
    pub cache_path: PathBuf,
    pub refresh_pages: bool,
    pub refresh_audio: bool,
    pub refresh_srt: bool,
}

impl Cache {
    /// Load cache from configuration
    pub fn load(config: CacheConfig) -> Result<Self> {
        let dir = match config.cache_dir {
            Some(dir) => dir,
            None => default_dir()?,
        };

        let cache_path = dir.join(INDEX_FILENAME);
        let data = if cache_path.exists() {
            let content = std::fs::read_to_string(&cache_path)?;
            serde_json::from_str(&content)?
        } else {
            CacheData::default()
        };

        Ok(Self {
            data,
            cache_path,
            refresh_pages: config.refresh_pages,
            refresh_audio: config.refresh_audio,
            refresh_srt: config.refresh_srt,
        })
    }

    /// Get cached YouTube URLs for a page (respects refresh_pages flag)
    pub fn get_page_urls(&self, page_url: &str) -> Option<&[String]> {
        if self.refresh_pages {
            return None;
        }
        self.data.pages.get(page_url).map(|v| v.as_slice())
    }

    /// Set cached YouTube URLs for a page
    pub fn set_page_urls(&mut self, page_url: String, urls: Vec<String>) {
        self.data.pages.insert(page_url, urls);
    }

    /// Get cached audio path for a URL (respects refresh_audio flag)
    pub fn get_audio_path(&self, key: &str) -> Option<&Path> {
        if self.refresh_audio {
            return None;
        }
        self.data.audio.get(key).map(|p| p.as_path())
    }

    /// Set cached audio path
    pub fn set_audio_path(&mut self, key: String, path: PathBuf) -> Result<()> {
        let canonical_path = path.canonicalize()?;
        self.data.audio.insert(key, canonical_path);
        Ok(())
    }

    /// Get cached SRT path for a file path (respects refresh_srt flag)
    pub fn get_srt_path(&self, key: &str) -> Option<&Path> {
        if self.refresh_srt {
            return None;
        }
        let canonical_key = PathBuf::from(key)
            .canonicalize()
            .ok()?
            .to_string_lossy()
            .to_string();
        self.data.srt.get(&canonical_key).map(|p| p.as_path())
    }

    /// Set cached SRT path
    pub fn set_srt_path(&mut self, key: String, path: PathBuf) -> Result<()> {
        let canonical_key = PathBuf::from(key)
            .canonicalize()?
            .to_string_lossy()
            .to_string();
        let canonical_path = path.canonicalize()?;
        self.data.srt.insert(canonical_key, canonical_path);
        Ok(())
    }

    /// Save cache to disk
    fn save(&self) -> Result<()> {
        if let Some(parent) = self.cache_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let content = serde_json::to_string_pretty(&self.data)?;
        std::fs::write(&self.cache_path, content)?;
        Ok(())
    }
}

impl Drop for Cache {
    fn drop(&mut self) {
        if let Err(e) = self.save() {
            tracing::error!(error = %e, "failed to save cache");
        }
    }
}

/// Run cache management command
pub fn run(cmd: CacheCommand) -> Result<()> {
    let dir = match cmd.cache_dir {
        Some(dir) => dir,
        None => default_dir()?,
    };

    match cmd.command {
        CacheSubcommand::Dir => {
            println!("{}", dir.display());
            Ok(())
        }
        CacheSubcommand::Clean { cache_type } => clean(dir, cache_type),
    }
}

fn clean(dir: PathBuf, cache_type: CacheType) -> Result<()> {
    let (result, target) = match cache_type {
        CacheType::All => (std::fs::remove_dir_all(&dir), dir),
        CacheType::Index => {
            let target = dir.join(INDEX_FILENAME);
            (std::fs::remove_file(&target), target)
        }
        CacheType::Models => {
            let target = dir.join(MODELS_DIR);
            (std::fs::remove_dir_all(&target), target)
        }
        CacheType::Ort => {
            let target = dir.join(ORT_DIR);
            (std::fs::remove_dir_all(&target), target)
        }
    };

    match result {
        Ok(()) => {
            println!("removed cache at {}", target.display());
            Ok(())
        }
        Err(e) if e.kind() == ErrorKind::NotFound => {
            println!("cache is already empty at {}", target.display());
            Ok(())
        }
        Err(e) => Err(e.into()),
    }
}
