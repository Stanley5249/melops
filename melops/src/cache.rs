//! Unified cache management for melops
//!
//! Cache directory structure:
//! ```text
//! <system_cache_dir>/melops/
//! ├── cache.json          # Multi-level cache (pages → audio → SRT)
//! ├── models/             # ONNX models from melops-export
//! └── ort/                # OpenVINO execution provider cache
//! ```

use crate::cli::CacheConfig;
use eyre::{OptionExt, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Cache filename
pub const CACHE_FILENAME: &str = "cache.json";

/// Get default cache directory
///
/// Returns system cache directory with "melops" subdirectory.
/// This is the unified cache root for all melops components.
pub fn default_dir() -> Result<PathBuf> {
    dirs::cache_dir()
        .map(|d| d.join("melops"))
        .ok_or_eyre("failed to determine cache directory")
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

        let cache_path = dir.join(CACHE_FILENAME);
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
