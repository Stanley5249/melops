//! Cache management command for melops.
//!
//! Cache directory structure:
//! ```text
//! <system_cache_dir>/melops/
//! ├── artifacts/
//! │   ├── pages/          # Web page URL → media URLs
//! │   ├── downloads/      # Media URL → audio paths
//! │   └── transcriptions/ # Audio data → transcription segments
//! ├── models/             # ONNX models from melops-export
//! └── ort/                # OpenVINO execution provider cache
//! ```

use clap::{Args, Subcommand, ValueEnum};
use eyre::{OptionExt, Result};
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use crate::cli::CacheArgs;

/// Cache operation strategy for resource access.
///
/// Controls how cached artifacts are accessed:
/// - `Use`: Read-only, fails if not cached
/// - `Auto`: Default behavior, computes if missing
/// - `Force`: Forces recomputation, ignores cache
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
pub enum CacheStrategy {
    /// Use existing cached value only (error if missing)
    Use,
    /// Use cache or compute if missing (default)
    #[default]
    Auto,
    /// Always recompute and update cache
    Force,
}

/// Artifact subdirectory names
const ARTIFACTS_DIR: &str = "artifacts";
const PAGES_DIR: &str = "pages";
const DOWNLOADS_DIR: &str = "downloads";
const TRANSCRIPTIONS_DIR: &str = "transcriptions";

/// Models directory name
pub const MODELS_DIR: &str = "models";

/// ONNX Runtime cache directory name
pub const ORT_DIR: &str = "ort";

/// Validated cache directory wrapper.
#[derive(Clone, Debug)]
pub struct CacheDir {
    base: PathBuf,
    artifacts: PathBuf,
    pages: PathBuf,
    downloads: PathBuf,
    transcriptions: PathBuf,
    models: PathBuf,
    ort: PathBuf,
}

impl CacheDir {
    /// Create cache directory from optional path
    ///
    /// When None, uses system cache directory.
    pub fn new(cache_dir: Option<PathBuf>) -> Result<Self> {
        let base = match cache_dir {
            Some(dir) => dir,
            None => dirs::cache_dir()
                .map(|d| d.join("melops"))
                .ok_or_eyre("failed to determine cache directory")?,
        };

        let artifacts = base.join(ARTIFACTS_DIR);
        let pages = artifacts.join(PAGES_DIR);
        let downloads = artifacts.join(DOWNLOADS_DIR);
        let transcriptions = artifacts.join(TRANSCRIPTIONS_DIR);
        let models = base.join(MODELS_DIR);
        let ort = base.join(ORT_DIR);

        Ok(CacheDir {
            base,
            artifacts,
            pages,
            downloads,
            transcriptions,
            models,
            ort,
        })
    }

    /// Get path to pages cache (web page URL → media URLs)
    pub fn pages(&self) -> &Path {
        &self.pages
    }

    /// Get path to downloads cache (media URL → audio paths)
    pub fn downloads(&self) -> &Path {
        &self.downloads
    }

    /// Get path to transcriptions cache (audio data → segments)
    pub fn transcriptions(&self) -> &Path {
        &self.transcriptions
    }

    /// Get path to artifacts cache directory
    pub fn artifacts(&self) -> &Path {
        &self.artifacts
    }

    /// Get path to models cache directory
    pub fn models(&self) -> &Path {
        &self.models
    }

    /// Get path to a model in the cache
    pub fn model(&self, model_id: &str) -> PathBuf {
        self.models.join(model_id)
    }

    /// Get path to ONNX Runtime cache
    pub fn ort(&self) -> &Path {
        &self.ort
    }
}

impl TryFrom<crate::cli::CacheArgs> for CacheDir {
    type Error = eyre::Report;

    fn try_from(args: crate::cli::CacheArgs) -> Result<Self> {
        Self::new(args.cache_dir)
    }
}

impl CacheDir {
    /// Get the base cache directory path
    pub fn base(&self) -> &Path {
        &self.base
    }
}

/// CLI arguments for cache management
#[derive(Args, Debug)]
pub struct CacheCommand {
    #[command(flatten)]
    pub cache_args: CacheArgs,

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

#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub enum CacheType {
    /// All cache (entire melops directory)
    All,
    /// All artifacts (pages, downloads, transcriptions)
    #[default]
    Artifacts,
    /// Web page URL cache
    Pages,
    /// Downloaded audio cache
    Downloads,
    /// Transcription results cache
    Transcriptions,
    /// Exported models
    Models,
    /// ONNX Runtime cache
    Ort,
}

impl CacheType {
    /// Get the path for this cache type
    pub fn path<'a>(&self, cache_dir: &'a CacheDir) -> &'a Path {
        match self {
            CacheType::All => cache_dir.base(),
            CacheType::Artifacts => cache_dir.artifacts(),
            CacheType::Pages => cache_dir.pages(),
            CacheType::Downloads => cache_dir.downloads(),
            CacheType::Transcriptions => cache_dir.transcriptions(),
            CacheType::Models => cache_dir.models(),
            CacheType::Ort => cache_dir.ort(),
        }
    }
}

/// Run cache management command
pub fn run(cmd: CacheCommand) -> Result<()> {
    let cache_dir = CacheDir::try_from(cmd.cache_args)?;

    match cmd.command {
        CacheSubcommand::Dir => {
            println!("{}", cache_dir.base().display());
            Ok(())
        }
        CacheSubcommand::Clean { cache_type } => clean(&cache_dir, cache_type),
    }
}

fn clean(cache_dir: &CacheDir, cache_type: CacheType) -> Result<()> {
    let target = cache_type.path(cache_dir);

    match std::fs::remove_dir_all(target) {
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
