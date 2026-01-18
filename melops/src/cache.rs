//! Cache management command for melops.
//!
//! Cache directory structure:
//! ```text
//! <system_cache_dir>/melops/
//! ├── index.json          # Artifact tracking (pages → audio → SRT)
//! ├── models/             # ONNX models from melops-export
//! └── ort/                # OpenVINO execution provider cache
//! ```

use clap::{Args, Subcommand, ValueEnum};
use eyre::{OptionExt, Result};
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

/// Validated cache directory wrapper.
#[derive(Clone, Debug)]
pub struct CacheDir(PathBuf);

impl CacheDir {
    /// Get the cache directory path
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.0
    }

    /// Join a path to the cache directory
    #[must_use]
    pub fn join(&self, path: impl AsRef<Path>) -> PathBuf {
        self.0.join(path)
    }
}

impl TryFrom<crate::cli::CacheArgs> for CacheDir {
    type Error = eyre::Report;

    fn try_from(args: crate::cli::CacheArgs) -> Result<Self> {
        let path = match args.cache_dir {
            Some(dir) => dir,
            None => default_dir()?,
        };
        Ok(CacheDir(path))
    }
}

/// CLI arguments for cache management
#[derive(Args, Debug)]
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

#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub enum CacheType {
    /// All cache (entire melops directory)
    All,
    /// Index file (index.json) - tracks artifact mappings
    #[default]
    Index,
    /// Exported models
    Models,
    /// ONNX Runtime cache
    Ort,
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
        CacheSubcommand::Clean { cache_type } => clean(&dir, cache_type),
    }
}

fn clean(dir: &Path, cache_type: CacheType) -> Result<()> {
    let (result, target) = match cache_type {
        CacheType::All => (std::fs::remove_dir_all(dir), dir.to_path_buf()),
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
