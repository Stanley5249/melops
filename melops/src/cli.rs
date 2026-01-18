//! CLI argument definitions using clap.
//! CLI argument parsing and command execution.

use crate::cache::CacheCommand;
use crate::cap::CapCommand;
use crate::dl::DlCommand;
use crate::index::CacheStrategy;
use crate::web::WebCommand;
use clap::{Args, Parser, Subcommand, ValueEnum};
use eyre::Result;
use melops_asr::chunk::{ChunkConfig, DEFAULT_CHUNK_DURATION, DEFAULT_CHUNK_OVERLAP};
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(name = "mel")]
#[command(about = "Audio captioning and download tools")]
#[command(version)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    /// Generate captions from audio file to SRT subtitles
    Cap(CapCommand),

    /// Download and generate captions from audio URL
    Dl(DlCommand),

    /// Scrape web page for YouTube URLs and batch process
    Web(WebCommand),

    /// Manage cache directory
    Cache(CacheCommand),
}

/// Model source type selection.
#[derive(Debug, Default, Clone, Copy, ValueEnum)]
pub enum ModelSource {
    /// Automatically detect: try cache first, fall back to HuggingFace Hub
    #[default]
    Auto,
    /// Load from local filesystem path
    Path,
    /// Load from HuggingFace cache only (error if not cached)
    Cache,
    /// Download from HuggingFace Hub (may use cache)
    Api,
}

/// CLI arguments for model loading configuration.
#[derive(Args, Debug)]
pub struct ModelArgs {
    /// Model repository ID or local path
    #[arg(long)]
    pub model_id: String,

    /// Model source type (auto, path, cache, api)
    #[arg(long, value_enum, default_value_t = ModelSource::default())]
    pub model_source: ModelSource,
}

/// CLI arguments for chunk configuration.
#[derive(Args, Clone, Copy, Debug, Default)]
pub struct ChunkArgs {
    /// Chunk duration in seconds for long audio
    #[arg(long, default_value_t = DEFAULT_CHUNK_DURATION)]
    pub duration: f32,

    /// Chunk overlap in seconds
    #[arg(long, default_value_t = DEFAULT_CHUNK_OVERLAP)]
    pub overlap: f32,
}

impl TryFrom<ChunkArgs> for ChunkConfig {
    type Error = melops_asr::error::Error;

    fn try_from(args: ChunkArgs) -> Result<Self, Self::Error> {
        ChunkConfig::new(args.duration, args.overlap)
    }
}

impl From<ChunkConfig> for ChunkArgs {
    fn from(config: ChunkConfig) -> Self {
        ChunkArgs {
            duration: config.duration(),
            overlap: config.overlap(),
        }
    }
}

/// Shared caption generation options.
#[derive(Args, Debug)]
pub struct CaptionArgs {
    /// Show preview of first and last subtitles
    #[arg(long)]
    pub preview: bool,

    #[command(flatten)]
    pub chunk_args: ChunkArgs,
}

/// CLI arguments for cache directory configuration.
#[derive(Args, Clone, Debug, Default)]
pub struct CacheArgs {
    /// Cache directory (default: system cache directory)
    #[arg(long)]
    pub cache_dir: Option<PathBuf>,
}

/// CLI arguments for index cache operations.
#[derive(Args, Clone, Copy, Debug, Default)]
pub struct IndexArgs {
    /// Cache operation for web page URLs
    #[arg(long, value_enum, default_value_t = CacheStrategy::GetOrInsert)]
    pub cache_pages: CacheStrategy,

    /// Cache operation for audio files
    #[arg(long, value_enum, default_value_t = CacheStrategy::GetOrInsert)]
    pub cache_audio: CacheStrategy,

    /// Cache operation for SRT files
    #[arg(long, value_enum, default_value_t = CacheStrategy::GetOrInsert)]
    pub cache_srt: CacheStrategy,
}

/// CLI arguments for download configuration.
#[derive(Args, Clone, Debug, Default)]
pub struct DownloadArgs {
    /// Output directory for downloaded audio (default: system download directory)
    #[arg(short, long)]
    pub output_dir: Option<PathBuf>,
}

/// Execute CLI command - separated for testing.
pub async fn run(cli: Cli) -> Result<()> {
    tracing::debug!(?cli, "parsed arguments");

    match cli.command {
        Commands::Cap(args) => crate::cap::run(args).await,
        Commands::Dl(args) => crate::dl::run(args).await,
        Commands::Web(args) => crate::web::run(args).await,
        Commands::Cache(args) => crate::cache::run(args),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_cap_command() {
        let cli = Cli::parse_from(["mel", "cap", "--model-id", "test_model", "audio.wav"]);

        match &cli.command {
            Commands::Cap(crate::cap::CapCommand {
                path,
                output: None,
                cache_args: CacheArgs { cache_dir: None },
                ..
            }) if path.to_str() == Some("audio.wav") => {}
            _ => panic!("unexpected command: {:?}", cli.command),
        }
    }

    #[test]
    fn parses_cap_with_output() {
        let cli = Cli::parse_from([
            "mel",
            "cap",
            "--model-id",
            "test_model",
            "audio.wav",
            "-o",
            "output.srt",
        ]);

        match &cli.command {
            Commands::Cap(crate::cap::CapCommand {
                path,
                output: Some(output),
                cache_args: CacheArgs { cache_dir: None },
                ..
            }) if path.to_str() == Some("audio.wav") && output.to_str() == Some("output.srt") => {}
            _ => panic!("unexpected command: {:?}", cli.command),
        }
    }

    #[test]
    fn parses_dl_command() {
        let cli = Cli::parse_from([
            "mel",
            "dl",
            "--model-id",
            "test_model",
            "https://example.com/video",
        ]);

        match &cli.command {
            Commands::Dl(crate::dl::DlCommand {
                url,
                download_args: DownloadArgs { output_dir: None },
                cache_args: CacheArgs { cache_dir: None },
                ..
            }) if url == "https://example.com/video" => {}
            _ => panic!("unexpected command: {:?}", cli.command),
        }
    }

    #[test]
    fn parses_dl_with_output() {
        let cli = Cli::parse_from([
            "mel",
            "dl",
            "--model-id",
            "test_model",
            "https://example.com/video",
            "-o",
            "/tmp/output",
        ]);

        match &cli.command {
            Commands::Dl(crate::dl::DlCommand {
                url,
                download_args:
                    DownloadArgs {
                        output_dir: Some(output_dir),
                    },
                cache_args: CacheArgs { cache_dir: None },
                ..
            }) if url == "https://example.com/video"
                && output_dir.to_str() == Some("/tmp/output") => {}
            _ => panic!("unexpected command: {:?}", cli.command),
        }
    }
}
