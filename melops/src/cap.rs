//! Cap subcommand - generate captions from audio file to SRT.

use crate::cache::Cache;
use crate::cli::{CacheArgs, CacheConfig, CaptionArgs, ModelArgs};
use crate::config::ModelConfig;
use crate::segment::Segmenter;
use crate::srt::{display_subtitles, preview_subtitles, to_subtitles};
use clap::Args;
use eyre::{Context, Result};
use melops_asr::audio::read_audio_mono;
use melops_asr::chunk::ChunkConfig;
use melops_asr::models::tdt::core::TdtModel;
use melops_asr::traits::AsrModel;
use std::path::PathBuf;

/// CLI arguments for caption generation.
#[derive(Args, Debug)]
pub struct CapCommand {
    /// Path to input WAV file
    pub path: PathBuf,

    /// Output SRT path (default: same as input with .srt extension)
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    #[command(flatten)]
    pub model_args: ModelArgs,

    #[command(flatten)]
    pub caption_args: CaptionArgs,

    #[command(flatten)]
    pub cache_args: CacheArgs,
}

/// Validated configuration for caption generation.
#[derive(Debug)]
pub struct CapConfig {
    pub path: PathBuf,
    pub output: PathBuf,
    pub preview: bool,
    pub chunk_config: ChunkConfig,
}

/// Generate captions with pre-loaded model and update cache
pub async fn caption(config: &CapConfig, model: &TdtModel, cache: &mut Cache) -> Result<()> {
    let cache_key = config.path.to_string_lossy();

    // Check cache for existing SRT (respects refresh_srt flag)
    if let Some(srt_path) = cache.get_srt_path(&cache_key) {
        tracing::info!(path = ?srt_path.display(), "srt file already exists in cache");
        return Ok(());
    }

    // Check if output exists on disk (only when not refreshing)
    if !cache.refresh_srt && config.output.exists() {
        tracing::info!(path = ?config.output.display(), "srt file already exists");
        // Update cache
        cache.set_srt_path(cache_key.to_string(), config.output.clone())?;
        return Ok(());
    }

    tracing::info!(
        input = ?config.path.display(),
        output = ?config.output.display(),
        "generating captions"
    );

    // Load audio
    let audio = read_audio_mono(&config.path)
        .wrap_err_with(|| format!("failed to load audio: {:?}", config.path.display()))?;

    // Transcribe
    let segments = model
        .transcribe_chunked(&audio, config.chunk_config)
        .await
        .wrap_err("transcription failed")?;

    // Regroup segments for comfortable speed
    let segments = Segmenter::COMFORTABLE.regroup(&segments);

    // Convert to subtitles
    let subtitles = to_subtitles(&segments);

    // Write SRT file
    tracing::info!(path = ?config.output.display(), "write srt file");
    std::fs::write(&config.output, display_subtitles(&subtitles))
        .wrap_err_with(|| format!("failed to write srt: {:?}", config.output.display()))?;

    // Update cache
    cache.set_srt_path(cache_key.to_string(), config.output.clone())?;

    // Preview
    if config.preview {
        print!("{}", preview_subtitles(&subtitles, 2, 2));
    }

    Ok(())
}

/// Entry point for cap command
pub async fn run(command: CapCommand) -> Result<()> {
    // Validate command into configs
    let path = command.path;
    let output = command.output.unwrap_or_else(|| path.with_extension("srt"));

    let model_config = ModelConfig::try_from(command.model_args)?;

    let cap_config = CapConfig {
        path,
        output,
        preview: command.caption_args.preview,
        chunk_config: command.caption_args.chunk_args.try_into()?,
    };

    // Load cache (application state)
    let cache_config = CacheConfig::from(command.cache_args);
    let mut cache = Cache::load(cache_config)?;

    // Load model
    let model = model_config.load()?;

    // Generate captions
    caption(&cap_config, &model, &mut cache).await
}
