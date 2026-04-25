//! Cap subcommand - generate captions from audio file to SRT.

use crate::cache::CacheDir;
use crate::cache::CacheStrategy;
use crate::cli::{CacheArgs, CaptionArgs, ModelArgs};
use crate::config::ModelConfig;
use crate::segment::Segmenter;
use crate::srt::{display_subtitles, preview_subtitles, to_subtitles};
use clap::Args;
use eyre::OptionExt;
use eyre::{Context, Result};
use melops_asr::models::tdt::core::TdtModel;
use melops_asr::traits::AsrModel;
use melops_asr::types::Segment;
use melops_audio::ffmpeg::Ffmpeg;
use melops_audio::segment::SegmentConfig;
use std::path::Path;
use std::path::PathBuf;
use tokio::sync::broadcast;

/// CLI arguments for caption generation.
#[derive(Args, Debug)]
pub struct CapCommand {
    /// Path to audio file (any format supported by ffmpeg: wav, mp3, m4a, mkv, …)
    pub path: PathBuf,

    /// Output SRT path (default: same as input with .srt extension)
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    #[command(flatten)]
    pub cache_args: CacheArgs,

    #[command(flatten)]
    pub caption_args: CaptionArgs,

    #[command(flatten)]
    pub model_args: ModelArgs,
}

/// Validated configuration for caption generation.
#[derive(Debug)]
pub struct CapConfig {
    pub audio_path: PathBuf,
    pub output_path: PathBuf,
    pub preview: bool,
    pub segment_config: SegmentConfig,
}

async fn write_cache(dir: &Path, key: &str, segments: &[Segment]) {
    tracing::debug!("writing segments to cache");

    let data = match serde_json::to_string(segments) {
        Ok(data) => data,
        Err(err) => {
            tracing::warn!(%err, "failed to serialize segments for cache");
            return;
        }
    };

    if let Err(err) = cacache::write(dir, key, &data).await {
        tracing::warn!(%err, "failed to write cache");
    }
}

/// Generate captions with pre-loaded model and cache index
pub async fn caption(
    config: &CapConfig,
    model: &TdtModel,
    cache_dir: &CacheDir,
    strategy: CacheStrategy,
) -> Result<()> {
    let dir = cache_dir.transcriptions();

    let mut ffmpeg = Ffmpeg::new(config.audio_path.clone())
        .await
        .wrap_err("failed to initialize ffmpeg")?;

    let key = ffmpeg.cache_key();

    let reader = ffmpeg
        .reader()
        .ok_or_eyre("ffmpeg stdout is not available")?;

    let result: Result<Vec<Segment>> = match strategy {
        CacheStrategy::Use => {
            tracing::debug!("reading segments from cache");
            let bytes = cacache::read(&dir, &key).await?;
            Ok(serde_json::from_slice(&bytes)?)
        }
        CacheStrategy::Auto => match cacache::read(&dir, &key).await {
            Ok(bytes) => {
                tracing::info!("using cached transcription");
                Ok(serde_json::from_slice(&bytes)?)
            }
            Err(e) => {
                tracing::info!("no cached transcription found, transcribing");
                Ok(model
                    .transcribe(reader, config.segment_config)
                    .await
                    .wrap_err("failed to transcribe")
                    .wrap_err(e)?)
            }
        },
        CacheStrategy::Force => {
            tracing::info!("forcing transcription, ignoring cache");

            Ok(model
                .transcribe(reader, config.segment_config)
                .await
                .wrap_err("failed to transcribe")?)
        }
    };

    let segments = result?;

    if strategy != CacheStrategy::Use {
        write_cache(dir, &key, &segments).await;
    }

    // Regroup segments for comfortable speed
    let segments = Segmenter::COMFORTABLE.regroup(&segments);

    // Convert to subtitles
    let subtitles = to_subtitles(&segments);

    // Write SRT file
    std::fs::write(&config.output_path, display_subtitles(&subtitles))
        .wrap_err("failed to save subtitles")?;

    tracing::info!(path = ?config.output_path.display(), "save subtitle");

    // Preview if requested
    if config.preview {
        print!("{}", preview_subtitles(&subtitles.to_vec(), 2, 2));
    }

    Ok(())
}

/// Summary of caption generation from channel.
#[derive(Debug, Clone, Copy, Default)]
pub struct CaptionSummary {
    pub completed: usize,
    pub failed: usize,
}

impl CaptionSummary {
    /// Returns true if all captions succeeded (no failures).
    pub fn is_success(&self) -> bool {
        self.failed == 0
    }
}

/// Caption audio files from channel with sequential processing.
///
/// Receives audio paths from channel and generates captions sequentially.
/// Continues on error (logs warnings, counts failures).
///
/// Returns summary when channel closes (sender dropped).
pub async fn caption_from_channel(
    mut rx: broadcast::Receiver<PathBuf>,
    model: TdtModel,
    cache_dir: CacheDir,
    cache_cap: CacheStrategy,
    segment_config: SegmentConfig,
    preview: bool,
) -> CaptionSummary {
    let mut summary = CaptionSummary::default();

    while let Ok(audio_path) = rx.recv().await {
        let cap_config = CapConfig {
            audio_path: audio_path.clone(),
            output_path: audio_path.with_extension("srt"),
            preview,
            segment_config,
        };

        match caption(&cap_config, &model, &cache_dir, cache_cap).await {
            Ok(_) => summary.completed += 1,
            Err(e) => {
                tracing::warn!(path = ?audio_path.display(), error = %e, "caption failed");
                summary.failed += 1;
            }
        }
    }

    summary
}

/// Entry point for cap command
pub async fn run(command: CapCommand) -> Result<()> {
    // Validate command into configs
    let audio_path = command.path;
    let output_path = command
        .output
        .unwrap_or_else(|| audio_path.with_extension("srt"));

    let model_config = ModelConfig::try_from(command.model_args)?;
    let cache_dir = command.cache_args.try_into()?;

    let cap_config = CapConfig {
        audio_path,
        output_path,
        preview: command.caption_args.preview,
        segment_config: command.caption_args.segment_args.try_into()?,
    };

    // Load model
    let model = model_config.load()?;

    // Generate captions
    caption(
        &cap_config,
        &model,
        &cache_dir,
        command.caption_args.cache_cap,
    )
    .await
}
