//! Cap subcommand - generate captions from audio file to SRT.

use crate::cli::{CacheArgs, CaptionArgs, IndexArgs, ModelArgs};
use crate::config::ModelConfig;
use crate::index::ArtifactCache;
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
    pub cache_args: CacheArgs,

    #[command(flatten)]
    pub caption_args: CaptionArgs,

    #[command(flatten)]
    pub index_args: IndexArgs,

    #[command(flatten)]
    pub model_args: ModelArgs,
}

/// Validated configuration for caption generation.
#[derive(Debug)]
pub struct CapConfig {
    pub audio_path: PathBuf,
    pub output_path: PathBuf,
    pub preview: bool,
    pub chunk_config: ChunkConfig,
}

/// Generate captions with pre-loaded model and cache index
pub async fn caption(
    config: &CapConfig,
    model: &TdtModel,
    index: &mut ArtifactCache,
) -> Result<()> {
    // Get cache key
    let cache_key = config
        .audio_path
        .canonicalize()?
        .to_string_lossy()
        .into_owned();

    // Access SRT path from cache or generate new one
    let srt_path = index
        .ensure_srt(cache_key, async || {
            // Load audio
            let audio = read_audio_mono(&config.audio_path).wrap_err_with(|| {
                format!("failed to load audio: {:?}", config.audio_path.display())
            })?;

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
            tracing::info!(path = ?config.output_path.display(), "write srt file");

            std::fs::write(&config.output_path, display_subtitles(&subtitles)).wrap_err_with(
                || format!("failed to write srt: {:?}", config.output_path.display()),
            )?;

            // Canonicalize output path
            Ok(config.output_path.canonicalize()?)
        })
        .await?;

    // Preview if requested
    if config.preview {
        // Read subtitles from file
        let content = std::fs::read_to_string(srt_path)
            .wrap_err_with(|| format!("failed to read srt: {:?}", srt_path.display()))?;

        let subtitles =
            srtlib::Subtitles::parse_from_str(content).wrap_err("failed to parse srt")?;

        print!("{}", preview_subtitles(&subtitles.to_vec(), 2, 2));
    }

    Ok(())
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
        chunk_config: command.caption_args.chunk_args.try_into()?,
    };

    // Load index
    let mut index = ArtifactCache::load(cache_dir, command.index_args)?;

    // Load model
    let model = model_config.load()?;

    // Generate captions
    caption(&cap_config, &model, &mut index).await
}
