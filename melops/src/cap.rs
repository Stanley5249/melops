//! Cap subcommand - generate captions from audio file to SRT.

use crate::cache::CacheDir;
use crate::cli::{CacheArgs, CaptionArgs, IndexArgs, ModelArgs};
use crate::config::ModelConfig;
use crate::index::CacheStrategy;
use crate::segment::Segmenter;
use crate::srt::{display_subtitles, preview_subtitles, to_subtitles};
use cacache::Integrity;
use clap::Args;
use eyre::{Context, Result};
use melops_asr::audio::read_audio_mono;
use melops_asr::chunk::ChunkConfig;
use melops_asr::models::tdt::core::TdtModel;
use melops_asr::traits::AsrModel;
use melops_asr::types::Segment;
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
    cache_dir: &CacheDir,
    strategy: CacheStrategy,
) -> Result<()> {
    let dir = cache_dir.artifact();

    // Load audio
    let audio = read_audio_mono(&config.audio_path).wrap_err("failed to load audio")?;

    let key = Integrity::from(bytemuck::cast_slice(&audio)).to_string();

    let result: Result<Vec<Segment>> = match strategy {
        CacheStrategy::Get => {
            let bytes = cacache::read(&dir, &key).await?;
            Ok(serde_json::from_slice(&bytes)?)
        }
        CacheStrategy::GetOrInsert => match cacache::read(&dir, &key).await {
            Ok(bytes) => Ok(serde_json::from_slice(&bytes)?),
            Err(err) => Ok(transcribe(&audio, &config.chunk_config, model)
                .await
                .wrap_err(err)?),
        },
        CacheStrategy::Replace => Ok(transcribe(&audio, &config.chunk_config, model).await?),
    };

    let segments = result?;

    if strategy != CacheStrategy::Get {
        let data = serde_json::to_string(&segments)?;
        cacache::write(&dir, &key, &data).await?;
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

pub async fn transcribe(
    audio: &[f32],
    chunk_config: &ChunkConfig,
    model: &TdtModel,
) -> Result<Vec<Segment>> {
    let segments = model
        .transcribe_chunked(&audio, chunk_config)
        .await
        .wrap_err("failed to transcribe")?;

    Ok(segments)
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

    // Load model
    let model = model_config.load()?;

    // Generate captions
    caption(
        &cap_config,
        &model,
        &cache_dir,
        command.index_args.cache_srt,
    )
    .await
}
