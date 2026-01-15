//! Cap subcommand - generate captions from audio file to SRT.

use crate::cli::{CaptionArgs, ModelArgs};
use crate::config::ModelConfig;
use crate::segment::Segmenter;
use crate::srt::{self, display_subtitles};
use clap::Args;
use eyre::{Context, Result};
use melops_asr::audio::read_audio_mono;
use melops_asr::chunk::ChunkConfig;
use melops_asr::models::tdt::core::TdtModel;
use melops_asr::traits::AsrModel;
use srtlib::Subtitle;
use std::path::{Path, PathBuf};
use tokio::runtime::Builder;

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
}

/// Resolved configuration for caption generation.
#[derive(Debug)]
pub struct CapConfig {
    pub path: PathBuf,
    pub output: PathBuf,
    pub model_config: ModelConfig,
    pub preview: bool,
    pub chunk_config: ChunkConfig,
}

impl TryFrom<CapCommand> for CapConfig {
    type Error = eyre::Error;

    fn try_from(args: CapCommand) -> Result<Self> {
        let output = args
            .output
            .unwrap_or_else(|| args.path.with_extension("srt"));

        Ok(Self {
            path: args.path,
            output,
            model_config: args.model_args.try_into()?,
            preview: args.caption_args.preview,
            chunk_config: args.caption_args.chunk_args.into(),
        })
    }
}

pub fn execute(config: CapConfig) -> Result<()> {
    // Resolve output path
    let output = config.output;

    tracing::info!(
        input = ?config.path.display(),
        output = ?output.display(),
        "generating captions"
    );

    let subtitles = caption_from_wav_file(&config.path, config.model_config, config.chunk_config)?;

    tracing::info!(path = ?output.display(), "write srt file");

    // Write to file
    std::fs::write(&output, display_subtitles(&subtitles))
        .wrap_err_with(|| format!("failed to write srt: {:?}", output.display()))?;

    // Display preview or full output to stdout
    if config.preview {
        print!("{}", srt::preview_subtitles(&subtitles, 2, 2));
    }

    Ok(())
}

/// Perform ASR on WAV file and return captions as subtitles.
fn caption_from_wav_file(
    wav_path: &Path,
    model_config: ModelConfig,
    chunk_config: ChunkConfig,
) -> Result<Vec<Subtitle>> {
    let audio = read_audio_mono(wav_path)
        .wrap_err_with(|| format!("failed to load audio: {:?}", wav_path.display()))?;

    let builder = crate::ort::build_session()?;
    let model = TdtModel::from_repo(&model_config.repo, builder)?;

    let segments = Builder::new_current_thread()
        .build()?
        .block_on(model.transcribe_chunked(&audio, chunk_config))
        .wrap_err("transcription failed")?;

    // Regroup segments for comfortable speed
    let segments = Segmenter::COMFORTABLE.regroup(&segments);

    let subtitles = srt::to_subtitles(&segments);

    Ok(subtitles)
}
