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
#[allow(unused_imports)]
use ort::ep::*;
use ort::session::Session;
use ort::session::builder::SessionBuilder;
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

    let builder = build_session()?;
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

/// Build execution config with execution providers configured by Cargo features.
///
/// Configures ONNX Runtime session with execution providers in priority order. The first
/// available provider is used; CPU is always available as fallback.
///
/// # Execution Providers
///
/// Enabled via Cargo features:
/// - `cuda` - NVIDIA CUDA
/// - `tensorrt` - NVIDIA TensorRT
/// - `openvino` - Intel OpenVINO
/// - `directml` - DirectML (Windows)
/// - `coreml` - CoreML (macOS)
///
/// Ensure required hardware, drivers, and runtime dependencies are installed for the
/// desired provider.
fn build_session() -> ort::Result<SessionBuilder> {
    let eps = [
        #[cfg(feature = "cuda")]
        CUDAExecutionProvider::default().build(),
        #[cfg(feature = "tensorrt")]
        TensorRTExecutionProvider::default().build(),
        #[cfg(feature = "openvino")]
        OpenVINOExecutionProvider::default()
            .with_device_type("HETERO:GPU,CPU")
            .with_cache_dir(".cache/ort")
            .with_precision("FP16")
            .build(),
        #[cfg(feature = "directml")]
        DirectMLExecutionProvider::default().build(),
        #[cfg(feature = "coreml")]
        CoreMLExecutionProvider::default().build(),
    ];

    // Async sessions use intra thread pools
    let builder = Session::builder()?.with_execution_providers(eps)?;

    #[cfg(not(feature = "openvino"))]
    let builder = builder
        .with_intra_threads(0)?
        .with_intra_op_spinning(true)?;

    // According to the OpenVINO documentation, disabling optimization can often yield better results
    #[cfg(feature = "openvino")]
    let builder = builder
        .with_intra_threads(2)?
        .with_intra_op_spinning(true)?
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Disable)?;

    Ok(builder)
}
