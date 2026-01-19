//! Dl subcommand - download and generate captions from audio URL.

use crate::cache::CacheDir;
use crate::cap::{CapConfig, caption};
use crate::cli::{CacheArgs, CaptionArgs, DownloadArgs, IndexArgs, ModelArgs};
use crate::config::ModelConfig;
use crate::index::ArtifactCache;
use clap::Args;
use color_eyre::Section;
use eyre::{OptionExt, Result, WrapErr, eyre};
use melops_dl::asr::AudioFormat;
use melops_dl::params::DownloadParams;
use std::path::PathBuf;

/// CLI arguments for download and caption generation.
#[derive(Args, Debug)]
pub struct DlCommand {
    /// URL to download
    pub url: String,

    #[command(flatten)]
    pub cache_args: CacheArgs,

    #[command(flatten)]
    pub caption_args: CaptionArgs,

    #[command(flatten)]
    pub download_args: DownloadArgs,

    #[command(flatten)]
    pub index_args: IndexArgs,

    #[command(flatten)]
    pub model_args: ModelArgs,
}

/// Validated download configuration.
#[derive(Debug)]
pub struct DownloadConfig {
    pub url: String,
    pub output_dir: Option<PathBuf>,
}

impl DownloadConfig {
    /// Transform to application state (DownloadParams)
    pub fn to_params(&self) -> DownloadParams {
        let mut params: DownloadParams = AudioFormat::Pcm16.into();
        if let Some(dir) = &self.output_dir {
            // SAFETY: AudioFormat::Pcm16 always initializes paths to Some
            params.paths = params.paths.map(|p| p.with_home(dir));
        }
        params
    }
}

/// Download audio from URL and update index
pub async fn download(config: &DownloadConfig, index: &mut ArtifactCache) -> Result<PathBuf> {
    let url = config.url.clone();

    let path = index
        .ensure_audio(url.clone(), async || {
            tracing::info!(url = %url, "downloading audio");

            let params = config.to_params();

            let (file_path, _info) =
                melops_dl::dl::download(&url, params).wrap_err("failed to download audio")?;

            let audio_path = file_path.ok_or_eyre("yt-dlp did not return downloaded file path")?;

            // Verify file exists
            if !audio_path.exists() {
                return Err(eyre!(
                    "audio downloaded but file not found: {:?}",
                    audio_path.display()
                ));
            }

            // Canonicalize path for consistency
            let audio_path = audio_path
                .canonicalize()
                .wrap_err("failed to canonicalize audio path")?;

            tracing::info!(downloaded = ?audio_path.display(), "audio downloaded");

            Ok(audio_path)
        })
        .await?;

    Ok(path.clone())
}

/// Entry point for dl command
pub async fn run(command: DlCommand) -> Result<()> {
    // Validate command into configs
    let download_config = DownloadConfig {
        url: command.url,
        output_dir: command.download_args.output_dir,
    };

    let model_config = ModelConfig::try_from(command.model_args)?;
    let cache_dir: CacheDir = command.cache_args.try_into()?;

    // Load index
    let mut index = ArtifactCache::load(cache_dir.clone(), command.index_args)?;

    // Download
    let audio_path = download(&download_config, &mut index).await?;

    tracing::info!(
        downloaded = ?audio_path.display(),
        "audio downloaded, starting captioning"
    );

    // Create caption config
    let cap_config = CapConfig {
        audio_path: audio_path.clone(),
        output_path: audio_path.with_extension("srt"),
        preview: command.caption_args.preview,
        chunk_config: command.caption_args.chunk_args.try_into()?,
    };

    // Load model and caption
    let model = model_config.load()?;
    caption(&cap_config, &model, &cache_dir, index.cache_srt)
        .await
        .with_note(|| {
            format!(
                "audio downloaded successfully to: {:?}",
                audio_path.display()
            )
        })
        .with_suggestion(|| format!("mel cap {:?}", audio_path.display()))
}
