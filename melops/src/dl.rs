//! Dl subcommand - download and generate captions from audio URL.

use crate::cache::{CacheDir, CacheStrategy};
use crate::cap::{CapConfig, caption};
use crate::cli::{CacheArgs, CaptionArgs, DownloadArgs, ModelArgs};
use crate::config::ModelConfig;
use clap::Args;
use color_eyre::Section;
use eyre::{OptionExt, Result, WrapErr, ensure};
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

/// Download audio from URL
async fn download_audio(url: &str, params: DownloadParams) -> Result<PathBuf> {
    let (file_path, _info) =
        melops_dl::dl::download(url, params).wrap_err("failed to download audio")?;

    let audio_path = file_path.ok_or_eyre("yt-dlp did not return downloaded file path")?;

    ensure!(
        audio_path.exists(),
        "audio file does not exist after download: {}",
        audio_path.display()
    );

    tracing::info!(file = ?audio_path.display(), "save audio");

    Ok(audio_path)
}

/// Download audio from URL using cache
pub async fn download(
    config: &DownloadConfig,
    cache_dir: &CacheDir,
    strategy: CacheStrategy,
) -> Result<Vec<PathBuf>> {
    let dir = cache_dir.downloads();
    let key = config.url.as_str();

    let result: Result<Vec<PathBuf>> = match strategy {
        CacheStrategy::Use => {
            let bytes = cacache::read(&dir, &key).await?;
            Ok(serde_json::from_slice(&bytes)?)
        }
        CacheStrategy::Auto => match cacache::read(&dir, &key).await {
            Ok(bytes) => Ok(serde_json::from_slice(&bytes)?),
            Err(err) => {
                let params = config.to_params();
                let path = download_audio(&config.url, params).await.wrap_err(err)?;
                Ok(vec![path])
            }
        },
        CacheStrategy::Force => {
            let params = config.to_params();
            let path = download_audio(&config.url, params).await?;
            Ok(vec![path])
        }
    };

    let paths = result?;

    if strategy != CacheStrategy::Use {
        let data = serde_json::to_string(&paths)?;
        cacache::write(&dir, &key, &data).await?;
    }

    Ok(paths)
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

    // Download
    let audio_paths =
        download(&download_config, &cache_dir, command.download_args.cache_dl).await?;

    let cache_srt = command.caption_args.cache_cap;

    // Load model once for all files
    let model = model_config.load()?;

    // Process each audio file
    for audio_path in &audio_paths {
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

        // Generate captions
        caption(&cap_config, &model, &cache_dir, cache_srt)
            .await
            .with_note(|| {
                format!(
                    "audio downloaded successfully to: {:?}",
                    audio_path.display()
                )
            })
            .with_suggestion(|| format!("mel cap {:?}", audio_path.display()))?;
    }

    Ok(())
}
