//! Dl subcommand - download and generate captions from audio URL.

use crate::cache::Cache;
use crate::cap::{CapConfig, caption};
use crate::cli::{CacheArgs, CacheConfig, CaptionArgs, DownloadArgs, ModelArgs};
use crate::config::ModelConfig;
use clap::Args;
use color_eyre::Section;
use eyre::{OptionExt, Result, WrapErr, eyre};
use melops_dl::asr::AudioFormat;
use melops_dl::dl::DownloadOptions;
use std::path::PathBuf;

/// CLI arguments for download and caption generation.
#[derive(Args, Debug)]
pub struct DlCommand {
    /// URL to download
    pub url: String,

    #[command(flatten)]
    pub download_args: DownloadArgs,

    #[command(flatten)]
    pub model_args: ModelArgs,

    #[command(flatten)]
    pub caption_args: CaptionArgs,

    #[command(flatten)]
    pub cache_args: CacheArgs,
}

/// Validated download configuration.
#[derive(Debug)]
pub struct DownloadConfig {
    pub url: String,
    pub output_dir: Option<PathBuf>,
}

impl DownloadConfig {
    /// Transform to application state (DownloadOptions)
    pub fn to_options(&self) -> DownloadOptions {
        let mut opts: DownloadOptions = AudioFormat::Pcm16.into();
        if let Some(dir) = &self.output_dir {
            opts.paths = Some(opts.paths.expect("paths should be some").with_home(dir));
        }
        opts
    }
}

/// Download audio from URL and update cache
pub fn download(config: &DownloadConfig, cache: &mut Cache) -> Result<PathBuf> {
    // Check cache first
    if let Some(audio_path) = cache.get_audio_path(&config.url) {
        if audio_path.exists() {
            tracing::info!(path = ?audio_path.display(), "using cached audio");
            return Ok(audio_path.to_path_buf());
        }
        tracing::info!(
            path = ?audio_path.display(),
            "cached audio file not found, re-downloading"
        );
    }

    tracing::info!(url = %config.url, "downloading audio");

    let options = config.to_options();

    let (file_path, _info) =
        melops_dl::dl::download(&config.url, options).wrap_err("failed to download audio")?;

    let audio_path = file_path.ok_or_eyre("yt-dlp did not return downloaded file path")?;

    // Verify file exists
    if !audio_path.exists() {
        return Err(eyre!(
            "audio downloaded but file not found: {:?}",
            audio_path.display()
        ));
    }

    tracing::info!(downloaded = ?audio_path.display(), "audio downloaded");

    // Update cache
    cache.set_audio_path(config.url.clone(), audio_path.clone())?;

    Ok(audio_path)
}

/// Entry point for dl command
pub async fn run(command: DlCommand) -> Result<()> {
    // Validate command into configs
    let download_config = DownloadConfig {
        url: command.url,
        output_dir: command.download_args.output_dir,
    };

    let model_config = ModelConfig::try_from(command.model_args)?;

    // Load cache (application state)
    let cache_config = CacheConfig::from(command.cache_args);
    let mut cache = Cache::load(cache_config)?;

    // Download
    let audio_path = download(&download_config, &mut cache)?;

    tracing::info!(
        downloaded = ?audio_path.display(),
        "audio downloaded, starting captioning"
    );

    // Create caption config
    let cap_config = CapConfig {
        path: audio_path.clone(),
        output: audio_path.with_extension("srt"),
        preview: command.caption_args.preview,
        chunk_config: command.caption_args.chunk_args.into(),
    };

    // Load model and caption
    let model = model_config.load()?;
    caption(&cap_config, &model, &mut cache)
        .await
        .with_note(|| {
            format!(
                "audio downloaded successfully to: {:?}",
                audio_path.display()
            )
        })
        .with_suggestion(|| format!("mel cap {:?}", audio_path.display()))
}
