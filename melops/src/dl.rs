//! Dl subcommand - download and generate captions from audio URL.

use crate::cache::{CacheDir, CacheStrategy};
use crate::cap::caption_from_channel;
use crate::cli::{CacheArgs, CaptionArgs, DownloadArgs, ModelArgs};
use crate::config::ModelConfig;
use clap::Args;
use eyre::{Result, WrapErr};
use melops_dl::asr::AudioFormat;
use melops_dl::params::DownloadParams;
use std::path::{Path, PathBuf};
use tokio::sync::broadcast;

/// Summary of download operations.
#[derive(Debug, Clone, Copy, Default)]
pub struct DownloadSummary {
    pub completed: usize,
    pub failed: usize,
}

impl DownloadSummary {
    /// Returns true if all downloads succeeded (no failures).
    pub fn is_success(&self) -> bool {
        self.failed == 0
    }
}

/// Broadcast channel buffer size for download → caption pipeline.
///
/// Sized to accommodate typical single-file downloads with buffer for processing lag.
/// If downloads produce multiple files quickly, a larger buffer prevents blocking.
const DOWNLOAD_PATH_CHANNEL_SIZE: usize = 128;

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
#[derive(Clone, Debug)]
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

/// Validate that all cached paths still exist on disk
pub fn validate_cache(paths: &[PathBuf]) -> bool {
    if paths.is_empty() {
        tracing::debug!("cache contains no paths");
        return false;
    }

    let mut is_valid = true;

    for path in paths {
        if !path.exists() {
            tracing::debug!(path = ?path.display(), "cached file does not exist");
            is_valid = false;
        }
    }

    if is_valid {
        tracing::debug!(count = paths.len(), "all cached files exist");
    }

    is_valid
}

async fn write_cache(dir: &Path, key: &str, paths: &[PathBuf]) {
    tracing::debug!(count = paths.len(), "writing audio paths to cache");

    let data = match serde_json::to_string(paths) {
        Ok(data) => data,
        Err(err) => {
            tracing::warn!(%err, "failed to serialize paths for cache");
            return;
        }
    };

    if let Err(err) = cacache::write(dir, key, &data).await {
        tracing::warn!(%err, "failed to write cache");
    }
}

/// Download audio and collect paths synchronously in blocking task
///
/// Creates internal subscription for collection, spawns blocking download,
/// collects all paths during download (paths broadcast to caption worker simultaneously).
async fn download_and_collect(
    url: &str,
    params: DownloadParams,
    broadcast_tx: broadcast::Sender<PathBuf>,
) -> Result<Vec<PathBuf>> {
    // Subscribe to broadcast channel for collection
    let mut collect_rx = broadcast_tx.subscribe();

    // Spawn blocking download task (sends to broadcast)
    let mut download_task = tokio::task::spawn_blocking({
        let url = url.to_string();
        move || melops_dl::dl::download(&url, params, broadcast_tx)
    });

    // Collect paths while download is running
    let mut paths = Vec::new();
    loop {
        tokio::select! {
            // Keep collecting paths as they arrive
            val = collect_rx.recv() => {
                if let Ok(path) = val {
                    paths.push(path);
                }
            }
            // Stop when download completes
            res = &mut download_task => {
                res.wrap_err("download task panicked")?
                    .wrap_err("failed to download audio")?;
                break;
            }
        }
    }

    // Collect any remaining paths after download completes
    while let Ok(path) = collect_rx.try_recv() {
        paths.push(path);
    }

    Ok(paths)
}

/// Download audio from URL using cache
///
/// Paths are broadcast through the channel as they arrive (for caption worker),
/// and also collected for caching purposes.
pub async fn download(
    config: &DownloadConfig,
    cache_dir: &CacheDir,
    strategy: CacheStrategy,
    tx: broadcast::Sender<PathBuf>,
) -> Result<()> {
    let dir = cache_dir.downloads();
    let key = config.url.as_str();
    let params = config.to_params();

    match strategy {
        CacheStrategy::Use => {
            tracing::debug!("reading audio paths from cache");
            let bytes = cacache::read(&dir, &key).await?;
            let paths: Vec<PathBuf> = serde_json::from_slice(&bytes)?;

            // Broadcast cached paths
            for path in &paths {
                if let Err(err) = tx.send(path.clone()) {
                    tracing::warn!(%err, "failed to broadcast cached path");
                }
            }
        }

        CacheStrategy::Auto => match cacache::read(&dir, &key).await {
            Ok(bytes) => {
                let paths: Vec<PathBuf> = serde_json::from_slice(&bytes)?;

                if validate_cache(&paths) {
                    tracing::info!("using cached audio files");
                    for path in &paths {
                        if let Err(err) = tx.send(path.clone()) {
                            tracing::warn!(%err, "failed to broadcast cached path");
                        }
                    }
                } else {
                    tracing::info!("cached files invalidated, downloading");
                    let paths = download_and_collect(&config.url, params, tx).await?;
                    write_cache(dir, key, &paths).await;
                }
            }
            Err(err) => {
                tracing::info!("cache miss or error, downloading");
                let paths = download_and_collect(&config.url, params, tx)
                    .await
                    .wrap_err(err)?;
                write_cache(dir, key, &paths).await;
            }
        },

        CacheStrategy::Force => {
            tracing::info!("forcing download, ignoring cache");
            let paths = download_and_collect(&config.url, params, tx).await?;
            write_cache(dir, key, &paths).await;
        }
    }

    Ok(())
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
    let cache_srt = command.caption_args.cache_cap;
    let cache_dl = command.download_args.cache_dl;

    // Fast failing: validate chunk config before expensive operations
    let segment_config = command.caption_args.segment_args.try_into()?;

    // Load model once for all files
    let model = model_config.load()?;

    // Create broadcast channel for download → caption pipeline
    let (tx, rx) = broadcast::channel::<PathBuf>(DOWNLOAD_PATH_CHANNEL_SIZE);

    // Spawn caption worker (using shared abstraction)
    let caption_task = tokio::spawn(caption_from_channel(
        rx,
        model,
        cache_dir.clone(),
        cache_srt,
        segment_config,
        command.caption_args.preview,
    ));

    // Download with caching (broadcasts paths as they arrive)
    let download_result = download(&download_config, &cache_dir, cache_dl, tx).await;

    // Check download errors first (prerequisite for captions)
    download_result?;

    // Wait for caption task to complete
    let caption_summary = caption_task.await.wrap_err("caption task panicked")?;

    if !caption_summary.is_success() {
        tracing::warn!(
            completed = caption_summary.completed,
            failed = caption_summary.failed,
            "captioning completed with some failures"
        );
    }

    Ok(())
}
