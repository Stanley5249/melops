//! Web scraping and batch processing command

use crate::cache::{CacheDir, CacheStrategy};
use crate::cap::caption_from_channel;
use crate::cli::{CacheArgs, CaptionArgs, DownloadArgs, ModelArgs, WebArgs};
use crate::config::ModelConfig;
use crate::dl::DownloadConfig;
use clap::Args;
use eyre::{Result, WrapErr, eyre};
use futures::stream::{self, StreamExt, TryStreamExt};
use std::path::{Path, PathBuf};
use tokio::sync::broadcast;

/// Maximum concurrent URL resolution requests
const MAX_CONCURRENT_RESOLVES: usize = 64;

/// Broadcast channel buffer size for download → caption pipeline.
///
/// Sized to accommodate typical single-file downloads with buffer for processing lag.
/// If downloads produce multiple files quickly, a larger buffer prevents blocking.
const DOWNLOAD_PATH_CHANNEL_SIZE: usize = 128;

#[derive(Args, Debug)]
pub struct WebCommand {
    /// URL of the web page to scrape
    pub url: String,

    /// Extract and display URLs only, without downloading or processing
    #[arg(long)]
    pub dry_run: bool,

    /// Process only the first N URLs
    #[arg(long)]
    pub limit: Option<usize>,

    #[command(flatten)]
    pub cache_args: CacheArgs,

    #[command(flatten)]
    pub caption_args: CaptionArgs,

    #[command(flatten)]
    pub download_args: DownloadArgs,

    #[command(flatten)]
    pub web_args: WebArgs,

    #[command(flatten)]
    pub model_args: ModelArgs,
}

/// Validated configuration for web scraping operations
#[derive(Clone, Debug)]
pub struct WebConfig {
    pub page_url: String,
    pub output_dir: Option<PathBuf>,
}

/// Fetch HTML and extract YouTube URLs from page
async fn fetch_and_extract_urls(page_url: &str) -> Result<Vec<String>> {
    tracing::info!(url = %page_url, "fetching page");
    let html = melops_web::fetch_page(page_url)?;

    tracing::info!("extracting youtube urls");
    let urls = melops_web::extract_youtube_urls(&html)?;

    if urls.is_empty() {
        return Err(eyre!("no youtube urls found on page: {}", page_url));
    }

    tracing::info!(count = urls.len(), "resolving youtube urls");

    let resolved_urls: Vec<String> = stream::iter(&urls)
        .map(async |url| {
            tracing::debug!(url = %url, "resolving url");
            melops_web::resolve_url(url).await
        })
        .buffer_unordered(MAX_CONCURRENT_RESOLVES)
        .try_collect()
        .await?;

    Ok(resolved_urls)
}

async fn write_page_cache(dir: &Path, key: &str, urls: &[String]) {
    tracing::debug!("writing page urls to cache");

    let data = match serde_json::to_string(urls) {
        Ok(data) => data,
        Err(err) => {
            tracing::warn!(%err, "failed to serialize urls for cache");
            return;
        }
    };

    if let Err(err) = cacache::write(dir, key, &data).await {
        tracing::warn!(%err, "failed to write cache");
    }
}

/// Fetch and extract YouTube URLs from page using cache
async fn fetch_page_urls(
    page_url: &str,
    cache_dir: &CacheDir,
    strategy: CacheStrategy,
) -> Result<Vec<String>> {
    let dir = cache_dir.pages();
    let key = page_url;

    let result: Result<Vec<String>> = match strategy {
        CacheStrategy::Use => {
            tracing::debug!("reading page urls from cache");
            let bytes = cacache::read(&dir, &key).await?;
            Ok(serde_json::from_slice(&bytes)?)
        }
        CacheStrategy::Auto => match cacache::read(&dir, &key).await {
            Ok(bytes) => {
                tracing::info!("using cached page urls");
                Ok(serde_json::from_slice(&bytes)?)
            }
            Err(err) => {
                tracing::info!("no cached page found, fetching");
                Ok(fetch_and_extract_urls(page_url).await.wrap_err(err)?)
            }
        },
        CacheStrategy::Force => {
            tracing::info!("forcing page fetch, ignoring cache");
            Ok(fetch_and_extract_urls(page_url).await?)
        }
    };

    let urls = result?;

    if strategy != CacheStrategy::Use {
        write_page_cache(dir, key, &urls).await;
    }

    Ok(urls)
}

/// Download audio files from URLs sequentially using cache
///
/// Calls dl::download() for each URL, utilizing the same cache mechanism.
/// Paths are broadcast through the channel as they arrive.
async fn download_batch(
    urls: &[String],
    output_dir: &Option<PathBuf>,
    cache_dir: &CacheDir,
    cache_dl: CacheStrategy,
    tx: broadcast::Sender<PathBuf>,
) -> Result<()> {
    // Process each URL sequentially
    for (i, url) in urls.iter().enumerate() {
        tracing::info!(
            current = i + 1,
            total = urls.len(),
            url = %url,
            "downloading video"
        );

        let config = DownloadConfig {
            url: url.clone(),
            output_dir: output_dir.clone(),
        };

        // Use the same download logic as dl command
        crate::dl::download(&config, cache_dir, cache_dl, tx.clone())
            .await
            .wrap_err_with(|| format!("failed to download: {}", url))?;
    }

    Ok(())
}

/// Entry point for web command
pub async fn run(command: WebCommand) -> Result<()> {
    // Validate command into configs
    let web_config = WebConfig {
        page_url: command.url,
        output_dir: command.download_args.output_dir,
    };

    let model_config = ModelConfig::try_from(command.model_args)?;
    let cache_dir: CacheDir = command.cache_args.try_into()?;

    let cache_web = command.web_args.cache_web;
    let cache_dl = command.download_args.cache_dl;
    let cache_cap = command.caption_args.cache_cap;

    // Fetch YouTube URLs
    let youtube_urls = fetch_page_urls(&web_config.page_url, &cache_dir, cache_web).await?;

    // If dry run, output all URLs and exit
    if command.dry_run {
        for url in &youtube_urls {
            println!("{}", url);
        }
        return Ok(());
    }

    // Apply limit for processing
    let count = command.limit.unwrap_or(usize::MAX).min(youtube_urls.len());

    tracing::info!(count, "processing youtube urls");

    // Fast-fail: validate chunk config before expensive operations
    let chunk_config = command.caption_args.chunk_args.try_into()?;

    // Load model once for all videos
    let model = model_config.load()?;

    // Create broadcast channel for download → caption pipeline
    let (tx, rx) = broadcast::channel::<PathBuf>(DOWNLOAD_PATH_CHANNEL_SIZE);

    // Spawn caption worker (using shared abstraction)
    let caption_task = tokio::spawn(caption_from_channel(
        rx,
        model,
        cache_dir.clone(),
        cache_cap,
        chunk_config,
        command.caption_args.preview,
    ));

    // Download with caching (broadcasts paths as they arrive)
    let download_result = download_batch(
        &youtube_urls[..count],
        &web_config.output_dir,
        &cache_dir,
        cache_dl,
        tx,
    )
    .await;

    // Wait for caption task to complete
    let caption_summary = caption_task.await??;

    // Propagate errors after both tasks finish
    download_result?;

    if caption_summary.is_success() {
        tracing::info!(
            completed = caption_summary.completed,
            "completed processing all videos"
        );
    } else {
        tracing::warn!(
            completed = caption_summary.completed,
            failed = caption_summary.failed,
            "completed processing with some caption failures"
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::ModelSource;

    #[test]
    fn creates_web_command() {
        let cmd = WebCommand {
            url: "https://example.com/course".to_string(),
            dry_run: false,
            limit: None,
            cache_args: CacheArgs { cache_dir: None },
            caption_args: CaptionArgs {
                preview: false,
                cache_cap: Default::default(),
                chunk_args: Default::default(),
            },
            download_args: DownloadArgs {
                output_dir: None,
                cache_dl: Default::default(),
            },
            web_args: Default::default(),
            model_args: ModelArgs {
                model_id: "test_model".to_string(),
                model_source: ModelSource::Auto,
            },
        };

        assert_eq!(cmd.url, "https://example.com/course");
    }

    #[test]
    fn creates_web_config() {
        let config = WebConfig {
            page_url: "https://example.com/course".to_string(),
            output_dir: Some(PathBuf::from("/output")),
        };

        assert_eq!(config.page_url, "https://example.com/course");
        assert_eq!(config.output_dir, Some(PathBuf::from("/output")));
    }
}
