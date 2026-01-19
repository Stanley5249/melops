//! Web scraping and batch processing command

use crate::cache::CacheDir;
use crate::cap::{CapConfig, caption};
use crate::cli::{CacheArgs, CaptionArgs, DownloadArgs, IndexArgs, ModelArgs};
use crate::config::ModelConfig;
use crate::dl::{DownloadConfig, download};
use crate::index::ArtifactCache;
use clap::Args;
use eyre::{Result, eyre};
use futures::stream::{self, StreamExt, TryStreamExt};
use std::path::PathBuf;

/// Maximum concurrent URL resolution requests
const MAX_CONCURRENT_RESOLVES: usize = 64;

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
    pub index_args: IndexArgs,

    #[command(flatten)]
    pub model_args: ModelArgs,
}

/// Validated configuration for web scraping operations
#[derive(Debug)]
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

/// Fetch and extract YouTube URLs from page, update index
async fn fetch_page_urls(page_url: &str, index: &mut ArtifactCache) -> Result<Vec<String>> {
    let page_url = page_url.to_string();

    // Access URLs from cache or fetch new ones
    let urls = index
        .ensure_pages(page_url.clone(), async || {
            fetch_and_extract_urls(&page_url).await
        })
        .await?;

    Ok(urls.clone())
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

    // Load index
    let mut index = ArtifactCache::load(cache_dir.clone(), command.index_args)?;

    // Fetch YouTube URLs
    let youtube_urls = fetch_page_urls(&web_config.page_url, &mut index).await?;

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

    // Load model once for all videos
    let model = model_config.load()?;

    // Process each video
    for (i, youtube_url) in youtube_urls.iter().take(count).enumerate() {
        tracing::info!(
            current = i + 1,
            total = count,
            url = %youtube_url,
            "processing video"
        );

        // Download audio
        let download_config = DownloadConfig {
            url: youtube_url.clone(),
            output_dir: web_config.output_dir.clone(),
        };

        let audio_path = download(&download_config, &mut index).await?;

        // Generate captions
        let cap_config = CapConfig {
            audio_path: audio_path.clone(),
            output_path: audio_path.with_extension("srt"),
            preview: command.caption_args.preview,
            chunk_config: command.caption_args.chunk_args.try_into()?,
        };

        caption(&cap_config, &model, &cache_dir, index.cache_srt).await?;
    }

    tracing::info!("completed processing all videos");
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
                chunk_args: Default::default(),
            },
            download_args: DownloadArgs { output_dir: None },
            index_args: Default::default(),
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
