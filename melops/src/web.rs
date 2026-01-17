//! Web scraping and batch processing command

use crate::cache::Cache;
use crate::cap::{CapConfig, caption};
use crate::cli::{CacheArgs, CacheConfig, CaptionArgs, DownloadArgs, ModelArgs};
use crate::config::ModelConfig;
use crate::dl::{DownloadConfig, download};
use clap::Args;
use eyre::{Result, eyre};
use std::path::PathBuf;

#[derive(Args, Debug)]
pub struct WebCommand {
    /// URL of the web page to scrape
    pub url: String,

    /// Extract and display URLs only, without downloading or processing
    #[arg(long)]
    pub dry_run: bool,

    #[command(flatten)]
    pub download_args: DownloadArgs,

    #[command(flatten)]
    pub model_args: ModelArgs,

    #[command(flatten)]
    pub caption_args: CaptionArgs,

    #[command(flatten)]
    pub cache_args: CacheArgs,
}

/// Validated configuration for web scraping operations
#[derive(Debug)]
pub struct WebConfig {
    pub page_url: String,
    pub output_dir: Option<PathBuf>,
}

/// Fetch and extract YouTube URLs from page, update cache
fn fetch_page_urls(page_url: &str, cache: &mut Cache) -> Result<Vec<String>> {
    // Check cache first
    if let Some(urls) = cache.get_page_urls(page_url) {
        tracing::info!(count = urls.len(), "using cached youtube urls");
        return Ok(urls.to_vec());
    }

    tracing::info!(url = %page_url, "fetching page");
    let html = melops_web::fetch_page(page_url)?;

    tracing::info!("extracting youtube urls");
    let urls = melops_web::extract_youtube_urls(&html)?;

    if urls.is_empty() {
        return Err(eyre!("no youtube urls found on page: {}", page_url));
    }

    tracing::info!(count = urls.len(), "found youtube urls");

    let mut resolved_urls = Vec::with_capacity(urls.len());
    for url in &urls {
        tracing::debug!(url = %url, "resolving url");
        let resolved = melops_web::resolve_url(url)?;
        resolved_urls.push(resolved);
    }

    // Update cache
    cache.set_page_urls(page_url.to_string(), resolved_urls.clone());

    Ok(resolved_urls)
}

/// Entry point for web command
pub async fn run(command: WebCommand) -> Result<()> {
    // Validate command into configs
    let web_config = WebConfig {
        page_url: command.url,
        output_dir: command.download_args.output_dir,
    };

    let model_config = ModelConfig::try_from(command.model_args)?;

    // Load cache (application state)
    let cache_config = CacheConfig::from(command.cache_args);
    let mut cache = Cache::load(cache_config)?;

    // Fetch YouTube URLs
    let youtube_urls = fetch_page_urls(&web_config.page_url, &mut cache)?;

    tracing::info!(count = youtube_urls.len(), "processing youtube urls");

    // If dry run, output URLs and exit
    if command.dry_run {
        for url in &youtube_urls {
            println!("{}", url);
        }
        return Ok(());
    }

    // Load model once for all videos
    let model = model_config.load()?;

    // Process each video
    for (i, youtube_url) in youtube_urls.iter().enumerate() {
        tracing::info!(
            current = i + 1,
            total = youtube_urls.len(),
            url = %youtube_url,
            "processing video"
        );

        // Download audio
        let download_config = DownloadConfig {
            url: youtube_url.clone(),
            output_dir: web_config.output_dir.clone(),
        };

        let audio_path = download(&download_config, &mut cache)?;

        // Generate captions
        let cap_config = CapConfig {
            path: audio_path.clone(),
            output: audio_path.with_extension("srt"),
            preview: command.caption_args.preview,
            chunk_config: command.caption_args.chunk_args.into(),
        };

        caption(&cap_config, &model, &mut cache).await?;
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
            download_args: DownloadArgs { output_dir: None },
            model_args: ModelArgs {
                model_id: "test_model".to_string(),
                model_source: ModelSource::Auto,
            },
            caption_args: CaptionArgs {
                preview: false,
                chunk_args: Default::default(),
            },
            cache_args: CacheArgs {
                cache_dir: None,
                refresh_pages: false,
                refresh_audio: false,
                refresh_srt: false,
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
