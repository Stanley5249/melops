//! yt-dlp metadata output types.
//!
//! Metadata extracted from yt-dlp info dict. Fields are mostly optional because:
//! - URLs can resolve to videos, playlists, channels, or live streams
//! - Different extractors (YouTube, Vimeo, etc.) provide different metadata
//! - Privacy settings or platform limitations may hide certain fields
//!
//! See: <https://github.com/yt-dlp/yt-dlp#output-template>

use pyo3::prelude::*;

/// Essential metadata from yt-dlp info dict.
///
/// Subset of fields from `YoutubeDL.sanitize_info()`. Full dict available via JSON.
///
/// **Note**: Only `id` and `title` are guaranteed. All other fields may be `None` depending on:
/// - Content type (video vs playlist vs channel vs live stream)
/// - Extractor capabilities (different platforms provide different metadata)
/// - Content availability (deleted videos, private videos, etc.)
///
/// See: <https://github.com/yt-dlp/yt-dlp#output-template>
#[derive(Clone, Debug, FromPyObject)]
#[pyo3(from_item_all)]
pub struct DownloadInfo {
    /// Video ID (platform-specific, required)
    pub id: String,
    /// Video title (required)
    pub title: String,
    /// Extractor name (e.g., `Youtube`, `Vimeo`)
    pub extractor_key: Option<String>,
    /// Uploader full name
    pub uploader: Option<String>,
    /// Uploader username or channel ID
    pub uploader_id: Option<String>,
    /// Duration in seconds
    pub duration: Option<f64>,
    /// Video webpage URL
    pub webpage_url: Option<String>,
    /// Video description text
    pub description: Option<String>,
    /// Upload date in UTC (`YYYYMMDD`)
    pub upload_date: Option<String>,
    /// View count
    pub view_count: Option<i64>,
    /// Number of likes
    pub like_count: Option<i64>,
    /// Age restriction (`0` = none)
    pub age_limit: Option<i64>,
}
