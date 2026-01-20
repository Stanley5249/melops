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
/// **Design Decision**: Only `id` is kept as a required field because:
/// - Even `title` is not guaranteed (some extractors or deleted/private videos may not provide it)
/// - All other fields depend on content type, extractor capabilities, and availability
/// - Keeping minimal struct reduces parsing failures and complexity
/// - Users can access full metadata via the `.info.json` file if needed
///
/// Fields like `title`, `extractor_key`, `uploader`, etc. can be accessed from the
/// `.info.json` file that yt-dlp generates alongside the downloaded media.
///
/// See: <https://github.com/yt-dlp/yt-dlp#output-template>
#[derive(Clone, Debug, FromPyObject)]
#[pyo3(from_item_all)]
pub struct DownloadInfo {
    /// Video ID (platform-specific).
    ///
    /// This is the only guaranteed field across all extractors and content types.
    /// Used for file naming and identifying the downloaded content.
    pub id: String,
}
