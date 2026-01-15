//! Core types for melops-asr

use eyre::{ContextCompat, Result, WrapErr};
use hf_hub::CacheRepo;
use hf_hub::api::sync::ApiRepo;
use std::path::PathBuf;

/// Token with timing information from decoder output.
#[derive(Clone, Debug)]
pub struct TokenDuration {
    /// Token ID (not blank)
    pub token_id: usize,
    /// Encoder frame index where token was emitted
    pub frame_index: usize,
    /// Duration prediction (number of frames to skip)
    pub duration: usize,
}

impl TokenDuration {
    pub fn new(token_id: usize, frame_index: usize, duration: usize) -> Self {
        Self {
            token_id,
            frame_index,
            duration,
        }
    }
}

/// Text segment with timestamps.
///
/// Represents a portion of transcribed text with start and end times in seconds.
#[derive(Clone, Debug)]
pub struct Segment {
    /// Transcribed text
    pub text: String,
    /// Start time in seconds
    pub start: f32,
    /// End time in seconds
    pub end: f32,
}

impl Segment {
    /// Create a new segment
    pub fn new(text: impl Into<String>, start: f32, end: f32) -> Self {
        Self {
            text: text.into(),
            start,
            end,
        }
    }
}

/// Model repository sources.
#[derive(Debug)]
pub enum ModelRepo {
    /// Local filesystem path
    Path(PathBuf),
    /// HuggingFace cache repository
    Cache(CacheRepo),
    /// HuggingFace API repository
    Api(ApiRepo),
}

impl ModelRepo {
    /// Resolve a file name to its full path in this repository.
    pub fn resolve(&self, file_name: &str) -> Result<PathBuf> {
        match self {
            ModelRepo::Path(path) => Ok(path.join(file_name)),
            ModelRepo::Cache(cache_repo) => cache_repo
                .get(file_name)
                .wrap_err(format!("failed to download from cache: {file_name}")),
            ModelRepo::Api(api_repo) => api_repo
                .get(file_name)
                .wrap_err(format!("failed to download from api: {file_name}")),
        }
    }

    /// Try resolving multiple file names, return first successful match.
    pub fn resolve_any(&self, candidates: &[&str]) -> Result<PathBuf> {
        use eyre::OptionExt;
        candidates
            .iter()
            .find_map(|name| self.resolve(name).ok())
            .ok_or_eyre("no model found from candidates")
    }
}
