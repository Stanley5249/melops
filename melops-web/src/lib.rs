//! Web scraping and URL extraction for melops
//!
//! This crate provides utilities for fetching web pages, extracting YouTube URLs,
//! and resolving URLs to canonical form.

mod error;
mod extract;
mod fetch;
mod resolve;

pub use error::{Error, Result};
pub use extract::extract_youtube_urls;
pub use fetch::fetch_page;
pub use resolve::resolve_url;
