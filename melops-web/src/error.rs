//! Error types for melops-web operations

use thiserror::Error;

/// Error types for melops-web operations
#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    Http(#[from] reqwest::Error),

    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// Result type for melops-web operations
pub type Result<T> = std::result::Result<T, Error>;
